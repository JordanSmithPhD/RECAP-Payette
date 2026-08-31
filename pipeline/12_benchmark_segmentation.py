#!/usr/bin/env python3
"""Run a common-split, post-hoc segmentation architecture benchmark.

This stage is intentionally separate from the historical reconstruction.  It
uses the 561 surviving positive-only chips, discards their fourth (alpha)
channel, and compares three standard architectures under the same training
contract.  The resulting metrics are a reviewer-requested architecture
comparison; they are not a replacement for an independent replication test.

The test partition is evaluated only after a checkpoint has been selected by
validation parking-class mean IoU.  Interrupted runs can be continued with
``--resume`` only when the immutable configuration, source-code hashes, input
checksums, package versions, and split bytes still match.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


WORKING_ROOT = Path(__file__).resolve().parents[1]
if str(WORKING_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKING_ROOT))

import project_paths
from pipeline.parking_common import (
    CLASS_NAMES,
    ContractError,
    build_spatial_partitions,
    parse_chip_position,
    sha256_file,
    validate_partition_rows,
)


MODEL_BUILDERS = {
    "deeplabv3plus_resnet34": "DeepLabV3Plus",
    "unet_resnet34": "Unet",
    "fpn_resnet34": "FPN",
}
CLASS_WEIGHTS = (1.0, 50.0, 50.0, 10.0, 10.0, 10.0)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PARKING_CLASS_IDS = (1, 2)
SPLIT_FIELDNAMES = (
    "chip_id",
    "column",
    "row",
    "spatial_group",
    "partition",
    "classes",
    "valid_pixels",
    "image_path",
    "mask_path",
    "image_sha256",
    "mask_sha256",
)
HISTORY_FIELDNAMES = (
    "epoch",
    "train_loss",
    "validation_mean_iou",
    "validation_parking_mean_iou",
    *(f"validation_iou_{name}" for name in CLASS_NAMES),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chips-root",
        type=Path,
        default=project_paths.SEGMENTATION_CHIP_ROOT,
        help="Surviving legacy chip root containing images/ and masks/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_paths.RESULTS_ROOT / "reviewer_benchmarks" / "segmentation",
    )
    parser.add_argument("--expected-chip-count", type=int, default=561)
    parser.add_argument("--split-seed", type=int, default=9401)
    parser.add_argument("--spatial-group-pixels", type=int, default=4096)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20261244, help="Training and augmentation seed.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_BUILDERS),
        default=list(MODEL_BUILDERS),
        help="Architectures to run. The default is the complete three-model comparison.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue only a checksum-identical interrupted run; completed test metrics are never recomputed.",
    )
    return parser


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_save(torch: Any, path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("numpy", "torch", "torchvision", "segmentation-models-pytorch"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _git_state() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=WORKING_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=WORKING_ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout.strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"revision": None, "dirty": None}


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def legacy_unit_rgb(image: Any, *, chip_id: str = "<array>") -> tuple[Any, Any]:
    """Return unit-scaled RGB plus validity decoded from the alpha band.

    This helper deliberately does not divide by 255.  The recovered Earth
    Engine RGB values are already scaled to [0, 1], while channel four is the
    GDAL alpha channel with values 0 (invalid/padding) and 255 (valid).
    """

    import numpy as np

    values = np.asarray(image)
    if values.ndim != 3 or values.shape[-1] != 4:
        raise ContractError(f"Legacy chip {chip_id} must be HWC with exactly four channels; found {values.shape}.")
    rgb = values[..., :3]
    alpha = values[..., 3]
    if not bool(np.isfinite(alpha).all()):
        raise ContractError(f"Legacy alpha for {chip_id} contains non-finite values.")
    alpha_is_zero = np.isclose(alpha, 0.0, rtol=0.0, atol=1e-6)
    alpha_is_valid = np.isclose(alpha, 255.0, rtol=0.0, atol=1e-6)
    if not bool((alpha_is_zero | alpha_is_valid).all()):
        observed = np.unique(alpha)
        raise ContractError(
            f"Legacy channel four for {chip_id} must be binary alpha {{0,255}}; "
            f"observed sample {observed[:10].tolist()}."
        )
    if not bool(alpha_is_valid.any()):
        raise ContractError(f"Legacy chip {chip_id} has no alpha-valid pixels.")
    valid_rgb = rgb[alpha_is_valid]
    if not bool(np.isfinite(valid_rgb).all()):
        raise ContractError(f"Legacy RGB for {chip_id} contains non-finite values in alpha-valid pixels.")
    lower, upper = float(valid_rgb.min()), float(valid_rgb.max())
    if lower < 0.0 or upper > 1.01:
        raise ContractError(
            f"Legacy RGB for {chip_id} must already be unit-scaled; observed valid range [{lower}, {upper}]."
        )
    return rgb.astype(np.float32, copy=False), alpha_is_valid


def deterministic_augment(
    image: Any,
    mask: Any,
    valid: Any,
    *,
    chip_id: str,
    epoch: int,
    seed: int,
) -> tuple[Any, Any, Any]:
    """Apply paired flips and a quarter-turn using only a stable sample key."""

    import numpy as np

    digest = hashlib.sha256(f"{seed}:{epoch}:{chip_id}".encode("utf-8")).digest()
    result_image, result_mask, result_valid = image, mask, valid
    if digest[0] & 1:
        result_image = np.flip(result_image, axis=1)
        result_mask = np.flip(result_mask, axis=1)
        result_valid = np.flip(result_valid, axis=1)
    if digest[1] & 1:
        result_image = np.flip(result_image, axis=0)
        result_mask = np.flip(result_mask, axis=0)
        result_valid = np.flip(result_valid, axis=0)
    turns = digest[2] % 4
    if turns:
        result_image = np.rot90(result_image, k=turns, axes=(0, 1))
        result_mask = np.rot90(result_mask, k=turns, axes=(0, 1))
        result_valid = np.rot90(result_valid, k=turns, axes=(0, 1))
    return (
        np.ascontiguousarray(result_image),
        np.ascontiguousarray(result_mask),
        np.ascontiguousarray(result_valid),
    )


def discover_inventory(chips_root: Path, *, expected_count: int) -> list[tuple[str, Path, Path]]:
    image_dir, mask_dir = chips_root / "images", chips_root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ and masks/ beneath {chips_root}.")
    image_paths = {path.stem: path for path in image_dir.glob("*.npy") if not path.name.startswith("._")}
    mask_paths = {path.stem: path for path in mask_dir.glob("*.npy") if not path.name.startswith("._")}
    if image_paths.keys() != mask_paths.keys():
        raise ContractError(
            "Image/mask inventories differ "
            f"(images-only={len(image_paths.keys() - mask_paths.keys())}, "
            f"masks-only={len(mask_paths.keys() - image_paths.keys())})."
        )
    if not image_paths:
        raise ContractError(f"No NumPy chips found beneath {chips_root}.")
    if expected_count <= 0:
        raise ContractError("expected-chip-count must be positive.")
    if len(image_paths) != expected_count:
        raise ContractError(
            f"Expected the {expected_count}-chip frozen legacy corpus; found {len(image_paths)} paired chips."
        )
    return [(chip_id, image_paths[chip_id], mask_paths[chip_id]) for chip_id in sorted(image_paths)]


def build_split_rows(
    inventory: Sequence[tuple[str, Path, Path]],
    *,
    chips_root: Path,
    split_seed: int,
    group_pixels: int,
    train_fraction: float,
    validation_fraction: float,
    validate_images: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate/hash the frozen chips and create deterministic group assignments."""

    import numpy as np

    assignments = build_spatial_partitions(
        (chip_id for chip_id, _, _ in inventory),
        seed=split_seed,
        group_pixels=group_pixels,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    rows: list[dict[str, Any]] = []
    class_pixels = {partition: Counter() for partition in ("train", "validation", "test")}
    class_positive_chips = {partition: Counter() for partition in ("train", "validation", "test")}
    valid_pixel_counts = Counter()
    total_pixel_counts = Counter()
    groups = {partition: set() for partition in ("train", "validation", "test")}
    dataset_digest = hashlib.sha256()
    for chip_id, image_path, mask_path in inventory:
        column, row = parse_chip_position(chip_id)
        image_hash, mask_hash = sha256_file(image_path), sha256_file(mask_path)
        image = np.load(image_path, allow_pickle=False, mmap_mode="r")
        mask = np.load(mask_path, allow_pickle=False, mmap_mode="r")
        if image.shape != (512, 512, 4) or mask.shape != (512, 512):
            raise ContractError(f"Invalid image/mask shapes for {chip_id}: {image.shape}/{mask.shape}.")
        rgb, valid = legacy_unit_rgb(image, chip_id=chip_id)
        if validate_images:
            # Force the lazy memory map to verify all RGB values while retaining
            # alpha-derived validity for loss and metric masking.
            del rgb
        all_classes = np.unique(mask)
        if all_classes.size == 0 or int(all_classes.min()) < 0 or int(all_classes.max()) >= len(CLASS_NAMES):
            raise ContractError(f"Mask {chip_id} contains invalid class IDs: {all_classes.tolist()}.")
        unique, counts = np.unique(mask[valid], return_counts=True)
        spatial_group, partition = assignments[chip_id]
        groups[partition].add(spatial_group)
        valid_pixels = int(valid.sum())
        valid_pixel_counts[partition] += valid_pixels
        total_pixel_counts[partition] += int(mask.size)
        for class_id, count in zip(unique, counts, strict=True):
            class_pixels[partition][int(class_id)] += int(count)
            if int(class_id) > 0:
                class_positive_chips[partition][int(class_id)] += 1
        image_relative = image_path.relative_to(chips_root).as_posix()
        mask_relative = mask_path.relative_to(chips_root).as_posix()
        row_payload = {
            "chip_id": chip_id,
            "column": column,
            "row": row,
            "spatial_group": spatial_group,
            "partition": partition,
            "classes": ",".join(str(int(value)) for value in unique if int(value) > 0),
            "valid_pixels": valid_pixels,
            "image_path": image_relative,
            "mask_path": mask_relative,
            "image_sha256": image_hash,
            "mask_sha256": mask_hash,
        }
        rows.append(row_payload)
        dataset_digest.update(_canonical_json_bytes(row_payload))
        dataset_digest.update(b"\n")

    validate_partition_rows(rows, corrected=True)
    for partition in ("train", "validation", "test"):
        missing_parking = [class_id for class_id in PARKING_CLASS_IDS if class_pixels[partition][class_id] == 0]
        if missing_parking:
            raise ContractError(
                f"The frozen split has no pixels for parking classes {missing_parking} in {partition}; "
                "choose and document a different split seed/group size before training."
            )
    stats = {
        "dataset_sha256": dataset_digest.hexdigest(),
        "chips": {partition: sum(row["partition"] == partition for row in rows) for partition in groups},
        "spatial_groups": {partition: len(groups[partition]) for partition in groups},
        "valid_pixels": {partition: int(valid_pixel_counts[partition]) for partition in groups},
        "invalid_pixels": {
            partition: int(total_pixel_counts[partition] - valid_pixel_counts[partition]) for partition in groups
        },
        "class_pixels": {
            partition: {str(class_id): int(class_pixels[partition][class_id]) for class_id in range(len(CLASS_NAMES))}
            for partition in class_pixels
        },
        "positive_chips": {
            partition: {
                str(class_id): int(class_positive_chips[partition][class_id]) for class_id in range(len(CLASS_NAMES))
            }
            for partition in class_positive_chips
        },
    }
    return rows, stats


def render_split_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SPLIT_FIELDNAMES, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def metrics_from_confusion(matrix: Any) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(matrix, dtype=np.int64)
    expected = (len(CLASS_NAMES), len(CLASS_NAMES))
    if values.shape != expected or bool((values < 0).any()):
        raise ContractError(f"Confusion matrix must be nonnegative with shape {expected}; found {values.shape}.")
    intersection = np.diag(values).astype(np.float64)
    union = values.sum(axis=1) + values.sum(axis=0) - intersection
    iou = np.divide(intersection, union, out=np.full(len(CLASS_NAMES), np.nan), where=union > 0)
    parking_values = iou[list(PARKING_CLASS_IDS)]
    if not bool(np.isfinite(parking_values).all()):
        raise ContractError("Both parking classes must be scorable for checkpoint selection and test evaluation.")
    return {
        "mean_iou": float(np.nanmean(iou)),
        "parking_mean_iou": float(parking_values.mean()),
        "per_class_iou": {name: float(value) for name, value in zip(CLASS_NAMES, iou, strict=True)},
    }


def choose_device(torch: Any, requested: str) -> Any:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if requested == "mps" and not mps_available:
        raise RuntimeError("MPS was requested but is unavailable.")
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available:
        return torch.device("mps")
    return torch.device("cpu")


def configure_randomness(torch: Any, np: Any, seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_dataset_class(torch: Any, np: Any) -> type:
    class LegacyChipDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            chips_root: Path,
            rows: Sequence[Mapping[str, Any]],
            *,
            augment: bool,
            seed: int,
        ) -> None:
            self.chips_root = chips_root
            self.rows = list(rows)
            self.augment = augment
            self.seed = seed
            self.epoch = 0
            self.mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
            self.std = np.asarray(IMAGENET_STD, dtype=np.float32)

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
            row = self.rows[index]
            chip_id = str(row["chip_id"])
            image = np.load(self.chips_root / str(row["image_path"]), allow_pickle=False)
            mask = np.load(self.chips_root / str(row["mask_path"]), allow_pickle=False)
            image, valid = legacy_unit_rgb(image, chip_id=chip_id)
            if self.augment:
                image, mask, valid = deterministic_augment(
                    image,
                    mask,
                    valid,
                    chip_id=chip_id,
                    epoch=self.epoch,
                    seed=self.seed,
                )
            image = image.copy()
            image[~valid] = 0.0
            image = (image - self.mean) / self.std
            image[~valid] = 0.0
            image = np.ascontiguousarray(np.moveaxis(image, -1, 0), dtype=np.float32)
            mask = np.ascontiguousarray(mask, dtype=np.int64)
            valid = np.ascontiguousarray(valid, dtype=bool)
            return torch.from_numpy(image), torch.from_numpy(mask), torch.from_numpy(valid)

    return LegacyChipDataset


def build_model(smp: Any, model_name: str) -> Any:
    builder = getattr(smp, MODEL_BUILDERS[model_name])
    return builder(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=len(CLASS_NAMES),
    )


def state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


def confusion_from_loader(model: Any, loader: Any, *, torch: Any, np: Any, device: Any) -> Any:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for images, masks, valid_masks in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu().numpy()
            truth = masks.numpy()
            valid = valid_masks.numpy().astype(bool)
            encoded = truth[valid] * len(CLASS_NAMES) + predictions[valid]
            matrix += np.bincount(encoded, minlength=len(CLASS_NAMES) ** 2).reshape(matrix.shape)
    return matrix


def _history_row(epoch: int, train_loss: float, metrics: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_mean_iou": metrics["mean_iou"],
        "validation_parking_mean_iou": metrics["parking_mean_iou"],
    }
    row.update(
        {
            f"validation_iou_{name}": metrics["per_class_iou"][name]
            for name in CLASS_NAMES
        }
    )
    return row


def _load_checkpoint(torch: Any, path: Path, device: Any) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch versions before weights_only was introduced
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ContractError(f"Checkpoint is not a mapping: {path}.")
    return payload


def _summary_row_from_metrics(model_name: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "model": model_name,
        "encoder": "resnet34",
        "encoder_initialization": "imagenet",
        "parameters": metrics["parameters"],
        "best_epoch": metrics["best_epoch"],
        "validation_parking_mean_iou": metrics["validation_parking_mean_iou"],
        "test_mean_iou": metrics["test"]["mean_iou"],
        "test_parking_mean_iou": metrics["test"]["parking_mean_iou"],
        "training_runtime_seconds": metrics["training_runtime_seconds"],
        "test_runtime_seconds": metrics["test_runtime_seconds"],
        "best_weights_sha256": metrics["best_weights_sha256"],
    }
    row.update({f"test_iou_{name}": metrics["test"]["per_class_iou"][name] for name in CLASS_NAMES})
    return row


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"Expected a JSON object: {path}.")
    return payload


def _initialize_or_validate_run(
    *,
    output_dir: Path,
    resume: bool,
    config: Mapping[str, Any],
    split_text: str,
) -> str:
    config_path, split_path = output_dir / "config.json", output_dir / "split.csv"
    if output_dir.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite existing benchmark directory: {output_dir}")
    if not output_dir.exists() and resume:
        raise FileNotFoundError(f"Cannot resume because the benchmark directory does not exist: {output_dir}")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        _atomic_write_text(split_path, split_text)
        _atomic_write_json(config_path, config)
    else:
        if not config_path.is_file() or not split_path.is_file():
            raise ContractError("Unsafe resume: config.json and split.csv must both exist.")
        observed_config = _read_json(config_path)
        if _canonical_json_bytes(observed_config) != _canonical_json_bytes(config):
            raise ContractError("Unsafe resume: immutable configuration, inputs, code, or package versions changed.")
        if split_path.read_text(encoding="utf-8") != split_text:
            raise ContractError("Unsafe resume: split.csv bytes changed.")
    return sha256_file(config_path)


def _train_and_test_model(
    *,
    model_name: str,
    args: argparse.Namespace,
    rows_by_partition: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
    config_sha256: str,
    reference_encoder_hash: str | None,
    torch: Any,
    np: Any,
    smp: Any,
    device: Any,
) -> tuple[dict[str, Any], str]:
    model_dir = output_dir / "models" / model_name
    history_path = model_dir / "history.csv"
    best_path = model_dir / "best.pth"
    last_path = model_dir / "last.pth"
    test_metrics_path = model_dir / "test_metrics.json"
    confusion_path = model_dir / "test_confusion.csv"
    model_dir.mkdir(parents=True, exist_ok=True)

    if test_metrics_path.is_file():
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite completed model result: {test_metrics_path}")
        metrics = _read_json(test_metrics_path)
        if metrics.get("config_sha256") != config_sha256:
            raise ContractError(f"Unsafe resume: {model_name} test metrics use another configuration.")
        if not best_path.is_file() or sha256_file(best_path) != metrics.get("best_weights_sha256"):
            raise ContractError(f"Unsafe resume: {model_name} best weights changed after test evaluation.")
        encoder_hash = str(metrics["encoder_initialization_sha256"])
        if reference_encoder_hash is not None and encoder_hash != reference_encoder_hash:
            raise ContractError("Models do not share identical ResNet34 ImageNet encoder initialization.")
        return metrics, encoder_hash

    configure_randomness(torch, np, args.seed)
    model = build_model(smp, model_name)
    encoder_hash = state_dict_sha256(model.encoder.state_dict())
    if reference_encoder_hash is not None and encoder_hash != reference_encoder_hash:
        raise ContractError("Models do not share identical ResNet34 ImageNet encoder initialization.")
    model = model.to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=device)

    Dataset = build_dataset_class(torch, np)
    train_dataset = Dataset(args.chips_root, rows_by_partition["train"], augment=True, seed=args.seed)
    validation_dataset = Dataset(args.chips_root, rows_by_partition["validation"], augment=False, seed=args.seed)
    test_dataset = Dataset(args.chips_root, rows_by_partition["test"], augment=False, seed=args.seed)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_epoch = 0
    elapsed_before_resume = 0.0
    if last_path.is_file():
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite interrupted checkpoint: {last_path}")
        checkpoint = _load_checkpoint(torch, last_path, device)
        metadata = checkpoint.get("metadata", {})
        if metadata.get("config_sha256") != config_sha256 or metadata.get("model") != model_name:
            raise ContractError(f"Unsafe resume: {model_name} checkpoint configuration does not match.")
        if metadata.get("encoder_initialization_sha256") != encoder_hash:
            raise ContractError(f"Unsafe resume: {model_name} pretrained encoder initialization changed.")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint["history"])
        completed_epoch = int(checkpoint["completed_epoch"])
        if len(history) != completed_epoch:
            raise ContractError(f"Unsafe resume: {model_name} history/checkpoint epoch counts differ.")
        start_epoch = completed_epoch + 1
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        elapsed_before_resume = float(checkpoint.get("training_runtime_seconds", 0.0))

    session_started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(args.seed + epoch)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers,
            generator=generator,
        )
        model.train()
        loss_numerator_total = 0.0
        loss_denominator_total = 0.0
        for images, masks, valid_masks in train_loader:
            images, masks, valid_masks = images.to(device), masks.to(device), valid_masks.to(device).bool()
            optimizer.zero_grad(set_to_none=True)
            pixel_losses = torch.nn.functional.cross_entropy(
                model(images),
                masks,
                weight=class_weights,
                reduction="none",
            )
            if not bool(valid_masks.any().item()):
                raise ContractError(f"{model_name} received a training batch with no alpha-valid pixels.")
            numerator = pixel_losses[valid_masks].sum()
            denominator = class_weights[masks[valid_masks]].sum()
            loss = numerator / denominator
            loss.backward()
            optimizer.step()
            loss_numerator_total += float(numerator.detach().cpu().item())
            loss_denominator_total += float(denominator.detach().cpu().item())
        validation_confusion = confusion_from_loader(
            model,
            validation_loader,
            torch=torch,
            np=np,
            device=device,
        )
        validation_metrics = metrics_from_confusion(validation_confusion)
        row = _history_row(epoch, loss_numerator_total / loss_denominator_total, validation_metrics)
        history.append(row)
        score = float(validation_metrics["parking_mean_iou"])
        if score > best_score:
            best_score, best_epoch = score, epoch
            _atomic_torch_save(
                torch,
                best_path,
                {
                    "state_dict": model.state_dict(),
                    "metadata": {
                        "schema_version": 1,
                        "contract": "post-hoc-common-split-segmentation-benchmark",
                        "model": model_name,
                        "encoder": "resnet34",
                        "encoder_weights": "imagenet",
                        "input_channels": 3,
                        "classes": list(CLASS_NAMES),
                        "config_sha256": config_sha256,
                        "encoder_initialization_sha256": encoder_hash,
                        "selection_partition": "validation",
                        "selection_metric": "parking_mean_iou",
                        "selection_epoch": best_epoch,
                        "selection_value": best_score,
                    },
                },
            )
        elapsed = elapsed_before_resume + (time.perf_counter() - session_started)
        _atomic_torch_save(
            torch,
            last_path,
            {
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "completed_epoch": epoch,
                "history": history,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "training_runtime_seconds": elapsed,
                "metadata": {
                    "config_sha256": config_sha256,
                    "model": model_name,
                    "encoder_initialization_sha256": encoder_hash,
                },
            },
        )
        _atomic_write_csv(history_path, HISTORY_FIELDNAMES, history)
        print(
            f"model={model_name} epoch={epoch}/{args.epochs} train_loss={row['train_loss']:.6f} "
            f"validation_parking_mean_iou={score:.6f}",
            flush=True,
        )

    if not last_path.is_file() or not best_path.is_file():
        raise ContractError(f"No complete training checkpoint exists for {model_name}.")
    checkpoint = _load_checkpoint(torch, last_path, device)
    if int(checkpoint["completed_epoch"]) != args.epochs:
        raise ContractError(f"{model_name} stopped before the configured final epoch.")
    history = list(checkpoint["history"])
    best_epoch = int(checkpoint["best_epoch"])
    best_score = float(checkpoint["best_score"])
    training_runtime = float(checkpoint.get("training_runtime_seconds", 0.0))
    best = _load_checkpoint(torch, best_path, device)
    model.load_state_dict(best["state_dict"])

    # This is the only test-loader construction and evaluation in the stage.
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_started = time.perf_counter()
    test_confusion = confusion_from_loader(model, test_loader, torch=torch, np=np, device=device)
    test_runtime = time.perf_counter() - test_started
    test_metrics = metrics_from_confusion(test_confusion)
    best_hash = sha256_file(best_path)
    metrics = {
        "schema_version": 1,
        "contract": "post-hoc-common-split-segmentation-benchmark",
        "model": model_name,
        "encoder": "resnet34",
        "encoder_weights": "imagenet",
        "parameters": parameters,
        "config_sha256": config_sha256,
        "encoder_initialization_sha256": encoder_hash,
        "best_weights_sha256": best_hash,
        "best_epoch": best_epoch,
        "validation_parking_mean_iou": best_score,
        "training_runtime_seconds": training_runtime,
        "test_runtime_seconds": test_runtime,
        "test_partition_evaluations": 1,
        "test": test_metrics,
        "test_confusion": test_confusion.tolist(),
    }
    confusion_rows = [
        {"truth_class": name, **{predicted: int(test_confusion[i, j]) for j, predicted in enumerate(CLASS_NAMES)}}
        for i, name in enumerate(CLASS_NAMES)
    ]
    _atomic_write_csv(confusion_path, ("truth_class", *CLASS_NAMES), confusion_rows)
    _atomic_write_json(test_metrics_path, metrics)
    return metrics, encoder_hash


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ContractError("epochs, batch-size, and learning-rate must be positive.")
    if args.num_workers < 0:
        raise ContractError("num-workers cannot be negative.")
    if args.spatial_group_pixels <= 0:
        raise ContractError("spatial-group-pixels must be positive.")
    if not 0.0 < args.train_fraction < 1.0:
        raise ContractError("train-fraction must lie strictly between zero and one.")
    if not 0.0 <= args.validation_fraction < 1.0 or args.train_fraction + args.validation_fraction >= 1.0:
        raise ContractError("validation-fraction must leave a positive test fraction.")
    if len(set(args.models)) != len(args.models):
        raise ContractError("models cannot contain duplicates.")


def run(args: argparse.Namespace) -> int:
    _validate_args(args)
    args.chips_root = args.chips_root.expanduser().resolve(strict=False)
    args.output_dir = args.output_dir.expanduser().resolve(strict=False)
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing benchmark directory: {args.output_dir}")

    inventory = discover_inventory(args.chips_root, expected_count=args.expected_chip_count)
    rows, split_stats = build_split_rows(
        inventory,
        chips_root=args.chips_root,
        split_seed=args.split_seed,
        group_pixels=args.spatial_group_pixels,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    split_text = render_split_csv(rows)
    script_path = Path(__file__).resolve()
    common_path = script_path.parent / "parking_common.py"
    versions = _package_versions()
    missing_packages = [name for name, version in versions.items() if version is None]
    if missing_packages:
        raise RuntimeError(f"Benchmark environment is missing packages: {missing_packages}.")
    try:
        import numpy as np
        import segmentation_models_pytorch as smp
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("Benchmark requires NumPy, Torch, torchvision, and segmentation-models-pytorch.") from exc
    device = choose_device(torch, args.device)
    config = {
        "schema_version": 1,
        "contract": "post-hoc-common-split-segmentation-benchmark",
        "scientific_scope": (
            "Architecture comparison on the surviving positive-only legacy corpus; "
            "not an independent replication test."
        ),
        "chips_root": str(args.chips_root),
        "expected_chip_count": args.expected_chip_count,
        "dataset_sha256": split_stats["dataset_sha256"],
        "split_csv_sha256": hashlib.sha256(split_text.encode("utf-8")).hexdigest(),
        "split": {
            "seed": args.split_seed,
            "spatial_group_pixels": args.spatial_group_pixels,
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "test_fraction": 1.0 - args.train_fraction - args.validation_fraction,
        },
        "input": {
            "channels": "RGB only; legacy alpha channel stripped",
            "radiometry": "RGB already unit-scaled; no division by 255",
            "normalization": "ImageNet mean/std after unit-scale validation",
        },
        "models": list(args.models),
        "encoder": "resnet34",
        "encoder_weights": "imagenet",
        "classes": list(CLASS_NAMES),
        "class_weights": list(CLASS_WEIGHTS),
        "selection_partition": "validation",
        "selection_metric": "parking_mean_iou(classes=1,2)",
        "test_policy": "evaluate each selected checkpoint once after all model selection",
        "augmentation": "deterministic horizontal flip, vertical flip, and rotations in {0,90,180,270}",
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "training_seed": args.seed,
        "device_requested": args.device,
        "device_effective": str(device),
        "deterministic_algorithms": True,
        "code_sha256": {
            script_path.name: sha256_file(script_path),
            common_path.name: sha256_file(common_path),
        },
        "python": {"version": sys.version, "platform": platform.platform()},
        "packages": versions,
    }
    config_sha256 = _initialize_or_validate_run(
        output_dir=args.output_dir,
        resume=args.resume,
        config=config,
        split_text=split_text,
    )
    rows_by_partition = {
        partition: [row for row in rows if row["partition"] == partition]
        for partition in ("train", "validation", "test")
    }

    summary_rows: list[dict[str, Any]] = []
    model_provenance: dict[str, Any] = {}
    reference_encoder_hash: str | None = None
    benchmark_started = time.perf_counter()
    for model_name in args.models:
        metrics, encoder_hash = _train_and_test_model(
            model_name=model_name,
            args=args,
            rows_by_partition=rows_by_partition,
            output_dir=args.output_dir,
            config_sha256=config_sha256,
            reference_encoder_hash=reference_encoder_hash,
            torch=torch,
            np=np,
            smp=smp,
            device=device,
        )
        reference_encoder_hash = reference_encoder_hash or encoder_hash
        summary_rows.append(_summary_row_from_metrics(model_name, metrics))
        model_provenance[model_name] = {
            "encoder_initialization_sha256": encoder_hash,
            "best_weights_sha256": metrics["best_weights_sha256"],
            "history_sha256": sha256_file(args.output_dir / "models" / model_name / "history.csv"),
            "test_metrics_sha256": sha256_file(args.output_dir / "models" / model_name / "test_metrics.json"),
            "test_confusion_sha256": sha256_file(args.output_dir / "models" / model_name / "test_confusion.csv"),
        }

    summary_fields = (
        "model",
        "encoder",
        "encoder_initialization",
        "parameters",
        "best_epoch",
        "validation_parking_mean_iou",
        "test_mean_iou",
        "test_parking_mean_iou",
        *(f"test_iou_{name}" for name in CLASS_NAMES),
        "training_runtime_seconds",
        "test_runtime_seconds",
        "best_weights_sha256",
    )
    summary_path = args.output_dir / "summary.csv"
    if summary_path.exists() and args.resume:
        import io

        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=summary_fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
        if summary_path.read_text(encoding="utf-8") != buffer.getvalue():
            raise ContractError("Unsafe resume: existing summary.csv differs from completed per-model metrics.")
    elif summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite {summary_path}.")
    else:
        _atomic_write_csv(summary_path, summary_fields, summary_rows)

    provenance_path = args.output_dir / "provenance.json"
    provenance = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "contract": "post-hoc-common-split-segmentation-benchmark",
        "config_sha256": config_sha256,
        "split_sha256": sha256_file(args.output_dir / "split.csv"),
        "summary_sha256": sha256_file(summary_path),
        "dataset_sha256": split_stats["dataset_sha256"],
        "code_sha256": config["code_sha256"],
        "packages": versions,
        "git": _git_state(),
        "split_statistics": split_stats,
        "shared_encoder_initialization_sha256": reference_encoder_hash,
        "models": model_provenance,
        "wall_runtime_seconds_this_invocation": time.perf_counter() - benchmark_started,
    }
    if provenance_path.exists():
        if not args.resume:
            raise FileExistsError(f"Refusing to overwrite {provenance_path}.")
        observed = _read_json(provenance_path)
        stable_keys = (
            "config_sha256",
            "split_sha256",
            "summary_sha256",
            "dataset_sha256",
            "code_sha256",
            "packages",
            "split_statistics",
            "shared_encoder_initialization_sha256",
            "models",
        )
        if any(observed.get(key) != _json_safe(provenance.get(key)) for key in stable_keys):
            raise ContractError("Unsafe resume: completed provenance differs from current artifacts.")
    else:
        _atomic_write_json(provenance_path, provenance)

    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "provenance": str(provenance_path),
                "models": list(args.models),
                "test_partition_evaluations_per_model": 1,
            },
            indent=2,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ContractError, FileExistsError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
