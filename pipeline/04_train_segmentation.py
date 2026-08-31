#!/usr/bin/env python3
"""Train one recovered DeepLabV3+ parking-segmentation variant.

Two deliberately different scientific contracts are supported:

``historical``
    Reconstructs the notebook's RGB-plus-alpha/double-division input and uses
    every surviving chip.  Any evaluation reported by this command is labeled
    ``resubstitution``; it is never described as validation or test evidence.

``corrected``
    Requires an immutable split manifest, trains on ``train`` only, evaluates
    model selection on ``validation`` only, and enables deterministic Torch
    settings.  The ``test`` partition is validated but never opened here.

Torch, segmentation-models-pytorch, Albumentations, NumPy, and Pandas are
imported only after argument validation so ``--help`` works in a lightweight
repository-audit environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


WORKING_ROOT = Path(__file__).resolve().parents[1]
if str(WORKING_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKING_ROOT))

import project_paths
from pipeline.parking_common import (
    CLASS_NAMES,
    BandObservation,
    ContractError,
    load_run_manifest,
    is_ignored_metadata_path,
    normalize_chip,
    refuse_existing_output,
    sha256_file,
    validate_band_contract,
    validate_partition_rows,
    verify_manifest_artifact,
    write_run_manifest,
)


VARIANTS: dict[str, dict[str, Any]] = {
    "v1_base": {"weighted": False, "augment": False, "parking_repeats": 0},
    "v2_weighted": {"weighted": True, "augment": False, "parking_repeats": 0},
    "v3_augmented": {"weighted": True, "augment": True, "parking_repeats": 20},
}
CLASS_WEIGHTS = (1.0, 50.0, 50.0, 10.0, 10.0, 10.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a recovered parking segmentation model under an explicit historical "
            "or corrected scientific contract."
        )
    )
    parser.add_argument("--contract", choices=("historical", "corrected"), required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument(
        "--chips-root",
        type=Path,
        default=None,
        help="Directory containing images/ and masks/ NumPy chips; defaults are contract-specific.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Parquet or CSV split manifest; defaults to the configured corrected split.",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help="Corrected stage-03 run manifest; defaults to <chips-root>/run_manifest.json.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help=(
            "New run directory. Corrected defaults to runs/segmentation/<parking-run-id>/<variant>; "
            "historical defaults to runs/segmentation/historical-<variant>."
        ),
    )
    parser.add_argument(
        "--band-contract",
        choices=("rgba-alpha", "rgb", "rgbnir"),
        help="Historical requires rgba-alpha; the corrected contract requires rgbnir.",
    )
    parser.add_argument(
        "--normalization",
        choices=("historical-double-divide", "unit-rgb", "unit-multiband"),
        help="Explicit radiometric transform; defaults are derived from --contract/--band-contract.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--allow-existing-run-dir",
        action="store_true",
        help="Allow an existing directory, but never overwrite any managed output within it.",
    )
    return parser


def resolve_default_paths(args: argparse.Namespace) -> argparse.Namespace:
    if args.contract == "corrected":
        args.chips_root = args.chips_root or project_paths.PARKING_CHIP_RUN_ROOT
        args.split_manifest = args.split_manifest or args.chips_root / "splits.parquet"
        args.dataset_manifest = args.dataset_manifest or args.chips_root / "run_manifest.json"
        args.run_dir = args.run_dir or project_paths.PARKING_MODEL_RUN_ROOT / args.variant
    else:
        args.chips_root = args.chips_root or project_paths.SEGMENTATION_CHIP_ROOT
        args.split_manifest = args.split_manifest or project_paths.SEGMENTATION_SPLITS
        args.run_dir = args.run_dir or (project_paths.RUNS_ROOT / "segmentation" / f"historical-{args.variant}")
    return args


def resolve_radiometry(args: argparse.Namespace) -> tuple[str, str, int]:
    if args.contract == "historical":
        band_contract = args.band_contract or "rgba-alpha"
        normalization = args.normalization or "historical-double-divide"
        if band_contract != "rgba-alpha" or normalization != "historical-double-divide":
            raise ContractError("Historical reconstruction is fixed to rgba-alpha plus historical-double-divide.")
        return band_contract, normalization, 4

    band_contract = args.band_contract or "rgbnir"
    if band_contract != "rgbnir":
        raise ContractError("The corrected contract is fixed to genuine ordered RGB+NIR imagery.")
    normalization = args.normalization or "unit-multiband"
    if normalization != "unit-multiband":
        raise ContractError("Corrected rgbnir requires --normalization unit-multiband.")
    return band_contract, normalization, 4


def _read_split_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("Reading the split manifest requires pandas and pyarrow for Parquet.") from exc

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        raise ContractError("Split manifest must be .parquet, .pq, or .csv.")
    required = {"chip_id", "partition"}
    missing = required.difference(frame.columns)
    if missing:
        raise ContractError(f"Split manifest is missing columns: {sorted(missing)}.")
    rows = frame.to_dict(orient="records")
    validate_partition_rows(rows, corrected=True)
    return rows


def discover_chip_ids(chips_root: Path, *, require_valid_masks: bool = False) -> list[str]:
    image_dir = chips_root / "images"
    mask_dir = chips_root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ and masks/ beneath {chips_root}.")
    image_ids = {
        path.stem for path in image_dir.glob("*.npy") if not is_ignored_metadata_path(path.relative_to(image_dir))
    }
    mask_ids = {
        path.stem for path in mask_dir.glob("*.npy") if not is_ignored_metadata_path(path.relative_to(mask_dir))
    }
    if image_ids != mask_ids:
        raise ContractError(
            f"Image/mask chip IDs differ (images-only={len(image_ids - mask_ids)}, "
            f"masks-only={len(mask_ids - image_ids)})."
        )
    if not image_ids:
        raise ContractError(f"No .npy chips found beneath {chips_root}.")
    if require_valid_masks:
        valid_dir = chips_root / "valid_masks"
        if not valid_dir.is_dir():
            raise FileNotFoundError(f"Corrected chips require valid_masks/ beneath {chips_root}.")
        valid_ids = {
            path.stem
            for path in valid_dir.glob("*.npy")
            if not is_ignored_metadata_path(path.relative_to(valid_dir))
        }
        if valid_ids != image_ids:
            raise ContractError(
                f"Image/valid-mask chip IDs differ (images-only={len(image_ids - valid_ids)}, "
                f"valid-only={len(valid_ids - image_ids)})."
            )
    return sorted(image_ids)


def validate_corrected_dataset_contract(payload: Mapping[str, Any], chips_root: Path) -> None:
    """Require checksum-bound valid masks and source-level ordered RGB+NIR identity."""

    verify_manifest_artifact(payload, chips_root / "valid_masks")
    extra = payload.get("extra", {})
    if extra.get("validity_contract") != "source-validity-plus-finite-radiometry":
        raise ContractError("Corrected dataset manifest lacks the source-validity mask contract.")
    raw = extra.get("band_observation")
    if not isinstance(raw, Mapping):
        raise ContractError("Corrected dataset manifest lacks source band observations.")
    try:
        observation = BandObservation(
            int(raw["channels"]),
            tuple(float(value) for value in raw["minima"]),
            tuple(float(value) for value in raw["maxima"]),
            tuple(str(value) for value in raw.get("color_interpretations", ())),
            tuple(str(value) for value in raw.get("band_names", ())),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("Corrected dataset band observations are malformed.") from exc
    validate_band_contract(observation, "rgbnir", require_identity=True)


def select_partitions(
    all_ids: Sequence[str], contract: str, split_rows: Sequence[Mapping[str, Any]] | None
) -> tuple[list[str], list[str], dict[str, int]]:
    if contract == "historical":
        return list(all_ids), list(all_ids), {"resubstitution": len(all_ids)}
    assert split_rows is not None
    known = set(all_ids)
    manifest_ids = {str(row["chip_id"]) for row in split_rows}
    missing_files = manifest_ids.difference(known)
    unassigned_files = known.difference(manifest_ids)
    if missing_files or unassigned_files:
        raise ContractError(
            f"Split/chip inventory mismatch (manifest-only={len(missing_files)}, chips-only={len(unassigned_files)})."
        )
    by_partition: dict[str, list[str]] = {name: [] for name in ("train", "validation", "test")}
    for row in split_rows:
        by_partition[str(row["partition"])].append(str(row["chip_id"]))
    counts = {name: len(values) for name, values in by_partition.items()}
    return sorted(by_partition["train"]), sorted(by_partition["validation"]), counts


def normalize_hwc(
    array: Any,
    band_contract: str,
    normalization: str,
    *,
    identity_verified: bool = False,
) -> Any:
    import numpy as np

    values = np.asarray(array)
    # NumPy chips do not retain Rasterio color interpretation metadata.  The
    # historical dataset contract nevertheless establishes band 4 as the GDAL
    # alpha channel, and V3 geometric/color augmentation can make that channel
    # spatially variable.  Preserve that declared provenance explicitly rather
    # than misclassifying a varying/padded alpha band as NIR.
    interpretations = ("red", "green", "blue", "alpha") if band_contract == "rgba-alpha" else ()
    observation = BandObservation.from_array(values, interpretations)
    try:
        indexes = validate_band_contract(observation, band_contract)
    except ContractError as exc:
        if not (
            identity_verified
            and band_contract == "rgbnir"
            and str(exc) == "Band 4 is constant and cannot be accepted as NIR."
        ):
            raise
        indexes = (0, 1, 2, 3)
    selected = values[..., indexes]
    if normalization == "historical-double-divide":
        if max(observation.maxima[:3]) > 1.01:
            raise ContractError(
                "Historical double-divide expects Earth Engine RGB already scaled to 0-1; "
                f"observed RGB maximum is {max(observation.maxima[:3])}."
            )
        return selected.astype(np.float32) / 255.0
    if normalization == "unit-rgb":
        return normalize_chip(selected, normalization)
    if normalization != "unit-multiband":
        raise ContractError(f"Unknown normalization: {normalization}.")
    selected = selected.astype(np.float32)
    lower = float(np.nanmin(selected))
    upper = float(np.nanmax(selected))
    if lower < 0:
        raise ContractError(f"Multiband chip contains negative values (minimum {lower}).")
    if upper <= 1.01:
        return selected
    if upper <= 255.0:
        return selected / 255.0
    raise ContractError(f"Multiband maximum {upper} is outside the supported unit/byte range.")


def configure_randomness(contract: str, seed: int) -> dict[str, Any]:
    """Configure RNGs; corrected mode additionally requires deterministic kernels."""

    if contract == "corrected":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import numpy as np
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Training requires numpy and torch; install the segmentation environment before running."
        ) from exc

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    deterministic = contract == "corrected"
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return {
        "seed": seed,
        "deterministic_algorithms": deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def choose_device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if requested == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is unavailable.")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_augmentation(seed: int) -> Any:
    try:
        import albumentations as A
    except ModuleNotFoundError as exc:
        raise RuntimeError("The augmented variant requires albumentations.") from exc
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.1, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ],
        additional_targets={"valid_mask": "mask"},
    )
    if hasattr(transform, "set_random_seed"):
        transform.set_random_seed(seed)
    return transform


def build_dataset_class(torch: Any) -> type:
    import numpy as np

    class ChipDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            chips_root: Path,
            chip_ids: Sequence[str],
            *,
            band_contract: str,
            normalization: str,
            validity_required: bool,
            augmentation: Any | None = None,
            parking_repeats: int = 0,
        ) -> None:
            self.image_dir = chips_root / "images"
            self.mask_dir = chips_root / "masks"
            self.valid_dir = chips_root / "valid_masks"
            self.band_contract = band_contract
            self.normalization = normalization
            self.validity_required = validity_required
            self.augmentation = augmentation
            ids: list[str] = []
            for chip_id in sorted(chip_ids):
                ids.append(chip_id)
                if parking_repeats:
                    mask = np.load(self.mask_dir / f"{chip_id}.npy", allow_pickle=False)
                    if np.isin(mask, (1, 2)).any():
                        ids.extend([chip_id] * parking_repeats)
            self.ids = ids

        def __len__(self) -> int:
            return len(self.ids)

        def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
            chip_id = self.ids[index]
            image = np.load(self.image_dir / f"{chip_id}.npy", allow_pickle=False)
            mask = np.load(self.mask_dir / f"{chip_id}.npy", allow_pickle=False)
            valid = (
                np.load(self.valid_dir / f"{chip_id}.npy", allow_pickle=False)
                if self.validity_required
                else np.ones(mask.shape, dtype=bool)
            )
            if image.ndim != 3 or mask.ndim != 2 or image.shape[:2] != mask.shape:
                raise ContractError(f"Invalid image/mask shapes for {chip_id}: {image.shape}/{mask.shape}.")
            if valid.ndim != 2 or valid.shape != mask.shape:
                raise ContractError(f"Invalid valid-mask shape for {chip_id}: {valid.shape}/{mask.shape}.")
            valid = valid.astype(bool, copy=False)
            if not bool(valid.any()):
                raise ContractError(f"Chip {chip_id} has no valid source pixels.")
            classes = np.unique(mask)
            if classes.min() < 0 or classes.max() >= len(CLASS_NAMES):
                raise ContractError(f"Mask {chip_id} contains out-of-range classes: {classes.tolist()}.")
            if self.augmentation is not None:
                transformed = self.augmentation(
                    image=image,
                    mask=mask,
                    valid_mask=valid.astype(np.uint8),
                )
                image, mask = transformed["image"], transformed["mask"]
                valid = transformed["valid_mask"].astype(bool)
            image = normalize_hwc(
                image,
                self.band_contract,
                self.normalization,
                identity_verified=self.validity_required,
            )
            image = np.ascontiguousarray(np.moveaxis(image, -1, 0), dtype=np.float32)
            mask = np.ascontiguousarray(mask, dtype=np.int64)
            valid = np.ascontiguousarray(valid, dtype=bool)
            return torch.from_numpy(image), torch.from_numpy(mask), torch.from_numpy(valid)

    return ChipDataset


def build_model(torch: Any, smp: Any, input_channels: int) -> Any:
    # Reproduce the notebook's explicit mean-RGB initialization when a fourth
    # spectral/input channel is requested.
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=len(CLASS_NAMES),
    )
    if input_channels == 4:
        original = model.encoder.conv1
        replacement = torch.nn.Conv2d(
            4,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=original.bias is not None,
        )
        with torch.no_grad():
            replacement.weight[:, :3] = original.weight
            replacement.weight[:, 3] = original.weight[:, :3].mean(dim=1)
            if replacement.bias is not None and original.bias is not None:
                replacement.bias.copy_(original.bias)
        model.encoder.conv1 = replacement
    return model


def confusion_from_loader(model: Any, loader: Any, torch: Any, device: Any) -> Any:
    import numpy as np

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


def masked_cross_entropy(
    logits: Any,
    targets: Any,
    valid: Any,
    *,
    torch: Any,
    class_weights: Any | None,
) -> tuple[Any, float, float]:
    """Return valid-pixel cross entropy and detached aggregation terms."""

    valid = valid.bool()
    if not bool(valid.any().item()):
        raise ContractError("A training batch contains no valid source pixels.")
    pixel_losses = torch.nn.functional.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        reduction="none",
    )
    numerator = pixel_losses[valid].sum()
    denominator = (
        valid.sum().to(dtype=numerator.dtype) if class_weights is None else class_weights[targets[valid]].sum()
    )
    if float(denominator.detach().cpu().item()) <= 0:
        raise ContractError("Masked cross-entropy has a nonpositive denominator.")
    return (
        numerator / denominator,
        float(numerator.detach().cpu().item()),
        float(denominator.detach().cpu().item()),
    )


def metrics_from_confusion(matrix: Any) -> tuple[float, list[float]]:
    import numpy as np

    intersection = np.diag(matrix).astype(float)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - intersection
    per_class = np.divide(intersection, union, out=np.full_like(intersection, np.nan), where=union > 0)
    return float(np.nanmean(per_class)), [float(value) for value in per_class]


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    resolve_default_paths(args)
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0 or args.num_workers < 0:
        raise ContractError("Epochs, batch size, and learning rate must be positive; workers cannot be negative.")
    if args.contract == "corrected" and args.num_workers != 0:
        raise ContractError(
            "Corrected deterministic training currently requires --num-workers 0 so augmentation RNG state "
            "cannot diverge between worker processes."
        )
    band_contract, normalization, input_channels = resolve_radiometry(args)
    run_dir = args.run_dir.resolve(strict=False)
    if run_dir.exists() and not args.allow_existing_run_dir:
        refuse_existing_output(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    weights_path = run_dir / "weights" / ("final.pth" if args.contract == "historical" else "best.pth")
    history_path = run_dir / "history.csv"
    metrics_path = run_dir / "training_metrics.csv"
    manifest_path = run_dir / "run_manifest.json"
    for output in (weights_path, history_path, metrics_path, manifest_path):
        refuse_existing_output(output)

    all_ids = discover_chip_ids(args.chips_root, require_valid_masks=args.contract == "corrected")
    split_rows = _read_split_rows(args.split_manifest) if args.contract == "corrected" else None
    train_ids, evaluation_ids, partition_counts = select_partitions(all_ids, args.contract, split_rows)
    if not train_ids or not evaluation_ids:
        raise ContractError("Training and model-selection partitions must both be non-empty.")

    dataset_manifest: Path | None = None
    dataset_manifest_sha256: str | None = None
    split_sha256: str | None = None
    corridor_manifest_sha256: str | None = None
    train_ids_sha256 = hashlib.sha256("\n".join(train_ids).encode("utf-8")).hexdigest()
    evaluation_ids_sha256 = hashlib.sha256("\n".join(evaluation_ids).encode("utf-8")).hexdigest()
    if args.contract == "corrected":
        dataset_manifest = args.dataset_manifest or args.chips_root / "run_manifest.json"
        dataset_payload = load_run_manifest(
            dataset_manifest,
            expected_stage="03_build_segmentation_dataset",
            expected_contract="corrected",
        )
        verify_manifest_artifact(dataset_payload, args.chips_root / "images")
        verify_manifest_artifact(dataset_payload, args.chips_root / "masks")
        verify_manifest_artifact(dataset_payload, args.split_manifest)
        validate_corrected_dataset_contract(dataset_payload, args.chips_root)
        dataset_manifest_sha256 = sha256_file(dataset_manifest)
        split_sha256 = sha256_file(args.split_manifest)
        corridor_manifest_sha256 = dataset_payload.get("extra", {}).get("corridor_manifest_sha256")
        if not corridor_manifest_sha256:
            raise ContractError("Corrected dataset manifest lacks canonical corridor ancestry.")

    randomness = configure_randomness(args.contract, args.seed)
    try:
        import numpy as np
        import torch
        import segmentation_models_pytorch as smp
    except ModuleNotFoundError as exc:
        raise RuntimeError("Training requires numpy, torch, and segmentation-models-pytorch.") from exc

    device = choose_device(torch, args.device)
    variant = VARIANTS[args.variant]
    augmentation = build_augmentation(args.seed) if variant["augment"] else None
    Dataset = build_dataset_class(torch)
    train_dataset = Dataset(
        args.chips_root,
        train_ids,
        band_contract=band_contract,
        normalization=normalization,
        validity_required=args.contract == "corrected",
        augmentation=augmentation,
        parking_repeats=int(variant["parking_repeats"]),
    )
    evaluation_dataset = Dataset(
        args.chips_root,
        evaluation_ids,
        band_contract=band_contract,
        normalization=normalization,
        validity_required=args.contract == "corrected",
    )
    if args.contract == "historical" and len(train_dataset) < args.batch_size:
        raise ContractError("Historical drop-last training requires at least one complete batch.")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=args.contract == "historical",
        num_workers=args.num_workers,
        generator=generator,
    )
    evaluation_loader = torch.utils.data.DataLoader(
        evaluation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    model = build_model(torch, smp, input_channels).to(device)
    class_weights = torch.tensor(CLASS_WEIGHTS, device=device) if variant["weighted"] else None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    scope = "resubstitution" if args.contract == "historical" else "validation"
    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_epoch = 0
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        loss_denominator_total = 0.0
        for images, masks, valid_masks in train_loader:
            images, masks, valid_masks = images.to(device), masks.to(device), valid_masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, loss_numerator, loss_denominator = masked_cross_entropy(
                model(images),
                masks,
                valid_masks,
                torch=torch,
                class_weights=class_weights,
            )
            loss.backward()
            optimizer.step()
            loss_total += loss_numerator
            loss_denominator_total += loss_denominator
        matrix = confusion_from_loader(model, evaluation_loader, torch, device)
        mean_iou, per_class = metrics_from_confusion(matrix)
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": loss_total / loss_denominator_total,
            "evaluation_scope": scope,
            f"{scope}_mean_iou": mean_iou,
        }
        row.update({f"{scope}_iou_{name}": value for name, value in zip(CLASS_NAMES, per_class, strict=True)})
        history.append(row)
        should_save = epoch == args.epochs if args.contract == "historical" else mean_iou > best_score
        if should_save:
            best_score, best_epoch = mean_iou, epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "metadata": {
                        "schema_version": 1,
                        "contract": args.contract,
                        "variant": args.variant,
                        "band_contract": band_contract,
                        "normalization": normalization,
                        "input_channels": input_channels,
                        "classes": list(CLASS_NAMES),
                        "dataset_manifest_sha256": dataset_manifest_sha256,
                        "split_sha256": split_sha256,
                        "train_ids_sha256": train_ids_sha256,
                        "selection_ids_sha256": evaluation_ids_sha256,
                        "corridor_manifest_sha256": corridor_manifest_sha256,
                        "validity_contract": (
                            "source-validity-plus-finite-radiometry" if args.contract == "corrected" else None
                        ),
                        "selection_scope": scope,
                        "selection_metric": "mean_iou",
                        "selection_epoch": best_epoch,
                        "selection_value": best_score,
                    },
                },
                weights_path,
            )
        print(
            f"epoch={epoch}/{args.epochs} train_loss={row['train_loss']:.6f} {scope}_mean_iou={mean_iou:.6f}",
            flush=True,
        )

    history_fields = list(history[0])
    write_csv(history_path, history_fields, history)
    metric_rows = [
        {
            "contract": args.contract,
            "evaluation_scope": scope,
            "metric": "mean_iou",
            "class_id": "",
            "class_name": "all_present_classes",
            "value": best_score,
            "selected_epoch": best_epoch,
        }
    ]
    best_row = history[best_epoch - 1]
    metric_rows.extend(
        {
            "contract": args.contract,
            "evaluation_scope": scope,
            "metric": "iou",
            "class_id": class_id,
            "class_name": name,
            "value": best_row[f"{scope}_iou_{name}"],
            "selected_epoch": best_epoch,
        }
        for class_id, name in enumerate(CLASS_NAMES)
    )
    write_csv(
        metrics_path,
        ("contract", "evaluation_scope", "metric", "class_id", "class_name", "value", "selected_epoch"),
        metric_rows,
    )

    arguments = vars(args).copy()
    arguments.update(
        {
            "band_contract": band_contract,
            "normalization": normalization,
            "run_dir": run_dir,
            "dataset_manifest_effective": dataset_manifest,
        }
    )
    inputs = [args.chips_root]
    if args.contract == "corrected":
        assert dataset_manifest is not None
        inputs.extend((args.split_manifest, dataset_manifest))
    write_run_manifest(
        manifest_path,
        workflow_root=project_paths.WORKFLOW_ROOT,
        stage="04_train_segmentation",
        contract=args.contract,
        arguments=arguments,
        inputs=inputs,
        outputs=(weights_path, history_path, metrics_path),
        extra={
            "evaluation_scope": scope,
            "partition_counts": partition_counts,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "split_sha256": split_sha256,
            "train_ids_sha256": train_ids_sha256,
            "selection_ids_sha256": evaluation_ids_sha256,
            "corridor_manifest_sha256": corridor_manifest_sha256,
            "training_examples_after_oversampling": len(train_dataset),
            "validity_contract": ("source-validity-plus-finite-radiometry" if args.contract == "corrected" else None),
            "variant_configuration": {
                "weighted_loss": bool(variant["weighted"]),
                "augmentation": bool(variant["augment"]),
                "parking_repeats": int(variant["parking_repeats"]),
                "effective_class_weights": (list(CLASS_WEIGHTS) if variant["weighted"] else [1.0] * len(CLASS_NAMES)),
            },
            "best_epoch": best_epoch,
            "best_mean_iou": best_score,
            "randomness": randomness,
            "device": str(device),
            "versions": {
                "torch": getattr(torch, "__version__", None),
                "segmentation_models_pytorch": getattr(smp, "__version__", None),
                "numpy": getattr(np, "__version__", None),
            },
        },
    )
    print(json.dumps({"weights": str(weights_path), "history": str(history_path), "scope": scope}, indent=2))
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
