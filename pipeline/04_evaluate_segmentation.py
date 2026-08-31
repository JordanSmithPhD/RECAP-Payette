#!/usr/bin/env python3
"""Evaluate a parking-segmentation checkpoint with honest partition labels.

Historical notebook checkpoints may only be evaluated as ``resubstitution``.
Corrected checkpoints require an immutable split manifest and an explicit
``validation`` or ``test`` partition.  No code path calls historical results
validation or test metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections.abc import Mapping, Sequence
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
    parse_chip_position,
    refuse_existing_output,
    sha256_file,
    validate_band_contract,
    validate_corrected_corridor_grid,
    validate_corrected_dem_slope_contract,
    validate_open_corrected_corridor_raster,
    validate_partition_rows,
    verify_manifest_artifact,
    write_run_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate DeepLabV3+ without relabeling resubstitution evidence as validation/test."
    )
    parser.add_argument("--contract", choices=("historical", "corrected"), required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Raw historical state_dict or metadata-bearing corrected checkpoint.",
    )
    parser.add_argument("--chips-root", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help="Corrected stage-03 run manifest; defaults to <chips-root>/run_manifest.json.",
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        help="Corrected stage-04 training manifest that checksum-binds --weights.",
    )
    parser.add_argument(
        "--partition",
        choices=("resubstitution", "validation", "test"),
        help="Historical is fixed to resubstitution; corrected requires validation or test.",
    )
    parser.add_argument("--band-contract", choices=("rgba-alpha", "rgb", "rgbnir"))
    parser.add_argument(
        "--normalization",
        choices=("historical-double-divide", "unit-rgb", "unit-multiband"),
    )
    parser.add_argument(
        "--evaluation-mask",
        choices=("all", "slope"),
        default="slope",
        help="Evaluate all chip pixels or only pixels inside the <=5%% slope constraint.",
    )
    parser.add_argument("--reference-raster", type=Path, default=None)
    parser.add_argument("--slope-vector", type=Path, default=None)
    parser.add_argument(
        "--imagery-manifest",
        type=Path,
        help="Corrected stage-01 corridor manifest that checksum-binds --reference-raster.",
    )
    parser.add_argument(
        "--slope-manifest",
        type=Path,
        help="Corrected stage-01 DEM/slope manifest that checksum-binds --slope-vector.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--allow-existing-output-dir", action="store_true")
    return parser


def resolve_default_paths(args: argparse.Namespace) -> argparse.Namespace:
    if args.contract == "corrected":
        args.weights = args.weights or (project_paths.PARKING_MODEL_RUN_ROOT / "v3_augmented" / "weights" / "best.pth")
        args.chips_root = args.chips_root or project_paths.PARKING_CHIP_RUN_ROOT
        args.split_manifest = args.split_manifest or args.chips_root / "splits.parquet"
        args.dataset_manifest = args.dataset_manifest or args.chips_root / "run_manifest.json"
        args.training_manifest = args.training_manifest or args.weights.parent.parent / "run_manifest.json"
        args.reference_raster = args.reference_raster or project_paths.PARKING_CORRIDOR_RASTER
        args.slope_vector = args.slope_vector or project_paths.PARKING_SLOPE_LE5_CORRIDOR
        args.imagery_manifest = args.imagery_manifest or project_paths.PARKING_CORRIDOR_MANIFEST
        args.slope_manifest = args.slope_manifest or project_paths.PARKING_SLOPE_MANIFEST
    else:
        args.weights = args.weights or project_paths.DEEPLAB_V3_WEIGHTS
        args.chips_root = args.chips_root or project_paths.SEGMENTATION_CHIP_ROOT
        args.split_manifest = args.split_manifest or project_paths.SEGMENTATION_SPLITS
        args.reference_raster = args.reference_raster or project_paths.NAIP_CORRIDOR_RASTER
        args.slope_vector = args.slope_vector or project_paths.SLOPE_LT5_CLIPPED
    return args


def resolve_contract(args: argparse.Namespace) -> tuple[str, str, int, str]:
    if args.contract == "historical":
        partition = args.partition or "resubstitution"
        band_contract = args.band_contract or "rgba-alpha"
        normalization = args.normalization or "historical-double-divide"
        if partition != "resubstitution":
            raise ContractError("Historical notebook evidence may only be labeled resubstitution.")
        if band_contract != "rgba-alpha" or normalization != "historical-double-divide":
            raise ContractError("Historical evaluation requires rgba-alpha/historical-double-divide.")
        return band_contract, normalization, 4, partition

    if args.partition not in {"validation", "test"}:
        raise ContractError("Corrected evaluation requires explicit --partition validation or test.")
    band_contract = args.band_contract or "rgbnir"
    if band_contract != "rgbnir":
        raise ContractError("The corrected contract is fixed to genuine ordered RGB+NIR imagery.")
    normalization = args.normalization or "unit-multiband"
    if normalization != "unit-multiband":
        raise ContractError("Corrected rgbnir requires unit-multiband normalization.")
    return band_contract, normalization, 4, args.partition


def _read_split_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("Split manifests require pandas and pyarrow for Parquet.") from exc
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        raise ContractError("Split manifest must be .parquet, .pq, or .csv.")
    missing = {"chip_id", "partition"}.difference(frame.columns)
    if missing:
        raise ContractError(f"Split manifest is missing columns: {sorted(missing)}.")
    rows = frame.to_dict(orient="records")
    validate_partition_rows(rows, corrected=True)
    return rows


def discover_chip_ids(chips_root: Path, *, require_valid_masks: bool = False) -> list[str]:
    image_dir, mask_dir = chips_root / "images", chips_root / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ and masks/ beneath {chips_root}.")
    images = {
        path.stem for path in image_dir.glob("*.npy") if not is_ignored_metadata_path(path.relative_to(image_dir))
    }
    masks = {
        path.stem for path in mask_dir.glob("*.npy") if not is_ignored_metadata_path(path.relative_to(mask_dir))
    }
    if images != masks:
        raise ContractError(
            f"Image/mask inventory mismatch (images-only={len(images - masks)}, masks-only={len(masks - images)})."
        )
    if not images:
        raise ContractError("No image/mask chips were found.")
    if require_valid_masks:
        valid_dir = chips_root / "valid_masks"
        if not valid_dir.is_dir():
            raise FileNotFoundError(f"Corrected chips require valid_masks/ beneath {chips_root}.")
        valid_ids = {
            path.stem
            for path in valid_dir.glob("*.npy")
            if not is_ignored_metadata_path(path.relative_to(valid_dir))
        }
        if valid_ids != images:
            raise ContractError(
                f"Image/valid-mask inventory mismatch (images-only={len(images - valid_ids)}, "
                f"valid-only={len(valid_ids - images)})."
            )
    return sorted(images)


def select_ids(
    all_ids: Sequence[str], contract: str, partition: str, rows: Sequence[Mapping[str, Any]] | None
) -> list[str]:
    if contract == "historical":
        return list(all_ids)
    assert rows is not None
    known, manifest_ids = set(all_ids), {str(row["chip_id"]) for row in rows}
    if known != manifest_ids:
        raise ContractError(
            f"Split/chip inventory mismatch (manifest-only={len(manifest_ids - known)}, "
            f"chips-only={len(known - manifest_ids)})."
        )
    selected = sorted(str(row["chip_id"]) for row in rows if str(row["partition"]) == partition)
    if not selected:
        raise ContractError(f"No chips assigned to {partition}.")
    return selected


def normalize_hwc(
    array: Any,
    band_contract: str,
    normalization: str,
    color_interpretations: Sequence[str] = (),
    *,
    identity_verified: bool = False,
) -> Any:
    import numpy as np

    values = np.asarray(array)
    observation = BandObservation.from_array(values, color_interpretations)
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
                "Historical double-divide expects Earth Engine RGB already in 0-1; "
                f"observed RGB maximum is {max(observation.maxima[:3])}."
            )
        return selected.astype(np.float32) / 255.0
    if normalization == "unit-rgb":
        return normalize_chip(selected, normalization)
    if normalization != "unit-multiband":
        raise ContractError(f"Unknown normalization: {normalization}.")
    selected = selected.astype(np.float32)
    lower, upper = float(np.nanmin(selected)), float(np.nanmax(selected))
    if lower < 0:
        raise ContractError(f"Multiband chip contains negative values (minimum {lower}).")
    if upper <= 1.01:
        return selected
    if upper <= 255.0:
        return selected / 255.0
    raise ContractError(f"Multiband maximum {upper} is outside the supported range.")


def choose_device(torch: Any, requested: str) -> Any:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is unavailable.")
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_randomness(contract: str, seed: int, torch: Any, np: Any) -> dict[str, Any]:
    if contract == "corrected":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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


def build_dataset_class(torch: Any) -> type:
    import numpy as np

    class ChipDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            root: Path,
            ids: Sequence[str],
            band_contract: str,
            normalization: str,
            *,
            validity_required: bool,
        ) -> None:
            self.image_dir, self.mask_dir = root / "images", root / "masks"
            self.valid_dir = root / "valid_masks"
            self.ids = list(ids)
            self.band_contract, self.normalization = band_contract, normalization
            self.validity_required = validity_required
            self.color_interpretations = ("red", "green", "blue", "alpha") if band_contract == "rgba-alpha" else ()

        def __len__(self) -> int:
            return len(self.ids)

        def __getitem__(self, index: int) -> tuple[Any, Any, Any, str]:
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
                raise ContractError(f"Mask {chip_id} has invalid classes: {classes.tolist()}.")
            image = normalize_hwc(
                image,
                self.band_contract,
                self.normalization,
                self.color_interpretations,
                identity_verified=self.validity_required,
            )
            image = np.ascontiguousarray(np.moveaxis(image, -1, 0), dtype=np.float32)
            return (
                torch.from_numpy(image),
                torch.from_numpy(mask.astype(np.int64)),
                torch.from_numpy(np.ascontiguousarray(valid, dtype=bool)),
                chip_id,
            )

    return ChipDataset


def build_model(torch: Any, smp: Any, input_channels: int) -> Any:
    model = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=len(CLASS_NAMES))
    if input_channels == 4:
        original = model.encoder.conv1
        model.encoder.conv1 = torch.nn.Conv2d(
            4,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=original.bias is not None,
        )
    return model


def load_checkpoint(
    weights: Path,
    *,
    torch: Any,
    model: Any,
    contract: str,
    band_contract: str,
    normalization: str,
    input_channels: int,
) -> dict[str, Any]:
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    try:
        payload = torch.load(weights, map_location="cpu", weights_only=False)
    except TypeError:  # Older Torch
        payload = torch.load(weights, map_location="cpu")
    metadata: dict[str, Any] = {}
    if isinstance(payload, Mapping) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        metadata = dict(payload.get("metadata") or {})
    else:
        state_dict = payload
    if contract == "corrected":
        if not metadata:
            raise ContractError("Corrected evaluation requires a metadata-bearing checkpoint.")
        expected = {
            "contract": contract,
            "band_contract": band_contract,
            "normalization": normalization,
            "input_channels": input_channels,
            "classes": list(CLASS_NAMES),
            "validity_contract": "source-validity-plus-finite-radiometry",
        }
        mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
        if mismatches:
            raise ContractError(f"Checkpoint contract mismatch: {mismatches}.")
    elif metadata.get("contract") not in (None, "historical"):
        raise ContractError("A corrected checkpoint cannot be evaluated under the historical contract.")
    model.load_state_dict(state_dict)
    return metadata


def validate_corrected_dataset_lineage(
    *,
    chips_root: Path,
    split_manifest: Path,
    dataset_manifest: Path,
) -> tuple[dict[str, Any], str, str]:
    """Verify stage-03 outputs and return immutable hashes for checkpoint matching."""

    payload = load_run_manifest(
        dataset_manifest,
        expected_stage="03_build_segmentation_dataset",
        expected_contract="corrected",
    )
    verify_manifest_artifact(payload, chips_root / "images")
    verify_manifest_artifact(payload, chips_root / "masks")
    verify_manifest_artifact(payload, chips_root / "valid_masks")
    verify_manifest_artifact(payload, split_manifest)
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
    return payload, sha256_file(dataset_manifest), sha256_file(split_manifest)


def _require_projected_metre_crs(value: Any, *, label: str) -> str:
    try:
        from pyproj import CRS
    except ModuleNotFoundError as exc:
        raise RuntimeError("Corrected evaluation provenance checks require pyproj.") from exc
    try:
        crs = CRS.from_user_input(value)
    except Exception as exc:
        raise ContractError(f"{label} is not a readable CRS: {value!r}.") from exc
    units = {str(axis.unit_name).lower() for axis in crs.axis_info if axis.unit_name}
    if not crs.is_projected or not units or not all("met" in unit for unit in units):
        raise ContractError(f"{label} must be projected with metre axes; found {crs.to_string()} ({sorted(units)}).")
    return crs.to_string()


def validate_corrected_stage_lineage(
    args: argparse.Namespace,
    *,
    band_contract: str,
    normalization: str,
) -> tuple[Path, Path, Path]:
    """Verify the training, corridor-imagery, and <=5% slope handoffs."""

    training_payload = load_run_manifest(
        args.training_manifest,
        expected_stage="04_train_segmentation",
        expected_contract="corrected",
    )
    verify_manifest_artifact(training_payload, args.weights)
    training_arguments = training_payload.get("arguments", {})
    expected_training = {"band_contract": band_contract, "normalization": normalization}
    mismatches = {
        key: (training_arguments.get(key), expected)
        for key, expected in expected_training.items()
        if training_arguments.get(key) != expected
    }
    if mismatches:
        raise ContractError(f"Training run manifest contract mismatch: {mismatches}.")
    if training_payload.get("extra", {}).get("validity_contract") != ("source-validity-plus-finite-radiometry"):
        raise ContractError("Corrected training manifest lacks the source-validity contract.")

    imagery_payload = load_run_manifest(
        args.imagery_manifest,
        expected_stage="01_build_corridor",
        expected_contract=f"projected-geometry-{band_contract}",
    )
    verify_manifest_artifact(imagery_payload, args.reference_raster)
    expected_names = ["R", "G", "B"] if band_contract == "rgb" else ["R", "G", "B", "N"]
    imagery_extra = imagery_payload.get("extra", {})
    raster_crs, _ = validate_corrected_corridor_grid(
        imagery_extra,
        expected_band_names=expected_names,
    )
    corridor_manifest_sha256 = sha256_file(args.imagery_manifest)
    if training_payload.get("extra", {}).get("corridor_manifest_sha256") != corridor_manifest_sha256:
        raise ContractError("Training weights and evaluation imagery do not originate from the same corridor run.")
    _require_projected_metre_crs(
        raster_crs,
        label="Corridor manifest raster_output_crs",
    )
    try:
        import rasterio
    except ModuleNotFoundError as exc:
        raise RuntimeError("Corrected evaluation raster-grid validation requires rasterio.") from exc
    with rasterio.open(args.reference_raster) as source:
        validate_open_corrected_corridor_raster(
            source,
            imagery_extra,
            expected_band_names=("R", "G", "B", "N"),
        )

    slope_payload = load_run_manifest(
        args.slope_manifest,
        expected_stage="01_build_dem_slope",
        expected_contract="corrected-complete-3dep-grid-projected-area-slope-le",
    )
    validate_corrected_dem_slope_contract(slope_payload)
    verify_manifest_artifact(slope_payload, args.slope_vector)
    slope_arguments = slope_payload.get("arguments", {})
    try:
        threshold = float(slope_arguments.get("slope_threshold_pct"))
    except (TypeError, ValueError) as exc:
        raise ContractError("Slope manifest has no numeric slope_threshold_pct.") from exc
    if threshold != 5.0 or slope_arguments.get("resolved_slope_comparison") != "le":
        raise ContractError("Corrected evaluation requires a documented slope <= 5% mask.")
    if slope_payload.get("extra", {}).get("threshold_operator") != "<=":
        raise ContractError("Slope manifest threshold operator does not match <= 5%.")
    if slope_payload.get("extra", {}).get("corridor_manifest_sha256") != sha256_file(args.imagery_manifest):
        raise ContractError("Slope and imagery manifests do not originate from the same immutable corridor run.")
    _require_projected_metre_crs(
        slope_arguments.get("metric_crs"),
        label="Slope manifest metric_crs",
    )
    return args.training_manifest, args.imagery_manifest, args.slope_manifest


def verify_checkpoint_dataset_hashes(
    metadata: Mapping[str, Any],
    *,
    dataset_manifest_sha256: str,
    split_sha256: str,
    train_ids: Sequence[str],
    selection_ids: Sequence[str],
    corridor_manifest_sha256: str,
) -> None:
    """Refuse evaluation if weights were selected from a different dataset/split."""

    expected = {
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "split_sha256": split_sha256,
        "train_ids_sha256": hashlib.sha256("\n".join(sorted(train_ids)).encode("utf-8")).hexdigest(),
        "selection_ids_sha256": hashlib.sha256("\n".join(sorted(selection_ids)).encode("utf-8")).hexdigest(),
        "corridor_manifest_sha256": corridor_manifest_sha256,
    }
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatches:
        raise ContractError(f"Checkpoint dataset lineage mismatch: {mismatches}.")


class SlopeMasker:
    """Rasterize the recovered <=5% vector constraint for individual chips."""

    def __init__(self, reference_raster: Path, slope_vector: Path) -> None:
        try:
            import geopandas as gpd
            import rasterio
        except ModuleNotFoundError as exc:
            raise RuntimeError("Slope-masked evaluation requires geopandas and rasterio.") from exc
        if not reference_raster.is_file() or not slope_vector.is_file():
            raise FileNotFoundError(f"Missing reference raster or slope vector: {reference_raster}, {slope_vector}")
        self._rasterio = rasterio
        self.source = rasterio.open(reference_raster)
        self.slopes = gpd.read_file(slope_vector)
        if self.slopes.crs != self.source.crs:
            self.slopes = self.slopes.to_crs(self.source.crs)
        self.index = self.slopes.sindex

    def close(self) -> None:
        self.source.close()

    def mask(self, chip_id: str, height: int, width: int) -> Any:
        import numpy as np
        from rasterio.features import rasterize
        from rasterio.windows import Window
        from shapely.geometry import box

        column, row = parse_chip_position(chip_id)
        window = Window(column, row, width, height)
        bounds = self.source.window_bounds(window)
        footprint = box(*bounds)
        possible = list(self.index.intersection(footprint.bounds))
        geometries = [
            geometry
            for geometry in self.slopes.geometry.iloc[possible]
            if geometry is not None and not geometry.is_empty and geometry.intersects(footprint)
        ]
        if not geometries:
            return np.zeros((height, width), dtype=bool)
        return rasterize(
            ((geometry, 1) for geometry in geometries),
            out_shape=(height, width),
            transform=self.source.window_transform(window),
            fill=0,
            dtype="uint8",
        ).astype(bool)


def metrics_from_confusion(matrix: Any) -> tuple[float, list[float], list[int]]:
    import numpy as np

    intersection = np.diag(matrix).astype(float)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - intersection
    iou = np.divide(intersection, union, out=np.full_like(intersection, np.nan), where=union > 0)
    return float(np.nanmean(iou)), [float(value) for value in iou], [int(value) for value in union]


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    resolve_default_paths(args)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ContractError("Batch size must be positive and num-workers cannot be negative.")
    band_contract, normalization, input_channels, partition = resolve_contract(args)
    dataset_manifest: Path | None = None
    dataset_manifest_sha256: str | None = None
    split_sha256: str | None = None
    corrected_stage_manifests: tuple[Path, Path, Path] | None = None
    corrected_corridor_manifest_sha256: str | None = None
    if args.contract == "corrected":
        if args.evaluation_mask != "slope":
            raise ContractError("Corrected evaluation requires the verified <=5% slope mask.")
        dataset_manifest = args.dataset_manifest
        dataset_payload, dataset_manifest_sha256, split_sha256 = validate_corrected_dataset_lineage(
            chips_root=args.chips_root,
            split_manifest=args.split_manifest,
            dataset_manifest=dataset_manifest,
        )
        corrected_stage_manifests = validate_corrected_stage_lineage(
            args,
            band_contract=band_contract,
            normalization=normalization,
        )
        corrected_corridor_manifest_sha256 = sha256_file(args.imagery_manifest)
        if dataset_payload.get("extra", {}).get("corridor_manifest_sha256") != corrected_corridor_manifest_sha256:
            raise ContractError("Evaluation dataset and imagery do not originate from the same corridor run.")
    default_output_dir = (
        project_paths.PARKING_RESULTS_RUN_ROOT / "segmentation" / f"{partition}_evaluation"
        if args.contract == "corrected"
        else project_paths.RESULTS_ROOT / "segmentation" / f"historical-{partition}-evaluation"
    )
    output_dir = (args.output_dir or default_output_dir).resolve(strict=False)
    if output_dir.exists() and not args.allow_existing_output_dir:
        refuse_existing_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{partition}_metrics.csv"
    confusion_path = output_dir / f"{partition}_confusion_matrix.csv"
    manifest_path = output_dir / "run_manifest.json"
    for output in (metrics_path, confusion_path, manifest_path):
        refuse_existing_output(output)

    all_ids = discover_chip_ids(args.chips_root, require_valid_masks=args.contract == "corrected")
    rows = _read_split_rows(args.split_manifest) if args.contract == "corrected" else None
    selected_ids = select_ids(all_ids, args.contract, partition, rows)

    if args.contract == "corrected":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import numpy as np
        import torch
        import segmentation_models_pytorch as smp
    except ModuleNotFoundError as exc:
        raise RuntimeError("Evaluation requires numpy, torch, and segmentation-models-pytorch.") from exc
    randomness = configure_randomness(args.contract, args.seed, torch, np)
    device = choose_device(torch, args.device)
    Dataset = build_dataset_class(torch)
    dataset = Dataset(
        args.chips_root,
        selected_ids,
        band_contract,
        normalization,
        validity_required=args.contract == "corrected",
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    model = build_model(torch, smp, input_channels)
    checkpoint_metadata = load_checkpoint(
        args.weights,
        torch=torch,
        model=model,
        contract=args.contract,
        band_contract=band_contract,
        normalization=normalization,
        input_channels=input_channels,
    )
    if args.contract == "corrected":
        assert dataset_manifest_sha256 is not None and split_sha256 is not None
        assert corrected_corridor_manifest_sha256 is not None
        assert rows is not None
        verify_checkpoint_dataset_hashes(
            checkpoint_metadata,
            dataset_manifest_sha256=dataset_manifest_sha256,
            split_sha256=split_sha256,
            train_ids=[str(row["chip_id"]) for row in rows if str(row["partition"]) == "train"],
            selection_ids=[str(row["chip_id"]) for row in rows if str(row["partition"]) == "validation"],
            corridor_manifest_sha256=corrected_corridor_manifest_sha256,
        )
    model.to(device).eval()
    masker = SlopeMasker(args.reference_raster, args.slope_vector) if args.evaluation_mask == "slope" else None
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    evaluated_pixels = 0
    source_valid_pixels = 0
    try:
        with torch.no_grad():
            for images, masks, source_valid_masks, chip_ids in loader:
                predictions = model(images.to(device)).argmax(dim=1).cpu().numpy()
                truth = masks.numpy()
                source_valid_batch = source_valid_masks.numpy().astype(bool)
                for index, chip_id in enumerate(chip_ids):
                    source_valid = source_valid_batch[index]
                    slope_valid = (
                        masker.mask(str(chip_id), truth.shape[1], truth.shape[2])
                        if masker is not None
                        else np.ones(truth.shape[1:], dtype=bool)
                    )
                    valid = source_valid & slope_valid
                    source_valid_pixels += int(source_valid.sum())
                    evaluated_pixels += int(valid.sum())
                    encoded = truth[index][valid] * len(CLASS_NAMES) + predictions[index][valid]
                    matrix += np.bincount(encoded, minlength=len(CLASS_NAMES) ** 2).reshape(matrix.shape)
    finally:
        if masker is not None:
            masker.close()
    if evaluated_pixels == 0:
        raise ContractError("Evaluation mask selected zero pixels.")

    mean_iou, per_class, unions = metrics_from_confusion(matrix)
    metric_rows: list[dict[str, Any]] = [
        {
            "contract": args.contract,
            "evaluation_scope": partition,
            "metric": "mean_iou",
            "class_id": "",
            "class_name": "all_present_classes",
            "value": mean_iou,
            "union_pixels": sum(unions),
            "chip_count": len(selected_ids),
            "evaluated_pixels": evaluated_pixels,
        }
    ]
    metric_rows.extend(
        {
            "contract": args.contract,
            "evaluation_scope": partition,
            "metric": "iou",
            "class_id": class_id,
            "class_name": name,
            "value": value,
            "union_pixels": unions[class_id],
            "chip_count": len(selected_ids),
            "evaluated_pixels": evaluated_pixels,
        }
        for class_id, (name, value) in enumerate(zip(CLASS_NAMES, per_class, strict=True))
    )
    fields = (
        "contract",
        "evaluation_scope",
        "metric",
        "class_id",
        "class_name",
        "value",
        "union_pixels",
        "chip_count",
        "evaluated_pixels",
    )
    write_csv(metrics_path, fields, metric_rows)
    confusion_rows = [
        {
            "truth_class": CLASS_NAMES[row],
            **{f"pred_{name}": int(matrix[row, col]) for col, name in enumerate(CLASS_NAMES)},
        }
        for row in range(len(CLASS_NAMES))
    ]
    write_csv(confusion_path, ("truth_class", *(f"pred_{name}" for name in CLASS_NAMES)), confusion_rows)

    arguments = vars(args).copy()
    arguments.update(
        {
            "band_contract": band_contract,
            "normalization": normalization,
            "partition": partition,
            "output_dir": output_dir,
            "dataset_manifest_effective": dataset_manifest,
        }
    )
    inputs = [args.weights, args.chips_root]
    if args.contract == "corrected":
        assert dataset_manifest is not None
        inputs.extend((args.split_manifest, dataset_manifest))
        assert corrected_stage_manifests is not None
        inputs.extend(corrected_stage_manifests)
    if args.evaluation_mask == "slope":
        inputs.extend((args.reference_raster, args.slope_vector))
    write_run_manifest(
        manifest_path,
        workflow_root=project_paths.WORKFLOW_ROOT,
        stage="04_evaluate_segmentation",
        contract=args.contract,
        arguments=arguments,
        inputs=inputs,
        outputs=(metrics_path, confusion_path),
        extra={
            "evaluation_scope": partition,
            "historical_result_is_resubstitution": args.contract == "historical",
            "chip_count": len(selected_ids),
            "evaluated_pixels": evaluated_pixels,
            "source_valid_pixels": source_valid_pixels,
            "validity_contract": ("source-validity-plus-finite-radiometry" if args.contract == "corrected" else None),
            "mean_iou": mean_iou,
            "device": str(device),
            "checkpoint_metadata": checkpoint_metadata,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "split_sha256": split_sha256,
            "corridor_manifest_sha256": corrected_corridor_manifest_sha256,
            "historical_alpha_identity": ("declared-rgba-contract" if args.contract == "historical" else None),
            "randomness": randomness,
            "versions": {
                "torch": getattr(torch, "__version__", None),
                "segmentation_models_pytorch": getattr(smp, "__version__", None),
                "numpy": getattr(np, "__version__", None),
            },
        },
    )
    print(json.dumps({"metrics": str(metrics_path), "evaluation_scope": partition, "mean_iou": mean_iou}, indent=2))
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
