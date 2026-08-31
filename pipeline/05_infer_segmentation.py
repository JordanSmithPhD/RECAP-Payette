#!/usr/bin/env python3
"""Run parking segmentation under explicit historical or corrected windows.

Corrected inference uses a fixed grid with a context halo, so merely retilling
an otherwise identical source cannot change predictions. Historical mode
defaults to the GeoTIFF's true internal blocks to reproduce the notebook.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


WORKING_ROOT = Path(__file__).resolve().parents[1]
if str(WORKING_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKING_ROOT))

import project_paths
from pipeline.parking_common import (
    CLASS_NAMES,
    BandObservation,
    ContractError,
    load_run_manifest,
    normalize_chip,
    refuse_existing_output,
    resolve_manifest_artifact_path,
    sha256_file,
    validate_band_contract,
    validate_corrected_corridor_grid,
    validate_corrected_dem_slope_contract,
    validate_open_corrected_corridor_raster,
    verify_manifest_artifact,
    write_run_manifest,
)


PREDICTION_NODATA = 255


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer parking classes under explicit historical/corrected data and window contracts."
    )
    parser.add_argument("--contract", choices=("historical", "corrected"), required=True)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument(
        "--training-manifest",
        type=Path,
        help="Corrected stage-04 run manifest that checksum-binds --weights.",
    )
    parser.add_argument("--input-raster", type=Path, default=None)
    parser.add_argument(
        "--imagery-manifest",
        type=Path,
        help="Corrected stage-01 corridor manifest that checksum-binds --input-raster and its band order.",
    )
    parser.add_argument("--slope-vector", type=Path, default=None)
    parser.add_argument(
        "--slope-manifest",
        type=Path,
        help="Corrected stage-01 DEM/slope manifest that checksum-binds --slope-vector.",
    )
    parser.add_argument(
        "--slope-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the slope constraint; corrected runs require a verified <=5%% mask (default: enabled).",
    )
    parser.add_argument("--output", type=Path, help="New GeoTIFF path; existing files are refused.")
    parser.add_argument("--summary", type=Path, help="Class-pixel CSV; defaults beside --output.")
    parser.add_argument("--manifest", type=Path, help="Run manifest; defaults beside --output.")
    parser.add_argument("--band-contract", choices=("rgba-alpha", "rgb", "rgbnir"))
    parser.add_argument(
        "--normalization",
        choices=("historical-double-divide", "unit-rgb", "unit-multiband"),
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Maximum model input size including context (default: 512).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Context halo on each side; output core is tile_size-2*overlap (default: 64).",
    )
    parser.add_argument(
        "--window-mode",
        choices=("internal-blocks", "fixed-grid"),
        help=(
            "Historical defaults to the notebook's GeoTIFF internal blocks; corrected defaults to fixed-grid "
            "cores with a context halo. Corrected runs may not use internal blocks."
        ),
    )
    parser.add_argument("--model-stride", type=int, default=16, help="Pad model inputs to this multiple.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Opt-in model batch size for historical internal-block inference only (default: 1). "
            "Corrected and fixed-grid runs require 1."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser


def resolve_default_paths(args: argparse.Namespace) -> argparse.Namespace:
    if args.contract == "corrected":
        args.weights = args.weights or (project_paths.PARKING_MODEL_RUN_ROOT / "v3_augmented" / "weights" / "best.pth")
        args.training_manifest = args.training_manifest or args.weights.parent.parent / "run_manifest.json"
        args.input_raster = args.input_raster or project_paths.PARKING_CORRIDOR_RASTER
        args.imagery_manifest = args.imagery_manifest or project_paths.PARKING_CORRIDOR_MANIFEST
        args.slope_vector = args.slope_vector or project_paths.PARKING_SLOPE_LE5_CORRIDOR
        args.slope_manifest = args.slope_manifest or project_paths.PARKING_SLOPE_MANIFEST
        args.output = args.output or project_paths.PARKING_PREDICTIONS
    else:
        args.weights = args.weights or project_paths.DEEPLAB_V3_WEIGHTS
        args.input_raster = args.input_raster or project_paths.NAIP_CORRIDOR_RASTER
        args.slope_vector = args.slope_vector or project_paths.SLOPE_LT5_CLIPPED
    return args


def resolve_contract(args: argparse.Namespace) -> tuple[str, str, int]:
    if args.contract == "historical":
        band_contract = args.band_contract or "rgba-alpha"
        normalization = args.normalization or "historical-double-divide"
        if band_contract != "rgba-alpha" or normalization != "historical-double-divide":
            raise ContractError("Historical inference requires rgba-alpha/historical-double-divide.")
        return band_contract, normalization, 4
    band_contract = args.band_contract or "rgbnir"
    if band_contract != "rgbnir":
        raise ContractError("The corrected contract is fixed to genuine ordered RGB+NIR imagery.")
    normalization = args.normalization or "unit-multiband"
    if normalization != "unit-multiband":
        raise ContractError("Corrected rgbnir requires unit-multiband normalization.")
    return band_contract, normalization, 4


def normalize_hwc(
    array: Any,
    *,
    band_contract: str,
    normalization: str,
    color_interpretations: Sequence[str],
    band_names: Sequence[str] = (),
    require_band_identity: bool = False,
    valid_mask: Any | None = None,
) -> Any:
    import numpy as np

    values = np.asarray(array)
    observation = BandObservation.from_array(values, color_interpretations, band_names)
    try:
        indexes = validate_band_contract(
            observation,
            band_contract,
            require_identity=require_band_identity,
        )
    except ContractError as exc:
        if not (
            require_band_identity
            and band_contract == "rgbnir"
            and str(exc) == "Band 4 is constant and cannot be accepted as NIR."
        ):
            raise
        # Ordered R/G/B/N identity is carried by checksum-bound GeoTIFF band
        # descriptions. A single spatial window need not contain NIR variance.
        indexes = (0, 1, 2, 3)
    selected = values[..., indexes].astype(np.float32)
    valid = np.ones(selected.shape[:2], dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != selected.shape[:2]:
        raise ContractError(f"Validity mask shape {valid.shape} does not match raster window {selected.shape[:2]}.")
    if not bool(valid.any()):
        return np.zeros_like(selected, dtype=np.float32)
    selected[~valid] = 0
    if normalization == "historical-double-divide":
        # Pass color metadata to the validator above because alpha can vary at
        # cutline edges; NumPy chips alone have no such metadata.
        rgb_maximum = max(observation.maxima[:3])
        if rgb_maximum > 1.01:
            raise ContractError(
                "Historical double-divide expects Earth Engine RGB already in 0-1; "
                f"observed RGB maximum is {rgb_maximum}."
            )
        return selected / 255.0
    if normalization == "unit-rgb":
        return normalize_chip(selected, normalization)
    if normalization != "unit-multiband":
        raise ContractError(f"Unknown normalization: {normalization}.")
    valid_values = selected[valid]
    lower, upper = float(np.nanmin(valid_values)), float(np.nanmax(valid_values))
    if lower < 0:
        raise ContractError(f"Multiband raster contains negative values (minimum {lower}).")
    if upper <= 1.01:
        return selected
    if upper <= 255.0:
        return selected / 255.0
    raise ContractError(f"Multiband maximum {upper} is outside the supported unit/byte range.")


def validate_raster_band_identity(
    channel_count: int,
    *,
    color_interpretations: Sequence[str],
    band_names: Sequence[str],
    band_contract: str,
    require_identity: bool,
) -> tuple[int, ...]:
    """Validate source-level band identity without requiring local pixel variance."""

    observation = BandObservation(
        channel_count,
        tuple(0.0 for _ in range(channel_count)),
        tuple(1.0 for _ in range(channel_count)),
        tuple(color_interpretations),
        tuple(band_names),
    )
    return validate_band_contract(observation, band_contract, require_identity=require_identity)


def configure_randomness(contract: str, seed: int) -> dict[str, Any]:
    if contract == "corrected":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import numpy as np
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("Inference requires numpy and torch.") from exc
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
    path: Path,
    *,
    torch: Any,
    model: Any,
    contract: str,
    band_contract: str,
    normalization: str,
    input_channels: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Weights not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    metadata: dict[str, Any] = {}
    if isinstance(payload, Mapping) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        metadata = dict(payload.get("metadata") or {})
    else:
        state_dict = payload
    if contract == "corrected":
        if not metadata:
            raise ContractError("Corrected inference requires a metadata-bearing checkpoint.")
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
        raise ContractError("A corrected checkpoint cannot be run under the historical contract.")
    model.load_state_dict(state_dict)
    return metadata


def _require_projected_metre_crs(value: Any, *, label: str) -> str:
    try:
        from pyproj import CRS
    except ModuleNotFoundError as exc:
        raise RuntimeError("Corrected provenance checks require pyproj.") from exc

    try:
        crs = CRS.from_user_input(value)
    except Exception as exc:
        raise ContractError(f"{label} is not a readable CRS: {value!r}.") from exc
    units = {str(axis.unit_name).lower() for axis in crs.axis_info if axis.unit_name}
    if not crs.is_projected or not units or not all("met" in unit for unit in units):
        raise ContractError(f"{label} must be projected with metre axes; found {crs.to_string()} ({sorted(units)}).")
    return crs.to_string()


def validate_corrected_lineage(
    args: argparse.Namespace,
    *,
    band_contract: str,
    normalization: str,
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any]]:
    """Checksum-bind corrected inference to reviewed stage-01 and stage-04 products."""

    missing = [
        option
        for option, value in (
            ("--training-manifest", args.training_manifest),
            ("--imagery-manifest", args.imagery_manifest),
            ("--slope-manifest", args.slope_manifest),
        )
        if value is None
    ]
    if missing:
        raise ContractError(f"Corrected inference requires explicit provenance: {', '.join(missing)}.")

    training_manifest = Path(args.training_manifest)
    imagery_manifest = Path(args.imagery_manifest)
    slope_manifest = Path(args.slope_manifest)

    training_payload = load_run_manifest(
        training_manifest,
        expected_stage="04_train_segmentation",
        expected_contract="corrected",
    )
    verify_manifest_artifact(training_payload, args.weights)
    training_arguments = training_payload.get("arguments", {})
    expected_training = {
        "band_contract": band_contract,
        "normalization": normalization,
    }
    mismatches = {
        key: (training_arguments.get(key), value)
        for key, value in expected_training.items()
        if training_arguments.get(key) != value
    }
    if mismatches:
        raise ContractError(f"Training run manifest contract mismatch: {mismatches}.")
    if training_payload.get("extra", {}).get("validity_contract") != ("source-validity-plus-finite-radiometry"):
        raise ContractError("Corrected training manifest lacks the source-validity contract.")

    imagery_payload = load_run_manifest(
        imagery_manifest,
        expected_stage="01_build_corridor",
        expected_contract=f"projected-geometry-{band_contract}",
    )
    verify_manifest_artifact(imagery_payload, args.input_raster)
    expected_names = ["R", "G", "B"] if band_contract == "rgb" else ["R", "G", "B", "N"]
    imagery_extra = imagery_payload.get("extra", {})
    raster_crs, _ = validate_corrected_corridor_grid(
        imagery_extra,
        expected_band_names=expected_names,
    )
    if training_payload.get("extra", {}).get("corridor_manifest_sha256") != sha256_file(imagery_manifest):
        raise ContractError("Training weights and inference imagery do not originate from the same corridor run.")
    _require_projected_metre_crs(
        raster_crs,
        label="Corridor manifest raster_output_crs",
    )

    slope_payload = load_run_manifest(
        slope_manifest,
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
        raise ContractError("Corrected inference requires a documented slope <= 5% mask.")
    if slope_payload.get("extra", {}).get("threshold_operator") != "<=":
        raise ContractError("Slope manifest threshold operator does not match <= 5%.")
    if slope_payload.get("extra", {}).get("corridor_manifest_sha256") != sha256_file(imagery_manifest):
        raise ContractError("Slope and imagery manifests do not originate from the same immutable corridor run.")
    _require_projected_metre_crs(
        slope_arguments.get("metric_crs"),
        label="Slope manifest metric_crs",
    )

    return training_manifest, imagery_manifest, slope_manifest, training_payload, dict(imagery_extra)


def corrected_lineage_manifest_paths(
    lineage: tuple[Path, Path, Path, dict[str, Any], dict[str, Any]],
) -> tuple[Path, Path, Path]:
    """Return the three manifest paths from the five-part validated lineage."""

    if len(lineage) != 5:
        raise ContractError(f"Corrected inference lineage must contain five components; found {len(lineage)}.")
    return lineage[0], lineage[1], lineage[2]


def verify_checkpoint_training_lineage(
    metadata: Mapping[str, Any],
    training_payload: Mapping[str, Any],
) -> None:
    """Match checkpoint-embedded data/split hashes to the verified training run."""

    input_records = training_payload.get("inputs")
    if not isinstance(input_records, list):
        raise ContractError("Training manifest has no input artifact records.")

    def verified_record_for_hash(metadata_key: str) -> dict[str, Any]:
        checksum = metadata.get(metadata_key)
        matches = [
            record
            for record in input_records
            if isinstance(record, Mapping) and checksum and record.get("sha256") == checksum
        ]
        if len(matches) != 1:
            raise ContractError(f"Checkpoint {metadata_key} does not identify exactly one training-manifest input.")
        artifact = resolve_manifest_artifact_path(training_payload, matches[0])
        return verify_manifest_artifact(training_payload, artifact, section="inputs")

    dataset_record = verified_record_for_hash("dataset_manifest_sha256")
    split_record = verified_record_for_hash("split_sha256")
    extra = training_payload.get("extra", {})
    expected = {
        "dataset_manifest_sha256": dataset_record.get("sha256"),
        "split_sha256": split_record.get("sha256"),
        "train_ids_sha256": extra.get("train_ids_sha256"),
        "selection_ids_sha256": extra.get("selection_ids_sha256"),
        "corridor_manifest_sha256": extra.get("corridor_manifest_sha256"),
    }
    missing = [key for key, value in expected.items() if not value]
    if missing:
        raise ContractError(f"Training manifest is missing checkpoint-lineage fields: {missing}.")
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatches:
        raise ContractError(f"Checkpoint does not match its training run lineage: {mismatches}.")


class SlopeMasker:
    def __init__(self, slope_vector: Path, raster_crs: Any) -> None:
        try:
            import geopandas as gpd
        except ModuleNotFoundError as exc:
            raise RuntimeError("Slope masking requires geopandas.") from exc
        if not slope_vector.is_file():
            raise FileNotFoundError(f"Slope vector not found: {slope_vector}")
        self.slopes = gpd.read_file(slope_vector)
        if self.slopes.empty:
            raise ContractError("Slope constraint is empty.")
        if self.slopes.crs != raster_crs:
            self.slopes = self.slopes.to_crs(raster_crs)
        self.index = self.slopes.sindex

    def mask(self, source: Any, window: Any) -> Any:
        import numpy as np
        from rasterio.features import rasterize
        from shapely.geometry import box

        height, width = int(window.height), int(window.width)
        footprint = box(*source.window_bounds(window))
        possible = list(self.index.intersection(footprint.bounds))
        geometries = [
            geometry
            for geometry in self.slopes.geometry.iloc[possible]
            if geometry is not None and not geometry.is_empty and geometry.intersects(footprint)
        ]
        if not geometries:
            return np.zeros((height, width), dtype=np.uint8)
        return rasterize(
            ((geometry, 1) for geometry in geometries),
            out_shape=(height, width),
            transform=source.window_transform(window),
            fill=0,
            dtype="uint8",
        )


def iter_core_windows(width: int, height: int, tile_size: int, overlap: int) -> Any:
    """Yield (read bounds, core bounds, core offset within read) deterministically."""

    core_size = tile_size - 2 * overlap
    for core_row in range(0, height, core_size):
        core_height = min(core_size, height - core_row)
        for core_column in range(0, width, core_size):
            core_width = min(core_size, width - core_column)
            read_column = max(0, core_column - overlap)
            read_row = max(0, core_row - overlap)
            read_right = min(width, core_column + core_width + overlap)
            read_bottom = min(height, core_row + core_height + overlap)
            yield (
                (read_column, read_row, read_right - read_column, read_bottom - read_row),
                (core_column, core_row, core_width, core_height),
                (core_column - read_column, core_row - read_row),
            )


def iter_internal_block_windows(source: Any) -> Any:
    """Yield the source GeoTIFF's true band-1 block windows, as the notebook did."""

    for _, window in source.block_windows(1):
        values = (
            int(window.col_off),
            int(window.row_off),
            int(window.width),
            int(window.height),
        )
        yield values, values, (0, 0)


@dataclass(frozen=True)
class PreparedInferenceTile:
    """One independently normalized and padded source window awaiting inference."""

    tensor: Any
    read_valid: Any
    core_window: Any
    core_offset: tuple[int, int]
    unpadded_height: int
    unpadded_width: int
    padded_shape: tuple[int, int]


def iter_equal_padded_shape_batches(items: Any, batch_size: int) -> Any:
    """Yield order-preserving batches without ever mixing padded tensor shapes."""

    if batch_size <= 0:
        raise ContractError("batch-size must be positive.")
    pending: list[PreparedInferenceTile] = []
    pending_shape: tuple[int, int] | None = None
    for item in items:
        if pending and item.padded_shape != pending_shape:
            yield tuple(pending)
            pending = []
            pending_shape = None
        if not pending:
            pending_shape = item.padded_shape
        pending.append(item)
        if len(pending) == batch_size:
            yield tuple(pending)
            pending = []
            pending_shape = None
    if pending:
        yield tuple(pending)


def predict_prepared_tile_batch(batch: Sequence[PreparedInferenceTile], *, model: Any, device: Any, torch: Any) -> Any:
    """Infer an equal-shape batch and crop every prediction back to its source window."""

    if not batch:
        raise ContractError("Cannot infer an empty tile batch.")
    shapes = {item.padded_shape for item in batch}
    if len(shapes) != 1:
        raise ContractError(f"A model batch cannot mix padded shapes: {sorted(shapes)}.")
    inputs = torch.cat([item.tensor for item in batch], dim=0).to(device)
    predictions = model(inputs).argmax(dim=1).cpu().numpy().astype("uint8")
    return tuple(
        prediction[: item.unpadded_height, : item.unpadded_width]
        for item, prediction in zip(batch, predictions, strict=True)
    )


def compose_corrected_prediction(
    prediction: Any,
    source_valid: Any,
    slope_valid: Any | None = None,
) -> tuple[Any, Any]:
    """Encode invalid pixels as nodata while preserving valid background class 0."""

    import numpy as np

    values = np.asarray(prediction, dtype=np.uint8)
    valid = np.asarray(source_valid, dtype=bool)
    if values.shape != valid.shape:
        raise ContractError(f"Prediction/validity shapes differ: {values.shape}/{valid.shape}.")
    if slope_valid is not None:
        slope = np.asarray(slope_valid, dtype=bool)
        if slope.shape != values.shape:
            raise ContractError(f"Prediction/slope shapes differ: {values.shape}/{slope.shape}.")
        valid = valid & slope
    output = np.full(values.shape, PREDICTION_NODATA, dtype=np.uint8)
    output[valid] = values[valid]
    return output, valid


def resolve_window_mode(contract: str, requested: str | None) -> str:
    mode = requested or ("internal-blocks" if contract == "historical" else "fixed-grid")
    if contract == "corrected" and mode != "fixed-grid":
        raise ContractError("Corrected inference requires fixed-grid windows with an explicit context halo.")
    return mode


def validate_batch_size(contract: str, window_mode: str, batch_size: int) -> int:
    """Keep batched execution opt-in and outside the corrected/fixed-grid contract."""

    if batch_size <= 0:
        raise ContractError("batch-size must be positive.")
    if batch_size > 1 and (contract != "historical" or window_mode != "internal-blocks"):
        raise ContractError(
            "batch-size greater than 1 is available only for historical internal-blocks inference."
        )
    return batch_size


def write_summary(path: Path, contract: str, counts: Sequence[int]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("contract", "class_id", "class_name", "pixel_count"))
        writer.writeheader()
        for class_id, (name, count) in enumerate(zip(CLASS_NAMES, counts, strict=True)):
            writer.writerow({"contract": contract, "class_id": class_id, "class_name": name, "pixel_count": int(count)})


@contextmanager
def atomic_inference_raster(rasterio: Any, output: Path, profile: Mapping[str, Any]) -> Any:
    """Write a GeoTIFF beside its destination and publish it only when complete."""

    suffix = output.suffix or ".tif"
    partial = output.with_name(f".{output.stem}.{uuid4().hex}.partial{suffix}")
    refuse_existing_output(partial)
    try:
        with rasterio.open(partial, "w", **profile) as destination:
            yield destination
        os.replace(partial, output)
    except BaseException:
        partial.unlink(missing_ok=True)
        Path(f"{partial}.aux.xml").unlink(missing_ok=True)
        Path(f"{partial}.ovr").unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> int:
    resolve_default_paths(args)
    window_mode = resolve_window_mode(args.contract, args.window_mode)
    batch_size = validate_batch_size(args.contract, window_mode, args.batch_size)
    if window_mode == "fixed-grid" and (args.tile_size <= 0 or args.overlap < 0 or args.tile_size <= 2 * args.overlap):
        raise ContractError("tile-size must be positive and greater than twice overlap.")
    if args.model_stride <= 0:
        raise ContractError("model-stride must be positive.")
    band_contract, normalization, input_channels = resolve_contract(args)
    corrected_lineage: tuple[Path, Path, Path, dict[str, Any], dict[str, Any]] | None = None
    if args.contract == "corrected":
        if not args.slope_mask:
            raise ContractError("Corrected inference requires the verified <=5% slope mask.")
        corrected_lineage = validate_corrected_lineage(
            args,
            band_contract=band_contract,
            normalization=normalization,
        )
    default_name = (
        "historical_rgb_alpha_internal_blocks.tif"
        if args.contract == "historical" and window_mode == "internal-blocks"
        else f"{args.contract}_fixed_grid.tif"
    )
    output = (args.output or project_paths.SEGMENTATION_INFERENCE_ROOT / default_name).resolve(strict=False)
    summary = (args.summary or output.with_suffix(".class_pixels.csv")).resolve(strict=False)
    manifest = (args.manifest or output.with_suffix(".run_manifest.json")).resolve(strict=False)
    if len({output, summary, manifest}) != 3:
        raise ContractError("Output raster, summary CSV, and run manifest must use distinct paths.")
    for path in (output, summary, manifest):
        refuse_existing_output(path)
    if not args.input_raster.is_file():
        raise FileNotFoundError(f"Input raster not found: {args.input_raster}")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    randomness = configure_randomness(args.contract, args.seed)
    try:
        import numpy as np
        import rasterio
        import torch
        import torch.nn.functional as functional
        import segmentation_models_pytorch as smp
        from rasterio.windows import Window
    except ModuleNotFoundError as exc:
        raise RuntimeError("Inference requires numpy, rasterio, torch, and segmentation-models-pytorch.") from exc
    device = choose_device(torch, args.device)
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
    if corrected_lineage is not None:
        verify_checkpoint_training_lineage(checkpoint_metadata, corrected_lineage[3])
    model.to(device).eval()

    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    tile_count = 0
    written_pixels = 0
    source_valid_pixels = 0
    prediction_valid_pixels = 0
    nodata_pixels = 0
    with rasterio.open(args.input_raster) as source:
        if source.crs is None:
            raise ContractError("Input raster has no CRS.")
        if corrected_lineage is not None:
            validate_open_corrected_corridor_raster(
                source,
                corrected_lineage[4],
                expected_band_names=("R", "G", "B", "N"),
            )
        color_interpretations = tuple(str(getattr(value, "name", value)).lower() for value in source.colorinterp)
        band_names = tuple(str(value or "") for value in source.descriptions)
        validate_raster_band_identity(
            source.count,
            color_interpretations=color_interpretations,
            band_names=band_names,
            band_contract=band_contract,
            require_identity=args.contract == "corrected",
        )
        slope_masker = SlopeMasker(args.slope_vector, source.crs) if args.slope_mask else None
        profile = source.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="uint8",
            count=1,
            nodata=PREDICTION_NODATA if args.contract == "corrected" else None,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            bigtiff="YES",
        )
        with atomic_inference_raster(rasterio, output, profile) as destination:
            destination.set_band_description(1, "parking_class")
            destination.update_tags(
                1,
                CLASS_VALUES=",".join(str(value) for value in range(len(CLASS_NAMES))),
                CLASS_NAMES=",".join(CLASS_NAMES),
                NODATA_MEANING=(
                    "invalid-source-or-outside-slope-contract"
                    if args.contract == "corrected"
                    else "not-set-historical-reconstruction"
                ),
            )
            with torch.no_grad():
                windows = (
                    iter_internal_block_windows(source)
                    if window_mode == "internal-blocks"
                    else iter_core_windows(source.width, source.height, args.tile_size, args.overlap)
                )
                if batch_size == 1:
                    # Preserve the established singleton path byte-for-byte by default.
                    for read_values, core_values, core_offset in windows:
                        read_window = Window(*read_values)
                        core_window = Window(*core_values)
                        source_data = source.read(window=read_window)
                        read_valid = (
                            (source.dataset_mask(window=read_window) > 0) & np.all(np.isfinite(source_data), axis=0)
                            if args.contract == "corrected"
                            else np.ones(source_data.shape[1:], dtype=bool)
                        )
                        data = np.moveaxis(source_data, 0, -1)
                        if bool(read_valid.any()):
                            normalized = normalize_hwc(
                                data,
                                band_contract=band_contract,
                                normalization=normalization,
                                color_interpretations=color_interpretations,
                                band_names=band_names,
                                require_band_identity=args.contract == "corrected",
                                valid_mask=read_valid,
                            )
                            tensor = (
                                torch.from_numpy(
                                    np.ascontiguousarray(np.moveaxis(normalized, -1, 0), dtype=np.float32)
                                )
                                .unsqueeze(0)
                                .to(device)
                            )
                            height, width = tensor.shape[-2:]
                            pad_height = (args.model_stride - height % args.model_stride) % args.model_stride
                            pad_width = (args.model_stride - width % args.model_stride) % args.model_stride
                            if pad_height or pad_width:
                                tensor = functional.pad(tensor, (0, pad_width, 0, pad_height), mode="replicate")
                            prediction = (
                                model(tensor).argmax(dim=1)[0, :height, :width].cpu().numpy().astype(np.uint8)
                            )
                        else:
                            prediction = np.zeros(read_valid.shape, dtype=np.uint8)
                        offset_column, offset_row = core_offset
                        core_prediction = prediction[
                            offset_row : offset_row + int(core_window.height),
                            offset_column : offset_column + int(core_window.width),
                        ]
                        core_source_valid = read_valid[
                            offset_row : offset_row + int(core_window.height),
                            offset_column : offset_column + int(core_window.width),
                        ]
                        output_valid = core_source_valid.copy()
                        slope_valid = None
                        if slope_masker is not None:
                            slope_valid = slope_masker.mask(source, core_window).astype(bool)
                            if args.contract != "corrected":
                                core_prediction *= slope_valid
                        if args.contract == "corrected":
                            core, output_valid = compose_corrected_prediction(
                                core_prediction,
                                core_source_valid,
                                slope_valid,
                            )
                        else:
                            core = core_prediction
                        destination.write(core, 1, window=core_window)
                        valid_predictions = core[output_valid]
                        counts += np.bincount(valid_predictions, minlength=len(CLASS_NAMES))[: len(CLASS_NAMES)]
                        core_pixels = int(core.size)
                        written_pixels += core_pixels
                        source_valid_pixels += int(core_source_valid.sum())
                        prediction_valid_pixels += int(output_valid.sum())
                        nodata_pixels += core_pixels - int(output_valid.sum())
                        tile_count += 1
                else:
                    # Validation above confines this branch to historical internal blocks.
                    def prepared_tiles() -> Any:
                        for read_values, core_values, core_offset in windows:
                            read_window = Window(*read_values)
                            core_window = Window(*core_values)
                            source_data = source.read(window=read_window)
                            read_valid = np.ones(source_data.shape[1:], dtype=bool)
                            normalized = normalize_hwc(
                                np.moveaxis(source_data, 0, -1),
                                band_contract=band_contract,
                                normalization=normalization,
                                color_interpretations=color_interpretations,
                                band_names=band_names,
                                require_band_identity=False,
                                valid_mask=read_valid,
                            )
                            tensor = torch.from_numpy(
                                np.ascontiguousarray(np.moveaxis(normalized, -1, 0), dtype=np.float32)
                            ).unsqueeze(0)
                            height, width = tensor.shape[-2:]
                            pad_height = (args.model_stride - height % args.model_stride) % args.model_stride
                            pad_width = (args.model_stride - width % args.model_stride) % args.model_stride
                            if pad_height or pad_width:
                                tensor = functional.pad(tensor, (0, pad_width, 0, pad_height), mode="replicate")
                            yield PreparedInferenceTile(
                                tensor=tensor,
                                read_valid=read_valid,
                                core_window=core_window,
                                core_offset=core_offset,
                                unpadded_height=height,
                                unpadded_width=width,
                                padded_shape=(height + pad_height, width + pad_width),
                            )

                    for batch in iter_equal_padded_shape_batches(prepared_tiles(), batch_size):
                        predictions = predict_prepared_tile_batch(
                            batch,
                            model=model,
                            device=device,
                            torch=torch,
                        )
                        for item, prediction in zip(batch, predictions, strict=True):
                            offset_column, offset_row = item.core_offset
                            core_prediction = prediction[
                                offset_row : offset_row + int(item.core_window.height),
                                offset_column : offset_column + int(item.core_window.width),
                            ]
                            core_source_valid = item.read_valid[
                                offset_row : offset_row + int(item.core_window.height),
                                offset_column : offset_column + int(item.core_window.width),
                            ]
                            output_valid = core_source_valid.copy()
                            if slope_masker is not None:
                                slope_valid = slope_masker.mask(source, item.core_window).astype(bool)
                                core_prediction *= slope_valid
                            core = core_prediction
                            destination.write(core, 1, window=item.core_window)
                            valid_predictions = core[output_valid]
                            counts += np.bincount(valid_predictions, minlength=len(CLASS_NAMES))[: len(CLASS_NAMES)]
                            core_pixels = int(core.size)
                            written_pixels += core_pixels
                            source_valid_pixels += int(core_source_valid.sum())
                            prediction_valid_pixels += int(output_valid.sum())
                            nodata_pixels += core_pixels - int(output_valid.sum())
                            tile_count += 1

            expected_pixels = int(source.width) * int(source.height)
            if written_pixels != expected_pixels:
                raise ContractError(
                    f"Inference windows wrote {written_pixels} pixels, but the raster contains {expected_pixels}; "
                    "right/bottom edge coverage is incomplete."
                )
            if prediction_valid_pixels + nodata_pixels != expected_pixels:
                raise ContractError("Prediction validity/nodata accounting does not cover the output raster.")
            if int(counts.sum()) != prediction_valid_pixels:
                raise ContractError("Class counts do not equal the number of valid prediction pixels.")

    try:
        write_summary(summary, args.contract, counts)
        arguments = vars(args).copy()
        arguments.update(
            {
                "band_contract": band_contract,
                "normalization": normalization,
                "output": output,
                "summary": summary,
                "manifest": manifest,
                "window_mode_effective": window_mode,
            }
        )
        inputs = [args.weights, args.input_raster]
        if args.slope_mask:
            inputs.append(args.slope_vector)
        if corrected_lineage is not None:
            training_manifest, imagery_manifest, slope_manifest = corrected_lineage_manifest_paths(corrected_lineage)
            inputs.extend((training_manifest, imagery_manifest, slope_manifest))
        write_run_manifest(
            manifest,
            workflow_root=project_paths.WORKFLOW_ROOT,
            stage="05_infer_segmentation",
            contract=args.contract,
            arguments=arguments,
            inputs=inputs,
            outputs=(output, summary),
            extra={
                "window_contract": (
                    "source-geotiff-internal-blocks"
                    if window_mode == "internal-blocks"
                    else "fixed-core-with-context-halo"
                ),
                "window_mode": window_mode,
                "batch_size": batch_size,
                "tile_size": args.tile_size,
                "overlap": args.overlap,
                "core_size": args.tile_size - 2 * args.overlap if window_mode == "fixed-grid" else None,
                "tile_count": tile_count,
                "class_pixel_counts": {name: int(value) for name, value in zip(CLASS_NAMES, counts, strict=True)},
                "slope_mask_applied": bool(args.slope_mask),
                "source_valid_pixels": source_valid_pixels,
                "prediction_valid_pixels": prediction_valid_pixels,
                "nodata_pixels": nodata_pixels,
                "prediction_nodata_value": PREDICTION_NODATA if args.contract == "corrected" else None,
                "source_validity_contract": (
                    "source-validity-plus-finite-radiometry" if args.contract == "corrected" else None
                ),
                "corridor_manifest_sha256": (
                    sha256_file(corrected_lineage[1]) if corrected_lineage is not None else None
                ),
                "background_is_not_nodata": True,
                "randomness": randomness,
                "device": str(device),
                "checkpoint_metadata": checkpoint_metadata,
                "versions": {
                    "torch": getattr(torch, "__version__", None),
                    "segmentation_models_pytorch": getattr(smp, "__version__", None),
                    "numpy": getattr(np, "__version__", None),
                    "rasterio": getattr(rasterio, "__version__", None),
                },
            },
        )
    except BaseException:
        manifest.unlink(missing_ok=True)
        summary.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        raise
    print(json.dumps({"output": str(output), "summary": str(summary), "tiles": tile_count}, indent=2))
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
