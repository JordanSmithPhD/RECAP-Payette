"""Portable paths for the public RECAP-Payette workflows.

The repository contains code, configuration, schemas, and synthetic examples;
it does not contain primary imagery, camera photographs, labels, or trained
weights.  Relative defaults keep a clone self-contained.  Large or restricted
inputs can live anywhere by setting environment variables or a local
``config.yaml`` (which is ignored by Git).

Precedence is environment variable, ``config.yaml``, then repository default.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


WORKFLOW_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = WORKFLOW_ROOT / "config.yaml"


def _load_config() -> tuple[dict[str, Any], Path]:
    configured = os.environ.get("RECAP_PAYETTE_CONFIG")
    path = Path(configured).expanduser() if configured else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = WORKFLOW_ROOT / path
    path = path.resolve(strict=False)
    if not path.exists():
        return {}, path
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required when config.yaml is present") from exc
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}, path
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return loaded, path


_CONFIG, CONFIG_PATH = _load_config()


def _config_value(*keys: str) -> Any | None:
    value: Any = _CONFIG
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _path_setting(environment_name: str, keys: tuple[str, ...], default: str | Path) -> Path:
    raw: Any = os.environ.get(environment_name)
    if raw is None:
        raw = _config_value(*keys)
    if raw in (None, ""):
        raw = default
    path = Path(os.path.expandvars(str(raw))).expanduser()
    if not path.is_absolute():
        path = CONFIG_PATH.parent / path
    return path.resolve(strict=False)


def _name_setting(environment_name: str, keys: tuple[str, ...], default: str) -> str:
    raw: Any = os.environ.get(environment_name)
    if raw is None:
        raw = _config_value(*keys)
    value = str(raw if raw not in (None, "") else default).strip()
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{environment_name} must be one directory-safe name")
    return value


# Public-local roots.  Everything below data/ and outputs/ is ignored by Git.
DATA_ROOT = _path_setting("RECAP_PAYETTE_DATA_ROOT", ("paths", "data_root"), WORKFLOW_ROOT / "data")
OUTPUT_ROOT = _path_setting("RECAP_PAYETTE_OUTPUT_ROOT", ("paths", "output_root"), WORKFLOW_ROOT / "outputs")
CAMERA_ROOT = _path_setting(
    "RECAP_PAYETTE_CAMERA_ROOT", ("paths", "camera_root"), DATA_ROOT / "external_camera_archive"
)
TRAINING_ROOT = _path_setting(
    "RECAP_PAYETTE_TRAINING_ROOT", ("paths", "training_root"), DATA_ROOT / "external_training"
)

DATA_RAW_ROOT = DATA_ROOT / "raw"
DATA_INTERMEDIATE_ROOT = DATA_ROOT / "intermediate"
RESULTS_ROOT = OUTPUT_ROOT / "results"
FIGURES_ROOT = OUTPUT_ROOT / "figures"
MODELS_ROOT = OUTPUT_ROOT / "models"
RUNS_ROOT = OUTPUT_ROOT / "runs"
PROCESSING_TEMP_ROOT = OUTPUT_ROOT / "tmp"


# Versioned recommended parking run.
PARKING_RUN_ID = _name_setting("RECAP_PAYETTE_PARKING_RUN_ID", ("parking", "run_id"), "recommended_v1")
PARKING_DATA_RUN_ROOT = DATA_INTERMEDIATE_ROOT / "parking_runs" / PARKING_RUN_ID
PARKING_RESULTS_RUN_ROOT = RESULTS_ROOT / "parking_runs" / PARKING_RUN_ID
PARKING_MODEL_RUN_ROOT = RUNS_ROOT / "segmentation" / PARKING_RUN_ID
PARKING_IMAGERY_RUN_ROOT = PARKING_DATA_RUN_ROOT / "imagery"
PARKING_CORRIDOR_RUN_ROOT = PARKING_DATA_RUN_ROOT / "corridor"
PARKING_TERRAIN_RUN_ROOT = PARKING_DATA_RUN_ROOT / "terrain"
PARKING_LABEL_RUN_ROOT = PARKING_DATA_RUN_ROOT / "labels"
PARKING_CHIP_RUN_ROOT = PARKING_DATA_RUN_ROOT / "chips"
PARKING_INFERENCE_RUN_ROOT = PARKING_DATA_RUN_ROOT / "inference"
PARKING_PROVENANCE_RUN_ROOT = PARKING_DATA_RUN_ROOT / "provenance"

PARKING_EE_SUBMISSION_MANIFEST = PARKING_IMAGERY_RUN_ROOT / "naip.earth_engine.submission.json"
PARKING_EE_RUN_MANIFEST = PARKING_IMAGERY_RUN_ROOT / "naip.earth_engine.run.json"
PARKING_NAIP_MOSAIC = PARKING_IMAGERY_RUN_ROOT / "naip_mosaic_rgbnir.tif"
PARKING_NAIP_MOSAIC_MANIFEST = PARKING_IMAGERY_RUN_ROOT / "naip_mosaic_rgbnir.run.json"
PARKING_SELECTED_ROADS = PARKING_CORRIDOR_RUN_ROOT / "selected_roads.gpkg"
PARKING_SELECTED_ROADS_BUFFER = PARKING_CORRIDOR_RUN_ROOT / "selected_roads_500m_buffer.gpkg"
PARKING_CORRIDOR = PARKING_CORRIDOR_RUN_ROOT / "road_river_corridor_500m.gpkg"
PARKING_CORRIDOR_RASTER = PARKING_CORRIDOR_RUN_ROOT / "naip_corridor_rgbnir.tif"
PARKING_CORRIDOR_MANIFEST = PARKING_CORRIDOR_RUN_ROOT / "road_river_corridor_500m.run.json"
PARKING_DEM = PARKING_TERRAIN_RUN_ROOT / "dem_3dep_1m.tif"
PARKING_SLOPE_LE5 = PARKING_TERRAIN_RUN_ROOT / "slope_le5.gpkg"
PARKING_SLOPE_LE5_CORRIDOR = PARKING_TERRAIN_RUN_ROOT / "slope_le5_corridor.gpkg"
PARKING_SLOPE_MANIFEST = PARKING_TERRAIN_RUN_ROOT / "slope_le5_corridor.run.json"
PARKING_SAM_RAW_PROPOSALS = PARKING_LABEL_RUN_ROOT / "sam_raw_proposals.gpkg"
PARKING_CURATED_CANDIDATES = PARKING_LABEL_RUN_ROOT / "curated_candidates.gpkg"
PARKING_LABEL_VALIDATION_MANIFEST = PARKING_LABEL_RUN_ROOT / "curated_candidates.validation.run.json"
PARKING_SPLITS = PARKING_CHIP_RUN_ROOT / "splits.parquet"
PARKING_PREDICTIONS = PARKING_INFERENCE_RUN_ROOT / "segmentation_predictions.tif"
PARKING_CAPACITY_RUN_TABLE = PARKING_RESULTS_RUN_ROOT / "parking_by_reach.csv"
PARKING_CAPACITY_RUN_POLYGONS = PARKING_RESULTS_RUN_ROOT / "parking_by_reach.gpkg"
PARKING_AREA_COMPARISON_RUN_TABLE = PARKING_RESULTS_RUN_ROOT / "parking_area_comparison.csv"
PARKING_CAPACITY_RUN_MANIFEST = PARKING_RESULTS_RUN_ROOT / "parking_by_reach.run.json"
PARKING_ROAD_PROXIMITY_RUN_TABLE = PARKING_RESULTS_RUN_ROOT / "parking_road_proximity.csv"


# User-supplied source vectors and historical-replay artifacts.  These names
# encode the expected layout but carry no assumption that the files ship here.
NAIP_EXPORT_ROOT = DATA_RAW_ROOT / "naip_exports"
NHDPLUS_FLOWLINES = DATA_RAW_ROOT / "hydrography" / "flowlines.gpkg"
TIGER_ROADS = DATA_RAW_ROOT / "roads" / "tiger_primary_roads.gpkg"
PAYETTE_RIVER_SOI = DATA_RAW_ROOT / "hydrography" / "river_system_of_interest.gpkg"
STUDY_AREAS = DATA_RAW_ROOT / "study_areas" / "study_areas.gpkg"
MANUAL_PARKING_LABELS = DATA_RAW_ROOT / "labels" / "manual_parking_labels.gpkg"

NAIP_CORRIDOR_RASTER = DATA_INTERMEDIATE_ROOT / "historical" / "naip_corridor_rgba.tif"
SLOPE_LT5_CLIPPED = DATA_INTERMEDIATE_ROOT / "historical" / "slope_lt5_corridor.gpkg"
SAM_RAW_PROPOSALS = DATA_INTERMEDIATE_ROOT / "historical" / "sam_raw_proposals.gpkg"
CURATED_CANDIDATES = DATA_INTERMEDIATE_ROOT / "historical" / "curated_candidates.gpkg"
SEGMENTATION_CHIP_ROOT = DATA_INTERMEDIATE_ROOT / "historical" / "training_chips"
SEGMENTATION_SPLITS = SEGMENTATION_CHIP_ROOT / "splits.parquet"
SEGMENTATION_INFERENCE_ROOT = DATA_INTERMEDIATE_ROOT / "historical" / "inference"
DEEPLAB_V3_WEIGHTS = MODELS_ROOT / "historical" / "deeplab_v3_augmented.pth"
DEEPLAB_V3_PREDICTIONS = SEGMENTATION_INFERENCE_ROOT / "deeplab_v3_augmented_predictions.tif"
PARKING_PREDICTIONS_LEGACY = DEEPLAB_V3_PREDICTIONS
PARKING_ROAD_PROXIMITY = RESULTS_ROOT / "historical" / "parking_road_proximity.csv"


# Camera manifests, labels, and weights remain external or ignored.
MANIFEST_ROOT = DATA_INTERMEDIATE_ROOT / "camera"
DAYTIME_MANIFEST = MANIFEST_ROOT / "image_manifest_daylight.parquet"
BASELINE_DETECTIONS_MANIFEST = MANIFEST_ROOT / "annotation_pool_detections.parquet"
FINAL_DETECTIONS_MANIFEST = MANIFEST_ROOT / "full_detections.parquet"
DWELL_TRACKS = RESULTS_ROOT / "camera" / "tracks.parquet"
DWELL_OBSERVATIONS = RESULTS_ROOT / "camera" / "track_observations.parquet"
AL_BATCHES_ROOT = OUTPUT_ROOT / "annotation_batches"
YOLO_AL_V1_ROOT = TRAINING_ROOT / "yolo_round_1"
FINAL_WEIGHTS = MODELS_ROOT / "historical" / "operational_detector.pt"


def remap_camera_path(value: str | Path, camera_root: Path = CAMERA_ROOT) -> Path:
    """Resolve a portable manifest path beneath the configured archive root.

    Public manifests should store paths relative to ``camera_root``.  Absolute
    paths are preserved so a caller can report a missing or unauthorized file
    instead of silently redirecting it.
    """

    path = Path(str(value).strip()).expanduser()
    return path if path.is_absolute() else camera_root / path


def describe_layout() -> dict[str, str]:
    """Return effective roots for diagnostics and run manifests."""

    return {
        "workflow_root": str(WORKFLOW_ROOT),
        "data_root": str(DATA_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "camera_root": str(CAMERA_ROOT),
        "training_root": str(TRAINING_ROOT),
        "parking_run_id": PARKING_RUN_ID,
    }
