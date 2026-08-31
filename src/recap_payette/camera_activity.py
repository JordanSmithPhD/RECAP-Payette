"""Data-free camera tracking, hourly activity, and calibration utilities.

The functions in this module operate on normalized tabular inputs.  They do
not read camera photographs and do not require site-specific paths.  Two modes
are explicit:

``historical_replay``
    Reproduces the archived aggregation behavior: greedy IoU matching at 0.50,
    tracks never expire, singleton observations remain in the observation
    table but are omitted from the track summary, and hours without detections
    are absent from the hourly output.

``recommended``
    Expires inactive tracks, retains every track in the summary, and creates a
    complete site-hour-by-class grid from the image manifest.  A zero is only
    reported when at least one image in the hour was readable; hours with no
    readable image receive a missing activity value instead of a false zero.

The resulting ``relative_activity_index`` is a detection-conditioned index of
distinct tracks observed per monitored site-hour.  It is not occupancy,
visitation, throughput, or physical carrying capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd


ActivityMode = Literal["historical_replay", "recommended"]
CalibrationGrouping = Literal["class", "site_class"]

DEFAULT_ACTIVITY_CLASSES = ("vehicle", "watercraft")
INDICATOR_DEFINITION = (
    "Detection-conditioned number of distinct object tracks observed per "
    "monitored site-hour; a relative activity index, not an occupancy or "
    "visitation estimate."
)

MANIFEST_REQUIRED = (
    "image_id",
    "site_id",
    "captured_at",
    "acquired",
    "readable",
)
DETECTION_REQUIRED = (
    "detection_id",
    "image_id",
    "class_name",
    "confidence",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
)
OBSERVATION_COLUMNS = (
    "detection_id",
    "image_id",
    "site_id",
    "captured_at",
    "hour_start",
    "class_name",
    "confidence",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "track_id",
    "track_observation_ordinal",
    "seconds_since_previous_observation",
    "included_in_track_summary",
)
TRACK_SUMMARY_COLUMNS = (
    "site_id",
    "class_name",
    "track_id",
    "started_at",
    "ended_at",
    "elapsed_seconds",
    "detection_observations",
    "image_observations",
    "first_image_id",
    "last_image_id",
    "summary_rule",
)
HOURLY_COLUMNS = (
    "site_id",
    "class_name",
    "hour_start",
    "manifest_images",
    "images_acquired",
    "images_readable",
    "images_with_detections",
    "readable_coverage_fraction",
    "acquisition_coverage_fraction",
    "detected_object_observations",
    "relative_activity_index",
    "monitoring_status",
    "indicator_definition",
    "aggregation_mode",
)


class CameraActivityError(ValueError):
    """Raised when a tabular input violates the public workflow contract."""


@dataclass(frozen=True)
class ActivityOutputs:
    """Normalized outputs from tracking and hourly aggregation."""

    track_observations: pd.DataFrame
    track_summary: pd.DataFrame
    hourly_activity: pd.DataFrame


@dataclass(frozen=True)
class CalibrationOutputs:
    """Calibration diagnostics, matched rows, and hourly estimates."""

    calibration_models: pd.DataFrame
    calibration_matches: pd.DataFrame
    calibrated_hourly_activity: pd.DataFrame


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Parquet table from ``path``."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise CameraActivityError(
                "Reading Parquet requires pyarrow. Install the camera workflow extras."
            ) from exc
    raise CameraActivityError(f"Unsupported table format for {path}; use CSV or Parquet.")


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    """Write a CSV or Parquet table, creating its parent directory."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
        return
    if suffix in {".parquet", ".pq"}:
        try:
            frame.to_parquet(path, index=False)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise CameraActivityError(
                "Writing Parquet requires pyarrow. Install the camera workflow extras."
            ) from exc
        return
    raise CameraActivityError(f"Unsupported table format for {path}; use CSV or Parquet.")


def _require_columns(frame: pd.DataFrame, required: Sequence[str], table_name: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise CameraActivityError(f"{table_name} is missing required columns: {missing}")


def _nonempty_strings(series: pd.Series, column: str, table_name: str) -> pd.Series:
    if series.isna().any():
        raise CameraActivityError(f"{table_name}.{column} contains missing values.")
    result = series.astype(str).str.strip()
    if result.eq("").any():
        raise CameraActivityError(f"{table_name}.{column} contains empty values.")
    return result


def _coerce_boolean(series: pd.Series, column: str) -> pd.Series:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
    }
    if pd.api.types.is_bool_dtype(series):
        if series.isna().any():
            raise CameraActivityError(f"image_manifest.{column} contains missing values.")
        return series.astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    result = normalized.map(mapping)
    if result.isna().any():
        examples = sorted(normalized[result.isna()].dropna().unique().tolist())[:5]
        raise CameraActivityError(
            f"image_manifest.{column} must contain boolean values; invalid examples: {examples}"
        )
    return result.astype(bool)


def normalize_image_manifest(image_manifest: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the image manifest.

    ``captured_at`` values are converted to UTC.  Naive timestamps are treated
    as UTC; callers with local timestamps should attach an offset before use.
    Each row represents a scheduled/acquired image opportunity.  ``readable``
    cannot be true when ``acquired`` is false.
    """

    _require_columns(image_manifest, MANIFEST_REQUIRED, "image_manifest")
    frame = image_manifest.loc[:, list(MANIFEST_REQUIRED)].copy()
    frame["image_id"] = _nonempty_strings(frame["image_id"], "image_id", "image_manifest")
    frame["site_id"] = _nonempty_strings(frame["site_id"], "site_id", "image_manifest")
    if frame["image_id"].duplicated().any():
        examples = frame.loc[frame["image_id"].duplicated(False), "image_id"].unique()[:5]
        raise CameraActivityError(f"image_manifest.image_id must be unique; examples: {examples.tolist()}")
    parsed = pd.to_datetime(frame["captured_at"], errors="coerce", utc=True)
    if parsed.isna().any():
        raise CameraActivityError("image_manifest.captured_at contains invalid or missing timestamps.")
    frame["captured_at"] = parsed
    frame["acquired"] = _coerce_boolean(frame["acquired"], "acquired")
    frame["readable"] = _coerce_boolean(frame["readable"], "readable")
    if (frame["readable"] & ~frame["acquired"]).any():
        raise CameraActivityError("An image cannot be readable when acquired is false.")
    frame["hour_start"] = frame["captured_at"].dt.floor("h")
    return frame.sort_values(["site_id", "captured_at", "image_id"], kind="mergesort").reset_index(drop=True)


def normalize_detections(
    detections: pd.DataFrame,
    image_manifest: pd.DataFrame,
    *,
    class_names: Iterable[str] | None = DEFAULT_ACTIVITY_CLASSES,
) -> pd.DataFrame:
    """Validate normalized long-form detections against an image manifest."""

    _require_columns(detections, DETECTION_REQUIRED, "detections")
    frame = detections.loc[:, list(DETECTION_REQUIRED)].copy()
    frame["detection_id"] = _nonempty_strings(frame["detection_id"], "detection_id", "detections")
    frame["image_id"] = _nonempty_strings(frame["image_id"], "image_id", "detections")
    frame["class_name"] = (
        _nonempty_strings(frame["class_name"], "class_name", "detections").str.casefold()
    )
    if frame["detection_id"].duplicated().any():
        raise CameraActivityError("detections.detection_id must be unique.")

    if class_names is not None:
        allowed = {str(value).strip().casefold() for value in class_names}
        unexpected = sorted(set(frame["class_name"]).difference(allowed))
        if unexpected:
            raise CameraActivityError(
                f"Detections contain classes outside the monitored class list: {unexpected}"
            )

    numeric = ["confidence", "x_min", "y_min", "x_max", "y_max"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any() or not np.isfinite(frame[column]).all():
            raise CameraActivityError(f"detections.{column} must contain finite numeric values.")
    if ((frame["confidence"] < 0) | (frame["confidence"] > 1)).any():
        raise CameraActivityError("detections.confidence must fall within [0, 1].")
    if (frame["x_max"] <= frame["x_min"]).any() or (frame["y_max"] <= frame["y_min"]).any():
        raise CameraActivityError("Every detection bounding box must have positive width and height.")

    manifest_fields = image_manifest[["image_id", "site_id", "captured_at", "acquired", "readable"]]
    merged = frame.merge(manifest_fields, on="image_id", how="left", validate="many_to_one", indicator=True)
    missing = merged["_merge"].eq("left_only")
    if missing.any():
        examples = merged.loc[missing, "image_id"].unique()[:5]
        raise CameraActivityError(f"Detections reference unknown image_id values: {examples.tolist()}")
    merged = merged.drop(columns="_merge")
    if (~merged["acquired"] | ~merged["readable"]).any():
        examples = merged.loc[~merged["acquired"] | ~merged["readable"], "image_id"].unique()[:5]
        raise CameraActivityError(
            "Detections may only reference acquired, readable images; invalid image_id examples: "
            f"{examples.tolist()}"
        )
    merged["hour_start"] = merged["captured_at"].dt.floor("h")
    return merged.sort_values(
        ["site_id", "class_name", "captured_at", "image_id", "detection_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def intersection_over_union(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Return intersection over union for two ``(x_min, y_min, x_max, y_max)`` boxes."""

    ax1, ay1, ax2, ay2 = (float(value) for value in box_a)
    bx1, by1, bx2, by2 = (float(value) for value in box_b)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _track_detections(
    detections: pd.DataFrame,
    *,
    mode: ActivityMode,
    iou_threshold: float,
    max_gap_seconds: float,
) -> pd.DataFrame:
    if not 0 < iou_threshold <= 1:
        raise CameraActivityError("iou_threshold must fall within (0, 1].")
    if mode == "historical_replay" and not np.isclose(iou_threshold, 0.5):
        raise CameraActivityError("historical_replay fixes iou_threshold at 0.50.")
    if mode == "recommended" and max_gap_seconds <= 0:
        raise CameraActivityError("recommended mode requires max_gap_seconds > 0.")
    if detections.empty:
        return _empty_frame(OBSERVATION_COLUMNS)

    observations: list[dict[str, object]] = []
    for (site_id, class_name), group in detections.groupby(["site_id", "class_name"], sort=True):
        active: dict[str, dict[str, object]] = {}
        next_track = 1
        for (_, _), image_group in group.groupby(["captured_at", "image_id"], sort=True):
            image_group = image_group.sort_values("detection_id", kind="mergesort").reset_index(drop=True)
            captured_at = image_group.loc[0, "captured_at"]
            if mode == "recommended":
                expired = [
                    track_id
                    for track_id, state in active.items()
                    if (captured_at - state["last_seen"]).total_seconds() > max_gap_seconds
                ]
                for track_id in expired:
                    del active[track_id]

            boxes = image_group[["x_min", "y_min", "x_max", "y_max"]].to_numpy(dtype=float)
            candidates: list[tuple[float, str, str, int]] = []
            for detection_position, box in enumerate(boxes):
                detection_id = str(image_group.loc[detection_position, "detection_id"])
                for track_id, state in active.items():
                    iou = intersection_over_union(box, state["last_box"])
                    if iou >= iou_threshold:
                        candidates.append((-iou, track_id, detection_id, detection_position))
            candidates.sort()

            assigned: dict[int, str] = {}
            used_tracks: set[str] = set()
            for _, track_id, _, detection_position in candidates:
                if detection_position in assigned or track_id in used_tracks:
                    continue
                assigned[detection_position] = track_id
                used_tracks.add(track_id)

            for detection_position, row in image_group.iterrows():
                track_id = assigned.get(detection_position)
                if track_id is None:
                    track_id = f"{site_id}:{class_name}:{next_track:06d}"
                    next_track += 1
                    previous_seen = None
                    ordinal = 1
                else:
                    state = active[track_id]
                    previous_seen = state["last_seen"]
                    ordinal = int(state["observations"]) + 1
                gap = (
                    pd.NA
                    if previous_seen is None
                    else float((row["captured_at"] - previous_seen).total_seconds())
                )
                box = (row["x_min"], row["y_min"], row["x_max"], row["y_max"])
                active[track_id] = {
                    "last_box": box,
                    "last_seen": row["captured_at"],
                    "observations": ordinal,
                }
                observations.append(
                    {
                        "detection_id": row["detection_id"],
                        "image_id": row["image_id"],
                        "site_id": row["site_id"],
                        "captured_at": row["captured_at"],
                        "hour_start": row["hour_start"],
                        "class_name": row["class_name"],
                        "confidence": float(row["confidence"]),
                        "x_min": float(row["x_min"]),
                        "y_min": float(row["y_min"]),
                        "x_max": float(row["x_max"]),
                        "y_max": float(row["y_max"]),
                        "track_id": track_id,
                        "track_observation_ordinal": ordinal,
                        "seconds_since_previous_observation": gap,
                    }
                )
    result = pd.DataFrame(observations)
    return result.sort_values(
        ["site_id", "class_name", "captured_at", "image_id", "detection_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _summarize_tracks(observations: pd.DataFrame, mode: ActivityMode) -> tuple[pd.DataFrame, pd.DataFrame]:
    if observations.empty:
        observations = _empty_frame(OBSERVATION_COLUMNS)
        return observations, _empty_frame(TRACK_SUMMARY_COLUMNS)

    ordered = observations.sort_values(
        ["site_id", "class_name", "track_id", "captured_at", "image_id", "detection_id"],
        kind="mergesort",
    )
    summary = (
        ordered.groupby(["site_id", "class_name", "track_id"], sort=True, as_index=False)
        .agg(
            started_at=("captured_at", "first"),
            ended_at=("captured_at", "last"),
            detection_observations=("detection_id", "size"),
            image_observations=("image_id", "nunique"),
            first_image_id=("image_id", "first"),
            last_image_id=("image_id", "last"),
        )
    )
    summary["elapsed_seconds"] = (summary["ended_at"] - summary["started_at"]).dt.total_seconds()
    if mode == "historical_replay":
        summary = summary.loc[summary["detection_observations"] >= 2].copy()
        summary["summary_rule"] = "historical_minimum_two_detection_observations"
    else:
        summary["summary_rule"] = "recommended_all_tracks"
    summary = summary.loc[:, list(TRACK_SUMMARY_COLUMNS)].reset_index(drop=True)
    included = set(summary["track_id"])
    ordered["included_in_track_summary"] = ordered["track_id"].isin(included)
    ordered = ordered.loc[:, list(OBSERVATION_COLUMNS)].reset_index(drop=True)
    return ordered, summary


def _manifest_hour_coverage(image_manifest: pd.DataFrame) -> pd.DataFrame:
    working = image_manifest.copy()
    working["acquired_readable"] = working["acquired"] & working["readable"]
    coverage = (
        working.groupby(["site_id", "hour_start"], sort=True, as_index=False)
        .agg(
            manifest_images=("image_id", "size"),
            images_acquired=("acquired", "sum"),
            images_readable=("acquired_readable", "sum"),
        )
    )
    coverage["readable_coverage_fraction"] = np.where(
        coverage["images_acquired"] > 0,
        coverage["images_readable"] / coverage["images_acquired"],
        np.nan,
    )
    coverage["acquisition_coverage_fraction"] = (
        coverage["images_acquired"] / coverage["manifest_images"]
    )
    return coverage


def _detected_hour_metrics(observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame(
            columns=[
                "site_id",
                "class_name",
                "hour_start",
                "images_with_detections",
                "detected_object_observations",
                "relative_activity_index",
            ]
        )
    return (
        observations.groupby(["site_id", "class_name", "hour_start"], sort=True, as_index=False)
        .agg(
            images_with_detections=("image_id", "nunique"),
            detected_object_observations=("detection_id", "size"),
            relative_activity_index=("track_id", "nunique"),
        )
    )


def _summarize_hourly(
    image_manifest: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    mode: ActivityMode,
    class_names: Sequence[str],
) -> pd.DataFrame:
    coverage = _manifest_hour_coverage(image_manifest)
    detected = _detected_hour_metrics(observations)

    if mode == "historical_replay":
        if detected.empty:
            return _empty_frame(HOURLY_COLUMNS)
        hourly = detected.merge(coverage, on=["site_id", "hour_start"], how="left", validate="many_to_one")
        hourly["monitoring_status"] = "observed_detection"
    else:
        classes = pd.DataFrame({"class_name": list(class_names)})
        if coverage.empty or classes.empty:
            return _empty_frame(HOURLY_COLUMNS)
        coverage = coverage.assign(_join_key=1)
        classes = classes.assign(_join_key=1)
        hourly = coverage.merge(classes, on="_join_key", how="inner").drop(columns="_join_key")
        hourly = hourly.merge(
            detected,
            on=["site_id", "class_name", "hour_start"],
            how="left",
            validate="one_to_one",
        )
        metric_columns = [
            "images_with_detections",
            "detected_object_observations",
            "relative_activity_index",
        ]
        readable = hourly["images_readable"] > 0
        for column in metric_columns:
            hourly[column] = pd.to_numeric(hourly[column], errors="coerce").astype("Float64")
            hourly.loc[readable & hourly[column].isna(), column] = 0
        hourly["monitoring_status"] = np.select(
            [~readable, hourly["relative_activity_index"].fillna(0).gt(0)],
            ["unobserved_no_readable_image", "observed_detection"],
            default="observed_zero",
        )

    hourly["indicator_definition"] = INDICATOR_DEFINITION
    hourly["aggregation_mode"] = mode
    return hourly.loc[:, list(HOURLY_COLUMNS)].sort_values(
        ["site_id", "class_name", "hour_start"], kind="mergesort"
    ).reset_index(drop=True)


def build_camera_activity(
    image_manifest: pd.DataFrame,
    detections: pd.DataFrame,
    *,
    mode: ActivityMode = "recommended",
    class_names: Sequence[str] = DEFAULT_ACTIVITY_CLASSES,
    iou_threshold: float = 0.5,
    max_gap_seconds: float = 900.0,
) -> ActivityOutputs:
    """Track detections and calculate a detection-conditioned hourly index."""

    if mode not in {"historical_replay", "recommended"}:
        raise CameraActivityError(f"Unsupported aggregation mode: {mode}")
    normalized_classes = tuple(dict.fromkeys(str(value).strip().casefold() for value in class_names))
    if not normalized_classes or any(not value for value in normalized_classes):
        raise CameraActivityError("class_names must contain at least one non-empty class name.")
    manifest = normalize_image_manifest(image_manifest)
    normalized_detections = normalize_detections(
        detections,
        manifest,
        class_names=normalized_classes,
    )
    observations = _track_detections(
        normalized_detections,
        mode=mode,
        iou_threshold=iou_threshold,
        max_gap_seconds=max_gap_seconds,
    )
    observations, track_summary = _summarize_tracks(observations, mode)
    hourly = _summarize_hourly(
        manifest,
        observations,
        mode=mode,
        class_names=normalized_classes,
    )
    return ActivityOutputs(observations, track_summary, hourly)


def _normalize_hourly_activity(hourly_activity: pd.DataFrame) -> pd.DataFrame:
    required = ("site_id", "class_name", "hour_start", "relative_activity_index")
    _require_columns(hourly_activity, required, "hourly_activity")
    frame = hourly_activity.copy()
    frame["site_id"] = _nonempty_strings(frame["site_id"], "site_id", "hourly_activity")
    frame["class_name"] = _nonempty_strings(
        frame["class_name"], "class_name", "hourly_activity"
    ).str.casefold()
    frame["hour_start"] = pd.to_datetime(frame["hour_start"], errors="coerce", utc=True)
    if frame["hour_start"].isna().any():
        raise CameraActivityError("hourly_activity.hour_start contains invalid timestamps.")
    frame["relative_activity_index"] = pd.to_numeric(
        frame["relative_activity_index"], errors="coerce"
    )
    keys = ["site_id", "class_name", "hour_start"]
    if frame.duplicated(keys).any():
        raise CameraActivityError(f"hourly_activity must be unique by {keys}.")
    return frame


def normalize_manual_reference(manual_counts: pd.DataFrame) -> pd.DataFrame:
    """Validate neutral manual reference observations used for calibration.

    The input file may be named ``manual_counts.csv`` for convenience, but its
    required measurement column is ``manual_reference_value``. This keeps the
    calibration input explicit and independent of the detector output.
    """

    required = ("site_id", "class_name", "hour_start", "manual_reference_value")
    _require_columns(manual_counts, required, "manual_counts")
    frame = manual_counts.loc[:, list(required)].copy()
    frame["site_id"] = _nonempty_strings(frame["site_id"], "site_id", "manual_counts")
    frame["class_name"] = _nonempty_strings(
        frame["class_name"], "class_name", "manual_counts"
    ).str.casefold()
    frame["hour_start"] = pd.to_datetime(frame["hour_start"], errors="coerce", utc=True)
    if frame["hour_start"].isna().any():
        raise CameraActivityError("manual_counts.hour_start contains invalid timestamps.")
    frame["manual_reference_value"] = pd.to_numeric(
        frame["manual_reference_value"], errors="coerce"
    )
    if frame["manual_reference_value"].isna().any() or not np.isfinite(
        frame["manual_reference_value"]
    ).all():
        raise CameraActivityError("manual_counts.manual_reference_value must be finite and numeric.")
    if (frame["manual_reference_value"] < 0).any():
        raise CameraActivityError("manual_counts.manual_reference_value cannot be negative.")
    keys = ["site_id", "class_name", "hour_start"]
    if frame.duplicated(keys).any():
        raise CameraActivityError(f"manual_counts must be unique by {keys}.")
    return frame


def _fit_linear_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    include_intercept: bool,
    minimum_matches: int,
) -> dict[str, object]:
    n = int(len(x))
    if n < minimum_matches:
        return {
            "matched_site_hours": n,
            "intercept": np.nan,
            "slope": np.nan,
            "r_squared": np.nan,
            "model_status": "insufficient_matches",
        }
    if include_intercept:
        design = np.column_stack([np.ones(n), x])
        if np.linalg.matrix_rank(design) < 2:
            return {
                "matched_site_hours": n,
                "intercept": float(np.mean(y)),
                "slope": 0.0,
                "r_squared": np.nan,
                "model_status": "insufficient_index_variation",
            }
        intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    else:
        if np.allclose(x, 0):
            return {
                "matched_site_hours": n,
                "intercept": 0.0,
                "slope": np.nan,
                "r_squared": np.nan,
                "model_status": "insufficient_index_variation",
            }
        intercept = 0.0
        slope = float(np.dot(x, y) / np.dot(x, x))
    fitted = intercept + slope * x
    residual_sum = float(np.square(y - fitted).sum())
    total_sum = float(np.square(y - y.mean()).sum())
    r_squared = np.nan if np.isclose(total_sum, 0) else 1.0 - residual_sum / total_sum
    return {
        "matched_site_hours": n,
        "intercept": float(intercept),
        "slope": float(slope),
        "r_squared": float(r_squared),
        "model_status": "fit",
    }


def calibrate_relative_activity(
    hourly_activity: pd.DataFrame,
    manual_counts: pd.DataFrame,
    *,
    grouping: CalibrationGrouping = "class",
    include_intercept: bool = True,
    minimum_matches: int = 2,
) -> CalibrationOutputs:
    """Fit transparent linear checks against independent manual references.

    Calibration does not change the interpretation of the source index.  The
    generated estimate is called ``calibrated_reference_estimate`` and remains
    conditional on camera placement, readable imagery, detector behavior, and
    the supplied reference observations.
    """

    if grouping not in {"class", "site_class"}:
        raise CameraActivityError(f"Unsupported calibration grouping: {grouping}")
    if minimum_matches < 2:
        raise CameraActivityError("minimum_matches must be at least 2.")
    hourly = _normalize_hourly_activity(hourly_activity)
    manual = normalize_manual_reference(manual_counts)
    keys = ["site_id", "class_name", "hour_start"]
    matches = hourly.merge(manual, on=keys, how="inner", validate="one_to_one")
    matches = matches.loc[matches["relative_activity_index"].notna()].copy()

    group_columns = ["class_name"] if grouping == "class" else ["site_id", "class_name"]
    models: list[dict[str, object]] = []
    for group_key, group in matches.groupby(group_columns, sort=True):
        key_values = (group_key,) if not isinstance(group_key, tuple) else group_key
        row = dict(zip(group_columns, key_values, strict=True))
        row.update(
            _fit_linear_model(
                group["relative_activity_index"].to_numpy(dtype=float),
                group["manual_reference_value"].to_numpy(dtype=float),
                include_intercept=include_intercept,
                minimum_matches=minimum_matches,
            )
        )
        row["calibration_scope"] = grouping
        row["include_intercept"] = include_intercept
        models.append(row)

    model_columns = [
        *group_columns,
        "calibration_scope",
        "include_intercept",
        "matched_site_hours",
        "intercept",
        "slope",
        "r_squared",
        "model_status",
    ]
    model_frame = pd.DataFrame(models, columns=model_columns)
    if model_frame.empty:
        matches["calibrated_reference_estimate"] = np.nan
        calibrated = hourly.copy()
        calibrated["calibrated_reference_estimate"] = np.nan
        calibrated["calibration_model_status"] = pd.NA
        return CalibrationOutputs(model_frame, matches, calibrated)

    merge_columns = [*group_columns, "intercept", "slope", "model_status"]
    matches = matches.merge(model_frame[merge_columns], on=group_columns, how="left", validate="many_to_one")
    matches["calibrated_reference_estimate"] = (
        matches["intercept"] + matches["slope"] * matches["relative_activity_index"]
    )
    matches = matches.rename(columns={"model_status": "calibration_model_status"})

    calibrated = hourly.merge(model_frame[merge_columns], on=group_columns, how="left", validate="many_to_one")
    calibrated["calibrated_reference_estimate"] = (
        calibrated["intercept"] + calibrated["slope"] * calibrated["relative_activity_index"]
    )
    calibrated = calibrated.rename(columns={"model_status": "calibration_model_status"})
    calibrated = calibrated.drop(columns=["intercept", "slope"])
    return CalibrationOutputs(model_frame, matches, calibrated)
