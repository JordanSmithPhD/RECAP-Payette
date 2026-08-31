from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from recap_payette.camera_activity import (  # noqa: E402
    CameraActivityError,
    build_camera_activity,
    calibrate_relative_activity,
    intersection_over_union,
)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": ["i1", "i2", "i3", "i4", "i5"],
            "site_id": ["parking_a"] * 5,
            "captured_at": [
                "2025-06-01T12:00:00Z",
                "2025-06-01T12:05:00Z",
                "2025-06-01T13:00:00Z",
                "2025-06-01T14:00:00Z",
                "2025-06-01T15:00:00Z",
            ],
            "acquired": [True, True, True, True, False],
            "readable": [True, True, True, False, False],
        }
    )


def _detections() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "detection_id": ["d1", "d2", "d3"],
            "image_id": ["i1", "i2", "i3"],
            "class_name": ["vehicle", "vehicle", "vehicle"],
            "confidence": [0.9, 0.8, 0.95],
            "x_min": [0, 1, 1],
            "y_min": [0, 1, 1],
            "x_max": [10, 11, 11],
            "y_max": [10, 11, 11],
        }
    )


def test_intersection_over_union() -> None:
    assert intersection_over_union((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert intersection_over_union((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert intersection_over_union((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_historical_replay_never_expires_and_omits_zero_hours() -> None:
    outputs = build_camera_activity(
        _manifest(),
        _detections(),
        mode="historical_replay",
        class_names=("vehicle",),
    )
    assert outputs.track_observations["track_id"].nunique() == 1
    assert len(outputs.track_summary) == 1
    assert outputs.track_summary.iloc[0]["detection_observations"] == 3
    assert outputs.hourly_activity["hour_start"].nunique() == 2
    assert set(outputs.hourly_activity["monitoring_status"]) == {"observed_detection"}


def test_historical_replay_preserves_singletons_only_in_observations() -> None:
    singleton = _detections().iloc[[0]].copy()
    outputs = build_camera_activity(
        _manifest(), singleton, mode="historical_replay", class_names=("vehicle",)
    )
    assert len(outputs.track_observations) == 1
    assert not bool(outputs.track_observations.iloc[0]["included_in_track_summary"])
    assert outputs.track_summary.empty


def test_recommended_expires_tracks_and_distinguishes_zeros_from_missing_coverage() -> None:
    outputs = build_camera_activity(
        _manifest(),
        _detections(),
        mode="recommended",
        class_names=("vehicle",),
        max_gap_seconds=900,
    )
    assert outputs.track_observations["track_id"].nunique() == 2
    assert len(outputs.track_summary) == 2

    hourly = outputs.hourly_activity.set_index("hour_start")
    assert len(hourly) == 4
    assert hourly.loc[pd.Timestamp("2025-06-01T12:00:00Z"), "relative_activity_index"] == 1
    assert hourly.loc[pd.Timestamp("2025-06-01T13:00:00Z"), "relative_activity_index"] == 1
    assert pd.isna(hourly.loc[pd.Timestamp("2025-06-01T14:00:00Z"), "relative_activity_index"])
    assert pd.isna(hourly.loc[pd.Timestamp("2025-06-01T15:00:00Z"), "relative_activity_index"])
    assert hourly.loc[pd.Timestamp("2025-06-01T14:00:00Z"), "monitoring_status"] == (
        "unobserved_no_readable_image"
    )
    assert hourly.loc[pd.Timestamp("2025-06-01T15:00:00Z"), "acquisition_coverage_fraction"] == 0


def test_recommended_includes_true_zero_for_readable_hour() -> None:
    manifest = _manifest()
    manifest.loc[len(manifest)] = [
        "i6",
        "parking_a",
        "2025-06-01T16:00:00Z",
        True,
        True,
    ]
    outputs = build_camera_activity(
        manifest, _detections(), mode="recommended", class_names=("vehicle",)
    )
    row = outputs.hourly_activity.loc[
        outputs.hourly_activity["hour_start"].eq(pd.Timestamp("2025-06-01T16:00:00Z"))
    ].iloc[0]
    assert row["relative_activity_index"] == 0
    assert row["monitoring_status"] == "observed_zero"


def test_invalid_detection_on_unreadable_image_is_rejected() -> None:
    detection = _detections().iloc[[0]].copy()
    detection["image_id"] = "i4"
    with pytest.raises(CameraActivityError, match="readable"):
        build_camera_activity(_manifest(), detection, class_names=("vehicle",))


def test_calibration_recovers_simple_linear_relationship() -> None:
    hourly = pd.DataFrame(
        {
            "site_id": ["a", "a", "a"],
            "class_name": ["vehicle"] * 3,
            "hour_start": [
                "2025-06-01T12:00:00Z",
                "2025-06-01T13:00:00Z",
                "2025-06-01T14:00:00Z",
            ],
            "relative_activity_index": [0, 1, 2],
        }
    )
    manual_counts = pd.DataFrame(
        {
            "site_id": ["a", "a", "a"],
            "class_name": ["vehicle"] * 3,
            "hour_start": hourly["hour_start"],
            "manual_reference_value": [2, 5, 8],
        }
    )
    outputs = calibrate_relative_activity(hourly, manual_counts)
    model = outputs.calibration_models.iloc[0]
    assert model["intercept"] == pytest.approx(2)
    assert model["slope"] == pytest.approx(3)
    assert model["r_squared"] == pytest.approx(1)
    assert np.allclose(
        outputs.calibration_matches["calibrated_reference_estimate"], [2, 5, 8]
    )
