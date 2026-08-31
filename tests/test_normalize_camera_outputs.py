from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


PUBLIC_ROOT = Path(__file__).resolve().parents[1]


def _load_stage():
    path = PUBLIC_ROOT / "pipeline" / "09_normalize_camera_outputs.py"
    spec = importlib.util.spec_from_file_location("normalize_camera_outputs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalizer_removes_archive_root_and_preserves_inference_gaps(tmp_path: Path) -> None:
    stage = _load_stage()
    camera_root = tmp_path / "camera_archive"
    first = camera_root / "Parking" / "Alpha" / "first.jpg"
    second = camera_root / "Parking" / "Alpha" / "second.jpg"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    manifest = pd.DataFrame(
        [
            {
                "image_id": "image-1",
                "path": str(first),
                "site_id": "Alpha",
                "captured_at": "2024-07-01T12:00:00-06:00",
                "acquired": True,
                "readable": True,
                "daylight": True,
            },
            {
                "image_id": "image-2",
                "path": str(second),
                "site_id": "Alpha",
                "captured_at": "2024-07-01T12:01:00-06:00",
                "acquired": True,
                "readable": True,
                "daylight": True,
            },
        ]
    )
    inference = pd.DataFrame(
        [
            {
                "path": str(first),
                "count_total": 2,
                "detections": [
                    {"class_name": "car", "conf": 0.8, "xyxy": [1, 2, 4, 6]},
                    {"class_name": "boat", "conf": 0.9, "xyxy": [2, 3, 5, 7]},
                ],
            }
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    inference_path = tmp_path / "inference.parquet"
    output_dir = tmp_path / "normalized"
    manifest.to_csv(manifest_path, index=False)
    inference.to_parquet(inference_path, index=False)

    result = stage.run(
        argparse.Namespace(
            image_manifest=manifest_path,
            inference=inference_path,
            camera_root=camera_root,
            output_dir=output_dir,
            output_format="csv",
            daylight_only=True,
        )
    )

    assert result == 0
    normalized_manifest = pd.read_csv(output_dir / "image_manifest.csv")
    normalized_detections = pd.read_csv(output_dir / "detections.csv")
    assert normalized_manifest["path"].tolist() == [
        "Parking/Alpha/first.jpg",
        "Parking/Alpha/second.jpg",
    ]
    assert normalized_manifest["readable"].tolist() == [True, False]
    assert normalized_detections["class_name"].tolist() == ["vehicle", "watercraft"]
    assert str(camera_root) not in (output_dir / "image_manifest.csv").read_text(encoding="utf-8")
    summary = json.loads((output_dir / "normalization_run.json").read_text(encoding="utf-8"))
    assert summary["rows_marked_unreadable_after_inference"] == 1
    assert summary["paths_are_archive_relative"] is True


def test_normalizer_rejects_inference_path_outside_archive(tmp_path: Path) -> None:
    stage = _load_stage()
    camera_root = tmp_path / "camera_archive"
    camera_root.mkdir()
    outside = tmp_path / "outside.jpg"

    try:
        stage.portable_path(outside, camera_root)
    except ValueError as exc:
        assert "outside --camera-root" in str(exc)
    else:
        raise AssertionError("An out-of-archive inference path must be rejected")
