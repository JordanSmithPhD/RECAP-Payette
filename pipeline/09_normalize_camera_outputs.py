#!/usr/bin/env python3
"""Normalize stage-09 nested output for tracking and hourly aggregation.

The resumable inference engine retains one row per image with a nested
``detections`` column for historical replay. This adapter joins those rows to
the portable image manifest and writes the long-form schemas consumed by stage
10. It does not copy photographs or expose archive-root paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


CLASS_MAP = {
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "motorcycle": "vehicle",
    "vehicle": "vehicle",
    "boat": "watercraft",
    "watercraft": "watercraft",
}


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table type: {path}")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)


def portable_path(value: Any, camera_root: Path) -> str:
    path = Path(str(value).strip()).expanduser()
    if path.is_absolute():
        try:
            path = path.resolve(strict=False).relative_to(camera_root)
        except ValueError as exc:
            raise ValueError(f"Inference path is outside --camera-root: {path}") from exc
    return path.as_posix().lstrip("./")


def nested_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError(f"Nested detections must be a list, observed {type(value).__name__}")
    return value


def nested_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, dict):
        return value
    if hasattr(value, "keys"):
        return {str(key): value[key] for key in value.keys()}
    raise ValueError(f"Detection record must be a mapping, observed {type(value).__name__}")


def normalize_box(value: Any) -> tuple[float, float, float, float]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("Every detection xyxy value must contain four coordinates")
    box = tuple(float(item) for item in value)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"Detection has a non-positive bounding box: {box}")
    return box


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True, help="Stage-09 detections.parquet")
    parser.add_argument("--camera-root", type=Path, required=True, help="Used only to make stage-09 paths relative")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-format", choices=("csv", "parquet"), default="parquet")
    parser.add_argument(
        "--daylight-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When the manifest has a daylight field, retain daylight rows only (default: enabled).",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    manifest = read_table(args.image_manifest.expanduser().resolve(strict=True)).copy()
    inference = read_table(args.inference.expanduser().resolve(strict=True)).copy()
    camera_root = args.camera_root.expanduser().resolve(strict=True)
    required_manifest = {"image_id", "path", "site_id", "captured_at", "acquired", "readable"}
    required_inference = {"path", "detections", "count_total"}
    if missing := sorted(required_manifest.difference(manifest.columns)):
        raise ValueError(f"Image manifest is missing columns: {missing}")
    if missing := sorted(required_inference.difference(inference.columns)):
        raise ValueError(f"Inference table is missing columns: {missing}")
    if args.daylight_only and "daylight" in manifest.columns:
        daylight = manifest["daylight"].astype("string").str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
        if daylight.isna().any():
            raise ValueError("Manifest daylight values must be boolean")
        manifest = manifest.loc[daylight].copy()

    manifest["portable_path"] = [portable_path(value, camera_root) for value in manifest["path"]]
    inference["portable_path"] = [portable_path(value, camera_root) for value in inference["path"]]
    for label, frame in (("manifest", manifest), ("inference", inference)):
        if frame["portable_path"].duplicated().any():
            examples = frame.loc[frame["portable_path"].duplicated(False), "portable_path"].unique()[:5]
            raise ValueError(f"{label} paths are not unique: {examples.tolist()}")

    unknown = sorted(set(inference["portable_path"]).difference(manifest["portable_path"]))
    if unknown:
        raise ValueError(f"Inference rows do not match the image manifest: {unknown[:5]}")
    image_id_by_path = manifest.set_index("portable_path")["image_id"].astype(str).to_dict()
    successful_paths = set(inference["portable_path"])

    rows: list[dict[str, Any]] = []
    for _, image_row in inference.sort_values("portable_path", kind="mergesort").iterrows():
        records = nested_list(image_row["detections"])
        if int(image_row["count_total"]) != len(records):
            raise ValueError(f"count_total disagrees with nested detections for {image_row['portable_path']}")
        image_id = image_id_by_path[str(image_row["portable_path"])]
        for ordinal, raw in enumerate(records):
            record = nested_dict(raw)
            source_class = str(record.get("class_name", "")).strip().casefold()
            if source_class not in CLASS_MAP:
                raise ValueError(f"Unsupported detector class {source_class!r}")
            box = normalize_box(record.get("xyxy"))
            confidence = float(record.get("conf"))
            if not 0 <= confidence <= 1:
                raise ValueError("Detection confidence must fall within [0, 1]")
            detection_id = hashlib.sha256(f"{image_id}:{ordinal}".encode("utf-8")).hexdigest()[:24]
            rows.append(
                {
                    "detection_id": detection_id,
                    "image_id": image_id,
                    "class_name": CLASS_MAP[source_class],
                    "confidence": confidence,
                    "x_min": box[0],
                    "y_min": box[1],
                    "x_max": box[2],
                    "y_max": box[3],
                }
            )

    normalized_manifest = manifest[
        ["image_id", "site_id", "captured_at", "acquired", "readable", "portable_path"]
    ].rename(columns={"portable_path": "path"})
    # A manifest image with no inference row represents a decode/inference gap,
    # not a monitored zero. Preserve acquisition while marking it unreadable for
    # downstream coverage accounting.
    missing_inference = ~manifest["portable_path"].isin(successful_paths)
    normalized_manifest.loc[missing_inference, "readable"] = False
    normalized_detections = pd.DataFrame(
        rows,
        columns=(
            "detection_id",
            "image_id",
            "class_name",
            "confidence",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        ),
    )

    output_dir = args.output_dir.expanduser().resolve(strict=False)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing nonempty output directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = ".csv" if args.output_format == "csv" else ".parquet"
    manifest_path = output_dir / f"image_manifest{extension}"
    detections_path = output_dir / f"detections{extension}"
    write_table(normalized_manifest, manifest_path)
    write_table(normalized_detections, detections_path)
    summary = {
        "stage": "09_normalize_camera_outputs",
        "manifest_rows": len(normalized_manifest),
        "inference_rows": len(inference),
        "detection_rows": len(normalized_detections),
        "rows_marked_unreadable_after_inference": int(missing_inference.sum()),
        "paths_are_archive_relative": True,
    }
    (output_dir / "normalization_run.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
