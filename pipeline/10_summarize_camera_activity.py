#!/usr/bin/env python3
"""Track long-form detections and create site-hour relative activity indices."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from recap_payette.camera_activity import (  # noqa: E402
    DEFAULT_ACTIVITY_CLASSES,
    INDICATOR_DEFINITION,
    build_camera_activity,
    read_table,
    write_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a detection-conditioned relative activity index from a normalized image "
            "manifest and long-form object detections."
        )
    )
    parser.add_argument("--image-manifest", type=Path, required=True, help="Normalized CSV or Parquet manifest.")
    parser.add_argument("--detections", type=Path, required=True, help="Normalized long-form CSV or Parquet detections.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("historical_replay", "recommended"),
        default="recommended",
        help="Use corrected monitoring semantics by default; replay is retained for paper-era comparison.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_ACTIVITY_CLASSES),
        help="Monitored detector classes used to construct the complete recommended-mode grid.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=900.0,
        help="Recommended-mode track expiry; ignored in historical replay.",
    )
    parser.add_argument("--output-format", choices=("csv", "parquet"), default="csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing nonempty output directory {args.output_dir}")
    outputs = build_camera_activity(
        read_table(args.image_manifest),
        read_table(args.detections),
        mode=args.mode,
        class_names=args.classes,
        iou_threshold=args.iou_threshold,
        max_gap_seconds=args.max_gap_seconds,
    )
    extension = ".csv" if args.output_format == "csv" else ".parquet"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_table(outputs.track_observations, args.output_dir / f"track_observations{extension}")
    write_table(outputs.track_summary, args.output_dir / f"track_summary{extension}")
    write_table(outputs.hourly_activity, args.output_dir / f"hourly_activity{extension}")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "monitored_classes": [value.strip().casefold() for value in args.classes],
        "iou_threshold": args.iou_threshold,
        "max_gap_seconds": None if args.mode == "historical_replay" else args.max_gap_seconds,
        "indicator_definition": INDICATOR_DEFINITION,
        "source_tables": {
            "image_manifest": str(args.image_manifest),
            "detections": str(args.detections),
        },
        "output_rows": {
            "track_observations": len(outputs.track_observations),
            "track_summary": len(outputs.track_summary),
            "hourly_activity": len(outputs.hourly_activity),
        },
    }
    (args.output_dir / "activity_run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
