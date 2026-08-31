#!/usr/bin/env python3
"""Compare the relative camera activity index with neutral manual references."""

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
    calibrate_relative_activity,
    read_table,
    write_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit transparent linear calibration checks between hourly relative activity and "
            "independent manual reference observations."
        )
    )
    parser.add_argument("--hourly-activity", type=Path, required=True, help="Stage-10 hourly CSV or Parquet table.")
    parser.add_argument(
        "--manual-counts",
        type=Path,
        required=True,
        help=(
            "Neutral manual_counts CSV or Parquet table with site_id, class_name, hour_start, "
            "and manual_reference_value."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grouping", choices=("class", "site_class"), default="class")
    parser.add_argument(
        "--include-intercept",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit an intercept by default; use --no-include-intercept to constrain it to zero.",
    )
    parser.add_argument("--minimum-matches", type=int, default=2)
    parser.add_argument("--output-format", choices=("csv", "parquet"), default="csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing nonempty output directory {args.output_dir}")
    outputs = calibrate_relative_activity(
        read_table(args.hourly_activity),
        read_table(args.manual_counts),
        grouping=args.grouping,
        include_intercept=args.include_intercept,
        minimum_matches=args.minimum_matches,
    )
    extension = ".csv" if args.output_format == "csv" else ".parquet"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_table(outputs.calibration_models, args.output_dir / f"calibration_models{extension}")
    write_table(outputs.calibration_matches, args.output_dir / f"calibration_matches{extension}")
    write_table(
        outputs.calibrated_hourly_activity,
        args.output_dir / f"calibrated_hourly_activity{extension}",
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "grouping": args.grouping,
        "include_intercept": args.include_intercept,
        "minimum_matches": args.minimum_matches,
        "source_tables": {
            "hourly_activity": str(args.hourly_activity),
            "manual_counts": str(args.manual_counts),
        },
        "matched_rows": len(outputs.calibration_matches),
    }
    (args.output_dir / "calibration_run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
