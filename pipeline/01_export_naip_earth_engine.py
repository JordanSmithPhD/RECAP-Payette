#!/usr/bin/env python3
"""Submit and record the recovered Earth Engine NAIP export.

The historical Colab notebook queried ``USDA/NAIP/DOQQ`` from 2022-01-01
through 2025-12-31, scaled byte imagery to 0--1, and exported it to Drive in
EPSG:4326. This entry point retains those acquisition/radiometry choices while
defaulting the corrected run to explicit RGB+NIR; RGB remains available for
historical reconstruction. It never authenticates interactively, installs
packages, mounts Drive, or overwrites an export with the same local prefix.

Earth Engine and GeoPandas are imported only after argument parsing and all
local safety checks, so ``--help`` and module import work in a light-weight
environment. Nonterminal tasks retain only their immutable submission record;
``01_finalize_naip_export.py`` later polls that task ID and creates the final
manifest without resubmitting the export.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence


WORKING_ROOT = Path(__file__).resolve().parents[1]
if str(WORKING_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKING_ROOT))

from project_paths import (  # noqa: E402
    DATA_RAW_ROOT,
    NAIP_EXPORT_ROOT,
    PARKING_EE_RUN_MANIFEST,
    PARKING_EE_SUBMISSION_MANIFEST,
    PARKING_RUN_ID,
    WORKFLOW_ROOT,
)
from pipeline.parking_common import ContractError, refuse_existing_output, write_run_manifest  # noqa: E402


DEFAULT_ROI = DATA_RAW_ROOT / "Payette_River_Flowlines_BBox" / "Payette_River_Flowlines_BBox.shp"
TERMINAL_TASK_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "CANCEL_REQUESTED"})
DEFAULT_EXPORT_PREFIX = f"Payette_ROI_Imagery_{PARKING_RUN_ID}"
CORRECTED_EXPORT_NODATA = -9999.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a date-sorted USDA NAIP mosaic from Google Earth Engine. "
            "Authentication must already be configured with the Earth Engine CLI."
        )
    )
    parser.add_argument("--roi", type=Path, default=DEFAULT_ROI, help="Polygon vector used as the export region.")
    parser.add_argument("--project", default=os.environ.get("GEE_PROJECT_ID"), help="Google Cloud project ID.")
    parser.add_argument("--start-date", default="2022-01-01", help="Inclusive Earth Engine start date.")
    parser.add_argument(
        "--end-date",
        default="2025-12-31",
        help="Exclusive Earth Engine end date (the recovered notebook used 2025-12-31).",
    )
    parser.add_argument(
        "--band-contract",
        choices=("rgb", "rgbnir"),
        default="rgbnir",
        help=(
            "Bands to export. 'rgbnir' is the corrected run default; 'rgb' reconstructs the notebook source "
            "and must remain isolated from corrected model runs."
        ),
    )
    parser.add_argument(
        "--scale-m",
        type=float,
        default=None,
        help="Export resolution in metres; default reads nominal scale from the first matching NAIP image.",
    )
    parser.add_argument("--description", default=DEFAULT_EXPORT_PREFIX, help="Earth Engine task description.")
    parser.add_argument(
        "--file-prefix",
        default=DEFAULT_EXPORT_PREFIX,
        help="Run-specific Drive export file-name prefix; changing the parking run ID changes this default.",
    )
    parser.add_argument(
        "--drive-folder",
        default="Payette_Computer_Vision_GEE",
        help="Google Drive folder name understood by Earth Engine (not a local filesystem path).",
    )
    parser.add_argument(
        "--local-export-dir",
        type=Path,
        default=NAIP_EXPORT_ROOT,
        help="Local synced-Drive directory used only for collision checks and artifact discovery.",
    )
    parser.add_argument("--max-pixels", type=float, default=1e13, help="Earth Engine export maxPixels value.")
    parser.add_argument(
        "--wait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Poll until the Earth Engine task reaches a terminal state (default: wait).",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0, help="Task polling interval when waiting.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=86_400.0,
        help="Maximum wait time; use 0 for no timeout.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Final run-manifest path; defaults to the configured corrected-run manifest.",
    )
    parser.add_argument(
        "--submission-manifest",
        type=Path,
        default=None,
        help=(
            "Immutable submission record written immediately after task.start(); defaults to the configured "
            "corrected-run submission manifest."
        ),
    )
    return parser


def _union_geometry(frame: Any) -> Any:
    geometry = frame.geometry
    return geometry.union_all() if hasattr(geometry, "union_all") else geometry.unary_union


def _export_family(directory: Path, prefix: str) -> list[Path]:
    exact = directory / f"{prefix}.tif"
    shards = sorted(path for path in directory.glob(f"{prefix}-*.tif") if path.is_file())
    return ([exact] if exact.is_file() else []) + shards


def _wait_for_task(task: Any, *, poll_seconds: float, timeout_seconds: float) -> dict[str, Any]:
    if poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive.")
    if timeout_seconds < 0:
        raise ValueError("--timeout-seconds cannot be negative.")

    started = time.monotonic()
    while True:
        status = dict(task.status())
        state = str(status.get("state", "UNKNOWN")).upper()
        print(f"Earth Engine task {status.get('id', '<unknown>')}: {state}", flush=True)
        if state in TERMINAL_TASK_STATES:
            return status
        if timeout_seconds and time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(
                "Earth Engine export is still running. The immutable submission manifest preserves its "
                "task ID; run 01_finalize_naip_export.py later rather than resubmitting."
            )
        time.sleep(poll_seconds)


def _task_status(task: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    """Return the newest task status without hiding an earlier run failure."""

    try:
        return dict(task.status())
    except Exception as exc:  # Earth Engine transports expose several exception types
        status = dict(fallback)
        status["status_refresh_error"] = repr(exc)
        return status


def _run_outcome(status: dict[str, Any], error: BaseException | None, *, waited: bool) -> str:
    state = str(status.get("state", "")).upper()
    if state == "COMPLETED":
        return "completed"
    if state in TERMINAL_TASK_STATES:
        return "failed"
    if isinstance(error, TimeoutError):
        return "wait-timeout"
    if error is not None:
        return "failed"
    if not waited:
        return "submitted-not-awaited"
    return "failed"


def run(args: argparse.Namespace) -> int:
    roi_path = args.roi.expanduser().resolve(strict=False)
    local_export_dir = args.local_export_dir.expanduser().resolve(strict=False)
    manifest_path = (
        args.manifest.expanduser().resolve(strict=False) if args.manifest is not None else PARKING_EE_RUN_MANIFEST
    )
    submission_manifest_path = (
        args.submission_manifest.expanduser().resolve(strict=False)
        if args.submission_manifest is not None
        else PARKING_EE_SUBMISSION_MANIFEST
    )

    if not roi_path.is_file():
        raise FileNotFoundError(f"ROI vector does not exist: {roi_path}")
    if not args.file_prefix.strip():
        raise ValueError("--file-prefix cannot be empty.")
    if args.file_prefix != args.file_prefix.strip() or Path(args.file_prefix).name != args.file_prefix:
        raise ValueError("--file-prefix must be a plain filename prefix without whitespace or path separators.")
    if not args.drive_folder.strip():
        raise ValueError("--drive-folder cannot be empty.")
    if args.scale_m is not None and args.scale_m <= 0:
        raise ValueError("--scale-m must be positive.")
    if args.max_pixels <= 0:
        raise ValueError("--max-pixels must be positive.")

    # A large Earth Engine export is commonly split into many GeoTIFFs.  Treat
    # every matching prefix as one immutable output family and stop before any
    # remote task is submitted.
    existing_exports = _export_family(local_export_dir, args.file_prefix)
    if existing_exports:
        preview = ", ".join(str(path) for path in existing_exports[:3])
        suffix = " ..." if len(existing_exports) > 3 else ""
        raise FileExistsError(
            f"{len(existing_exports)} local export file(s) already use this prefix: {preview}{suffix}. "
            "Choose a new --file-prefix; existing imagery is immutable."
        )
    if manifest_path == submission_manifest_path:
        raise ValueError("--manifest and --submission-manifest must be different paths.")
    if args.band_contract == "rgb" and (
        args.file_prefix == DEFAULT_EXPORT_PREFIX
        or args.description == DEFAULT_EXPORT_PREFIX
        or manifest_path == PARKING_EE_RUN_MANIFEST.resolve(strict=False)
        or submission_manifest_path == PARKING_EE_SUBMISSION_MANIFEST.resolve(strict=False)
    ):
        raise ContractError(
            "Historical RGB export requires explicit isolated --description, --file-prefix, --manifest, and "
            "--submission-manifest values; corrected defaults are reserved for RGB+NIR."
        )
    refuse_existing_output(manifest_path)
    refuse_existing_output(submission_manifest_path)

    try:
        import ee
        import geopandas as gpd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This stage requires earthengine-api and geopandas. Install the parking/geospatial environment "
            "before submitting an export."
        ) from exc

    try:
        if args.project:
            ee.Initialize(project=args.project)
        else:
            ee.Initialize()
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Run `earthengine authenticate` outside this script and pass "
            "--project (or set GEE_PROJECT_ID)."
        ) from exc

    roi_frame = gpd.read_file(roi_path)
    if roi_frame.empty:
        raise ValueError(f"ROI vector is empty: {roi_path}")
    if roi_frame.crs is None:
        raise ValueError(f"ROI vector has no CRS: {roi_path}")
    roi_frame = roi_frame.to_crs("EPSG:4326")
    roi_geometry = _union_geometry(roi_frame)
    if roi_geometry.is_empty:
        raise ValueError(f"ROI geometry is empty after union: {roi_path}")
    roi = ee.Geometry(roi_geometry.__geo_interface__)

    collection = ee.ImageCollection("USDA/NAIP/DOQQ").filterBounds(roi).filterDate(args.start_date, args.end_date)
    image_count = int(collection.size().getInfo())
    if image_count == 0:
        raise RuntimeError(f"No USDA/NAIP/DOQQ images intersect the ROI from {args.start_date} to {args.end_date}.")

    bands = ["R", "G", "B"] if args.band_contract == "rgb" else ["R", "G", "B", "N"]
    # Earth Engine mosaic gives later collection entries priority.  The
    # recovered notebook sorted descending, which consequently gives older
    # valid pixels priority; retain that ordering only for historical RGB.
    # Corrected RGB+NIR uses the explicit latest-valid-pixel policy.
    sort_ascending = args.band_contract == "rgbnir"
    mosaic_priority = "latest-valid-pixel" if sort_ascending else "historical-descending-sort-oldest-valid-pixel"
    image = collection.sort("system:time_start", sort_ascending).mosaic().select(bands).divide(255).toFloat().clip(roi)
    if args.band_contract == "rgbnir":
        # Earth Engine's documented GeoTIFF nodata recipe requires both
        # materializing masked pixels with unmask() and declaring the same
        # sentinel in formatOptions. The export region bounds the fill created
        # by sameFootprint=False.
        image = image.unmask(value=CORRECTED_EXPORT_NODATA, sameFootprint=False)
    scale_m = args.scale_m
    if scale_m is None:
        scale_m = float(collection.first().select("R").projection().nominalScale().getInfo())

    export_options = {
        "image": image,
        "description": args.description,
        "folder": args.drive_folder,
        "fileNamePrefix": args.file_prefix,
        "region": roi,
        "scale": scale_m,
        "maxPixels": args.max_pixels,
        "crs": "EPSG:4326",
        "fileFormat": "GeoTIFF",
    }
    if args.band_contract == "rgbnir":
        # Zero is valid unit-scaled reflectance.  A corrected export must carry
        # an out-of-range sentinel so masked source holes cannot become valid
        # black pixels after Drive synchronization/mosaicking.
        export_options["formatOptions"] = {"noData": CORRECTED_EXPORT_NODATA}
    task = ee.batch.Export.image.toDrive(
        **export_options,
    )
    task.start()
    submitted_fallback = {"state": "SUBMITTED"}
    task_id = getattr(task, "id", None)
    if task_id:
        submitted_fallback["id"] = str(task_id)
    status = _task_status(task, submitted_fallback)
    common_extra = {
        "earth_engine_asset": "USDA/NAIP/DOQQ",
        "bands": bands,
        "image_count": image_count,
        "mosaic_priority": mosaic_priority,
        "collection_sort_ascending": sort_ascending,
        "scale_m": scale_m,
        "expected_local_prefix": local_export_dir / args.file_prefix,
        "pre_submission_local_match_count": 0,
        "output_radiometry": "float32 values scaled from byte DN by 1/255",
        "export_nodata_value": CORRECTED_EXPORT_NODATA if args.band_contract == "rgbnir" else None,
        "validity_metadata_contract": (
            "explicit-out-of-range-nodata" if args.band_contract == "rgbnir" else "historical-uncontrolled"
        ),
    }
    # This record is deliberately written before polling. If the process is
    # interrupted, the remote task ID and exact request still survive locally.
    write_run_manifest(
        submission_manifest_path,
        workflow_root=WORKFLOW_ROOT,
        stage="01_export_naip_earth_engine_submission",
        contract=f"naip-unit-scaled-{args.band_contract}",
        arguments=vars(args),
        inputs=[roi_path],
        outputs=[],
        extra={**common_extra, "task_status_at_submission": status},
    )
    print(f"Wrote immutable submission manifest: {submission_manifest_path}")

    run_error: BaseException | None = None
    try:
        if args.wait:
            status = _wait_for_task(task, poll_seconds=args.poll_seconds, timeout_seconds=args.timeout_seconds)
            if str(status.get("state", "")).upper() != "COMPLETED":
                raise RuntimeError(f"Earth Engine export did not complete successfully: {status}")
    except BaseException as exc:
        run_error = exc
    status = _task_status(task, status)
    task_state = str(status.get("state", "")).upper()
    if task_state == "COMPLETED" and isinstance(run_error, TimeoutError):
        # The final refresh won a race with the local timeout.
        run_error = None
    exported_files = _export_family(local_export_dir, args.file_prefix)
    if task_state in TERMINAL_TASK_STATES:
        write_run_manifest(
            manifest_path,
            workflow_root=WORKFLOW_ROOT,
            stage="01_export_naip_earth_engine",
            contract=f"naip-unit-scaled-{args.band_contract}",
            arguments=vars(args),
            inputs=[roi_path, submission_manifest_path],
            # A completed remote task and a locally stable Drive shard family
            # are separate events. Even a nonempty local observation can be a
            # partially synchronized family, so the mosaic stage owns the first
            # authoritative shard inventory and checksums.
            outputs=[],
            extra={
                **common_extra,
                "task_status": status,
                "run_outcome": _run_outcome(status, run_error, waited=args.wait),
                "local_export_count": len(exported_files),
                "shard_inventory_policy": "deferred-to-mosaic-after-drive-sync",
                "error": (
                    {"type": type(run_error).__name__, "message": str(run_error)} if run_error is not None else None
                ),
                "submission_manifest": submission_manifest_path,
            },
        )
        print(f"Wrote final run manifest: {manifest_path}")
    else:
        print(
            "Earth Engine task is nonterminal; no final manifest was written. "
            "Run pipeline/01_finalize_naip_export.py with the immutable submission manifest later."
        )

    if run_error is not None:
        raise run_error
    if not args.wait:
        print("Export submitted but not awaited; finalize its task ID before mosaicking local files.")
    elif not exported_files:
        print(
            "Earth Engine completed, but synced Drive files are not yet locally visible; verify Drive hydration later."
        )
    else:
        print(
            f"Observed {len(exported_files)} local TIFF(s); their authoritative inventory is deferred "
            "to the mosaic stage after Drive synchronization."
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
