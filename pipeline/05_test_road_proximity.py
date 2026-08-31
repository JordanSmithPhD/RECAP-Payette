"""Measure paved parking area within a specified distance of road centerlines.

This is the previously missing analysis behind the manuscript's “within 4 m”
claim. It writes a new, contract-labeled result; it cannot retroactively prove
the historical percentage because the original input polygons/run record are
not identified.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.parking_common import (  # noqa: E402
    ContractError,
    load_run_manifest,
    refuse_existing_output,
    sha256_file,
    verify_manifest_artifact,
    write_run_manifest,
)
from project_paths import (  # noqa: E402
    PARKING_CAPACITY_RUN_MANIFEST,
    PARKING_CAPACITY_RUN_POLYGONS,
    PARKING_CORRIDOR_MANIFEST,
    PARKING_ROAD_PROXIMITY,
    PARKING_ROAD_PROXIMITY_RUN_TABLE,
    PARKING_RESULTS_RUN_ROOT,
    PARKING_SELECTED_ROADS,
    RESULTS_ROOT,
    WORKFLOW_ROOT,
)


def proximity_statistics(total_area_m2: float, near_road_area_m2: float) -> dict[str, float]:
    if total_area_m2 <= 0:
        raise ContractError("Total paved-prediction area must be positive.")
    if near_road_area_m2 < 0 or near_road_area_m2 > total_area_m2 * (1 + 1e-9):
        raise ContractError("Near-road area must be between zero and total paved area.")
    return {
        "total_paved_area_m2": total_area_m2,
        "near_road_area_m2": near_road_area_m2,
        "near_road_fraction": near_road_area_m2 / total_area_m2,
        "near_road_percent": 100.0 * near_road_area_m2 / total_area_m2,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", choices=("historical", "corrected"), required=True)
    parser.add_argument(
        "--parking-polygons",
        type=Path,
        default=None,
        help="Stage-05 parking GeoPackage/Shapefile whose geometry represents boundary-accounted predictions.",
    )
    parser.add_argument("--parking-layer", default=None)
    parser.add_argument(
        "--summary-manifest",
        type=Path,
        help=(
            "Corrected stage-05 capacity-summary manifest that checksum-binds --parking-polygons; "
            "required in corrected mode."
        ),
    )
    parser.add_argument("--class-field", default="class_id")
    parser.add_argument("--paved-class", type=int, default=1)
    parser.add_argument(
        "--roads",
        type=Path,
        default=None,
        help=(
            "Road-centerline dataset. Corrected mode defaults to the versioned stage-01 selected roads; "
            "historical mode requires an explicit source because the published source is unknown."
        ),
    )
    parser.add_argument("--road-layer", default=None)
    parser.add_argument(
        "--roads-manifest",
        type=Path,
        help=(
            "Stage-01 corridor manifest that checksum-binds --roads. Required in corrected mode; "
            "defaults to the configured parking-run corridor manifest."
        ),
    )
    parser.add_argument("--distance-m", type=float, default=4.0)
    parser.add_argument("--metric-crs", default="EPSG:32611")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--near-road-geometry-output",
        type=Path,
        default=None,
    )
    return parser


def _resolve_default_paths(args: argparse.Namespace) -> argparse.Namespace:
    if args.contract == "corrected":
        args.parking_polygons = args.parking_polygons or PARKING_CAPACITY_RUN_POLYGONS
        args.summary_manifest = args.summary_manifest or PARKING_CAPACITY_RUN_MANIFEST
        args.roads = args.roads or PARKING_SELECTED_ROADS
        args.roads_manifest = args.roads_manifest or PARKING_CORRIDOR_MANIFEST
        args.output = args.output or PARKING_ROAD_PROXIMITY_RUN_TABLE
        args.near_road_geometry_output = args.near_road_geometry_output or (
            PARKING_RESULTS_RUN_ROOT / "paved_parking_within_road_buffer.gpkg"
        )
    else:
        if args.parking_polygons is None or args.roads is None:
            raise ContractError(
                "Historical road proximity requires explicit --parking-polygons and --roads; "
                "the published road source is unidentified."
            )
        args.output = args.output or PARKING_ROAD_PROXIMITY
        args.near_road_geometry_output = args.near_road_geometry_output or (
            RESULTS_ROOT / "paved_parking_within_road_buffer.gpkg"
        )
    return args


def _union(frame: Any) -> Any:
    return frame.geometry.union_all() if hasattr(frame.geometry, "union_all") else frame.geometry.unary_union


def _require_projected_metre_crs(crs: Any, *, label: str) -> None:
    units = {str(axis.unit_name).lower() for axis in crs.axis_info if axis.unit_name}
    if not crs.is_projected or not units or not all("met" in unit for unit in units):
        raise ContractError(f"{label} must be projected with metre axes; found {crs.to_string()} ({sorted(units)}).")


def _validate_summary_lineage(args: argparse.Namespace) -> Path | None:
    if args.contract == "historical" and args.summary_manifest is None:
        return None
    if args.summary_manifest is None:
        raise ContractError("Corrected road proximity requires --summary-manifest.")
    payload = load_run_manifest(
        args.summary_manifest,
        expected_stage="05_summarize_parking_capacity",
        expected_contract=args.contract,
    )
    verify_manifest_artifact(payload, args.parking_polygons)
    if args.contract == "corrected":
        arguments = payload.get("arguments", {})
        if arguments.get("assignment_mode_effective") != "clip":
            raise ContractError("Corrected road proximity requires boundary-clipped parking polygons.")
        try:
            from pyproj import CRS

            summary_crs = CRS.from_user_input(arguments.get("metric_crs"))
        except Exception as exc:
            raise ContractError("Summary manifest has no readable metric_crs.") from exc
        _require_projected_metre_crs(summary_crs, label="Summary manifest metric_crs")
    return args.summary_manifest


def _validate_roads_lineage(args: argparse.Namespace, summary_manifest: Path | None) -> Path | None:
    if args.contract == "historical" and args.roads_manifest is None:
        return None
    if args.roads_manifest is None:
        raise ContractError("Corrected road proximity requires --roads-manifest.")
    roads_payload = load_run_manifest(
        args.roads_manifest,
        expected_stage="01_build_corridor",
        expected_contract=("projected-geometry-rgbnir" if args.contract == "corrected" else None),
    )
    verify_manifest_artifact(roads_payload, args.roads)
    if args.contract == "corrected":
        if roads_payload.get("extra", {}).get("input_raster_provenance") != "verified":
            raise ContractError("Corrected roads are not bound to a verified corridor mosaic.")
        if summary_manifest is None:
            raise ContractError("Corrected roads require a verified capacity summary manifest.")
        summary_payload = load_run_manifest(
            summary_manifest,
            expected_stage="05_summarize_parking_capacity",
            expected_contract="corrected",
        )
        expected_hash = summary_payload.get("extra", {}).get("corridor_manifest_sha256")
        observed_hash = sha256_file(args.roads_manifest)
        if not expected_hash or expected_hash != observed_hash:
            raise ContractError("Road centerlines and capacity polygons do not originate from the same corridor run.")
    return args.roads_manifest


def run(args: argparse.Namespace) -> None:
    import geopandas as gpd
    from pyproj import CRS

    _resolve_default_paths(args)
    if args.distance_m <= 0:
        raise ContractError("distance_m must be positive.")
    manifest_path = args.output.with_suffix(".run.json")
    refuse_existing_output(args.output)
    refuse_existing_output(args.near_road_geometry_output)
    refuse_existing_output(manifest_path)
    if not args.parking_polygons.is_file():
        raise FileNotFoundError(args.parking_polygons)
    if not args.roads.is_file():
        raise FileNotFoundError(args.roads)
    summary_manifest = _validate_summary_lineage(args)
    roads_manifest = _validate_roads_lineage(args, summary_manifest)
    metric_crs = CRS.from_user_input(args.metric_crs)
    _require_projected_metre_crs(metric_crs, label="metric_crs")

    parking = gpd.read_file(args.parking_polygons, layer=args.parking_layer)
    roads = gpd.read_file(args.roads, layer=args.road_layer)
    if parking.crs is None or roads.crs is None:
        raise ContractError("Parking and road vectors must both declare a CRS.")
    if args.class_field not in parking.columns:
        raise ContractError(f"Parking class field {args.class_field!r} is absent.")
    paved = parking[parking[args.class_field].astype(int) == args.paved_class].copy()
    if paved.empty:
        raise ContractError(f"No paved-class {args.paved_class} polygons were found.")
    paved = paved.to_crs(metric_crs)
    roads = roads.to_crs(metric_crs)
    if roads.empty:
        raise ContractError("Road vector is empty.")

    # Union first so overlapping prediction/reach fragments cannot inflate the
    # numerator or denominator.
    paved_geometry = _union(paved)
    road_buffer = _union(roads).buffer(args.distance_m)
    near_geometry = paved_geometry.intersection(road_buffer)
    statistics = proximity_statistics(float(paved_geometry.area), float(near_geometry.area))
    row = {
        "contract": args.contract,
        "paved_class": args.paved_class,
        "distance_m": args.distance_m,
        **statistics,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow(row)
    near_frame = gpd.GeoDataFrame(
        [row],
        geometry=[near_geometry],
        crs=metric_crs,
    )
    args.near_road_geometry_output.parent.mkdir(parents=True, exist_ok=True)
    near_frame.to_file(args.near_road_geometry_output, layer="near_road_paved_parking", driver="GPKG")
    inputs = [args.parking_polygons, args.roads]
    if summary_manifest is not None:
        inputs.append(summary_manifest)
    if roads_manifest is not None:
        inputs.append(roads_manifest)
    write_run_manifest(
        manifest_path,
        workflow_root=WORKFLOW_ROOT,
        stage="05_test_road_proximity",
        contract=args.contract,
        arguments=vars(args),
        inputs=tuple(inputs),
        outputs=(args.output, args.near_road_geometry_output),
        extra={
            "measurement": "unioned paved-prediction area intersecting a buffered road-centerline union",
            "summary_provenance": "verified" if summary_manifest is not None else "missing-historical-provenance",
            "roads_provenance": "verified" if roads_manifest is not None else "missing-historical-provenance",
            "area_and_distance_units": "square metres and metres",
        },
    )
    print(f"Near-road paved area: {statistics['near_road_percent']:.3f}%")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run(args)
    except (ContractError, FileExistsError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
