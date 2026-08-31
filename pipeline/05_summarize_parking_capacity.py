"""Polygonize parking predictions and summarize area by named river reach.

The conversion from acres to vehicle spaces is never implicit: callers must
either supply a documented rate or request area-only output. Corrected runs
clip polygons to reach boundaries. The notebook's full-polygon `intersects`
assignment remains available only as an explicitly labeled historical mode.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.parking_common import (  # noqa: E402
    CLASS_NAMES,
    ContractError,
    capacity_spaces,
    load_run_manifest,
    refuse_existing_output,
    require_unique_nonempty_names,
    verify_manifest_artifact,
    write_run_manifest,
)
from project_paths import (  # noqa: E402
    DEEPLAB_V3_PREDICTIONS,
    PARKING_AREA_COMPARISON_RUN_TABLE,
    PARKING_CAPACITY_RUN_POLYGONS,
    PARKING_CAPACITY_RUN_TABLE,
    PARKING_PREDICTIONS,
    RESULTS_ROOT,
    STUDY_AREAS,
    WORKFLOW_ROOT,
)


SQUARE_FEET_PER_SQUARE_METRE = 10.76391041671
SQUARE_METRES_PER_ACRE = 4046.8564224
SURFACE_NAMES = {1: "paved", 2: "dirt_gravel"}
SOURCE_VALIDITY_CONTRACT = "source-validity-plus-finite-radiometry"
PREDICTION_NODATA = 255


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", choices=("historical", "corrected"), required=True)
    parser.add_argument("--prediction", type=Path, default=None)
    parser.add_argument("--study-areas", type=Path, default=STUDY_AREAS)
    parser.add_argument("--reach-name-field", default="area_name")
    parser.add_argument(
        "--manual-labels",
        type=Path,
        default=None,
        help=(
            "Optional independent reference polygons used only for a model-versus-manual area check. "
            "They are not required to summarize predicted parking opportunity."
        ),
    )
    parser.add_argument("--manual-class-field", default="type_b")
    parser.add_argument(
        "--minimum-area-sqft",
        type=float,
        default=None,
        help=(
            "Connected-component filter. Default: 3,000 for historical notebook reconstruction; "
            "0 for corrected manuscript-compatible pixel-area accounting."
        ),
    )
    parser.add_argument("--metric-crs", default="EPSG:32611")
    parser.add_argument("--tile-size", type=int, default=2048)
    parser.add_argument(
        "--assignment-mode",
        choices=("clip", "historical-intersects"),
        default=None,
        help="Defaults to clip for corrected runs and historical-intersects for historical reconstruction.",
    )
    capacity = parser.add_mutually_exclusive_group(required=True)
    capacity.add_argument(
        "--spaces-per-acre",
        type=float,
        help="Explicit, documented physical-capacity conversion; recorded in every output row.",
    )
    capacity.add_argument("--area-only", action="store_true", help="Report opportunity area without vehicle spaces.")
    parser.add_argument(
        "--capacity-rate-source",
        help="Citation, DOI/URL, or documented derivation; required whenever --spaces-per-acre is used.",
    )
    parser.add_argument(
        "--prediction-manifest",
        type=Path,
        help=(
            "Stage-05 inference manifest. Corrected runs require it; by default the command looks for "
            "<prediction-stem>.run_manifest.json beside the raster."
        ),
    )
    parser.add_argument("--output-table", type=Path, default=None)
    parser.add_argument(
        "--output-polygons",
        type=Path,
        default=None,
    )
    parser.add_argument("--area-comparison-output", type=Path, default=None)
    return parser


def resolve_default_paths(args: argparse.Namespace) -> argparse.Namespace:
    if args.contract == "corrected":
        args.prediction = args.prediction or PARKING_PREDICTIONS
        args.output_table = args.output_table or PARKING_CAPACITY_RUN_TABLE
        args.output_polygons = args.output_polygons or PARKING_CAPACITY_RUN_POLYGONS
        args.area_comparison_output = args.area_comparison_output or PARKING_AREA_COMPARISON_RUN_TABLE
    else:
        historical_root = RESULTS_ROOT / "parking_historical_reconstruction"
        args.prediction = args.prediction or DEEPLAB_V3_PREDICTIONS
        args.output_table = args.output_table or historical_root / "parking_capacity_by_study_area.csv"
        args.output_polygons = args.output_polygons or historical_root / "parking_opportunity_by_reach.gpkg"
        args.area_comparison_output = args.area_comparison_output or historical_root / "parking_area_comparison.csv"
    return args


def _validate_minimum_area_contract(contract: str, minimum_area_sqft: float) -> None:
    if minimum_area_sqft < 0:
        raise ContractError("minimum_area_sqft cannot be negative.")
    if contract == "corrected" and minimum_area_sqft != 0:
        raise ContractError(
            "Corrected capacity accounting requires minimum_area_sqft=0 so every valid parking pixel is retained."
        )


def _polygonize(args: argparse.Namespace) -> Any:
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.features import shapes
    from rasterio.windows import Window, transform as window_transform
    from shapely.geometry import shape

    records: list[dict[str, Any]] = []
    with rasterio.open(args.prediction) as source:
        if source.crs is None:
            raise ContractError("Prediction raster has no CRS.")
        for row in range(0, source.height, args.tile_size):
            height = min(args.tile_size, source.height - row)
            for column in range(0, source.width, args.tile_size):
                width = min(args.tile_size, source.width - column)
                window = Window(column, row, width, height)
                data = source.read(1, window=window)
                parking = np.isin(data, (1, 2))
                if not parking.any():
                    continue
                transform = window_transform(window, source.transform)
                for geometry, value in shapes(data.astype(np.uint8), mask=parking, transform=transform):
                    class_id = int(value)
                    if class_id in SURFACE_NAMES:
                        records.append({"class_id": class_id, "geometry": shape(geometry)})
        raster_crs = source.crs

    if not records:
        if args.contract != "corrected":
            raise ContractError("Prediction contains no class-1/class-2 parking pixels.")
        # An all-background result is a scientifically valid corrected result,
        # not a failed run. Preserve a typed spatial frame so downstream area
        # comparison and empty-vector output retain the prediction CRS.
        return gpd.GeoDataFrame(
            {
                "class_id": [],
                "area_m2": [],
                "surface": [],
                "source_polygon_id": [],
            },
            geometry=[],
            crs=raster_crs,
        ).to_crs(args.metric_crs)
    polygons = gpd.GeoDataFrame(records, crs=raster_crs).to_crs(args.metric_crs)
    # Merge pieces created at deterministic processing-tile boundaries before
    # applying the physical-size threshold.
    polygons = polygons.dissolve(by="class_id", as_index=False).explode(index_parts=False, ignore_index=True)
    polygons["area_m2"] = polygons.geometry.area
    minimum_m2 = args.minimum_area_sqft / SQUARE_FEET_PER_SQUARE_METRE
    polygons = polygons[polygons["area_m2"] >= minimum_m2].copy()
    if polygons.empty and args.contract != "corrected":
        raise ContractError("No predicted parking polygons remain after the minimum-area filter.")
    polygons["surface"] = polygons["class_id"].map(SURFACE_NAMES)
    polygons["source_polygon_id"] = range(len(polygons))
    return polygons


def _assign_reaches(
    polygons: Any,
    reaches: Any,
    name_field: str,
    mode: str,
    *,
    allow_empty: bool = False,
) -> Any:
    import geopandas as gpd

    reaches = reaches[[name_field, "geometry"]].copy()
    if polygons.empty and allow_empty:
        assigned = polygons.copy()
    elif mode == "clip":
        assigned = gpd.overlay(polygons, reaches, how="intersection", keep_geom_type=True)
    else:
        assigned = gpd.sjoin(polygons, reaches, how="inner", predicate="intersects").drop(columns=["index_right"])
    if assigned.empty and not allow_empty:
        raise ContractError("No parking polygons intersect the study reaches.")
    if name_field not in assigned.columns:
        assigned[name_field] = []
    assigned["assignment_contract"] = "boundary-clipped" if mode == "clip" else "historical-full-polygon-intersects"
    assigned["assigned_area_m2"] = assigned.geometry.area
    assigned["assigned_acres"] = assigned["assigned_area_m2"] / SQUARE_METRES_PER_ACRE
    return assigned


def _build_corrected_reach_summary(
    assigned: Any,
    reaches: Any,
    args: argparse.Namespace,
    assignment_mode: str,
) -> Any:
    """Return the complete declared-reach x parking-class accounting table."""

    import pandas as pd

    declared_reaches = list(reaches[args.reach_name_field])
    skeleton = pd.DataFrame(
        [
            {
                "reach_name": reach_name,
                "class_id": class_id,
                "surface": surface,
            }
            for reach_name in declared_reaches
            for class_id, surface in SURFACE_NAMES.items()
        ]
    )
    if assigned.empty:
        observed = pd.DataFrame(columns=["reach_name", "class_id", "surface", "acres"])
        assigned_total = 0.0
    else:
        observed = (
            assigned.groupby(
                [args.reach_name_field, "class_id", "surface"],
                as_index=False,
            )["assigned_acres"]
            .sum()
            .rename(
                columns={
                    args.reach_name_field: "reach_name",
                    "assigned_acres": "acres",
                }
            )
        )
        assigned_total = float(assigned["assigned_acres"].sum())

    table = skeleton.merge(
        observed,
        how="left",
        on=["reach_name", "class_id", "surface"],
        validate="one_to_one",
    )
    table["acres"] = pd.to_numeric(table["acres"], errors="raise").fillna(0.0)
    summarized_total = float(table["acres"].sum())
    tolerance = max(1e-12, abs(assigned_total) * 1e-12)
    if abs(summarized_total - assigned_total) > tolerance:
        raise ContractError("Corrected reach-summary rows do not preserve the assigned-area total.")

    table["spaces_per_acre"] = args.spaces_per_acre
    table["capacity_rate_source"] = args.capacity_rate_source
    if args.spaces_per_acre is None:
        table["estimated_spaces"] = None
    else:
        table["estimated_spaces"] = table["acres"].map(
            lambda acres: capacity_spaces(float(acres), args.spaces_per_acre)
        )
    table["contract"] = args.contract
    table["assignment_contract"] = (
        "boundary-clipped" if assignment_mode == "clip" else "historical-full-polygon-intersects"
    )
    table["minimum_area_sqft"] = args.minimum_area_sqft
    return table.sort_values(["reach_name", "class_id"]).reset_index(drop=True)


def _corrected_area_comparison_table(
    counts: dict[tuple[str, int], int],
    *,
    pixel_area_m2: float,
    minimum_area_sqft: float,
) -> Any:
    import pandas as pd

    rows = [
        {
            "source": source_name,
            "contract": "corrected",
            "class_id": class_id,
            "class_name": SURFACE_NAMES[class_id],
            "pixel_count": counts[(source_name, class_id)],
            "pixel_area_m2": pixel_area_m2,
            "acres": counts[(source_name, class_id)] * pixel_area_m2 / SQUARE_METRES_PER_ACRE,
            "minimum_area_sqft": minimum_area_sqft,
            "comparison_domain": "prediction-valid-domain-plus-study-areas-on-grid",
        }
        for source_name in ("model_prediction", "manual_label")
        for class_id in SURFACE_NAMES
    ]
    return pd.DataFrame(rows).sort_values(["class_name", "source"])


def _validate_corrected_reaches(reaches: Any, name_field: str) -> Any:
    """Require an unambiguous polygon partition for corrected reach accounting."""

    if reaches.empty:
        raise ContractError("Corrected study reaches cannot be empty.")
    if reaches.crs is None:
        raise ContractError("Corrected study reaches must declare a CRS.")

    geometries = reaches.geometry
    valid = geometries.notna() & ~geometries.is_empty & geometries.is_valid
    if not bool(valid.all()):
        invalid_rows = list(reaches.index[~valid][:10])
        raise ContractError(f"Corrected study reaches contain null, empty, or invalid geometry at rows {invalid_rows}.")
    polygonal = geometries.geom_type.isin(("Polygon", "MultiPolygon"))
    if not bool(polygonal.all()):
        invalid_rows = list(reaches.index[~polygonal][:10])
        invalid_types = list(geometries.loc[~polygonal].geom_type.astype("string").unique()[:10])
        raise ContractError(
            "Corrected study reaches must contain only Polygon/MultiPolygon geometry; "
            f"rows {invalid_rows} have types {invalid_types}."
        )
    positive_area = geometries.map(lambda geometry: float(geometry.area) > 0.0)
    if not bool(positive_area.all()):
        invalid_rows = list(reaches.index[~positive_area][:10])
        raise ContractError(f"Corrected study reaches must have positive area; invalid rows: {invalid_rows}.")

    labels = reaches[name_field] if name_field in reaches.columns else reaches.index.to_series()
    for left_position in range(len(reaches)):
        left = geometries.iloc[left_position]
        for right_position in range(left_position + 1, len(reaches)):
            right = geometries.iloc[right_position]
            if not left.intersects(right):
                continue
            overlap = left.intersection(right)
            if not overlap.is_empty and float(overlap.area) > 0.0:
                raise ContractError(
                    "Corrected study reaches have a positive-area pairwise overlap between "
                    f"{labels.iloc[left_position]!r} and {labels.iloc[right_position]!r}."
                )
    return reaches


def _validate_corrected_manual_labels(manual: Any, class_field: str) -> Any:
    """Reject every label that the corrected comparison would otherwise drop."""

    import numpy as np
    import pandas as pd

    valid = manual.geometry.notna() & ~manual.geometry.is_empty & manual.geometry.is_valid
    if not bool(valid.all()):
        invalid_rows = list(manual.index[~valid][:10])
        raise ContractError(
            "Corrected manual parking labels contain null, empty, or invalid geometry "
            f"at rows {invalid_rows}; no source labels may be silently discarded."
        )
    polygonal = manual.geometry.geom_type.isin(("Polygon", "MultiPolygon"))
    if not bool(polygonal.all()):
        invalid_rows = list(manual.index[~polygonal][:10])
        invalid_types = list(manual.geometry.loc[~polygonal].geom_type.astype("string").unique()[:10])
        raise ContractError(
            "Corrected manual parking labels must contain only Polygon/MultiPolygon geometry; "
            f"rows {invalid_rows} have types {invalid_types}."
        )
    positive_area = manual.geometry.map(lambda geometry: float(geometry.area) > 0.0)
    if not bool(positive_area.all()):
        invalid_rows = list(manual.index[~positive_area][:10])
        raise ContractError(f"Corrected manual parking labels must have positive area; invalid rows: {invalid_rows}.")
    class_values = pd.to_numeric(manual[class_field], errors="coerce")
    boolean_values = manual[class_field].map(lambda value: isinstance(value, (bool, np.bool_)))
    recognized = class_values.isin(SURFACE_NAMES) & ~boolean_values
    if not bool(recognized.all()):
        invalid_values = list(manual.loc[~recognized, class_field].astype("string").unique()[:10])
        raise ContractError(
            "Corrected manual parking labels contain missing or unrecognized class values "
            f"at rows {list(manual.index[~recognized][:10])}: {invalid_values}."
        )
    validated = manual.copy()
    validated["__class_id"] = class_values.astype(int)
    return validated


def _area_comparison(polygons: Any, reaches: Any, args: argparse.Namespace) -> Any:
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    if args.contract == "corrected":
        import rasterio
        from pyproj import CRS
        from rasterio.features import rasterize
        from rasterio.windows import Window, transform as window_transform

        manual = gpd.read_file(args.manual_labels)
        if manual.crs is None:
            raise ContractError(f"Manual labels have no CRS: {args.manual_labels}.")
        if args.manual_class_field not in manual.columns:
            raise ContractError(f"Manual class field {args.manual_class_field!r} is absent from {args.manual_labels}.")
        manual = _validate_corrected_manual_labels(manual, args.manual_class_field)
        manual = manual.to_crs(args.metric_crs)
        minimum_m2 = args.minimum_area_sqft / SQUARE_FEET_PER_SQUARE_METRE
        manual = manual.loc[manual.geometry.area >= minimum_m2].copy()
        if manual.empty:
            raise ContractError("No valid manual parking labels remain for the corrected comparison.")

        manual_unions = {
            class_id: (
                group.geometry.union_all() if hasattr(group.geometry, "union_all") else group.geometry.unary_union
            )
            for class_id, group in manual.groupby("__class_id")
        }
        if 1 in manual_unions and 2 in manual_unions:
            cross_class_overlap = manual_unions[1].intersection(manual_unions[2])
            if not cross_class_overlap.is_empty and float(cross_class_overlap.area) > 1e-6:
                raise ContractError("Corrected manual paved and dirt/gravel labels overlap; class area is ambiguous.")

        with rasterio.open(args.prediction) as source:
            if source.crs is None:
                raise ContractError("Prediction raster has no CRS for comparison-grid accounting.")
            source_crs = CRS.from_user_input(source.crs)
            units = {str(axis.unit_name).lower() for axis in source_crs.axis_info if axis.unit_name}
            if not source_crs.is_projected or not units or not all("met" in unit for unit in units):
                raise ContractError("Corrected comparison requires a prediction grid projected with metre axes.")
            pixel_area_m2 = abs(source.transform.a * source.transform.e - source.transform.b * source.transform.d)
            if pixel_area_m2 <= 0:
                raise ContractError("Prediction grid has nonpositive pixel area.")
            model_grid = polygons.to_crs(source.crs)
            manual_grid = manual.to_crs(source.crs)
            reaches_grid = reaches.to_crs(source.crs)
            reach_shapes = [geometry for geometry in reaches_grid.geometry if not geometry.is_empty]
            model_shapes = {
                class_id: [geometry for geometry in group.geometry if not geometry.is_empty]
                for class_id, group in model_grid.groupby("class_id")
            }
            manual_shapes = {
                int(class_id): [geometry for geometry in group.geometry if not geometry.is_empty]
                for class_id, group in manual_grid.groupby("__class_id")
            }
            counts = {
                (source_name, class_id): 0
                for source_name in ("model_prediction", "manual_label")
                for class_id in SURFACE_NAMES
            }
            for row in range(0, source.height, args.tile_size):
                height = min(args.tile_size, source.height - row)
                for column in range(0, source.width, args.tile_size):
                    width = min(args.tile_size, source.width - column)
                    window = Window(column, row, width, height)
                    valid_grid = source.read_masks(1, window=window) > 0
                    if not bool(valid_grid.any()):
                        continue
                    transform = window_transform(window, source.transform)
                    study_grid = rasterize(
                        ((geometry, 1) for geometry in reach_shapes),
                        out_shape=(height, width),
                        transform=transform,
                        fill=0,
                        dtype="uint8",
                    ).astype(bool)
                    domain = valid_grid & study_grid
                    if not bool(domain.any()):
                        continue
                    for class_id in SURFACE_NAMES:
                        for source_name, shapes_by_class in (
                            ("model_prediction", model_shapes),
                            ("manual_label", manual_shapes),
                        ):
                            class_shapes = shapes_by_class.get(class_id, [])
                            if not class_shapes:
                                continue
                            class_grid = rasterize(
                                ((geometry, 1) for geometry in class_shapes),
                                out_shape=(height, width),
                                transform=transform,
                                fill=0,
                                dtype="uint8",
                            ).astype(bool)
                            counts[(source_name, class_id)] += int(np.count_nonzero(class_grid & domain))

        return _corrected_area_comparison_table(
            counts,
            pixel_area_m2=pixel_area_m2,
            minimum_area_sqft=args.minimum_area_sqft,
        )

    rows: list[dict[str, Any]] = []
    for class_id, group in polygons.groupby("class_id"):
        rows.append(
            {
                "source": "model_prediction",
                "contract": args.contract,
                "class_id": int(class_id),
                "surface": SURFACE_NAMES[int(class_id)],
                "acres": float(group.geometry.area.sum() / SQUARE_METRES_PER_ACRE),
                "minimum_area_sqft": args.minimum_area_sqft,
            }
        )

    if args.manual_labels is not None and args.manual_labels.is_file():
        manual = gpd.read_file(args.manual_labels)
        if manual.crs is None:
            raise ContractError(f"Manual labels have no CRS: {args.manual_labels}.")
        manual = manual.to_crs(args.metric_crs)
        if args.manual_class_field not in manual.columns:
            raise ContractError(f"Manual class field {args.manual_class_field!r} is absent from {args.manual_labels}.")
        valid = manual.geometry.notna() & ~manual.geometry.is_empty & manual.geometry.is_valid
        manual = manual.loc[valid].copy()
        minimum_m2 = args.minimum_area_sqft / SQUARE_FEET_PER_SQUARE_METRE
        manual = manual.loc[manual.geometry.area >= minimum_m2].copy()
        for class_id, group in manual.groupby(args.manual_class_field):
            class_id = int(class_id)
            if class_id in SURFACE_NAMES:
                rows.append(
                    {
                        "source": "manual_label",
                        "contract": args.contract,
                        "class_id": class_id,
                        "surface": SURFACE_NAMES[class_id],
                        "acres": float(group.geometry.area.sum() / SQUARE_METRES_PER_ACRE),
                        "minimum_area_sqft": args.minimum_area_sqft,
                    }
                )
    return pd.DataFrame(rows).sort_values(["surface", "source"])


def _inspect_corrected_prediction_raster(prediction: Path) -> dict[str, Any]:
    try:
        import numpy as np
        import rasterio
    except ModuleNotFoundError as exc:
        raise ContractError("Validating corrected prediction accounting requires rasterio.") from exc

    with rasterio.open(prediction) as source:
        if source.count != 1:
            raise ContractError("Corrected prediction raster must contain exactly one classification band.")
        if source.nodata is None or float(source.nodata) != float(PREDICTION_NODATA):
            raise ContractError(f"Corrected prediction raster must declare nodata={PREDICTION_NODATA}.")
        total_pixels = int(source.width) * int(source.height)
        if total_pixels <= 0:
            raise ContractError("Corrected prediction raster has no pixels.")

        valid_pixels = 0
        mask_nodata_pixels = 0
        explicit_nodata_pixels = 0
        histogram = np.zeros(len(CLASS_NAMES), dtype=np.int64)
        for _, window in source.block_windows(1):
            values = source.read(1, window=window)
            valid = source.read_masks(1, window=window) > 0
            explicit_nodata = values == PREDICTION_NODATA
            if not np.array_equal(~valid, explicit_nodata):
                raise ContractError(
                    "Corrected prediction raster mask and explicit nodata=255 pixels do not identify the same domain."
                )
            selected = values[valid]
            if selected.size:
                numeric = selected.astype(np.float64, copy=False)
                if not bool(np.isfinite(numeric).all()) or not bool(np.equal(numeric, np.floor(numeric)).all()):
                    raise ContractError("Corrected prediction valid pixels must contain finite integer class IDs.")
                if float(numeric.min()) < 0 or float(numeric.max()) >= len(CLASS_NAMES):
                    raise ContractError(
                        f"Corrected prediction valid pixels may contain only class IDs 0..{len(CLASS_NAMES) - 1}."
                    )
                histogram += np.bincount(numeric.astype(np.int64), minlength=len(CLASS_NAMES))[: len(CLASS_NAMES)]
            valid_pixels += int(np.count_nonzero(valid))
            mask_nodata_pixels += int(np.count_nonzero(~valid))
            explicit_nodata_pixels += int(np.count_nonzero(explicit_nodata))

    if valid_pixels + mask_nodata_pixels != total_pixels:
        raise ContractError("Corrected prediction raster mask accounting does not equal width x height.")
    return {
        "total_pixels": total_pixels,
        "prediction_valid_pixels": valid_pixels,
        "mask_nodata_pixels": mask_nodata_pixels,
        "explicit_nodata_pixels": explicit_nodata_pixels,
        "class_pixel_counts": {name: int(count) for name, count in zip(CLASS_NAMES, histogram, strict=True)},
    }


def _validate_prediction_manifest(
    args: argparse.Namespace,
) -> tuple[Path | None, str, dict[str, Any] | None]:
    candidate = args.prediction_manifest
    if candidate is None:
        candidate = args.prediction.with_suffix(".run_manifest.json")
    if not candidate.is_file():
        if args.contract == "corrected":
            raise ContractError(
                "Corrected summaries require the stage-05 inference manifest; pass --prediction-manifest."
            )
        return None, "missing-historical-provenance", None

    payload = load_run_manifest(
        candidate,
        expected_stage="05_infer_segmentation",
        expected_contract=args.contract,
    )
    if args.contract == "corrected" and payload.get("arguments", {}).get("slope_mask") is not True:
        raise ContractError(
            "Corrected capacity summaries require an inference run with the declared slope mask enabled."
        )

    verify_manifest_artifact(payload, args.prediction)
    if args.contract == "corrected":
        extra = payload.get("extra")
        if not isinstance(extra, dict):
            raise ContractError("Corrected inference manifest has no extra metadata object.")
        if extra.get("source_validity_contract") != SOURCE_VALIDITY_CONTRACT:
            raise ContractError("Corrected inference manifest has the wrong source-validity contract.")
        if extra.get("prediction_nodata_value") != PREDICTION_NODATA:
            raise ContractError(f"Corrected inference manifest must declare nodata value {PREDICTION_NODATA}.")
        if extra.get("background_is_not_nodata") is not True:
            raise ContractError("Corrected inference manifest must declare that background is not nodata.")

        counts: dict[str, int] = {}
        for field in ("source_valid_pixels", "prediction_valid_pixels", "nodata_pixels"):
            value = extra.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(f"Corrected inference manifest extra.{field} must be a nonnegative integer.")
            counts[field] = value

        class_counts = extra.get("class_pixel_counts")
        if not isinstance(class_counts, dict) or set(class_counts) != set(CLASS_NAMES):
            raise ContractError("Corrected inference manifest class_pixel_counts keys must exactly match CLASS_NAMES.")
        for class_name, value in class_counts.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(
                    f"Corrected inference manifest class count {class_name!r} must be a nonnegative integer."
                )

        raster = _inspect_corrected_prediction_raster(args.prediction)
        expected_pixels = raster["total_pixels"]
        if counts["source_valid_pixels"] > expected_pixels:
            raise ContractError("Corrected inference source-valid count exceeds the prediction raster size.")
        if counts["prediction_valid_pixels"] > counts["source_valid_pixels"]:
            raise ContractError("Corrected inference prediction-valid count exceeds its source-valid count.")
        if counts["prediction_valid_pixels"] + counts["nodata_pixels"] != expected_pixels:
            raise ContractError("Corrected inference valid/nodata accounting does not equal prediction width x height.")
        if sum(class_counts.values()) != counts["prediction_valid_pixels"]:
            raise ContractError("Corrected inference class counts do not equal prediction_valid_pixels.")
        if counts["prediction_valid_pixels"] != raster["prediction_valid_pixels"]:
            raise ContractError(
                "Corrected inference prediction_valid_pixels does not match the checksum-bound raster mask."
            )
        if (
            counts["nodata_pixels"] != raster["mask_nodata_pixels"]
            or counts["nodata_pixels"] != raster["explicit_nodata_pixels"]
        ):
            raise ContractError(
                "Corrected inference nodata_pixels does not match the checksum-bound raster mask/nodata count."
            )
        if class_counts != raster["class_pixel_counts"]:
            raise ContractError(
                "Corrected inference class_pixel_counts do not match the checksum-bound raster histogram."
            )
    return candidate, "verified", payload


def _write_assigned_polygons(assigned: Any, output: Path) -> None:
    """Write assigned polygons, including a schema-only layer for zero detections."""

    try:
        assigned.to_file(output, layer="parking_by_reach", driver="GPKG")
        return
    except (NotImplementedError, ValueError):
        if not assigned.empty:
            raise

    # GeoPandas/Fiona versions differ on whether GeoDataFrame.to_file accepts
    # zero rows. Fiona can portably create the required schema-only GeoPackage.
    import fiona

    property_types = {
        "class_id": "int",
        "area_m2": "float",
        "surface": "str",
        "source_polygon_id": "int",
        "assignment_contract": "str",
        "assigned_area_m2": "float",
        "assigned_acres": "float",
    }
    schema = {
        "geometry": "Unknown",
        "properties": {
            column: property_types.get(column, "str") for column in assigned.columns if column != assigned.geometry.name
        },
    }
    crs_wkt = assigned.crs.to_wkt() if assigned.crs is not None else None
    with fiona.open(
        output,
        mode="w",
        driver="GPKG",
        layer="parking_by_reach",
        schema=schema,
        crs_wkt=crs_wkt,
    ):
        pass


def summarize(args: argparse.Namespace) -> None:
    import geopandas as gpd
    from pyproj import CRS

    resolve_default_paths(args)
    if args.minimum_area_sqft is None:
        args.minimum_area_sqft = 3000.0 if args.contract == "historical" else 0.0
    _validate_minimum_area_contract(args.contract, args.minimum_area_sqft)
    if args.tile_size <= 0:
        raise ContractError("tile_size must be positive.")
    if args.spaces_per_acre is not None:
        capacity_spaces(0.0, args.spaces_per_acre)
        if not args.capacity_rate_source or not args.capacity_rate_source.strip():
            raise ContractError("--capacity-rate-source is required with --spaces-per-acre.")
    elif args.capacity_rate_source:
        raise ContractError("--capacity-rate-source is only valid with --spaces-per-acre.")
    metric_crs = CRS.from_user_input(args.metric_crs)
    if not metric_crs.is_projected:
        raise ContractError("metric_crs must be projected for square-metre area calculations.")
    metric_units = {str(axis.unit_name).lower() for axis in metric_crs.axis_info if axis.unit_name}
    if metric_units and not all("met" in unit for unit in metric_units):
        raise ContractError(f"metric_crs axes must use metres, not {sorted(metric_units)}.")
    assignment_mode = args.assignment_mode or ("historical-intersects" if args.contract == "historical" else "clip")
    if args.contract == "corrected" and assignment_mode != "clip":
        raise ContractError("Corrected analysis requires boundary-clipped reach assignment.")
    manifest_path = args.output_table.with_suffix(".run.json")
    comparison_enabled = args.manual_labels is not None
    managed_outputs = [args.output_table, args.output_polygons, manifest_path]
    if comparison_enabled:
        managed_outputs.append(args.area_comparison_output)
    for output in managed_outputs:
        refuse_existing_output(output)
    if not args.prediction.is_file():
        raise FileNotFoundError(args.prediction)
    if not args.study_areas.is_file():
        raise FileNotFoundError(args.study_areas)
    if comparison_enabled and not args.manual_labels.is_file():
        raise FileNotFoundError(
            f"Manual-versus-model comparison input does not exist: {args.manual_labels}"
        )
    prediction_manifest, prediction_provenance, prediction_payload = _validate_prediction_manifest(args)

    polygons = _polygonize(args)
    reaches = gpd.read_file(args.study_areas)
    if args.contract == "corrected":
        reaches = _validate_corrected_reaches(reaches, args.reach_name_field)
    reaches = reaches.to_crs(args.metric_crs)
    if args.reach_name_field not in reaches.columns:
        raise ContractError(f"Reach-name field {args.reach_name_field!r} is absent from {args.study_areas}.")
    reaches[args.reach_name_field] = require_unique_nonempty_names(
        reaches[args.reach_name_field],
        args.reach_name_field,
    )
    assigned = _assign_reaches(
        polygons,
        reaches,
        args.reach_name_field,
        assignment_mode,
        allow_empty=args.contract == "corrected",
    )

    if args.contract == "corrected":
        table = _build_corrected_reach_summary(assigned, reaches, args, assignment_mode)
    else:
        table = (
            assigned.groupby(
                [args.reach_name_field, "class_id", "surface"],
                as_index=False,
            )["assigned_acres"]
            .sum()
            .rename(
                columns={
                    args.reach_name_field: "reach_name",
                    "assigned_acres": "acres",
                }
            )
        )
        table["spaces_per_acre"] = args.spaces_per_acre
        table["capacity_rate_source"] = args.capacity_rate_source
        if args.spaces_per_acre is None:
            table["estimated_spaces"] = None
        else:
            table["estimated_spaces"] = table["acres"].map(
                lambda acres: capacity_spaces(float(acres), args.spaces_per_acre)
            )
        table["contract"] = args.contract
        table["assignment_contract"] = (
            "boundary-clipped" if assignment_mode == "clip" else "historical-full-polygon-intersects"
        )
        table["minimum_area_sqft"] = args.minimum_area_sqft
        table = table.sort_values(["reach_name", "class_id"])

    comparison = _area_comparison(polygons, reaches, args) if comparison_enabled else None
    if args.contract == "corrected" and comparison is not None:
        observed_sources = set(comparison["source"].astype(str))
        expected_sources = {"manual_label", "model_prediction"}
        if observed_sources != expected_sources:
            raise ContractError(
                "Corrected area comparison requires both manual_label and model_prediction rows; "
                f"found {sorted(observed_sources)}."
            )
    corridor_manifest_sha256 = (
        prediction_payload.get("extra", {}).get("corridor_manifest_sha256") if prediction_payload is not None else None
    )
    if args.contract == "corrected" and not corridor_manifest_sha256:
        raise ContractError("Corrected inference manifest does not expose its corridor-manifest checksum.")
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    args.output_polygons.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_table, index=False)
    if comparison is not None:
        args.area_comparison_output.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(args.area_comparison_output, index=False)
    _write_assigned_polygons(assigned, args.output_polygons)

    inputs = [args.prediction, args.study_areas]
    if prediction_manifest is not None:
        inputs.append(prediction_manifest)
    if args.manual_labels is not None and args.manual_labels.is_file():
        inputs.append(args.manual_labels)
    outputs = [args.output_table, args.output_polygons]
    if comparison is not None:
        outputs.append(args.area_comparison_output)
    write_run_manifest(
        manifest_path,
        workflow_root=WORKFLOW_ROOT,
        stage="05_summarize_parking_capacity",
        contract=args.contract,
        arguments={**vars(args), "assignment_mode_effective": assignment_mode},
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        extra={
            "capacity_conversion_status": (
                "area-only" if args.spaces_per_acre is None else "explicit-user-supplied-rate"
            ),
            "capacity_rate_source": args.capacity_rate_source,
            "prediction_provenance": prediction_provenance,
            "corridor_manifest_sha256": corridor_manifest_sha256,
            "declared_reach_names": (list(reaches[args.reach_name_field]) if args.contract == "corrected" else None),
            "area_comparison_sources": (
                sorted(set(comparison["source"].astype(str))) if comparison is not None else []
            ),
            "area_comparison_domain": (
                (
                    "prediction-valid-domain-plus-study-areas-on-grid"
                    if args.contract == "corrected"
                    else "historical-unconstrained-vector-summary"
                )
                if comparison is not None
                else "not-requested"
            ),
            "warning": (
                "Historical intersects assignment can count a source polygon in multiple reaches."
                if assignment_mode == "historical-intersects"
                else None
            ),
        },
    )
    print(f"Wrote {len(table):,} reach/surface rows to {args.output_table}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summarize(args)
    except (ContractError, FileExistsError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
