#!/usr/bin/env python3
"""Build the recovered road/river analysis corridor and clip NAIP imagery.

This reconstructs notebook cells 5 and 14 from existing TIGER roads and the
manually selected Payette river system-of-interest layer.  The river subset is
an input, not silently regenerated: its historical selection rule remains
unknown.  All distances and areas are calculated in an explicitly projected
CRS. Corrected clipping retains genuine RGB+NIR without alpha; explicitly
isolated historical RGB clipping deliberately recreates the notebook's GDAL
alpha band so it can feed the historical RGBA dataset contract.
"""

from __future__ import annotations

import argparse
import math
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence


WORKING_ROOT = Path(__file__).resolve().parents[1]
if str(WORKING_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKING_ROOT))

from project_paths import (  # noqa: E402
    PARKING_CORRIDOR,
    PARKING_CORRIDOR_MANIFEST,
    PARKING_CORRIDOR_RASTER,
    PARKING_NAIP_MOSAIC,
    PARKING_NAIP_MOSAIC_MANIFEST,
    PARKING_SELECTED_ROADS,
    PARKING_SELECTED_ROADS_BUFFER,
    PAYETTE_RIVER_SOI,
    TIGER_ROADS,
    WORKFLOW_ROOT,
)
from pipeline.parking_common import (  # noqa: E402
    ContractError,
    load_run_manifest,
    refuse_existing_output,
    sha256_file,
    validate_corrected_mosaic_contract,
    verify_manifest_artifact,
    write_run_manifest,
)


# Recovered from notebook cell 5.  These are TIGER LINEARIDs for the selected
# Banks--Lowman Highway and related corridor segments.
DEFAULT_LINEAR_IDS = (
    "110448014479",
    "11036351348140",
    "110448014734",
    "1105599206204",
    "1104748137122",
    "1105599206409",
)
# Canonical scientific name used by the strict configuration consistency gate.
TARGET_LINEAR_IDS = DEFAULT_LINEAR_IDS
DEFAULT_INPUT_RASTER = PARKING_NAIP_MOSAIC
DEFAULT_INPUT_RASTER_MANIFEST = PARKING_NAIP_MOSAIC_MANIFEST
DEFAULT_SELECTED_ROADS = PARKING_SELECTED_ROADS
DEFAULT_SELECTED_ROADS_BUFFER = PARKING_SELECTED_ROADS_BUFFER
DEFAULT_CORRIDOR = PARKING_CORRIDOR
DEFAULT_CORRIDOR_RASTER = PARKING_CORRIDOR_RASTER
BAND_NAMES = {"rgb": ["R", "G", "B"], "rgbnir": ["R", "G", "B", "N"]}
CORRIDOR_NODATA = -9999.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select the recovered TIGER road segments, intersect 500 m road/river buffers, and clip NAIP."
    )
    parser.add_argument("--roads", type=Path, default=TIGER_ROADS, help="TIGER roads vector containing LINEARID.")
    parser.add_argument(
        "--river",
        type=Path,
        default=PAYETTE_RIVER_SOI,
        help="Reviewed Payette river system-of-interest vector (selection provenance must be supplied separately).",
    )
    parser.add_argument(
        "--linear-id",
        action="append",
        dest="linear_ids",
        default=None,
        help="TIGER LINEARID to select; repeat to replace the six recovered defaults.",
    )
    parser.add_argument("--buffer-m", type=float, default=500.0, help="Road and river buffer distance in metres.")
    parser.add_argument(
        "--simplify-m",
        type=float,
        default=5.0,
        help="Topology-preserving corridor simplification tolerance in metres.",
    )
    parser.add_argument(
        "--metric-crs",
        default="EPSG:32611",
        help="Projected CRS used for buffers, intersections, simplification, and area (default: WGS 84 / UTM 11N).",
    )
    parser.add_argument("--storage-crs", default="EPSG:4326", help="CRS used for saved corridor vector products.")
    parser.add_argument("--selected-roads-output", type=Path, default=DEFAULT_SELECTED_ROADS)
    parser.add_argument("--road-buffer-output", type=Path, default=DEFAULT_SELECTED_ROADS_BUFFER)
    parser.add_argument("--corridor-output", type=Path, default=DEFAULT_CORRIDOR)
    parser.add_argument(
        "--clip-raster",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip the input raster to the corridor (default: enabled).",
    )
    parser.add_argument("--input-raster", type=Path, default=DEFAULT_INPUT_RASTER)
    parser.add_argument(
        "--input-raster-manifest",
        type=Path,
        default=DEFAULT_INPUT_RASTER_MANIFEST,
        help=(
            "01_mosaic_naip_exports run manifest. Its stage, band contract, and checksum for --input-raster "
            "are enforced before output; defaults to the corrected run manifest."
        ),
    )
    parser.add_argument(
        "--no-input-raster-manifest",
        action="store_const",
        const=None,
        dest="input_raster_manifest",
        help="Disable mosaic-manifest enforcement only for an explicitly isolated historical reconstruction.",
    )
    parser.add_argument("--raster-output", type=Path, default=DEFAULT_CORRIDOR_RASTER)
    parser.add_argument(
        "--raster-resolution-m",
        type=float,
        default=None,
        help=(
            "Projected corridor-raster resolution in metres. With a verified mosaic, the default is the "
            "Earth Engine export scale recorded in its manifest; otherwise this must be supplied explicitly."
        ),
    )
    parser.add_argument(
        "--band-contract",
        choices=("rgb", "rgbnir"),
        default="rgbnir",
        help=(
            "Raster contract. Corrected RGBNIR requires a real fourth spectral band and never adds alpha. "
            "Explicitly isolated historical RGB deliberately adds the notebook's GDAL alpha band."
        ),
    )
    parser.add_argument(
        "--overviews",
        default="2,4,8,16,32,64,128,256,512",
        help="Comma-separated GDAL overview factors; pass an empty string to skip.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Run-manifest path; defaults beside the corridor vector.",
    )
    return parser


def _vector_family(path: Path) -> list[Path]:
    if path.suffix.lower() != ".shp":
        return [path] if path.exists() else []
    return sorted(candidate for candidate in path.parent.glob(f"{path.stem}.*") if candidate.is_file())


def _refuse_vector_output(path: Path) -> None:
    existing = _vector_family(path)
    if existing:
        raise FileExistsError(
            f"Vector output family already exists; choose a new path: {path} "
            f"(found {', '.join(item.name for item in existing[:5])})"
        )


def _remove_vector_output(path: Path) -> None:
    for member in _vector_family(path):
        member.unlink(missing_ok=True)


def _union_geometry(frame: Any) -> Any:
    geometry = frame.geometry
    return geometry.union_all() if hasattr(geometry, "union_all") else geometry.unary_union


def _write_vector(frame: Any, path: Path, *, layer: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gpkg":
        frame.to_file(path, layer=layer, driver="GPKG")
    elif path.suffix.lower() == ".shp":
        frame.to_file(path, driver="ESRI Shapefile")
    else:
        raise ValueError(f"Unsupported vector output extension for {path}; use .gpkg or .shp.")


def _parse_overviews(value: str) -> list[int]:
    if not value.strip():
        return []
    try:
        levels = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    except ValueError as exc:
        raise ValueError("--overviews must be comma-separated integers.") from exc
    if any(level <= 1 for level in levels):
        raise ValueError("Every overview factor must be greater than one.")
    return levels


def _clip_rgb_raster(
    *,
    input_path: Path,
    output_path: Path,
    cutline_path: Path,
    band_contract: str,
    output_crs: str,
    output_resolution_m: float,
    overview_levels: Iterable[int],
) -> dict[str, Any]:
    try:
        from osgeo import gdal
    except ModuleNotFoundError as exc:
        raise RuntimeError("Raster clipping requires the GDAL Python bindings.") from exc

    gdal.UseExceptions()
    source = gdal.Open(str(input_path), gdal.GA_ReadOnly)
    if source is None:
        raise RuntimeError(f"GDAL could not open input raster: {input_path}")
    source_band_count = 3 if band_contract == "rgb" else 4
    add_historical_alpha = band_contract == "rgb"
    required_output_bands = source_band_count + int(add_historical_alpha)
    if source.RasterCount < source_band_count:
        raise ValueError(
            f"{band_contract} requires {source_band_count} source bands; found {source.RasterCount} in {input_path}."
        )
    if band_contract == "rgbnir":
        fourth = source.GetRasterBand(4)
        interpretation = gdal.GetColorInterpretationName(fourth.GetColorInterpretation()).lower()
        if "alpha" in interpretation:
            raise ValueError(
                "The fourth source band is alpha, not NIR. Re-export genuine R,G,B,N imagery before RGBNIR clipping."
            )
        for index in range(1, 5):
            band = source.GetRasterBand(index)
            nodata = band.GetNoDataValue()
            if gdal.GetDataTypeName(band.DataType) != "Float32":
                raise ContractError(f"Corrected mosaic band {index} is not Float32: {input_path}.")
            if nodata is None or not math.isclose(float(nodata), CORRIDOR_NODATA, rel_tol=0.0, abs_tol=1e-9):
                raise ContractError(
                    f"Corrected mosaic band {index} does not declare nodata {CORRIDOR_NODATA}: {input_path}."
                )

    # Select only the declared source bands through an in-memory VRT.  In
    # particular, do not repeat the notebook's dstAlpha=True operation.
    vrt_path = f"/vsimem/payette_corridor_{uuid.uuid4().hex}.vrt"
    selected = None
    warped = None
    try:
        selected = gdal.Translate(
            vrt_path,
            source,
            options=gdal.TranslateOptions(format="VRT", bandList=list(range(1, source_band_count + 1))),
        )
        if selected is None:
            raise RuntimeError("GDAL failed to construct the selected-band VRT.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        warp_options = gdal.WarpOptions(
            format="GTiff",
            cutlineDSName=str(cutline_path),
            cropToCutline=True,
            dstSRS=output_crs,
            xRes=output_resolution_m,
            yRes=output_resolution_m,
            targetAlignedPixels=True,
            resampleAlg="near",
            dstAlpha=add_historical_alpha,
            # Zero is a valid unit-scaled reflectance. Use an out-of-range
            # Float32 sentinel so source validity never discards real black
            # pixels as if they were outside the cutline.
            dstNodata=CORRIDOR_NODATA,
            creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=2", "BIGTIFF=IF_SAFER"],
            multithread=True,
            warpOptions=["NUM_THREADS=ALL_CPUS"],
        )
        warped = gdal.Warp(str(output_path), selected, options=warp_options)
        if warped is None:
            raise RuntimeError("GDAL Warp returned no dataset.")
        warped.FlushCache()
        output_band_count = warped.RasterCount
        if output_band_count != required_output_bands:
            raise RuntimeError(
                f"Clipped raster has {output_band_count} bands; the declared contract requires {required_output_bands}."
            )
        geotransform = warped.GetGeoTransform()
        if not (
            abs(abs(float(geotransform[1])) - output_resolution_m) <= 1e-6
            and abs(abs(float(geotransform[5])) - output_resolution_m) <= 1e-6
        ):
            raise RuntimeError(
                "Clipped raster resolution does not match the requested projected metre grid: "
                f"{geotransform[1]}, {geotransform[5]} versus {output_resolution_m}."
            )
        output_names = BAND_NAMES[band_contract] if band_contract == "rgbnir" else ["R", "G", "B", "alpha"]
        interpretations = [gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand, gdal.GCI_Undefined]
        if add_historical_alpha:
            interpretations[-1] = gdal.GCI_AlphaBand
        output_nodata: list[float | None] = []
        output_types: list[str] = []
        for index, name in enumerate(output_names, start=1):
            band = warped.GetRasterBand(index)
            band.SetDescription(name)
            band.SetColorInterpretation(interpretations[index - 1])
            band.SetMetadataItem("PAYETTE_SEMANTIC_BAND", name)
            nodata = band.GetNoDataValue()
            output_nodata.append(float(nodata) if nodata is not None else None)
            output_types.append(gdal.GetDataTypeName(band.DataType))
        effective_band_contract = "rgba-alpha" if add_historical_alpha else "rgbnir"
        warped.SetMetadataItem("PAYETTE_BAND_CONTRACT", effective_band_contract)
        projection = warped.GetProjection()
        if not projection:
            raise ContractError("Clipped corridor raster has no projected CRS.")
        from osgeo import osr

        observed_srs = osr.SpatialReference()
        expected_srs = osr.SpatialReference()
        observed_srs.ImportFromWkt(projection)
        expected_srs.SetFromUserInput(output_crs)
        if hasattr(observed_srs, "SetAxisMappingStrategy"):
            observed_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            expected_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not bool(observed_srs.IsSame(expected_srs)):
            raise ContractError(f"Clipped corridor raster CRS does not equal {output_crs}.")
        if any(data_type != "Float32" for data_type in output_types):
            raise ContractError(f"Clipped corridor raster bands are not all Float32: {output_types!r}.")
        if band_contract == "rgbnir" and output_nodata != [CORRIDOR_NODATA] * 4:
            raise ContractError(
                f"Corrected corridor bands must all declare nodata {CORRIDOR_NODATA}; found {output_nodata!r}."
            )
        levels = list(overview_levels)
        if levels:
            warped.BuildOverviews("AVERAGE", levels)
        warped = None
        return {
            "band_count": output_band_count,
            "band_names": output_names,
            "band_contract": effective_band_contract,
            "crs_wkt": projection,
            "resolution_m": abs(float(geotransform[1])),
            "geotransform": [float(value) for value in geotransform],
            "nodata_values": output_nodata,
            "data_types": output_types,
        }
    finally:
        warped = None
        selected = None
        source = None
        gdal.Unlink(vrt_path)


def run(args: argparse.Namespace) -> int:
    roads_path = args.roads.expanduser().resolve(strict=False)
    river_path = args.river.expanduser().resolve(strict=False)
    selected_output = args.selected_roads_output.expanduser().resolve(strict=False)
    buffer_output = args.road_buffer_output.expanduser().resolve(strict=False)
    corridor_output = args.corridor_output.expanduser().resolve(strict=False)
    raster_input = args.input_raster.expanduser().resolve(strict=False)
    raster_manifest = (
        args.input_raster_manifest.expanduser().resolve(strict=False)
        if args.input_raster_manifest is not None
        else None
    )
    raster_output = args.raster_output.expanduser().resolve(strict=False)
    manifest_path = (
        args.manifest.expanduser().resolve(strict=False) if args.manifest is not None else PARKING_CORRIDOR_MANIFEST
    )

    linear_ids = tuple(str(value).strip() for value in (args.linear_ids or DEFAULT_LINEAR_IDS))
    if not linear_ids or any(not value for value in linear_ids):
        raise ValueError("At least one non-empty --linear-id is required.")
    if len(set(linear_ids)) != len(linear_ids):
        raise ValueError("--linear-id values must be unique.")
    if args.buffer_m <= 0:
        raise ValueError("--buffer-m must be positive.")
    if args.simplify_m < 0:
        raise ValueError("--simplify-m cannot be negative.")
    if not roads_path.is_file():
        raise FileNotFoundError(f"Road vector does not exist: {roads_path}")
    if not river_path.is_file():
        raise FileNotFoundError(f"River system-of-interest vector does not exist: {river_path}")
    if args.clip_raster and not raster_input.is_file():
        raise FileNotFoundError(f"Input NAIP raster does not exist: {raster_input}")
    if not args.clip_raster and raster_manifest is not None:
        raise ValueError("--input-raster-manifest is only meaningful when --clip-raster is enabled.")
    if args.band_contract == "rgb":
        corrected_defaults = {
            DEFAULT_SELECTED_ROADS.resolve(strict=False),
            DEFAULT_SELECTED_ROADS_BUFFER.resolve(strict=False),
            DEFAULT_CORRIDOR.resolve(strict=False),
            DEFAULT_CORRIDOR_RASTER.resolve(strict=False),
            DEFAULT_INPUT_RASTER.resolve(strict=False),
            DEFAULT_INPUT_RASTER_MANIFEST.resolve(strict=False),
            PARKING_CORRIDOR_MANIFEST.resolve(strict=False),
        }
        historical_targets = {selected_output, buffer_output, corridor_output, manifest_path}
        if args.clip_raster:
            historical_targets.update((raster_input, raster_output))
            if raster_manifest is not None:
                historical_targets.add(raster_manifest)
        if historical_targets & corrected_defaults:
            raise ContractError(
                "Historical RGB corridor reconstruction requires explicit isolated input/output/manifest paths; "
                "corrected RGB+NIR defaults are reserved."
            )

    durable_outputs = [selected_output, buffer_output, corridor_output]
    if args.clip_raster:
        durable_outputs.append(raster_output)
    durable_outputs.append(manifest_path)
    if len(set(durable_outputs)) != len(durable_outputs):
        raise ValueError("Selected-road, buffer, corridor, raster, and manifest outputs must all use distinct paths.")

    # Resolve every output before importing GeoPandas or writing a partial
    # product.  Shapefile sidecars are treated as one immutable output family.
    for output in (selected_output, buffer_output, corridor_output):
        _refuse_vector_output(output)
    if args.clip_raster:
        refuse_existing_output(raster_output)
    refuse_existing_output(manifest_path)
    overview_levels = _parse_overviews(args.overviews)

    verified_raster_record = None
    recorded_source_scale_m: float | None = None
    input_mosaic_manifest_sha256: str | None = None
    input_mosaic_pixel_valid_contract: str | None = None
    if raster_manifest is not None:
        mosaic_manifest = load_run_manifest(
            raster_manifest,
            expected_stage="01_mosaic_naip_exports",
            expected_contract=f"naip-unit-scaled-{args.band_contract}-mosaic",
        )
        mosaic_extra = mosaic_manifest.get("extra")
        if args.band_contract == "rgbnir":
            validate_corrected_mosaic_contract(mosaic_extra)
            input_mosaic_manifest_sha256 = sha256_file(raster_manifest)
            input_mosaic_pixel_valid_contract = mosaic_extra.get("pixel_valid_coverage_contract")
        recorded_band_names = mosaic_extra.get("raster_output_band_names") if isinstance(mosaic_extra, dict) else None
        if recorded_band_names != BAND_NAMES[args.band_contract]:
            raise ContractError(
                f"Input mosaic manifest band names {recorded_band_names!r} do not match "
                f"{BAND_NAMES[args.band_contract]}."
            )
        try:
            recorded_source_scale_m = float(mosaic_extra.get("source_scale_m"))
        except (TypeError, ValueError) as exc:
            raise ContractError("Input mosaic manifest has no numeric source_scale_m.") from exc
        if recorded_source_scale_m <= 0:
            raise ContractError(
                f"Input mosaic manifest source_scale_m must be positive; found {recorded_source_scale_m}."
            )
        verified_raster_record = verify_manifest_artifact(mosaic_manifest, raster_input, section="outputs")

    try:
        import geopandas as gpd
        from pyproj import CRS
    except ModuleNotFoundError as exc:
        raise RuntimeError("This stage requires geopandas, pyproj, and their vector I/O dependencies.") from exc

    metric_crs = CRS.from_user_input(args.metric_crs)
    if not metric_crs.is_projected:
        raise ValueError(f"--metric-crs must be projected for metre buffers and square-metre area: {args.metric_crs}")
    metric_units = {str(axis.unit_name).lower() for axis in metric_crs.axis_info if axis.unit_name}
    if metric_units and not all("met" in unit for unit in metric_units):
        raise ValueError(f"--metric-crs axes must use metres, not {sorted(metric_units)}: {args.metric_crs}")
    raster_resolution_m = args.raster_resolution_m
    if args.clip_raster:
        if raster_resolution_m is None:
            raster_resolution_m = recorded_source_scale_m
        if raster_resolution_m is None:
            raise ContractError(
                "Raster clipping without a verified mosaic manifest requires explicit --raster-resolution-m."
            )
        if raster_resolution_m <= 0:
            raise ValueError("--raster-resolution-m must be positive.")
        if (
            args.band_contract == "rgbnir"
            and recorded_source_scale_m is not None
            and not math.isclose(raster_resolution_m, recorded_source_scale_m, rel_tol=0.0, abs_tol=1e-9)
        ):
            raise ContractError(
                "Corrected corridor resolution must equal the verified Earth Engine export scale; "
                f"found {raster_resolution_m} versus {recorded_source_scale_m}."
            )

    roads = gpd.read_file(roads_path)
    if roads.empty:
        raise ValueError(f"Road vector is empty: {roads_path}")
    if roads.crs is None:
        raise ValueError(f"Road vector has no CRS: {roads_path}")
    if "LINEARID" not in roads.columns:
        raise KeyError(f"Road vector has no LINEARID column: {roads_path}")
    road_ids = roads["LINEARID"].astype(str).str.strip()
    selected = roads.loc[road_ids.isin(linear_ids)].copy()
    found_ids = set(selected["LINEARID"].astype(str).str.strip())
    missing_ids = sorted(set(linear_ids).difference(found_ids))
    if missing_ids:
        raise ValueError(
            f"Recovered LINEARIDs are absent from the road source: {missing_ids}. "
            "Do not silently build a different corridor from a different TIGER vintage."
        )
    selected["LINEARID"] = selected["LINEARID"].astype(str).str.strip()

    rivers = gpd.read_file(river_path)
    if rivers.empty:
        raise ValueError(f"River system-of-interest vector is empty: {river_path}")
    if rivers.crs is None:
        raise ValueError(f"River vector has no CRS: {river_path}")

    selected_metric = selected.to_crs(metric_crs)
    rivers_metric = rivers.to_crs(metric_crs)
    road_buffer_geometry = _union_geometry(selected_metric).buffer(args.buffer_m)
    river_buffer_geometry = _union_geometry(rivers_metric).buffer(args.buffer_m)
    corridor_geometry = road_buffer_geometry.intersection(river_buffer_geometry)
    if corridor_geometry.is_empty:
        raise RuntimeError("The recovered road and river buffers do not intersect.")
    if args.simplify_m:
        corridor_geometry = corridor_geometry.simplify(args.simplify_m, preserve_topology=True)
    if corridor_geometry.is_empty:
        raise RuntimeError("Corridor became empty after simplification.")

    road_buffer_metric = gpd.GeoDataFrame(
        {"buffer_m": [args.buffer_m], "area_m2": [float(road_buffer_geometry.area)]},
        geometry=[road_buffer_geometry],
        crs=metric_crs,
    )
    corridor_metric = gpd.GeoDataFrame(
        {
            "road_buf_m": [args.buffer_m],
            "riv_buf_m": [args.buffer_m],
            "simpl_m": [args.simplify_m],
            "area_m2": [float(corridor_geometry.area)],
        },
        geometry=[corridor_geometry],
        crs=metric_crs,
    )

    raster_details: dict[str, Any] | None = None
    try:
        _write_vector(selected.to_crs(args.storage_crs), selected_output, layer="selected_roads")
        _write_vector(road_buffer_metric.to_crs(args.storage_crs), buffer_output, layer="selected_roads_buffer")
        _write_vector(corridor_metric.to_crs(args.storage_crs), corridor_output, layer="road_river_corridor")

        if args.clip_raster:
            raster_details = _clip_rgb_raster(
                input_path=raster_input,
                output_path=raster_output,
                cutline_path=corridor_output,
                band_contract=args.band_contract,
                output_crs=metric_crs.to_string(),
                output_resolution_m=raster_resolution_m,
                overview_levels=overview_levels,
            )

        input_paths = [roads_path, river_path]
        output_paths = [selected_output, buffer_output, corridor_output]
        if args.clip_raster:
            input_paths.append(raster_input)
            if raster_manifest is not None:
                input_paths.append(raster_manifest)
            output_paths.append(raster_output)
        manifest_arguments = dict(vars(args))
        manifest_arguments["linear_ids"] = list(linear_ids)
        write_run_manifest(
            manifest_path,
            workflow_root=WORKFLOW_ROOT,
            stage="01_build_corridor",
            contract=("projected-geometry-rgba-alpha" if args.band_contract == "rgb" else "projected-geometry-rgbnir"),
            arguments=manifest_arguments,
            inputs=input_paths,
            outputs=output_paths,
            extra={
                "selected_feature_count": int(len(selected)),
                "selected_linear_ids": list(linear_ids),
                "metric_crs": metric_crs.to_string(),
                "road_buffer_area_m2": float(road_buffer_geometry.area),
                "corridor_area_m2": float(corridor_geometry.area),
                "raster_output_band_count": raster_details["band_count"] if raster_details else None,
                "raster_output_band_names": raster_details["band_names"] if raster_details else None,
                "raster_output_band_contract": raster_details["band_contract"] if raster_details else None,
                "raster_output_crs": metric_crs.to_string() if raster_details else None,
                "raster_output_crs_wkt": raster_details["crs_wkt"] if raster_details else None,
                "raster_output_resolution_m": raster_details["resolution_m"] if raster_details else None,
                "raster_output_geotransform": raster_details["geotransform"] if raster_details else None,
                "raster_resampling": "nearest" if args.clip_raster else None,
                "raster_nodata_value": CORRIDOR_NODATA if args.clip_raster else None,
                "raster_nodata_values_observed": raster_details["nodata_values"] if raster_details else None,
                "raster_data_types_observed": raster_details["data_types"] if raster_details else None,
                "historical_alpha_emulation": args.band_contract == "rgb" and args.clip_raster,
                "raster_resolution_source": (
                    "explicit-cli"
                    if args.clip_raster and args.raster_resolution_m is not None
                    else "verified-earth-engine-export-scale"
                    if args.clip_raster
                    else None
                ),
                "input_raster_provenance": "verified" if verified_raster_record is not None else "unverified",
                "input_raster_manifest": raster_manifest,
                "input_mosaic_manifest_sha256": input_mosaic_manifest_sha256,
                "input_mosaic_pixel_valid_contract": input_mosaic_pixel_valid_contract,
                "river_soi_selection_rule": "external reviewed input; not recoverable from notebook",
                "transactional_output_contract": "cleanup-uncommitted-stage-outputs-on-failure",
            },
        )
    except BaseException:
        for vector_output in (selected_output, buffer_output, corridor_output):
            _remove_vector_output(vector_output)
        if args.clip_raster:
            raster_output.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    print(f"Saved corridor: {corridor_output}")
    if args.clip_raster:
        print(f"Saved {args.band_contract.upper()} corridor raster: {raster_output}")
    print(f"Wrote run manifest: {manifest_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
