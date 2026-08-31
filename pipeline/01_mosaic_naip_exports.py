#!/usr/bin/env python3
"""Assemble completed Earth Engine NAIP export shards into one verified mosaic.

This stage accepts only a successful ``01_export_naip_earth_engine`` manifest,
discovers the TIFF family declared by its export prefix, validates the band
contract, and writes a checksummed single-raster product. Heavy GDAL imports
are deferred so ``--help`` remains usable in a lightweight environment.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


WORKING_ROOT = Path(__file__).resolve().parents[1]
if str(WORKING_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKING_ROOT))

from project_paths import NAIP_EXPORT_ROOT, PARKING_NAIP_MOSAIC, WORKFLOW_ROOT  # noqa: E402
from pipeline.parking_common import (  # noqa: E402
    ContractError,
    load_run_manifest,
    refuse_existing_output,
    resolve_manifest_artifact_path,
    sha256_file,
    verify_manifest_artifact,
    write_run_manifest,
)


DEFAULT_OUTPUT = PARKING_NAIP_MOSAIC
BAND_NAMES = {"rgb": ["R", "G", "B"], "rgbnir": ["R", "G", "B", "N"]}
MOSAIC_NODATA = -9999.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mosaic a completed Earth Engine NAIP TIFF family and preserve its verified band contract."
    )
    parser.add_argument(
        "--export-manifest",
        type=Path,
        required=True,
        help="Completed 01_export_naip_earth_engine run manifest (a submission manifest is not sufficient).",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=NAIP_EXPORT_ROOT,
        help="Local directory containing the synced TIFF family; the prefix is read from the export manifest.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="New single-raster GeoTIFF mosaic.")
    parser.add_argument(
        "--overviews",
        default="2,4,8,16,32,64,128,256,512",
        help="Comma-separated GDAL overview factors; pass an empty string to skip.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Run-manifest path; defaults beside --output.",
    )
    return parser


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


def _band_contract(payload: Mapping[str, Any]) -> str:
    contract = str(payload.get("contract", ""))
    prefix = "naip-unit-scaled-"
    if not contract.startswith(prefix):
        raise ContractError(f"Unsupported Earth Engine export contract: {contract!r}.")
    band_contract = contract.removeprefix(prefix)
    if band_contract not in BAND_NAMES:
        raise ContractError(f"Unsupported NAIP band contract in export manifest: {band_contract!r}.")
    extra = payload.get("extra")
    recorded_bands = extra.get("bands") if isinstance(extra, Mapping) else None
    if recorded_bands != BAND_NAMES[band_contract]:
        raise ContractError(
            f"Earth Engine band metadata {recorded_bands!r} does not match {band_contract!r} "
            f"({BAND_NAMES[band_contract]})."
        )
    return band_contract


def _require_completed_export(payload: Mapping[str, Any]) -> None:
    extra = payload.get("extra")
    if not isinstance(extra, Mapping):
        raise ContractError("Earth Engine manifest has no structured extra metadata.")
    task_status = extra.get("task_status")
    state = str(task_status.get("state", "")).upper() if isinstance(task_status, Mapping) else ""
    if state != "COMPLETED":
        raise ContractError(f"Earth Engine export is not completed (recorded task state: {state or 'missing'}).")
    outcome = extra.get("run_outcome")
    if outcome not in (None, "completed"):
        raise ContractError(f"Earth Engine run outcome is not completed: {outcome!r}.")


def _export_scale_m(payload: Mapping[str, Any]) -> float:
    extra = payload.get("extra")
    raw = extra.get("scale_m") if isinstance(extra, Mapping) else None
    try:
        scale_m = float(raw)
    except (TypeError, ValueError) as exc:
        raise ContractError("Earth Engine manifest has no numeric export scale_m.") from exc
    if scale_m <= 0:
        raise ContractError(f"Earth Engine export scale_m must be positive; found {scale_m}.")
    return scale_m


def _export_prefix(payload: Mapping[str, Any]) -> str:
    extra = payload.get("extra")
    raw = extra.get("expected_local_prefix") if isinstance(extra, Mapping) else None
    prefix = Path(str(raw)).name.strip() if raw else ""
    if not prefix or prefix in {".", ".."}:
        raise ContractError("Earth Engine manifest has no usable expected_local_prefix.")
    return prefix


def _discover_exports(export_dir: Path, prefix: str) -> list[Path]:
    exact = export_dir / f"{prefix}.tif"
    paths = ([exact.resolve(strict=False)] if exact.is_file() else []) + sorted(
        path.resolve(strict=False) for path in export_dir.glob(f"{prefix}-*.tif") if path.is_file()
    )
    if not paths:
        raise FileNotFoundError(f"No synced Earth Engine TIFFs match {prefix!r} in {export_dir}.")
    return paths


def _source_snapshot(sources: Sequence[Path]) -> tuple[tuple[str, int, int, str], ...]:
    """Capture metadata and content hashes around mosaic/manifest creation."""

    return tuple(
        (
            str(path.resolve(strict=False)),
            path.stat().st_size,
            path.stat().st_mtime_ns,
            sha256_file(path),
        )
        for path in sources
    )


def _roi_input(payload: Mapping[str, Any]) -> Path:
    records = payload.get("inputs")
    if not isinstance(records, list):
        raise ContractError("Earth Engine manifest has no input artifact list.")
    vector_suffixes = {".shp", ".gpkg", ".geojson", ".json", ".fgb"}
    candidates: list[Path] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        path = resolve_manifest_artifact_path(payload, record)
        # Submission/run JSON records are provenance, not the region vector.
        if path.suffix.lower() in vector_suffixes and not path.name.endswith((".submission.json", ".run.json")):
            candidates.append(path)
    if len(candidates) != 1:
        raise ContractError(
            f"Earth Engine manifest must identify exactly one ROI vector input; found {len(candidates)}."
        )
    verify_manifest_artifact(payload, candidates[0], section="inputs")
    return candidates[0]


def _verify_recorded_exports(payload: Mapping[str, Any], sources: Iterable[Path]) -> int:
    """Require a legacy exact family or an explicit mosaic-time inventory handoff.

    New terminal manifests never treat an immediate Drive observation as
    authoritative because synchronization can expose a partial shard family.
    Their collision-checked prefix hands inventory ownership to this stage.
    Legacy manifests with output records retain exact checksum enforcement.
    """

    source_set = {path.resolve(strict=False) for path in sources}
    records = payload.get("outputs")
    if not isinstance(records, list):
        raise ContractError("Earth Engine manifest has no outputs artifact list.")
    recorded_paths: set[Path] = set()
    for record in records:
        if not isinstance(record, Mapping):
            continue
        recorded = resolve_manifest_artifact_path(payload, record)
        recorded_paths.add(recorded)

    extra = payload.get("extra")
    if not isinstance(extra, Mapping):
        raise ContractError("Earth Engine manifest has no structured post-sync metadata.")
    inventory_policy = extra.get("shard_inventory_policy")
    if inventory_policy == "deferred-to-mosaic-after-drive-sync":
        if recorded_paths:
            raise ContractError(
                "A deferred Earth Engine shard policy cannot also contain authoritative output records."
            )
        if extra.get("pre_submission_local_match_count") != 0:
            raise ContractError(
                "Deferred shard discovery requires a collision check with zero pre-submission prefix matches."
            )
        return 0

    if recorded_paths:
        if recorded_paths != source_set:
            raise ContractError(
                "Discovered Earth Engine TIFF family does not exactly match the export manifest "
                f"(manifest-only={len(recorded_paths - source_set)}, discovered-only={len(source_set - recorded_paths)})."
            )
        for recorded in sorted(recorded_paths):
            verify_manifest_artifact(payload, recorded, section="outputs")
        return len(recorded_paths)

    if extra.get("local_export_count") != 0 or extra.get("pre_submission_local_match_count") != 0:
        raise ContractError(
            "An export manifest without shard checksums must explicitly record zero pre-submission and "
            "zero final local prefix matches."
        )
    return 0


def _inspect_sources(gdal: Any, sources: Sequence[Path], band_contract: str) -> dict[str, Any]:
    required_bands = len(BAND_NAMES[band_contract])
    projections: set[str] = set()
    data_types: set[str] = set()
    pixel_sizes: set[tuple[float, float]] = set()
    grid_origins: list[tuple[float, float]] = []
    band_descriptions: set[tuple[str, ...]] = set()
    nodata_values: set[tuple[float | None, ...]] = set()
    minima = [float("inf")] * required_bands
    maxima = [float("-inf")] * required_bands
    for source_path in sources:
        dataset = gdal.Open(str(source_path), gdal.GA_ReadOnly)
        if dataset is None:
            raise RuntimeError(f"GDAL could not open Earth Engine shard: {source_path}")
        if dataset.RasterCount != required_bands:
            raise ContractError(
                f"{source_path} has {dataset.RasterCount} bands; {band_contract} requires exactly {required_bands}."
            )
        projection = dataset.GetProjection()
        if not projection:
            raise ContractError(f"Earth Engine shard has no CRS: {source_path}")
        projections.add(projection)
        transform = dataset.GetGeoTransform()
        if (
            not transform
            or not math.isclose(float(transform[2]), 0.0, abs_tol=1e-15)
            or not math.isclose(float(transform[4]), 0.0, abs_tol=1e-15)
        ):
            raise ContractError(f"Earth Engine shard is not a north-up grid: {source_path}.")
        x_size, y_size = abs(float(transform[1])), abs(float(transform[5]))
        if x_size <= 0 or y_size <= 0:
            raise ContractError(f"Earth Engine shard has a nonpositive pixel size: {source_path}.")
        pixel_sizes.add((x_size, y_size))
        grid_origins.append((float(transform[0]), float(transform[3])))
        descriptions: list[str] = []
        source_nodata: list[float | None] = []
        for index in range(1, required_bands + 1):
            band = dataset.GetRasterBand(index)
            data_types.add(gdal.GetDataTypeName(band.DataType))
            descriptions.append(str(band.GetDescription() or "").strip())
            nodata = band.GetNoDataValue()
            source_nodata.append(float(nodata) if nodata is not None else None)
            minimum, maximum = (float(value) for value in band.ComputeRasterMinMax(False))
            if not math.isfinite(minimum) or not math.isfinite(maximum):
                raise ContractError(f"Earth Engine shard contains nonfinite band statistics: {source_path}.")
            if minimum < -1e-6 or maximum > 1.000001:
                raise ContractError(
                    f"Earth Engine shard radiometry is outside the declared 0-1 range in band {index}: "
                    f"[{minimum}, {maximum}] ({source_path})."
                )
            minima[index - 1] = min(minima[index - 1], minimum)
            maxima[index - 1] = max(maxima[index - 1], maximum)
        nodata_values.add(tuple(source_nodata))
        band_descriptions.add(tuple(descriptions))
        declared = [value.upper() for value in descriptions]
        if any(declared) and declared != BAND_NAMES[band_contract]:
            raise ContractError(
                f"Earth Engine shard band descriptions {descriptions!r} do not match "
                f"{BAND_NAMES[band_contract]}: {source_path}."
            )
        if band_contract == "rgbnir":
            interpretation = gdal.GetColorInterpretationName(dataset.GetRasterBand(4).GetColorInterpretation()).lower()
            if "alpha" in interpretation:
                raise ContractError(f"Band 4 is alpha rather than NIR in {source_path}.")
        dataset = None
    if len(projections) != 1:
        raise ContractError(f"Earth Engine shards do not share one CRS (found {len(projections)} projections).")
    if len(data_types) != 1:
        raise ContractError(f"Earth Engine shards do not share one GDAL data type: {sorted(data_types)}.")
    if data_types != {"Float32"}:
        raise ContractError(
            f"Earth Engine export manifest declares float32 unit-scaled imagery, but shards use {sorted(data_types)}."
        )
    if band_contract == "rgbnir":
        expected_nodata = (MOSAIC_NODATA,) * required_bands
        if nodata_values != {expected_nodata}:
            raise ContractError(
                "Corrected Earth Engine shards must declare the explicit out-of-range nodata value "
                f"{MOSAIC_NODATA} on every band; found {sorted(nodata_values, key=str)!r}."
            )
    if len(pixel_sizes) != 1:
        raise ContractError(f"Earth Engine shards do not share one pixel size: {sorted(pixel_sizes)}.")
    x_size, y_size = next(iter(pixel_sizes))
    base_x, base_y = grid_origins[0]
    for origin_x, origin_y in grid_origins[1:]:
        if not math.isclose((origin_x - base_x) / x_size, round((origin_x - base_x) / x_size), abs_tol=1e-6):
            raise ContractError("Earth Engine shard x origins are not aligned to one pixel grid.")
        if not math.isclose((origin_y - base_y) / y_size, round((origin_y - base_y) / y_size), abs_tol=1e-6):
            raise ContractError("Earth Engine shard y origins are not aligned to one pixel grid.")
    return {
        "source_projection": next(iter(projections)),
        "source_data_type": next(iter(data_types)),
        "source_pixel_size": [x_size, y_size],
        "source_band_descriptions": sorted([list(values) for values in band_descriptions]),
        "source_nodata_values": sorted([list(values) for values in nodata_values], key=str),
        "source_radiometry_minima": minima,
        "source_radiometry_maxima": maxima,
    }


def _validate_mosaic_roi_coverage(
    gdal: Any,
    output: Path,
    roi_path: Path,
    sources: Sequence[Path],
) -> dict[str, Any]:
    """Require the actual shard footprints to cover the checksum-bound ROI.

    A mosaic envelope is insufficient evidence of a complete Drive sync: two
    edge shards can establish the expected bounds while one or more interior
    shards are absent.  Coverage is therefore calculated from the union of the
    individual source footprints, independently of the zero-filled mosaic that
    GDAL can create between them.
    """

    try:
        from osgeo import ogr, osr
    except ModuleNotFoundError as exc:
        raise RuntimeError("Mosaic ROI validation requires GDAL/OGR Python bindings.") from exc
    if hasattr(ogr, "UseExceptions"):
        ogr.UseExceptions()
    if hasattr(osr, "UseExceptions"):
        osr.UseExceptions()

    raster = gdal.Open(str(output), gdal.GA_ReadOnly)
    vector = ogr.Open(str(roi_path), 0)
    if raster is None or vector is None:
        raise RuntimeError(f"Could not open mosaic/ROI for coverage validation: {output}, {roi_path}.")
    layer = vector.GetLayer(0)
    if layer is None or layer.GetFeatureCount() == 0:
        raise ContractError(f"ROI vector has no readable features: {roi_path}.")
    source_srs = layer.GetSpatialRef()
    raster_srs = osr.SpatialReference()
    if not raster.GetProjection() or raster_srs.ImportFromWkt(raster.GetProjection()) != 0:
        raise ContractError("Mosaic has no readable CRS for ROI coverage validation.")
    if source_srs is None:
        raise ContractError(f"ROI vector has no readable CRS: {roi_path}.")
    if hasattr(source_srs, "SetAxisMappingStrategy"):
        source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        raster_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(source_srs, raster_srs)
    roi_geometry = None
    layer.ResetReading()
    for feature in layer:
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty():
            continue
        projected = geometry.Clone()
        if projected.Transform(transform) != 0:
            raise ContractError(f"Could not transform ROI geometry into the mosaic CRS: {roi_path}.")
        roi_geometry = projected if roi_geometry is None else roi_geometry.Union(projected)
    if roi_geometry is None or roi_geometry.IsEmpty() or roi_geometry.Area() <= 0:
        raise ContractError(f"ROI vector has no nonempty polygonal area: {roi_path}.")
    envelope = roi_geometry.GetEnvelope()
    roi_bounds = (envelope[0], envelope[2], envelope[1], envelope[3])
    geotransform = raster.GetGeoTransform()
    if (
        not geotransform
        or not math.isclose(float(geotransform[2]), 0.0, abs_tol=1e-15)
        or not math.isclose(float(geotransform[4]), 0.0, abs_tol=1e-15)
    ):
        raise ContractError("Mosaic is not a north-up grid.")
    x0, px, _, y0, _, py = (float(value) for value in geotransform)
    x1 = x0 + raster.RasterXSize * px
    y1 = y0 + raster.RasterYSize * py
    raster_bounds = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    tolerance_x, tolerance_y = 2 * abs(px), 2 * abs(py)
    if (
        raster_bounds[0] > roi_bounds[0] + tolerance_x
        or raster_bounds[1] > roi_bounds[1] + tolerance_y
        or raster_bounds[2] < roi_bounds[2] - tolerance_x
        or raster_bounds[3] < roi_bounds[3] - tolerance_y
    ):
        raise ContractError(
            "Synced Earth Engine TIFF family does not cover the recorded ROI envelope; "
            "Drive synchronization may be incomplete."
        )

    shard_footprints = None
    for source_path in sources:
        source = gdal.Open(str(source_path), gdal.GA_ReadOnly)
        if source is None:
            raise RuntimeError(f"Could not reopen Earth Engine shard for footprint validation: {source_path}.")
        source_transform = source.GetGeoTransform()
        source_projection = source.GetProjection()
        if not source_transform or not source_projection:
            raise ContractError(f"Earth Engine shard lacks a usable grid/CRS: {source_path}.")
        sx0, spx, srx, sy0, sry, spy = (float(value) for value in source_transform)
        corners = [
            (sx0, sy0),
            (sx0 + source.RasterXSize * spx, sy0 + source.RasterXSize * sry),
            (
                sx0 + source.RasterXSize * spx + source.RasterYSize * srx,
                sy0 + source.RasterXSize * sry + source.RasterYSize * spy,
            ),
            (sx0 + source.RasterYSize * srx, sy0 + source.RasterYSize * spy),
        ]
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for point_x, point_y in [*corners, corners[0]]:
            ring.AddPoint_2D(point_x, point_y)
        footprint = ogr.Geometry(ogr.wkbPolygon)
        footprint.AddGeometry(ring)
        shard_srs = osr.SpatialReference()
        if shard_srs.ImportFromWkt(source_projection) != 0:
            raise ContractError(f"Earth Engine shard has an unreadable CRS: {source_path}.")
        if hasattr(shard_srs, "SetAxisMappingStrategy"):
            shard_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if not bool(shard_srs.IsSame(raster_srs)):
            shard_transform = osr.CoordinateTransformation(shard_srs, raster_srs)
            if footprint.Transform(shard_transform) != 0:
                raise ContractError(f"Could not transform shard footprint into mosaic CRS: {source_path}.")
        shard_footprints = footprint if shard_footprints is None else shard_footprints.Union(footprint)
        source = None
    if shard_footprints is None or shard_footprints.IsEmpty():
        raise ContractError("Earth Engine shard family has no usable spatial footprints.")

    # Permit only numerical/grid-edge slivers.  Buffering by just over half a
    # pixel cannot conceal an absent interior shard but avoids rejecting a
    # complete family for sub-pixel CRS-transform rounding.
    edge_tolerance = 0.51 * max(abs(px), abs(py))
    uncovered = roi_geometry.Difference(shard_footprints.Buffer(edge_tolerance))
    uncovered_area = 0.0 if uncovered is None or uncovered.IsEmpty() else float(uncovered.Area())
    roi_area = float(roi_geometry.Area())
    uncovered_fraction = uncovered_area / roi_area
    pixel_area = abs(px * py)
    if uncovered_area > 4 * pixel_area:
        raise ContractError(
            "Synced Earth Engine shard footprints leave an interior or boundary gap within the recorded ROI "
            f"({uncovered_area:.12g} CRS-square-units; {uncovered_fraction:.6%} of ROI). "
            "Wait for Drive synchronization to finish and rerun."
        )
    pixel_details = _validate_mosaic_pixel_coverage(gdal, ogr, raster, raster_srs, roi_geometry, output)
    raster = None
    vector = None
    return {
        "roi_input": roi_path,
        "roi_bounds_in_mosaic_crs": list(roi_bounds),
        "mosaic_bounds": list(raster_bounds),
        "mosaic_envelope_tolerance_pixels": 2,
        "shard_footprint_coverage_contract": ("union-covers-roi-with-half-pixel-edge-and-four-pixel-area-tolerance"),
        "shard_footprint_edge_tolerance_pixels": 0.51,
        "shard_footprint_count": len(sources),
        "roi_area_in_mosaic_crs_units": roi_area,
        "roi_uncovered_by_shard_footprints": uncovered_area,
        "roi_uncovered_fraction": uncovered_fraction,
        **pixel_details,
    }


def _validate_mosaic_pixel_coverage(
    gdal: Any,
    ogr: Any,
    raster: Any,
    raster_srs: Any,
    roi_geometry: Any,
    output: Path,
) -> dict[str, Any]:
    """Require every rasterized ROI pixel centre to be valid in every band.

    Footprint unions catch absent rectangular shards, but they cannot detect an
    internal source mask or an output driver that silently converted nodata to
    zero.  This blockwise check rasterizes the checksum-bound ROI on the mosaic
    lattice and verifies the output nodata/mask, finite radiometry, and range at
    every included pixel without loading the full corridor into memory.
    """

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pixel-valid mosaic coverage requires NumPy.") from exc

    if raster.RasterCount <= 0:
        raise ContractError(f"Mosaic has no raster bands: {output}.")
    band_nodata: list[float] = []
    band_mask_flags: list[int] = []
    for index in range(1, raster.RasterCount + 1):
        band = raster.GetRasterBand(index)
        nodata = band.GetNoDataValue()
        if nodata is None or not math.isclose(float(nodata), MOSAIC_NODATA, rel_tol=0.0, abs_tol=1e-9):
            raise ContractError(
                f"Mosaic band {index} does not declare the required nodata value {MOSAIC_NODATA}: {output}."
            )
        flags = int(band.GetMaskFlags())
        if flags == int(gdal.GMF_ALL_VALID):
            raise ContractError(
                f"Mosaic band {index} reports every pixel valid despite its explicit nodata contract: {output}."
            )
        band_nodata.append(float(nodata))
        band_mask_flags.append(flags)

    memory_vector_driver = ogr.GetDriverByName("MEM") or ogr.GetDriverByName("Memory")
    vector_memory = memory_vector_driver.CreateDataSource("payette_roi_pixel_mask")
    roi_layer = vector_memory.CreateLayer("roi", srs=raster_srs, geom_type=ogr.wkbUnknown)
    roi_feature = ogr.Feature(roi_layer.GetLayerDefn())
    roi_feature.SetGeometry(roi_geometry)
    if roi_layer.CreateFeature(roi_feature) != 0:
        raise RuntimeError("OGR could not create the temporary projected ROI feature.")
    roi_feature = None

    # Restrict iteration to the ROI envelope, then rasterize only one in-memory
    # block at a time.  This avoids a multi-gigabyte temporary mask for the
    # approximately metre-resolution Earth Engine export.
    geotransform = tuple(float(value) for value in raster.GetGeoTransform())
    x0, px, rx, y0, ry, py = geotransform
    if not math.isclose(rx, 0.0, abs_tol=1e-15) or not math.isclose(ry, 0.0, abs_tol=1e-15):
        raise ContractError("Mosaic pixel-valid coverage requires a north-up grid.")
    envelope = roi_geometry.GetEnvelope()
    x_start = max(0, math.floor((envelope[0] - x0) / px) - 1)
    x_stop = min(raster.RasterXSize, math.ceil((envelope[1] - x0) / px) + 1)
    y_start = max(0, math.floor((envelope[3] - y0) / py) - 1)
    y_stop = min(raster.RasterYSize, math.ceil((envelope[2] - y0) / py) + 1)
    block_x = 1024
    block_y = 1024
    roi_pixels = 0
    valid_pixels = 0
    memory_driver = gdal.GetDriverByName("MEM")
    for y_offset in range(y_start, y_stop, block_y):
        y_size = min(block_y, y_stop - y_offset)
        for x_offset in range(x_start, x_stop, block_x):
            x_size = min(block_x, x_stop - x_offset)
            mask_dataset = memory_driver.Create("", x_size, y_size, 1, gdal.GDT_Byte)
            if mask_dataset is None:
                raise RuntimeError("GDAL could not allocate an in-memory ROI mask block.")
            block_origin_x = x0 + x_offset * px + y_offset * rx
            block_origin_y = y0 + x_offset * ry + y_offset * py
            mask_dataset.SetGeoTransform((block_origin_x, px, rx, block_origin_y, ry, py))
            mask_dataset.SetProjection(raster.GetProjection())
            mask_band = mask_dataset.GetRasterBand(1)
            mask_band.Fill(0)
            roi_layer.ResetReading()
            if gdal.RasterizeLayer(mask_dataset, [1], roi_layer, burn_values=[1]) != 0:
                raise RuntimeError("GDAL could not rasterize an ROI mask block on the mosaic grid.")
            roi_values = mask_band.ReadAsArray()
            mask_dataset = None
            if roi_values is None:
                raise RuntimeError("GDAL could not read an in-memory ROI mask block.")
            roi_selected = np.asarray(roi_values) != 0
            selected_count = int(np.count_nonzero(roi_selected))
            if selected_count == 0:
                continue
            roi_pixels += selected_count
            valid = roi_selected.copy()
            for index, nodata in enumerate(band_nodata, start=1):
                band = raster.GetRasterBand(index)
                values = band.ReadAsArray(x_offset, y_offset, x_size, y_size)
                masks = band.GetMaskBand().ReadAsArray(x_offset, y_offset, x_size, y_size)
                if values is None or masks is None:
                    raise RuntimeError(f"GDAL could not read mosaic band/mask {index}.")
                array = np.asarray(values)
                valid &= np.asarray(masks) != 0
                valid &= np.isfinite(array)
                valid &= array != nodata
                valid &= (array >= -1e-6) & (array <= 1.000001)
            valid_pixels += int(np.count_nonzero(valid))

    vector_memory = None

    if roi_pixels <= 0:
        raise ContractError("The recorded ROI contains no pixel centres on the mosaic grid.")
    invalid_pixels = roi_pixels - valid_pixels
    if invalid_pixels:
        raise ContractError(
            "Mosaic valid-data masks leave missing, nodata, nonfinite, or out-of-range imagery inside the "
            f"recorded ROI ({invalid_pixels}/{roi_pixels} ROI pixel centres)."
        )
    return {
        "mosaic_nodata_value": MOSAIC_NODATA,
        "mosaic_band_mask_flags": band_mask_flags,
        "roi_pixel_centres_total": roi_pixels,
        "roi_pixel_centres_valid_all_bands": valid_pixels,
        "roi_pixel_centres_invalid": invalid_pixels,
        "pixel_valid_coverage_contract": "all-roi-pixel-centres-valid-all-bands-zero-tolerance",
    }


def _write_mosaic(
    gdal: Any,
    sources: Sequence[Path],
    output: Path,
    *,
    band_contract: str,
    export_manifest_sha256: str,
    overview_levels: Sequence[int],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="payette-naip-mosaic-", dir=output.parent) as temporary:
        temporary_root = Path(temporary)
        vrt_path = temporary_root / "source_family.vrt"
        partial_path = temporary_root / "mosaic.partial.tif"
        vrt = gdal.BuildVRT(
            str(vrt_path),
            [str(path) for path in sources],
            options=gdal.BuildVRTOptions(srcNodata=MOSAIC_NODATA, VRTNodata=MOSAIC_NODATA),
        )
        if vrt is None:
            raise RuntimeError("GDAL failed to build a VRT for the Earth Engine TIFF family.")
        vrt.FlushCache()
        vrt = None
        dataset = gdal.Translate(
            str(partial_path),
            str(vrt_path),
            options=gdal.TranslateOptions(
                format="GTiff",
                creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=2", "BIGTIFF=IF_SAFER"],
            ),
        )
        if dataset is None:
            raise RuntimeError("GDAL failed to create the NAIP mosaic.")
        dataset.SetMetadataItem("PAYETTE_BAND_CONTRACT", band_contract)
        dataset.SetMetadataItem("PAYETTE_SOURCE_EXPORT_MANIFEST_SHA256", export_manifest_sha256)
        interpretations = [gdal.GCI_RedBand, gdal.GCI_GreenBand, gdal.GCI_BlueBand, gdal.GCI_Undefined]
        for index, name in enumerate(BAND_NAMES[band_contract], start=1):
            band = dataset.GetRasterBand(index)
            band.SetNoDataValue(MOSAIC_NODATA)
            band.SetDescription(name)
            band.SetColorInterpretation(interpretations[index - 1])
            band.SetMetadataItem("PAYETTE_SEMANTIC_BAND", name)
        if overview_levels:
            dataset.BuildOverviews("AVERAGE", list(overview_levels))
        dataset.FlushCache()
        dataset = None
        if not partial_path.is_file():
            raise RuntimeError("GDAL reported success but did not create the NAIP mosaic.")
        os.replace(partial_path, output)


def run(args: argparse.Namespace) -> int:
    export_manifest = args.export_manifest.expanduser().resolve(strict=False)
    export_dir = args.export_dir.expanduser().resolve(strict=False)
    output = args.output.expanduser().resolve(strict=False)
    manifest_path = (
        args.manifest.expanduser().resolve(strict=False)
        if args.manifest is not None
        else output.with_name(f"{output.stem}.run.json")
    )
    if output == manifest_path:
        raise ValueError("--output and --manifest must be different paths.")
    if not export_dir.is_dir():
        raise FileNotFoundError(f"Earth Engine export directory does not exist: {export_dir}")
    refuse_existing_output(output)
    refuse_existing_output(manifest_path)
    overview_levels = _parse_overviews(args.overviews)

    payload = load_run_manifest(export_manifest, expected_stage="01_export_naip_earth_engine")
    _require_completed_export(payload)
    band_contract = _band_contract(payload)
    default_manifest = DEFAULT_OUTPUT.with_name(f"{DEFAULT_OUTPUT.stem}.run.json").resolve(strict=False)
    if band_contract == "rgb" and (output == DEFAULT_OUTPUT.resolve(strict=False) or manifest_path == default_manifest):
        raise ContractError(
            "Historical RGB mosaics require explicit isolated --output and --manifest paths; "
            "the RGB+NIR corrected defaults are reserved."
        )
    source_scale_m = _export_scale_m(payload)
    prefix = _export_prefix(payload)
    roi_path = _roi_input(payload)
    sources = _discover_exports(export_dir, prefix)
    if output in sources:
        raise ValueError("--output cannot be one of the Earth Engine source TIFFs.")
    verified_export_artifacts = _verify_recorded_exports(payload, sources)

    try:
        from osgeo import gdal
    except ModuleNotFoundError as exc:
        raise RuntimeError("NAIP mosaicking requires the GDAL Python bindings.") from exc
    gdal.UseExceptions()
    source_details = _inspect_sources(gdal, sources, band_contract)
    initial_sources = tuple(sources)
    initial_snapshot = _source_snapshot(initial_sources)
    export_manifest_sha256 = sha256_file(export_manifest)
    _write_mosaic(
        gdal,
        sources,
        output,
        band_contract=band_contract,
        export_manifest_sha256=export_manifest_sha256,
        overview_levels=overview_levels,
    )
    try:
        final_sources = tuple(_discover_exports(export_dir, prefix))
        if final_sources != initial_sources or _source_snapshot(final_sources) != initial_snapshot:
            raise ContractError(
                "Earth Engine shard family changed while the mosaic was being built; wait for Drive sync to "
                "stabilize and rerun."
            )
        coverage_details = _validate_mosaic_roi_coverage(gdal, output, roi_path, sources)
    except BaseException:
        # This file was created by the current invocation and has not been
        # committed to a manifest. Do not strand a scientifically invalid
        # output that would make the safe retry guard fail.
        output.unlink(missing_ok=True)
        raise

    try:
        write_run_manifest(
            manifest_path,
            workflow_root=WORKFLOW_ROOT,
            stage="01_mosaic_naip_exports",
            contract=f"naip-unit-scaled-{band_contract}-mosaic",
            arguments=vars(args),
            inputs=[export_manifest, *sources],
            outputs=[output],
            extra={
                "band_contract": band_contract,
                "raster_output_band_names": BAND_NAMES[band_contract],
                "source_file_prefix": prefix,
                "source_file_count": len(sources),
                "output_radiometry": "float32 values scaled from byte DN by 1/255",
                "source_scale_m": source_scale_m,
                "export_manifest_sha256": export_manifest_sha256,
                "export_artifacts_preverified_from_manifest": verified_export_artifacts,
                "export_binding_mode": (
                    "exact-export-manifest-shard-inventory"
                    if verified_export_artifacts
                    else "mosaic-time-inventory-after-completed-task"
                ),
                "mosaic_source_order": sources,
                "shard_family_stable_during_mosaic": True,
                "shard_family_content_hashes_stable_through_manifest": True,
                **source_details,
                **coverage_details,
            },
        )
        if _source_snapshot(final_sources) != initial_snapshot:
            raise ContractError(
                "Earth Engine shard content changed while the mosaic manifest was being committed; "
                "the output and manifest were removed."
            )
    except BaseException:
        manifest_path.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        raise
    print(f"Saved {band_contract.upper()} NAIP mosaic from {len(sources)} TIFF(s): {output}")
    print(f"Wrote run manifest: {manifest_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
