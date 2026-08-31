#!/usr/bin/env python3
"""Acquire the recovered 3DEP grid and derive the less-than-5% slope mask.

Notebook cells 18, 20, and 21 used 1 m USGS 3DEP data, a 0.02-degree
rectangular request grid, GDAL percent slope, and a strict slope threshold of
less than five percent. The manuscript describes less-than-or-equal-to five
percent. ``--contract historical`` therefore defaults to ``<`` while
``--contract corrected`` defaults to ``<=``; ``--slope-comparison`` remains an
explicit override. This implementation also requires a complete signature-keyed
chunk grid, streams polygonization to disk, and calculates every ``area_m2``
value in a projected CRS.

Use ``--dem-input`` to derive slope from an already reviewed DEM without any
network calls.  Heavy geospatial and service packages are imported only after
argument parsing and collision checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, Sequence


WORKING_ROOT = Path(__file__).resolve().parents[1]
if str(WORKING_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKING_ROOT))

from project_paths import (  # noqa: E402
    PARKING_CORRIDOR,
    PARKING_CORRIDOR_MANIFEST,
    PARKING_DEM,
    PARKING_SLOPE_LE5,
    PARKING_SLOPE_LE5_CORRIDOR,
    PARKING_SLOPE_MANIFEST,
    PARKING_TERRAIN_RUN_ROOT,
    WORKFLOW_ROOT,
)
from pipeline.parking_common import (  # noqa: E402
    ContractError,
    load_run_manifest,
    refuse_existing_output,
    sha256_file,
    validate_corrected_corridor_grid,
    verify_manifest_artifact,
    write_run_manifest,
)


RECOVERED_BOUNDS = (-116.28766600566757, 43.88355482041268, -115.491507, 44.514881)
DEM_CACHE_SIGNATURE_FILENAME = "request_signature.json"
DEFAULT_TERRAIN_ROOT = PARKING_TERRAIN_RUN_ROOT
DEFAULT_CORRIDOR_INPUT = PARKING_CORRIDOR
DEFAULT_DEM_OUTPUT = PARKING_DEM
DEFAULT_SLOPE_OUTPUT = PARKING_SLOPE_LE5
DEFAULT_CLIPPED_SLOPE_OUTPUT = PARKING_SLOPE_LE5_CORRIDOR
DEM_NODATA = -9999.0
# The signed Payette request box is inland mountain terrain. Requiring values
# above 100 m is deliberately conservative for the region and, critically,
# prevents an undeclared zero-fill hole from becoming a false 0% slope area.
DEM_MIN_VALID_ELEVATION_M = 100.0
DEM_MAX_VALID_ELEVATION_M = 10_000.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the recovered 1 m 3DEP grid (or use --dem-input), derive percent slope, "
            "apply the historical/corrected threshold contract, and clip it to the road/river corridor."
        )
    )
    parser.add_argument(
        "--contract",
        choices=("historical", "corrected"),
        required=True,
        help="Scientific contract: historical defaults to slope < threshold; corrected defaults to <= threshold.",
    )
    parser.add_argument(
        "--dem-input",
        type=Path,
        default=None,
        help=(
            "Existing projected DEM for explicitly isolated historical/review work only. Corrected runs must use "
            "the maintained signed py3dep acquisition so the USGS 3DEP source is provenance-bound."
        ),
    )
    parser.add_argument("--dem-output", type=Path, default=DEFAULT_DEM_OUTPUT, help="New mosaicked 3DEP GeoTIFF.")
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=DEFAULT_TERRAIN_ROOT / "dem_chunks",
        help="Resumable 3DEP chunk cache used only when --dem-input is absent.",
    )
    parser.add_argument(
        "--reuse-chunks",
        action="store_true",
        help="Validate and reuse matching grid_chunk_*.tif cache files; without this flag, any existing chunk stops the run.",
    )
    parser.add_argument("--min-x", type=float, default=RECOVERED_BOUNDS[0])
    parser.add_argument("--min-y", type=float, default=RECOVERED_BOUNDS[1])
    parser.add_argument("--max-x", type=float, default=RECOVERED_BOUNDS[2])
    parser.add_argument("--max-y", type=float, default=RECOVERED_BOUNDS[3])
    parser.add_argument("--grid-size-deg", type=float, default=0.02, help="Recovered request-grid width/height.")
    parser.add_argument(
        "--edge-padding-deg",
        type=float,
        default=0.001,
        help="Recovered small maximum-edge padding used by the Colab grid.",
    )
    parser.add_argument("--resolution-m", type=float, default=1.0, help="py3dep DEM resolution in metres.")
    parser.add_argument(
        "--metric-crs",
        default="EPSG:26911",
        help="Projected DEM/service CRS (recovered default: NAD83 / UTM zone 11N).",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum py3dep attempts per missing chunk.")
    parser.add_argument("--retry-delay-seconds", type=float, default=1.0)
    parser.add_argument("--corridor", type=Path, default=DEFAULT_CORRIDOR_INPUT)
    parser.add_argument(
        "--corridor-manifest",
        type=Path,
        default=None,
        help=(
            "Stage-01 corridor manifest. Corrected mode defaults to the configured run manifest and "
            "checksum-binds the exact corridor used for terrain clipping; historical mode leaves it optional."
        ),
    )
    parser.add_argument("--slope-threshold-pct", type=float, default=5.0)
    parser.add_argument(
        "--slope-comparison",
        choices=("lt", "le"),
        default=None,
        help=(
            "Explicit threshold override. If omitted, historical uses 'lt' (notebook) and corrected uses 'le' "
            "(manuscript text)."
        ),
    )
    parser.add_argument(
        "--minimum-polygon-area-m2",
        type=float,
        default=0.0,
        help="Optional polygon noise filter; zero reproduces the notebook implementation.",
    )
    parser.add_argument("--slope-output", type=Path, default=DEFAULT_SLOPE_OUTPUT)
    parser.add_argument("--clipped-output", type=Path, default=DEFAULT_CLIPPED_SLOPE_OUTPUT)
    parser.add_argument(
        "--clipped-storage-crs",
        default="EPSG:4326",
        help="Saved CRS for the clipped mask; area_m2 is calculated before this reprojection.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Run-manifest path; defaults beside the clipped slope output.",
    )
    return parser


def iter_grid_bounds(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    grid_size: float,
    edge_padding: float,
) -> Iterator[tuple[int, tuple[float, float, float, float]]]:
    """Yield the recovered rectangular grid in x-major/y-minor order.

    The notebook's ``np.arange`` plus ``max + 0.001`` construction resolves to
    40 by 32 cells (1,280 total) for the checked-in defaults.  The explicit
    integer indexing here avoids NumPy-version-dependent floating endpoints.
    """

    if not min_x < max_x or not min_y < max_y:
        raise ValueError("Grid minima must be smaller than maxima.")
    if grid_size <= 0:
        raise ValueError("Grid size must be positive.")
    if edge_padding < 0 or edge_padding >= grid_size:
        raise ValueError("Edge padding must be non-negative and smaller than the grid size.")

    padded_max_x = max_x + edge_padding
    padded_max_y = max_y + edge_padding
    x_count = math.ceil((padded_max_x - min_x) / grid_size)
    y_count = math.ceil((padded_max_y - min_y) / grid_size)
    index = 0
    for x_index in range(x_count):
        x1 = min_x + x_index * grid_size
        x2 = min(x1 + grid_size, padded_max_x)
        for y_index in range(y_count):
            y1 = min_y + y_index * grid_size
            y2 = min(y1 + grid_size, padded_max_y)
            if x2 > x1 and y2 > y1:
                yield index, (x1, y1, x2, y2)
                index += 1


def resolve_slope_comparison(contract: str, override: str | None) -> str:
    if contract not in {"historical", "corrected"}:
        raise ValueError(f"Unknown slope contract: {contract!r}.")
    if override is not None:
        if override not in {"lt", "le"}:
            raise ValueError(f"Unknown slope comparison: {override!r}.")
        return override
    return "lt" if contract == "historical" else "le"


def _dem_request_payload(
    args: argparse.Namespace,
    grid: Sequence[tuple[int, tuple[float, float, float, float]]],
) -> dict[str, Any]:
    return {
        "service": "USGS-3DEP-DEM",
        "grid_algorithm": "x-major-y-minor-v1",
        "bounds_wgs84": [args.min_x, args.min_y, args.max_x, args.max_y],
        "grid_size_degrees": args.grid_size_deg,
        "edge_padding_degrees": args.edge_padding_deg,
        "expected_grid_cells": len(grid),
        "resolution_metres": args.resolution_m,
        "output_crs": args.metric_crs,
    }


def _payload_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _ensure_dem_cache_signature(
    chunk_dir: Path,
    args: argparse.Namespace,
    grid: Sequence[tuple[int, tuple[float, float, float, float]]],
) -> tuple[Path, str]:
    """Create or enforce the immutable request signature for a DEM chunk cache."""

    request = _dem_request_payload(args, grid)
    digest = _payload_sha256(request)
    expected = {"schema_version": 1, "request": request, "request_sha256": digest}
    signature_path = chunk_dir / DEM_CACHE_SIGNATURE_FILENAME
    existing_chunks = sorted(chunk_dir.glob("grid_chunk_*.tif")) if chunk_dir.exists() else []
    if signature_path.exists():
        try:
            observed = json.loads(signature_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"Cannot read DEM cache request signature {signature_path}: {exc}") from exc
        if observed != expected:
            raise ContractError(
                f"DEM chunk cache request signature does not match the requested bounds/grid/resolution/CRS: "
                f"{signature_path}. Choose a new --chunk-dir; never mix requests in one cache."
            )
        return signature_path, digest
    if existing_chunks:
        raise ContractError(
            f"DEM chunk cache contains {len(existing_chunks)} TIFF(s) but no persisted request signature: "
            f"{chunk_dir}. Choose a new --chunk-dir; unkeyed chunks cannot be reused safely."
        )

    chunk_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="request_signature.",
        suffix=".partial.json",
        dir=chunk_dir,
        delete=False,
    ) as handle:
        partial_path = Path(handle.name)
        json.dump(expected, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(partial_path, signature_path)
    return signature_path, digest


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


def _metric_crs_for_service(value: str) -> str | int:
    upper = value.upper()
    if upper.startswith("EPSG:") and upper[5:].isdigit():
        return int(upper[5:])
    return value


def _densified_request_geometry(
    bounds_wgs84: tuple[float, float, float, float],
    destination_crs: Any,
) -> dict[str, Any]:
    """Return a densified projected polygon for one signed WGS84 request cell."""

    from rasterio.warp import transform

    min_x, min_y, max_x, max_y = bounds_wgs84
    steps = 21
    fractions = [index / steps for index in range(steps + 1)]
    coordinates: list[tuple[float, float]] = []
    coordinates.extend((min_x + (max_x - min_x) * fraction, min_y) for fraction in fractions)
    coordinates.extend((max_x, min_y + (max_y - min_y) * fraction) for fraction in fractions[1:])
    coordinates.extend((max_x - (max_x - min_x) * fraction, max_y) for fraction in fractions[1:])
    coordinates.extend((min_x, max_y - (max_y - min_y) * fraction) for fraction in fractions[1:])
    if coordinates[-1] != coordinates[0]:
        coordinates.append(coordinates[0])
    xs, ys = transform(
        "EPSG:4326",
        destination_crs,
        [coordinate[0] for coordinate in coordinates],
        [coordinate[1] for coordinate in coordinates],
    )
    return {"type": "Polygon", "coordinates": [[list(pair) for pair in zip(xs, ys, strict=True)]]}


def _dem_request_pixel_coverage(
    source: Any,
    bounds_wgs84: tuple[float, float, float, float],
    *,
    rasterio: Any,
    expected_resolution_m: float,
) -> tuple[int, int]:
    """Count request-polygon pixel centres and require valid plausible DEM data."""

    import numpy as np
    from rasterio.features import geometry_mask
    from rasterio.windows import Window, from_bounds

    nodata = source.nodata
    if nodata is None:
        raise ContractError("DEM raster has no declared nodata value.")
    nodata_value = float(nodata)
    if math.isfinite(nodata_value) and DEM_MIN_VALID_ELEVATION_M <= nodata_value <= DEM_MAX_VALID_ELEVATION_M:
        raise ContractError(
            f"DEM nodata {nodata_value} overlaps the accepted elevation range "
            f"[{DEM_MIN_VALID_ELEVATION_M}, {DEM_MAX_VALID_ELEVATION_M}] m."
        )

    geometry = _densified_request_geometry(bounds_wgs84, source.crs)
    ring = geometry["coordinates"][0]
    xs = [float(point[0]) for point in ring]
    ys = [float(point[1]) for point in ring]
    polygon_bounds = (min(xs), min(ys), max(xs), max(ys))
    tolerance = 0.51 * expected_resolution_m
    if (
        source.bounds.left > polygon_bounds[0] + tolerance
        or source.bounds.bottom > polygon_bounds[1] + tolerance
        or source.bounds.right < polygon_bounds[2] - tolerance
        or source.bounds.top < polygon_bounds[3] - tolerance
    ):
        raise ContractError("DEM raster does not cover the complete projected request polygon.")

    raw_window = from_bounds(*polygon_bounds, transform=source.transform)
    col_start = max(0, math.floor(raw_window.col_off) - 1)
    row_start = max(0, math.floor(raw_window.row_off) - 1)
    col_stop = min(source.width, math.ceil(raw_window.col_off + raw_window.width) + 1)
    row_stop = min(source.height, math.ceil(raw_window.row_off + raw_window.height) + 1)
    if col_stop <= col_start or row_stop <= row_start:
        raise ContractError("DEM request polygon has no intersecting raster cells.")
    window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
    values = source.read(1, window=window, masked=True)
    inside = geometry_mask(
        [geometry],
        out_shape=values.shape,
        transform=source.window_transform(window),
        invert=True,
        all_touched=False,
    )
    expected_pixels = int(np.count_nonzero(inside))
    if expected_pixels <= 0:
        raise ContractError("DEM request polygon contains no pixel centres on the declared grid.")
    data = np.asarray(values.data)
    invalid = np.ma.getmaskarray(values) | ~np.isfinite(data)
    invalid |= data < DEM_MIN_VALID_ELEVATION_M
    invalid |= data > DEM_MAX_VALID_ELEVATION_M
    invalid_pixels = int(np.count_nonzero(invalid & inside))
    if invalid_pixels:
        raise ContractError(
            "DEM request polygon contains masked, nodata, nonfinite, or implausible elevation pixels "
            f"({invalid_pixels}/{expected_pixels})."
        )
    return expected_pixels, expected_pixels - invalid_pixels


def _chunk_is_valid(
    path: Path,
    *,
    rasterio: Any,
    expected_crs: Any,
    expected_resolution_m: float,
    expected_bounds_wgs84: tuple[float, float, float, float],
) -> bool:
    try:
        from rasterio.warp import transform_bounds

        if path.stat().st_size <= 1_000:
            return False
        with rasterio.open(path) as source:
            if not (
                source.width > 0
                and source.height > 0
                and source.count == 1
                and source.crs is not None
                and source.crs == expected_crs
                and math.isclose(abs(source.transform.a), expected_resolution_m, rel_tol=1e-6, abs_tol=1e-6)
                and math.isclose(abs(source.transform.e), expected_resolution_m, rel_tol=1e-6, abs_tol=1e-6)
                and math.isclose(source.transform.b, 0.0, abs_tol=1e-12)
                and math.isclose(source.transform.d, 0.0, abs_tol=1e-12)
            ):
                return False
            expected_projected_bounds = transform_bounds(
                "EPSG:4326",
                source.crs,
                *expected_bounds_wgs84,
                densify_pts=21,
            )
            tolerance = 0.51 * expected_resolution_m
            if not (
                source.bounds.left <= expected_projected_bounds[0] + tolerance
                and source.bounds.bottom <= expected_projected_bounds[1] + tolerance
                and source.bounds.right >= expected_projected_bounds[2] - tolerance
                and source.bounds.top >= expected_projected_bounds[3] - tolerance
            ):
                return False
            _dem_request_pixel_coverage(
                source,
                expected_bounds_wgs84,
                rasterio=rasterio,
                expected_resolution_m=expected_resolution_m,
            )
            return True
    except (ContractError, OSError, ValueError, rasterio.errors.RasterioIOError):
        return False


def _normalize_downloaded_dem_chunk(
    raw_path: Path,
    normalized_path: Path,
    *,
    bounds_wgs84: tuple[float, float, float, float],
    expected_crs: Any,
    resolution_m: float,
    gdal: Any,
) -> list[float]:
    """Warp a provider response onto the signed common target lattice."""

    from rasterio.warp import transform_bounds

    gdal.UseExceptions()
    projected_bounds = transform_bounds(
        "EPSG:4326",
        expected_crs,
        *bounds_wgs84,
        densify_pts=21,
    )
    normalized = gdal.Warp(
        str(normalized_path),
        str(raw_path),
        options=gdal.WarpOptions(
            format="GTiff",
            dstSRS=str(expected_crs),
            outputBounds=projected_bounds,
            xRes=resolution_m,
            yRes=resolution_m,
            targetAlignedPixels=True,
            resampleAlg="bilinear",
            outputType=gdal.GDT_Float32,
            dstNodata=DEM_NODATA,
            creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_NEEDED"],
        ),
    )
    if normalized is None:
        raise RuntimeError("GDAL could not normalize a downloaded 3DEP chunk to the signed target lattice.")
    normalized.FlushCache()
    normalized = None
    if not normalized_path.is_file():
        raise RuntimeError("GDAL reported success without creating the normalized DEM chunk.")
    return [float(value) for value in projected_bounds]


def _download_and_mosaic_dem(args: argparse.Namespace, dem_output: Path, chunk_dir: Path) -> dict[str, Any]:
    try:
        import py3dep
        import rasterio
        from osgeo import gdal
        from pyproj import CRS
        from rasterio.warp import transform_bounds
        from shapely.geometry import box
    except ModuleNotFoundError as exc:
        raise RuntimeError("3DEP acquisition requires py3dep, rasterio, pyproj, shapely, rioxarray, and GDAL.") from exc

    metric_crs = CRS.from_user_input(args.metric_crs)
    if not metric_crs.is_projected:
        raise ValueError(f"--metric-crs must be projected: {args.metric_crs}")
    metric_units = {str(axis.unit_name).lower() for axis in metric_crs.axis_info if axis.unit_name}
    if metric_units and not all("met" in unit for unit in metric_units):
        raise ValueError(f"--metric-crs axes must use metres, not {sorted(metric_units)}: {args.metric_crs}")
    expected_raster_crs = rasterio.crs.CRS.from_user_input(metric_crs)
    grid = list(
        iter_grid_bounds(
            args.min_x,
            args.min_y,
            args.max_x,
            args.max_y,
            args.grid_size_deg,
            args.edge_padding_deg,
        )
    )
    if not grid:
        raise ValueError("The configured 3DEP request grid is empty.")

    signature_path, signature_sha256 = _ensure_dem_cache_signature(chunk_dir, args, grid)

    existing_chunks = sorted(chunk_dir.glob("grid_chunk_*.tif")) if chunk_dir.exists() else []
    if existing_chunks and not args.reuse_chunks:
        raise FileExistsError(
            f"Chunk cache already contains {len(existing_chunks)} file(s): {chunk_dir}. "
            "Pass --reuse-chunks to validate/reuse it, or choose a new --chunk-dir."
        )
    chunk_dir.mkdir(parents=True, exist_ok=True)

    expected_names = {f"grid_chunk_{index}.tif" for index, _ in grid}
    unexpected_chunks = [path for path in existing_chunks if path.name not in expected_names]
    if unexpected_chunks:
        raise ContractError(
            f"DEM cache has {len(unexpected_chunks)} unexpected grid chunk(s) despite a matching signature: "
            f"{chunk_dir}. Choose a clean --chunk-dir."
        )

    expected_paths: list[Path] = []
    reused = 0
    downloaded = 0
    failures: list[dict[str, Any]] = []
    service_crs = _metric_crs_for_service(args.metric_crs)
    for index, bounds in grid:
        chunk_path = chunk_dir / f"grid_chunk_{index}.tif"
        expected_paths.append(chunk_path)
        if (
            args.reuse_chunks
            and chunk_path.exists()
            and _chunk_is_valid(
                chunk_path,
                rasterio=rasterio,
                expected_crs=expected_raster_crs,
                expected_resolution_m=args.resolution_m,
                expected_bounds_wgs84=bounds,
            )
        ):
            reused += 1
            continue

        partial_path = chunk_path.with_suffix(".partial.tif")
        raw_partial_path = chunk_path.with_suffix(".download.partial.tif")
        success = False
        last_error: Exception | None = None
        for attempt in range(1, args.max_retries + 1):
            try:
                chunk = py3dep.get_map("DEM", box(*bounds), resolution=args.resolution_m, crs=service_crs)
                if chunk is None or not hasattr(chunk, "rio"):
                    raise RuntimeError("py3dep returned no rioxarray object.")
                chunk.rio.to_raster(raw_partial_path, compress="deflate", tiled=True)
                with rasterio.open(raw_partial_path, "r+") as downloaded_raster:
                    if downloaded_raster.nodata is None:
                        downloaded_raster.nodata = DEM_NODATA
                _normalize_downloaded_dem_chunk(
                    raw_partial_path,
                    partial_path,
                    bounds_wgs84=bounds,
                    expected_crs=expected_raster_crs,
                    resolution_m=args.resolution_m,
                    gdal=gdal,
                )
                if not _chunk_is_valid(
                    partial_path,
                    rasterio=rasterio,
                    expected_crs=expected_raster_crs,
                    expected_resolution_m=args.resolution_m,
                    expected_bounds_wgs84=bounds,
                ):
                    raise RuntimeError("Downloaded chunk failed raster/CRS validation.")
                os.replace(partial_path, chunk_path)
                raw_partial_path.unlink(missing_ok=True)
                downloaded += 1
                success = True
                break
            except Exception as exc:  # service errors vary across requests versions
                last_error = exc
                partial_path.unlink(missing_ok=True)
                raw_partial_path.unlink(missing_ok=True)
                if attempt < args.max_retries:
                    time.sleep(args.retry_delay_seconds)
        if not success:
            failures.append({"index": index, "bounds": list(bounds), "error": repr(last_error)})

    if failures:
        preview = "; ".join(f"{item['index']}: {item['error']}" for item in failures[:5])
        raise RuntimeError(
            f"{len(failures)} of {len(grid)} required 3DEP chunks failed. Refusing an incomplete mosaic. {preview}"
        )
    invalid = []
    for (_, bounds), path in zip(grid, expected_paths, strict=True):
        if not path.exists() or not _chunk_is_valid(
            path,
            rasterio=rasterio,
            expected_crs=expected_raster_crs,
            expected_resolution_m=args.resolution_m,
            expected_bounds_wgs84=bounds,
        ):
            invalid.append(path)
    if invalid:
        raise RuntimeError(f"Chunk-grid validation failed for {len(invalid)} file(s); refusing an incomplete mosaic.")

    gdal.UseExceptions()
    dem_output.parent.mkdir(parents=True, exist_ok=True)
    requested_bounds_wgs84 = (
        args.min_x,
        args.min_y,
        args.max_x + args.edge_padding_deg,
        args.max_y + args.edge_padding_deg,
    )
    projected_output_bounds = transform_bounds(
        "EPSG:4326",
        expected_raster_crs,
        *requested_bounds_wgs84,
        densify_pts=21,
    )
    dataset = gdal.Warp(
        str(dem_output),
        [str(path) for path in expected_paths],
        options=gdal.WarpOptions(
            format="GTiff",
            dstSRS=metric_crs.to_wkt(),
            outputBounds=projected_output_bounds,
            xRes=args.resolution_m,
            yRes=args.resolution_m,
            targetAlignedPixels=True,
            resampleAlg="near",
            outputType=gdal.GDT_Float32,
            dstNodata=DEM_NODATA,
            creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_NEEDED"],
            multithread=True,
            warpOptions=["NUM_THREADS=ALL_CPUS"],
        ),
    )
    if dataset is None:
        raise RuntimeError("GDAL failed to mosaic the complete 3DEP chunk grid.")
    dataset.FlushCache()
    dataset = None
    if not dem_output.is_file():
        raise RuntimeError(f"DEM mosaic was not created: {dem_output}")
    return {
        "grid_cells": len(grid),
        "chunks_reused": reused,
        "chunks_downloaded": downloaded,
        "chunk_request_signature_path": signature_path,
        "chunk_request_signature_sha256": signature_sha256,
        "dem_mosaic_crs": metric_crs.to_string(),
        "dem_mosaic_resolution_m": args.resolution_m,
        "dem_mosaic_resampling": "nearest",
        "dem_mosaic_nodata": DEM_NODATA,
        "dem_mosaic_target_aligned_pixels": True,
        "dem_mosaic_requested_bounds_wgs84": list(requested_bounds_wgs84),
        "dem_mosaic_projected_output_bounds": list(projected_output_bounds),
        "chunk_normalization_contract": "provider-response-to-target-crs-one-metre-aligned-bilinear",
        "chunk_normalization_resampling": "bilinear",
    }


def _validate_projected_dem(dem_path: Path, *, rasterio: Any) -> str:
    with rasterio.open(dem_path) as source:
        if source.crs is None:
            raise ValueError(f"DEM has no CRS: {dem_path}")
        if not source.crs.is_projected:
            raise ValueError(
                f"DEM CRS is geographic ({source.crs}); GDAL slope and area require a projected metre-based raster."
            )
        linear_units = str(getattr(source.crs, "linear_units", "")).lower()
        if linear_units and "met" not in linear_units:
            raise ValueError(f"DEM linear units are not metres ({linear_units!r}): {dem_path}")
        return source.crs.to_string()


def _validate_complete_dem_grid(
    dem_path: Path,
    args: argparse.Namespace,
    grid: Sequence[tuple[int, tuple[float, float, float, float]]],
) -> dict[str, Any]:
    """Validate the exact lattice and every request-polygon pixel centre."""

    try:
        import rasterio
        from rasterio.warp import transform_bounds
    except ModuleNotFoundError as exc:
        raise RuntimeError("Complete DEM validation requires rasterio and NumPy.") from exc

    with rasterio.open(dem_path) as source:
        dem_crs = _validate_projected_dem(dem_path, rasterio=rasterio)
        expected_crs = rasterio.crs.CRS.from_user_input(args.metric_crs)
        if source.crs != expected_crs:
            raise ContractError(
                f"DEM CRS {source.crs} does not equal the signed request CRS {expected_crs}: {dem_path}."
            )
        if not (
            math.isclose(abs(source.transform.a), args.resolution_m, rel_tol=1e-6, abs_tol=1e-6)
            and math.isclose(abs(source.transform.e), args.resolution_m, rel_tol=1e-6, abs_tol=1e-6)
            and math.isclose(source.transform.b, 0.0, abs_tol=1e-12)
            and math.isclose(source.transform.d, 0.0, abs_tol=1e-12)
        ):
            raise ContractError(f"DEM does not use the declared north-up {args.resolution_m} m grid: {dem_path}.")
        if not (
            math.isclose(
                source.transform.c / args.resolution_m,
                round(source.transform.c / args.resolution_m),
                abs_tol=1e-6,
            )
            and math.isclose(
                source.transform.f / args.resolution_m,
                round(source.transform.f / args.resolution_m),
                abs_tol=1e-6,
            )
        ):
            raise ContractError("DEM origin is not aligned to the signed metre-resolution lattice.")
        nodata = source.nodata
        if nodata is None or not math.isclose(float(nodata), DEM_NODATA, rel_tol=0.0, abs_tol=1e-9):
            raise ContractError(f"DEM must declare nodata {DEM_NODATA}; found {nodata!r}: {dem_path}.")
        requested_bounds = (
            args.min_x,
            args.min_y,
            args.max_x + args.edge_padding_deg,
            args.max_y + args.edge_padding_deg,
        )
        projected_bounds = transform_bounds(
            "EPSG:4326",
            source.crs,
            *requested_bounds,
            densify_pts=21,
        )
        tolerance = 0.51 * args.resolution_m
        if (
            source.bounds.left > projected_bounds[0] + tolerance
            or source.bounds.bottom > projected_bounds[1] + tolerance
            or source.bounds.right < projected_bounds[2] - tolerance
            or source.bounds.top < projected_bounds[3] - tolerance
        ):
            raise ContractError("DEM footprint does not cover the complete signed 3DEP request grid.")
        request_pixels = 0
        valid_request_pixels = 0
        for index, bounds in grid:
            try:
                expected_pixels, valid_pixels = _dem_request_pixel_coverage(
                    source,
                    bounds,
                    rasterio=rasterio,
                    expected_resolution_m=args.resolution_m,
                )
            except ContractError as exc:
                raise ContractError(f"DEM failed complete valid-data coverage for request cell {index}: {exc}") from exc
            request_pixels += expected_pixels
            valid_request_pixels += valid_pixels
        return {
            "dem_crs": dem_crs,
            "dem_bounds": list(source.bounds),
            "requested_bounds_wgs84": list(requested_bounds),
            "request_cells_valid": len(grid),
            "request_cells_expected": len(grid),
            "request_polygon_pixel_centres_expected": request_pixels,
            "request_polygon_pixel_centres_valid": valid_request_pixels,
            "dem_nodata_value": float(nodata),
            "dem_valid_elevation_range_m": [DEM_MIN_VALID_ELEVATION_M, DEM_MAX_VALID_ELEVATION_M],
            "dem_grid_validation": "exact-crs-lattice-and-all-request-polygon-pixel-centres-valid",
        }


def _vector_driver(path: Path) -> tuple[str, str | None]:
    if path.suffix.lower() == ".shp":
        return "ESRI Shapefile", None
    if path.suffix.lower() == ".gpkg":
        return "GPKG", path.stem
    raise ValueError(f"Unsupported vector output extension for {path}; use .shp or .gpkg.")


def _derive_slope_polygons(
    dem_path: Path,
    slope_output: Path,
    *,
    threshold_pct: float,
    comparison: str,
    minimum_area_m2: float,
) -> tuple[int, str]:
    try:
        import fiona
        import numpy as np
        import rasterio
        from osgeo import gdal
        from rasterio.features import shapes
        from shapely.geometry import shape
    except ModuleNotFoundError as exc:
        raise RuntimeError("Slope derivation requires GDAL, rasterio, Fiona, NumPy, and Shapely.") from exc

    dem_crs = _validate_projected_dem(dem_path, rasterio=rasterio)
    slope_output.parent.mkdir(parents=True, exist_ok=True)
    driver, layer = _vector_driver(slope_output)
    with tempfile.TemporaryDirectory(prefix="payette-slope-") as temporary:
        slope_raster = Path(temporary) / "slope_percent.tif"
        gdal.UseExceptions()
        dataset = gdal.DEMProcessing(
            str(slope_raster),
            str(dem_path),
            "slope",
            options=gdal.DEMProcessingOptions(
                format="GTiff",
                slopeFormat="percent",
                creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"],
            ),
        )
        if dataset is None:
            raise RuntimeError("GDAL DEMProcessing failed to create percent slope.")
        dataset.FlushCache()
        dataset = None

        schema = {"geometry": "Polygon", "properties": {"area_m2": "float:24.3"}}
        open_options: dict[str, Any] = {
            "mode": "w",
            "driver": driver,
            "schema": schema,
        }
        if layer is not None:
            open_options["layer"] = layer
        feature_count = 0
        with rasterio.open(slope_raster) as source:
            if source.crs is None or not source.crs.is_projected:
                raise ValueError("Derived slope raster does not retain the projected DEM CRS.")
            open_options["crs_wkt"] = source.crs.to_wkt()
            with fiona.open(slope_output, **open_options) as sink:
                for _, window in source.block_windows(1):
                    values = source.read(1, window=window)
                    below_threshold = values < threshold_pct if comparison == "lt" else values <= threshold_pct
                    valid = np.isfinite(values) & (values >= 0) & below_threshold
                    if source.nodata is not None:
                        valid &= values != source.nodata
                    if not bool(valid.any()):
                        continue
                    byte_mask = valid.astype("uint8")
                    for geometry, value in shapes(
                        byte_mask,
                        mask=valid,
                        transform=source.window_transform(window),
                    ):
                        if int(value) != 1:
                            continue
                        area_m2 = float(shape(geometry).area)
                        if area_m2 < minimum_area_m2:
                            continue
                        sink.write({"geometry": geometry, "properties": {"area_m2": area_m2}})
                        feature_count += 1
    if feature_count == 0:
        operator = "<" if comparison == "lt" else "<="
        raise RuntimeError(f"No polygons met the slope {operator} {threshold_pct}% contract.")
    return feature_count, dem_crs


def _clip_slope_polygons(
    slope_path: Path,
    corridor_path: Path,
    clipped_output: Path,
    *,
    storage_crs: str,
) -> int:
    try:
        import geopandas as gpd
        from pyproj import CRS
    except ModuleNotFoundError as exc:
        raise RuntimeError("Slope clipping requires GeoPandas and pyproj.") from exc

    corridor = gpd.read_file(corridor_path)
    if corridor.empty:
        raise ValueError(f"Corridor vector is empty: {corridor_path}")
    if corridor.crs is None:
        raise ValueError(f"Corridor vector has no CRS: {corridor_path}")

    slope_head = gpd.read_file(slope_path, rows=1)
    if slope_head.empty or slope_head.crs is None:
        raise ValueError(f"Slope vector has no readable feature/CRS: {slope_path}")
    slope_crs = CRS.from_user_input(slope_head.crs)
    if not slope_crs.is_projected:
        raise ValueError(f"Slope vector CRS must be projected before area calculation: {slope_head.crs}")

    corridor_projected = corridor.to_crs(slope_crs)
    candidates = gpd.read_file(slope_path, mask=corridor_projected)
    if candidates.empty:
        raise RuntimeError("No less-than-threshold slope polygons intersect the corridor.")
    if candidates.crs != corridor_projected.crs:
        candidates = candidates.to_crs(corridor_projected.crs)
    clipped = gpd.clip(candidates, corridor_projected)
    clipped = clipped.loc[~clipped.geometry.is_empty & clipped.geometry.notna()].copy()
    if clipped.empty:
        raise RuntimeError("Exact corridor clipping produced no slope polygons.")
    clipped["area_m2"] = clipped.geometry.area.astype(float)

    clipped_output.parent.mkdir(parents=True, exist_ok=True)
    driver, layer = _vector_driver(clipped_output)
    saved = clipped.to_crs(storage_crs)
    if driver == "GPKG":
        saved.to_file(clipped_output, layer=layer, driver=driver)
    else:
        saved.to_file(clipped_output, driver=driver)
    return int(len(saved))


def run(args: argparse.Namespace) -> int:
    dem_input = args.dem_input.expanduser().resolve(strict=False) if args.dem_input is not None else None
    dem_output = args.dem_output.expanduser().resolve(strict=False)
    chunk_dir = args.chunk_dir.expanduser().resolve(strict=False)
    corridor_path = args.corridor.expanduser().resolve(strict=False)
    corridor_manifest = args.corridor_manifest
    if args.contract == "corrected" and corridor_manifest is None:
        corridor_manifest = PARKING_CORRIDOR_MANIFEST
    corridor_manifest_path = (
        corridor_manifest.expanduser().resolve(strict=False) if corridor_manifest is not None else None
    )
    slope_output = args.slope_output.expanduser().resolve(strict=False)
    clipped_output = args.clipped_output.expanduser().resolve(strict=False)
    manifest_path = (
        args.manifest.expanduser().resolve(strict=False)
        if args.manifest is not None
        else clipped_output.with_name(f"{clipped_output.stem}.run.json")
    )
    slope_comparison = resolve_slope_comparison(args.contract, args.slope_comparison)

    if args.resolution_m <= 0:
        raise ValueError("--resolution-m must be positive.")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be positive.")
    if args.retry_delay_seconds < 0:
        raise ValueError("--retry-delay-seconds cannot be negative.")
    if args.slope_threshold_pct <= 0:
        raise ValueError("--slope-threshold-pct must be positive.")
    if args.minimum_polygon_area_m2 < 0:
        raise ValueError("--minimum-polygon-area-m2 cannot be negative.")
    if args.contract == "corrected" and slope_comparison != "le":
        raise ContractError("Corrected terrain construction is fixed to the manuscript's slope <= 5% contract.")
    if args.contract == "corrected" and args.minimum_polygon_area_m2 != 0:
        raise ContractError(
            "Corrected terrain construction requires --minimum-polygon-area-m2 0; nonzero blockwise component "
            "filtering is not invariant to raster tiling."
        )
    if not corridor_path.is_file():
        raise FileNotFoundError(f"Corridor vector does not exist: {corridor_path}")
    if dem_input is not None and not dem_input.is_file():
        raise FileNotFoundError(f"--dem-input does not exist: {dem_input}")
    if args.contract == "corrected" and dem_input is not None:
        raise ContractError(
            "Corrected terrain construction does not accept --dem-input because an arbitrary raster cannot prove "
            "USGS 3DEP acquisition provenance. Use the maintained signed py3dep request grid."
        )
    if args.contract == "historical":
        corrected_defaults = {
            DEFAULT_CORRIDOR_INPUT.resolve(strict=False),
            DEFAULT_SLOPE_OUTPUT.resolve(strict=False),
            DEFAULT_CLIPPED_SLOPE_OUTPUT.resolve(strict=False),
            PARKING_SLOPE_MANIFEST.resolve(strict=False),
        }
        historical_targets = {corridor_path, slope_output, clipped_output, manifest_path}
        if dem_input is None:
            corrected_defaults.update(
                {
                    DEFAULT_DEM_OUTPUT.resolve(strict=False),
                    (DEFAULT_TERRAIN_ROOT / "dem_chunks").resolve(strict=False),
                }
            )
            historical_targets.add(dem_output)
        corrected_defaults.add((DEFAULT_TERRAIN_ROOT / "dem_chunks").resolve(strict=False))
        historical_targets.add(chunk_dir)
        if historical_targets & corrected_defaults:
            raise ContractError(
                "Historical terrain reconstruction requires explicit isolated corridor/output/cache/manifest "
                "paths; corrected defaults are reserved."
            )

    corridor_provenance = "missing-historical-provenance"
    corridor_manifest_sha256: str | None = None
    if corridor_manifest_path is not None:
        corridor_payload = load_run_manifest(
            corridor_manifest_path,
            expected_stage="01_build_corridor",
            expected_contract=("projected-geometry-rgbnir" if args.contract == "corrected" else None),
        )
        verify_manifest_artifact(corridor_payload, corridor_path)
        if args.contract == "corrected":
            extra = corridor_payload.get("extra", {})
            validate_corrected_corridor_grid(
                extra,
                expected_band_names=("R", "G", "B", "N"),
            )
        corridor_provenance = "verified"
        corridor_manifest_sha256 = sha256_file(corridor_manifest_path)
    elif args.contract == "corrected":
        raise ContractError("Corrected terrain construction requires --corridor-manifest.")

    durable_outputs = [slope_output, clipped_output, manifest_path]
    if dem_input is None:
        durable_outputs.append(dem_output)
    if len(set(durable_outputs)) != len(durable_outputs):
        raise ValueError("DEM, slope, clipped-slope, and manifest outputs must all use distinct paths.")

    # Check every durable output before imports, network requests, or writes.
    if dem_input is None:
        refuse_existing_output(dem_output)
    _refuse_vector_output(slope_output)
    _refuse_vector_output(clipped_output)
    refuse_existing_output(manifest_path)

    grid = list(
        iter_grid_bounds(
            args.min_x,
            args.min_y,
            args.max_x,
            args.max_y,
            args.grid_size_deg,
            args.edge_padding_deg,
        )
    )
    if not grid:
        raise ContractError("The declared DEM request grid is empty.")

    acquisition: dict[str, Any]
    external_signature_path = chunk_dir / DEM_CACHE_SIGNATURE_FILENAME
    external_signature_preexisting = external_signature_path.exists()
    external_chunk_dir_preexisting = chunk_dir.exists()
    try:
        if dem_input is None:
            acquisition = _download_and_mosaic_dem(args, dem_output, chunk_dir)
            dem_path = dem_output
        else:
            signature_path, signature_sha256 = _ensure_dem_cache_signature(chunk_dir, args, grid)
            acquisition = {
                "mode": "existing-dem",
                "grid_cells": len(grid),
                "chunks_reused": None,
                "chunks_downloaded": None,
                "chunk_request_signature_path": signature_path,
                "chunk_request_signature_sha256": signature_sha256,
            }
            dem_path = dem_input

        dem_validation = _validate_complete_dem_grid(dem_path, args, grid)
        acquisition.update(dem_validation)

        slope_count, dem_crs = _derive_slope_polygons(
            dem_path,
            slope_output,
            threshold_pct=args.slope_threshold_pct,
            comparison=slope_comparison,
            minimum_area_m2=args.minimum_polygon_area_m2,
        )
        clipped_count = _clip_slope_polygons(
            slope_output,
            corridor_path,
            clipped_output,
            storage_crs=args.clipped_storage_crs,
        )

        inputs = [corridor_path]
        if corridor_manifest_path is not None:
            inputs.append(corridor_manifest_path)
        outputs = [slope_output, clipped_output]
        if dem_input is None:
            outputs.insert(0, dem_output)
        else:
            inputs.insert(0, dem_input)
        inputs.append(Path(acquisition["chunk_request_signature_path"]))
        write_run_manifest(
            manifest_path,
            workflow_root=WORKFLOW_ROOT,
            stage="01_build_dem_slope",
            contract=f"{args.contract}-complete-3dep-grid-projected-area-slope-{slope_comparison}",
            arguments={
                **vars(args),
                "corridor_manifest_effective": corridor_manifest_path,
                "resolved_slope_comparison": slope_comparison,
            },
            inputs=inputs,
            outputs=outputs,
            extra={
                **acquisition,
                "dem_crs": dem_crs,
                "slope_polygon_count": slope_count,
                "clipped_polygon_count": clipped_count,
                "threshold_operator": "<" if slope_comparison == "lt" else "<=",
                "scientific_contract": args.contract,
                "corridor_provenance": corridor_provenance,
                "corridor_manifest_sha256": corridor_manifest_sha256,
                "slope_comparison_was_overridden": args.slope_comparison is not None,
                "threshold_source_note": (
                    "Notebook cells 20-22 use <5%; manuscript text says <=5%. The selected operator is explicit."
                ),
                "area_units": "square metres calculated before storage reprojection",
                "transactional_output_contract": "cleanup-uncommitted-stage-outputs-on-failure",
            },
        )
    except BaseException:
        if dem_input is None:
            dem_output.unlink(missing_ok=True)
        elif not external_signature_preexisting:
            external_signature_path.unlink(missing_ok=True)
            if not external_chunk_dir_preexisting:
                try:
                    chunk_dir.rmdir()
                except OSError:
                    pass
        _remove_vector_output(slope_output)
        _remove_vector_output(clipped_output)
        manifest_path.unlink(missing_ok=True)
        raise
    print(f"Saved {slope_count} slope polygons: {slope_output}")
    print(f"Saved {clipped_count} corridor-clipped polygons: {clipped_output}")
    print(f"Wrote run manifest: {manifest_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
