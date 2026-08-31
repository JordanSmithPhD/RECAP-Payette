#!/usr/bin/env python3
"""Generate immutable raw SAM proposals within the flat-terrain corridor.

This is the portable reconstruction of notebook cell 22.  Its recovered ViT-H
and automatic-mask-generator parameters are defaults, but the notebook's
destructive write to ``candidate_objects.gpkg`` is forbidden.  Raw proposals
are written to ``project_paths.SAM_RAW_PROPOSALS`` without a label column;
manual curation must create a distinct copy retaining ``proposal_id``.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterator, Sequence


WORKING_ROOT = Path(__file__).resolve().parents[1]
if str(WORKING_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKING_ROOT))

from project_paths import (  # noqa: E402
    MODELS_ROOT,
    PARKING_CORRIDOR_MANIFEST,
    PARKING_CORRIDOR_RASTER,
    PARKING_CURATED_CANDIDATES,
    PARKING_SAM_RAW_PROPOSALS,
    PARKING_SLOPE_LE5_CORRIDOR,
    PARKING_SLOPE_MANIFEST,
    WORKFLOW_ROOT,
)
from pipeline.parking_common import (  # noqa: E402
    ContractError,
    load_run_manifest,
    refuse_existing_output,
    sha256_file,
    validate_corrected_corridor_grid,
    validate_corrected_dem_slope_contract,
    validate_open_corrected_corridor_raster,
    verify_manifest_artifact,
    write_run_manifest,
)


SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
DEFAULT_CHECKPOINT = MODELS_ROOT / "sam_vit_h_4b8939.pth"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the recovered SAM ViT-H automatic proposal loop over NAIP tiles that intersect the "
            "less-than-5-percent slope mask."
        )
    )
    parser.add_argument(
        "--contract",
        choices=("historical", "corrected"),
        default="corrected",
        help="Corrected mode requires checksum-bound stage-01 imagery and slope manifests.",
    )
    parser.add_argument("--naip", type=Path, default=PARKING_CORRIDOR_RASTER)
    parser.add_argument("--slope", type=Path, default=PARKING_SLOPE_LE5_CORRIDOR)
    parser.add_argument(
        "--imagery-manifest",
        type=Path,
        default=PARKING_CORRIDOR_MANIFEST,
        help="Stage-01 corridor manifest that checksum-binds --naip.",
    )
    parser.add_argument(
        "--slope-manifest",
        type=Path,
        default=PARKING_SLOPE_MANIFEST,
        help="Stage-01 DEM/slope manifest that checksum-binds --slope.",
    )
    parser.add_argument(
        "--no-input-manifests",
        action="store_true",
        help="Allow an isolated historical reconstruction without stage-01 manifests; prohibited when corrected.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-url", default=SAM_CHECKPOINT_URL)
    parser.add_argument(
        "--download-checkpoint",
        action="store_true",
        help="Download the recovered ViT-H checkpoint only if --checkpoint is missing.",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        default=None,
        help="Optional expected SHA-256. Supplying it is recommended for a new checkpoint acquisition.",
    )
    parser.add_argument("--model-type", default="vit_h", choices=("vit_h", "vit_l", "vit_b"))
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.86)
    parser.add_argument("--stability-score-thresh", type=float, default=0.92)
    parser.add_argument("--crop-n-layers", type=int, default=0)
    parser.add_argument("--min-mask-region-area", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="Recorded NumPy/PyTorch seed for deterministic components.")
    parser.add_argument(
        "--normalization",
        choices=("strict-unit-or-byte", "historical-auto"),
        default="strict-unit-or-byte",
        help=(
            "Strict mode accepts unit-scaled or byte RGB only. Historical mode also reproduces per-tile min/max "
            "stretching for other numeric ranges."
        ),
    )
    parser.add_argument(
        "--metric-crs",
        default="EPSG:32611",
        help="Projected CRS used exclusively for area_m2 calculation.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device. Auto chooses CUDA when available.",
    )
    parser.add_argument(
        "--continue-on-tile-error",
        action="store_true",
        help="Record and skip failed tiles. Default behavior stops so incomplete proposal runs are not silent.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PARKING_SAM_RAW_PROPOSALS,
        help="New raw proposal GeoPackage; it must differ from the curated label path.",
    )
    parser.add_argument(
        "--curated-labels",
        type=Path,
        default=PARKING_CURATED_CANDIDATES,
        help="Reserved manual-label path used only to enforce raw/curated separation.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Run-manifest path; defaults beside the raw proposal GeoPackage.",
    )
    return parser


def _validate_input_lineage(
    args: argparse.Namespace,
    *,
    naip_path: Path,
    slope_path: Path,
) -> tuple[Path, Path, str, float, dict[str, Any]] | None:
    """Bind corrected proposal generation to the exact stage-01 products."""

    if args.no_input_manifests:
        if args.contract == "corrected":
            raise ContractError("Corrected SAM generation may not use --no-input-manifests.")
        return None

    imagery_manifest = args.imagery_manifest.expanduser().resolve(strict=False)
    slope_manifest = args.slope_manifest.expanduser().resolve(strict=False)
    imagery_payload = load_run_manifest(
        imagery_manifest,
        expected_stage="01_build_corridor",
        expected_contract="projected-geometry-rgbnir",
    )
    verify_manifest_artifact(imagery_payload, naip_path)
    imagery_extra = imagery_payload.get("extra", {})
    raster_crs, raster_resolution_m = validate_corrected_corridor_grid(
        imagery_extra,
        expected_band_names=("R", "G", "B", "N"),
    )

    slope_payload = load_run_manifest(
        slope_manifest,
        expected_stage="01_build_dem_slope",
        expected_contract="corrected-complete-3dep-grid-projected-area-slope-le",
    )
    validate_corrected_dem_slope_contract(slope_payload)
    verify_manifest_artifact(slope_payload, slope_path)
    slope_arguments = slope_payload.get("arguments", {})
    try:
        threshold = float(slope_arguments.get("slope_threshold_pct"))
    except (TypeError, ValueError) as exc:
        raise ContractError("Slope manifest has no numeric slope_threshold_pct.") from exc
    if threshold != 5.0 or slope_arguments.get("resolved_slope_comparison") != "le":
        raise ContractError("Corrected SAM generation requires the documented slope <= 5% mask.")
    if slope_payload.get("extra", {}).get("threshold_operator") != "<=":
        raise ContractError("Slope manifest threshold operator does not match <= 5%.")
    if slope_payload.get("extra", {}).get("corridor_manifest_sha256") != sha256_file(imagery_manifest):
        raise ContractError("Slope and imagery manifests do not originate from the same immutable corridor run.")
    return imagery_manifest, slope_manifest, raster_crs, raster_resolution_m, dict(imagery_extra)


def _iter_windows(width: int, height: int, tile_size: int, Window: Any) -> Iterator[Any]:
    for row_offset in range(0, height, tile_size):
        for column_offset in range(0, width, tile_size):
            yield Window(
                column_offset,
                row_offset,
                min(tile_size, width - column_offset),
                min(tile_size, height - row_offset),
            )


def _to_uint8_rgb(
    image: Any,
    *,
    np: Any,
    normalization: str,
    valid_mask: Any | None = None,
) -> Any:
    """Convert rasterio CHW RGB to SAM HWC uint8 without using an alpha band."""

    if image.ndim != 3 or image.shape[0] < 3:
        raise ValueError(f"SAM requires at least three raster bands; found shape {image.shape}.")
    rgb = np.moveaxis(image[:3], 0, -1)
    valid = np.ones(rgb.shape[:2], dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != rgb.shape[:2]:
        raise ValueError(f"RGB valid mask shape {valid.shape} does not match tile shape {rgb.shape[:2]}.")
    if not bool(valid.any()):
        raise ValueError("RGB tile has no source-valid pixels.")
    if not bool(np.isfinite(rgb[valid]).all()):
        raise ValueError("Source-valid RGB pixels contain NaN or infinite values.")
    if rgb.dtype == np.uint8:
        converted = rgb.copy()
        converted[~valid] = 0
        return converted

    values = rgb.astype(np.float32, copy=False)
    minimum = float(values[valid].min())
    maximum = float(values[valid].max())
    safe_values = np.where(valid[..., None], values, 0.0)
    if minimum < 0:
        raise ValueError(f"RGB tile contains negative radiometry (minimum {minimum}).")
    if maximum <= 1.5:
        converted = np.rint(safe_values * 255.0).clip(0, 255).astype(np.uint8)
        converted[~valid] = 0
        return converted
    if maximum <= 255.0:
        converted = np.rint(safe_values).clip(0, 255).astype(np.uint8)
        converted[~valid] = 0
        return converted
    if normalization == "historical-auto" and maximum > minimum:
        converted = np.rint((safe_values - minimum) / (maximum - minimum) * 255.0).astype(np.uint8)
        converted[~valid] = 0
        return converted
    raise ValueError(
        f"RGB range [{minimum}, {maximum}] is neither unit-scaled nor byte-scaled. "
        "Inspect the raster or explicitly choose --normalization historical-auto for artifact reconstruction."
    )


def _download_checkpoint(path: Path, url: str) -> None:
    if not url:
        raise ValueError("--checkpoint-url cannot be empty when --download-checkpoint is used.")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    try:
        urllib.request.urlretrieve(url, partial)
        if not partial.is_file() or partial.stat().st_size <= 1_000:
            raise RuntimeError("Downloaded checkpoint is unexpectedly small or missing.")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _sdp_context(torch: Any, *, use_cuda: bool) -> Any:
    if use_cuda and hasattr(torch.backends.cuda, "sdp_kernel"):
        return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    return contextlib.nullcontext()


def run(args: argparse.Namespace) -> int:
    naip_path = args.naip.expanduser().resolve(strict=False)
    slope_path = args.slope.expanduser().resolve(strict=False)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=False)
    output_path = args.output.expanduser().resolve(strict=False)
    curated_path = args.curated_labels.expanduser().resolve(strict=False)
    manifest_path = (
        args.manifest.expanduser().resolve(strict=False)
        if args.manifest is not None
        else output_path.with_name(f"{output_path.stem}.run.json")
    )

    if output_path.suffix.lower() != ".gpkg":
        raise ValueError("Raw SAM proposals must be stored in one .gpkg file.")
    if output_path == curated_path:
        raise ValueError(
            "Raw SAM output cannot be the curated-label GeoPackage. Choose a distinct --output and copy proposals "
            "into a separate curation artifact."
        )
    if output_path in {naip_path, slope_path, checkpoint_path, manifest_path}:
        raise ValueError("Raw-proposal output must differ from every input and from the run manifest.")
    if manifest_path in {naip_path, slope_path, checkpoint_path, curated_path}:
        raise ValueError("Run-manifest output must differ from every input and from the reserved curated-label path.")
    if args.contract == "historical":
        corrected_defaults = {
            PARKING_CORRIDOR_RASTER.resolve(strict=False),
            PARKING_SLOPE_LE5_CORRIDOR.resolve(strict=False),
            PARKING_CORRIDOR_MANIFEST.resolve(strict=False),
            PARKING_SLOPE_MANIFEST.resolve(strict=False),
            PARKING_SAM_RAW_PROPOSALS.resolve(strict=False),
            PARKING_CURATED_CANDIDATES.resolve(strict=False),
        }
        historical_paths = {
            naip_path,
            slope_path,
            output_path,
            curated_path,
            manifest_path,
        }
        if not args.no_input_manifests:
            historical_paths.update(
                {
                    args.imagery_manifest.expanduser().resolve(strict=False),
                    args.slope_manifest.expanduser().resolve(strict=False),
                }
            )
        if historical_paths & corrected_defaults:
            raise ContractError(
                "Historical SAM reconstruction requires explicit isolated imagery/slope/output/label/manifest "
                "paths; corrected defaults are reserved."
            )
        if not args.no_input_manifests:
            raise ContractError(
                "Historical SAM reconstruction currently requires --no-input-manifests because the recovered "
                "stage-01 historical provenance is incomplete."
            )
    if args.tile_size <= 0 or args.points_per_side <= 0:
        raise ValueError("--tile-size and --points-per-side must be positive.")
    if not 0 <= args.pred_iou_thresh <= 1 or not 0 <= args.stability_score_thresh <= 1:
        raise ValueError("SAM IoU and stability thresholds must fall between zero and one.")
    if args.crop_n_layers < 0 or args.min_mask_region_area < 0:
        raise ValueError("Crop layers and minimum mask area cannot be negative.")
    if not 0 <= args.seed <= 2**32 - 1:
        raise ValueError("--seed must fall between 0 and 2**32-1.")
    if not naip_path.is_file():
        raise FileNotFoundError(f"NAIP corridor raster does not exist: {naip_path}")
    if not slope_path.is_file():
        raise FileNotFoundError(f"Slope constraint vector does not exist: {slope_path}")
    input_lineage = _validate_input_lineage(
        args,
        naip_path=naip_path,
        slope_path=slope_path,
    )
    refuse_existing_output(output_path)
    refuse_existing_output(manifest_path)

    if not checkpoint_path.is_file():
        if not args.download_checkpoint:
            raise FileNotFoundError(
                f"SAM checkpoint is missing: {checkpoint_path}. Supply --checkpoint or explicitly allow "
                "--download-checkpoint."
            )
        _download_checkpoint(checkpoint_path, args.checkpoint_url)
    if args.checkpoint_sha256:
        observed = sha256_file(checkpoint_path)
        if observed.lower() != args.checkpoint_sha256.lower():
            raise ValueError(
                f"SAM checkpoint SHA-256 mismatch: expected {args.checkpoint_sha256.lower()}, observed {observed}."
            )

    try:
        import geopandas as gpd
        import numpy as np
        import rasterio
        import torch
        from pyproj import CRS
        from rasterio.features import rasterize, shapes
        from rasterio.windows import Window
        from rasterio.windows import bounds as window_bounds
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        from shapely.geometry import box, shape
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SAM proposal generation requires PyTorch, segment-anything, GeoPandas, rasterio, NumPy, pyproj, and Shapely."
        ) from exc

    metric_crs = CRS.from_user_input(args.metric_crs)
    if not metric_crs.is_projected:
        raise ValueError(f"--metric-crs must be projected for area_m2: {args.metric_crs}")
    metric_units = {str(axis.unit_name).lower() for axis in metric_crs.axis_info if axis.unit_name}
    if metric_units and not all("met" in unit for unit in metric_units):
        raise ValueError(f"--metric-crs axes must use metres, not {sorted(metric_units)}: {args.metric_crs}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but PyTorch reports no CUDA device.")
    device = "cuda" if (args.device == "cuda" or args.device == "auto" and torch.cuda.is_available()) else "cpu"
    use_cuda = device == "cuda"
    if use_cuda:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    if args.model_type not in sam_model_registry:
        raise ValueError(f"segment-anything does not provide model type {args.model_type!r}.")
    model = sam_model_registry[args.model_type](checkpoint=str(checkpoint_path))
    model.to(device=device, dtype=torch.float32)
    model.eval()
    generator = SamAutomaticMaskGenerator(
        model=model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        min_mask_region_area=args.min_mask_region_area,
    )

    slope = gpd.read_file(slope_path)
    if slope.empty:
        raise ValueError(f"Slope constraint layer is empty: {slope_path}")
    if slope.crs is None:
        raise ValueError(f"Slope constraint layer has no CRS: {slope_path}")

    proposals: list[dict[str, Any]] = []
    tile_errors: list[dict[str, str | int]] = []
    processed_tiles = 0
    eligible_tiles = 0
    raster_width = 0
    raster_height = 0
    raster_crs: Any = None
    source_valid_pixel_count = 0
    proposal_domain_pixel_count = 0
    with rasterio.open(naip_path) as source:
        if source.crs is None:
            raise ValueError(f"NAIP raster has no CRS: {naip_path}")
        if source.count < 3:
            raise ValueError(f"NAIP raster has {source.count} bands; SAM requires RGB.")
        raster_width = source.width
        raster_height = source.height
        raster_crs = source.crs
        if args.contract == "corrected":
            assert input_lineage is not None
            validate_open_corrected_corridor_raster(
                source,
                input_lineage[4],
                expected_band_names=("R", "G", "B", "N"),
            )
            expected_crs = CRS.from_user_input(input_lineage[2])
            observed_crs = CRS.from_user_input(source.crs)
            units = {str(axis.unit_name).lower() for axis in observed_crs.axis_info if axis.unit_name}
            resolution_m = input_lineage[3]
            if (
                not observed_crs.equals(expected_crs)
                or not observed_crs.is_projected
                or not units
                or not all("met" in unit for unit in units)
            ):
                raise ContractError("Corrected SAM imagery does not use the projected metre CRS declared by stage 01.")
            if (
                not math.isclose(abs(source.transform.a), resolution_m, abs_tol=1e-6)
                or not math.isclose(abs(source.transform.e), resolution_m, abs_tol=1e-6)
                or not math.isclose(source.transform.b, 0.0, abs_tol=1e-12)
                or not math.isclose(source.transform.d, 0.0, abs_tol=1e-12)
            ):
                raise ContractError(
                    "Corrected SAM imagery transform does not match the manifest's north-up projected metre grid."
                )
            metric_area_crs = CRS.from_user_input(args.metric_crs)
            if not metric_area_crs.equals(observed_crs):
                raise ContractError(
                    "Corrected SAM --metric-crs must equal the projected corridor raster CRS so proposal areas "
                    "and pixel-domain geometry use one grid."
                )
        if slope.crs != source.crs:
            slope = slope.to_crs(source.crs)
        slope_index = slope.sindex
        total_tiles = math.ceil(source.width / args.tile_size) * math.ceil(source.height / args.tile_size)
        print(f"Processing {total_tiles} raster tiles on {device}.")

        for tile_number, window in enumerate(
            _iter_windows(source.width, source.height, args.tile_size, Window), start=1
        ):
            processed_tiles += 1
            tile_col = int(window.col_off)
            tile_row = int(window.row_off)
            try:
                tile_bounds = window_bounds(window, source.transform)
                tile_geometry = box(*tile_bounds)
                possible = list(slope_index.intersection(tile_geometry.bounds))
                if not possible or not bool(slope.geometry.iloc[possible].intersects(tile_geometry).any()):
                    continue
                eligible_tiles += 1
                image = source.read(window=window)
                if image.size == 0:
                    continue
                tile_transform = source.window_transform(window)
                proposal_domain = None
                if args.contract == "corrected":
                    source_valid = source.dataset_mask(window=window) > 0
                    source_valid &= np.isfinite(image[:4]).all(axis=0)
                    if not bool(source_valid.any()):
                        continue
                    slope_shapes = [
                        (geometry, 1)
                        for geometry in slope.geometry.iloc[possible]
                        if geometry is not None and not geometry.is_empty
                    ]
                    slope_pixels = rasterize(
                        slope_shapes,
                        out_shape=(int(window.height), int(window.width)),
                        transform=tile_transform,
                        fill=0,
                        dtype=np.uint8,
                    ).astype(bool)
                    proposal_domain = source_valid & slope_pixels
                    source_valid_pixel_count += int(source_valid.sum())
                    proposal_domain_pixel_count += int(proposal_domain.sum())
                    if not bool(proposal_domain.any()):
                        continue
                    rgb = _to_uint8_rgb(
                        image,
                        np=np,
                        normalization=args.normalization,
                        valid_mask=source_valid,
                    )
                else:
                    if float(np.nanmax(image)) == 0:
                        continue
                    rgb = _to_uint8_rgb(image, np=np, normalization=args.normalization)

                autocast = torch.autocast(device_type="cuda", enabled=False) if use_cuda else contextlib.nullcontext()
                with autocast:
                    with _sdp_context(torch, use_cuda=use_cuda):
                        masks = generator.generate(rgb)
                if not masks:
                    continue

                for mask_data in masks:
                    segmentation = mask_data["segmentation"].astype(bool)
                    if proposal_domain is not None:
                        segmentation &= proposal_domain
                    segmentation = segmentation.astype(np.uint8)
                    if not bool(segmentation.any()):
                        continue
                    for geometry, value in shapes(
                        segmentation,
                        mask=segmentation > 0,
                        transform=tile_transform,
                    ):
                        if int(value) != 1:
                            continue
                        component_pixels = int(
                            np.count_nonzero(
                                rasterize(
                                    [(geometry, 1)],
                                    out_shape=segmentation.shape,
                                    transform=tile_transform,
                                    fill=0,
                                    dtype=np.uint8,
                                )
                                & segmentation
                            )
                        )
                        if component_pixels <= 0:
                            continue
                        polygon = shape(geometry)
                        if not polygon.is_valid:
                            polygon = polygon.buffer(0)
                        if polygon.is_empty:
                            continue
                        matches = list(slope_index.intersection(polygon.bounds))
                        if not matches or not bool(slope.geometry.iloc[matches].intersects(polygon).any()):
                            continue
                        proposals.append(
                            {
                                "geometry": polygon,
                                # The recovered tile identifier is explicitly
                                # <column>_<row>, matching later chip IDs.
                                "tile": f"{tile_col}_{tile_row}",
                                "tile_col": tile_col,
                                "tile_row": tile_row,
                                "mask_px": component_pixels,
                                "pred_iou": float(mask_data.get("predicted_iou", float("nan"))),
                                "stability": float(mask_data.get("stability_score", float("nan"))),
                            }
                        )
            except Exception as exc:
                error = {"tile": tile_number, "tile_col": tile_col, "tile_row": tile_row, "error": repr(exc)}
                tile_errors.append(error)
                if not args.continue_on_tile_error:
                    raise RuntimeError(
                        f"SAM failed at tile {tile_number} ({tile_col}_{tile_row}); no partial GeoPackage was written."
                    ) from exc

    if not proposals:
        raise RuntimeError("SAM produced no proposals intersecting the slope constraint.")
    if tile_errors and not args.continue_on_tile_error:
        raise RuntimeError("Tile failures occurred in strict mode.")

    frame = gpd.GeoDataFrame(proposals, geometry="geometry", crs=raster_crs)
    frame.insert(0, "proposal_id", [f"sam_{index:08d}" for index in range(1, len(frame) + 1)])
    metric = frame.to_crs(metric_crs)
    frame["area_m2"] = metric.geometry.area.to_numpy(dtype=float)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(output_path, layer="sam_raw_proposals", driver="GPKG")

    manifest_inputs = [naip_path, slope_path, checkpoint_path]
    if input_lineage is not None:
        manifest_inputs.extend(input_lineage[:2])
    write_run_manifest(
        manifest_path,
        workflow_root=WORKFLOW_ROOT,
        stage="02_generate_sam_candidates",
        contract=f"{args.contract}-immutable-raw-proposals-projected-area",
        arguments=vars(args),
        inputs=manifest_inputs,
        outputs=[output_path],
        extra={
            "proposal_count": int(len(frame)),
            "processed_tile_count": processed_tiles,
            "eligible_tile_count": eligible_tiles,
            "tile_error_count": len(tile_errors),
            "tile_errors": tile_errors,
            "device": device,
            "raster_width": raster_width,
            "raster_height": raster_height,
            "raster_crs": str(raster_crs),
            "metric_crs": metric_crs.to_string(),
            "source_valid_pixel_count_across_eligible_tiles": source_valid_pixel_count,
            "proposal_domain_pixel_count": proposal_domain_pixel_count,
            "proposal_domain_contract": (
                "source-validity-plus-finite-radiometry-intersect-slope-le5-on-source-grid"
                if args.contract == "corrected"
                else "historical-vector-intersection"
            ),
            "slope_rasterization": ("pixel-centre-default-all_touched-false" if args.contract == "corrected" else None),
            "proposal_id_contract": "sam_<one-based eight-digit generation order>",
            "mask_px_contract": "per-output-connected-component-after-proposal-domain-clipping",
            "label_column_present": False,
            "input_provenance": ("verified" if input_lineage is not None else "missing-historical-provenance"),
            "corridor_manifest_sha256": (sha256_file(input_lineage[0]) if input_lineage is not None else None),
        },
    )
    print(f"Saved {len(frame)} immutable raw proposals: {output_path}")
    print(f"Wrote run manifest: {manifest_path}")
    print(f"Create curated labels as a separate copy retaining proposal_id: {curated_path}")
    if tile_errors:
        print("The proposal artifact is incomplete because one or more tiles failed; see the run manifest.")
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
