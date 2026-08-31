"""Build segmentation chips and an auditable partition manifest.

The recovered notebook retained only positive 512-pixel chips and never made a
split. This entry point can reconstruct that historical dataset, labeled only
as `resubstitution`, or build a corrected RGB+NIR dataset with
deterministic negative sampling and spatially grouped train/validation/test
partitions.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.parking_common import (  # noqa: E402
    BandObservation,
    ContractError,
    build_spatial_partitions,
    load_run_manifest,
    refuse_existing_output,
    sha256_file,
    validate_band_contract,
    validate_corrected_corridor_grid,
    validate_open_corrected_corridor_raster,
    validate_partition_rows,
    verify_manifest_artifact,
    write_run_manifest,
)
from project_paths import (  # noqa: E402
    CURATED_CANDIDATES,
    NAIP_CORRIDOR_RASTER,
    PARKING_CHIP_RUN_ROOT,
    PARKING_CORRIDOR_MANIFEST,
    PARKING_CORRIDOR_RASTER,
    PARKING_CURATED_CANDIDATES,
    PARKING_LABEL_VALIDATION_MANIFEST,
    WORKFLOW_ROOT,
)


@dataclass(frozen=True)
class ChipWindow:
    chip_id: str
    row: int
    column: int
    classes: tuple[int, ...]
    valid_fraction: float = 1.0

    @property
    def has_positive(self) -> bool:
        return bool(self.classes)


def iter_chip_windows(
    width: int,
    height: int,
    chip_size: int,
    *,
    include_partial: bool,
) -> Any:
    """Yield row/column plus in-raster size for deterministic chip coverage."""

    if width <= 0 or height <= 0 or chip_size <= 0:
        raise ContractError("Raster dimensions and chip_size must be positive.")
    row_stop = height if include_partial else max(0, height - chip_size + 1)
    column_stop = width if include_partial else max(0, width - chip_size + 1)
    for row in range(0, row_stop, chip_size):
        read_height = min(chip_size, height - row)
        for column in range(0, column_stop, chip_size):
            read_width = min(chip_size, width - column)
            yield row, column, read_height, read_width


def _read_padded_chip(
    source: Any,
    band_indexes: Sequence[int],
    *,
    row: int,
    column: int,
    read_height: int,
    read_width: int,
    chip_size: int,
    window_class: Any,
) -> tuple[Any, Any, Any, Any]:
    """Read one chip and pad out-of-raster edges as explicitly invalid pixels."""

    import numpy as np

    read_window = window_class(column, row, read_width, read_height)
    full_window = window_class(column, row, chip_size, chip_size)
    raw = source.read(band_indexes, window=read_window)
    source_valid = source.dataset_mask(window=read_window) > 0
    source_valid &= np.all(np.isfinite(raw), axis=0)
    image = np.zeros((len(band_indexes), chip_size, chip_size), dtype=raw.dtype)
    valid = np.zeros((chip_size, chip_size), dtype=bool)
    image[:, :read_height, :read_width] = raw
    valid[:read_height, :read_width] = source_valid
    image[:, ~valid] = 0
    return image, valid, read_window, full_window


def stable_negative_selection(
    windows: Sequence[ChipWindow],
    *,
    seed: int,
    negative_ratio: float,
) -> set[str]:
    """Select a fixed number of negatives using a content-independent hash."""

    if negative_ratio < 0:
        raise ContractError("negative_ratio cannot be negative.")
    positives = sum(window.has_positive for window in windows)
    negative_target = math.ceil(positives * negative_ratio)
    negatives = [window.chip_id for window in windows if not window.has_positive]
    ranked = sorted(
        negatives,
        key=lambda chip_id: hashlib.sha256(f"{seed}:negative:{chip_id}".encode("utf-8")).digest(),
    )
    return set(ranked[: min(negative_target, len(ranked))])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", choices=("historical", "corrected"), required=True)
    parser.add_argument(
        "--imagery",
        type=Path,
        default=None,
        help="Defaults to the versioned corrected corridor raster, or the legacy raster in historical mode.",
    )
    parser.add_argument(
        "--imagery-manifest",
        type=Path,
        help="Corrected stage-01 corridor manifest that binds the imagery checksum and ordered bands.",
    )
    parser.add_argument(
        "--band-contract",
        choices=("rgb", "rgbnir"),
        default=None,
        help="Corrected source contract. Only rgbnir is canonical and accepted for corrected runs.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Defaults to versioned curated labels, or the surviving legacy labels in historical mode.",
    )
    parser.add_argument(
        "--labels-manifest",
        type=Path,
        help="Corrected stage-02 adjudicated-lineage validation run manifest.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "A new, non-existing run directory. Corrected mode defaults to the configured parking run; "
            "historical reconstruction requires an explicit isolated path."
        ),
    )
    parser.add_argument("--label-layer", default=None)
    parser.add_argument("--class-field", default="class_id")
    parser.add_argument("--chip-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--spatial-group-pixels", type=int, default=4096)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--required-partition-class",
        action="append",
        type=int,
        default=None,
        help="Corrected mode: class required in every partition; defaults to paved (1) and dirt/gravel (2).",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Corrected mode only: number of seeded negative chips per positive chip.",
    )
    parser.add_argument(
        "--minimum-negative-valid-fraction",
        type=float,
        default=0.50,
        help="Corrected mode: exclude negative chips mostly outside the clipped-raster footprint.",
    )
    parser.add_argument(
        "--overlap-policy",
        choices=("error", "last"),
        default=None,
        help="Defaults to 'last' for reconstruction and 'error' for corrected analysis.",
    )
    return parser


def resolve_default_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Apply contract-specific paths without ever targeting legacy outputs for corrected work."""

    if args.contract == "corrected":
        args.imagery = args.imagery or PARKING_CORRIDOR_RASTER
        args.imagery_manifest = args.imagery_manifest or PARKING_CORRIDOR_MANIFEST
        args.labels = args.labels or PARKING_CURATED_CANDIDATES
        args.labels_manifest = args.labels_manifest or PARKING_LABEL_VALIDATION_MANIFEST
        args.output_root = args.output_root or PARKING_CHIP_RUN_ROOT
    else:
        args.imagery = args.imagery or NAIP_CORRIDOR_RASTER
        args.labels = args.labels or CURATED_CANDIDATES
        if args.output_root is None:
            raise ContractError(
                "Historical dataset reconstruction requires an explicit new --output-root; "
                "the surviving chip directory is immutable evidence."
            )
    return args


def _query_labels(labels: Any, bounds_geometry: Any) -> Any:
    indexes = labels.sindex.query(bounds_geometry, predicate="intersects")
    return labels.iloc[list(indexes)]


def color_interpretation_names(values: Sequence[Any]) -> tuple[str, ...]:
    """Return stable Rasterio color names across enum string representations."""

    return tuple(str(getattr(value, "name", value)).lower() for value in values)


def _mask_for_window(
    labels: Any,
    *,
    bounds_geometry: Any,
    transform: Any,
    size: int,
    class_field: str,
    overlap_policy: str,
) -> Any:
    import numpy as np
    from rasterio.features import rasterize

    candidates = _query_labels(labels, bounds_geometry)
    candidates = candidates[candidates[class_field].fillna(0).astype(int) > 0]
    if candidates.empty:
        return np.zeros((size, size), dtype=np.uint8)

    shapes = [
        (geometry, int(class_id))
        for geometry, class_id in zip(candidates.geometry, candidates[class_field], strict=True)
        if geometry is not None and not geometry.is_empty
    ]
    if overlap_policy == "error":
        occupied = np.zeros((size, size), dtype=np.uint8)
        for geometry, class_id in shapes:
            class_pixels = rasterize(
                [(geometry, 1)],
                out_shape=(size, size),
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )
            if np.any((occupied > 0) & (class_pixels > 0)):
                raise ContractError(
                    f"Overlapping positive labels detected in a chip (class {class_id}); "
                    "resolve them or explicitly request --overlap-policy last."
                )
            occupied[class_pixels > 0] = 1

    return rasterize(
        shapes,
        out_shape=(size, size),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )


def build_dataset(args: argparse.Namespace) -> None:
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import rasterio
    from pyproj import CRS
    from rasterio.windows import Window, bounds as window_bounds, transform as window_transform
    from shapely.geometry import box

    resolve_default_paths(args)
    if args.chip_size <= 0:
        raise ContractError("chip_size must be positive.")
    if args.contract == "historical" and args.negative_ratio != 1.0:
        raise ContractError("negative_ratio is not used in historical positive-only reconstruction.")
    if args.contract == "historical" and args.band_contract is not None:
        raise ContractError("Historical reconstruction is fixed to the recovered RGBA/alpha source.")
    if args.contract == "corrected" and args.band_contract not in (None, "rgbnir"):
        raise ContractError("The corrected contract is fixed to genuine ordered RGB+NIR imagery.")
    if not 0 <= args.minimum_negative_valid_fraction <= 1:
        raise ContractError("minimum_negative_valid_fraction must be between zero and one.")
    refuse_existing_output(args.output_root)
    if not args.imagery.is_file():
        raise FileNotFoundError(args.imagery)
    if not args.labels.is_file():
        raise FileNotFoundError(args.labels)

    imagery_manifest: Path | None = None
    labels_manifest: Path | None = None
    expected_raster_crs: Any | None = None
    expected_raster_resolution_m: float | None = None
    if args.contract == "corrected":
        imagery_manifest = args.imagery_manifest
        if imagery_manifest is None:
            raise ContractError("Corrected dataset construction requires --imagery-manifest.")
        expected_band_contract = "rgbnir"
        imagery_payload = load_run_manifest(
            imagery_manifest,
            expected_stage="01_build_corridor",
            expected_contract=f"projected-geometry-{expected_band_contract}",
        )
        verify_manifest_artifact(imagery_payload, args.imagery)
        expected_names = ["R", "G", "B"] if expected_band_contract == "rgb" else ["R", "G", "B", "N"]
        imagery_extra = imagery_payload.get("extra", {})
        declared_raster_crs, declared_resolution_m = validate_corrected_corridor_grid(
            imagery_extra,
            expected_band_names=expected_names,
        )
        try:
            expected_raster_crs = CRS.from_user_input(declared_raster_crs)
            metric_crs = CRS.from_user_input(imagery_extra["metric_crs"])
            expected_raster_resolution_m = declared_resolution_m
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(
                "Corrected corridor manifest lacks a readable projected raster CRS/resolution contract."
            ) from exc
        units = {str(axis.unit_name).lower() for axis in expected_raster_crs.axis_info if axis.unit_name}
        if (
            not expected_raster_crs.is_projected
            or not units
            or not all("met" in unit for unit in units)
            or not expected_raster_crs.equals(metric_crs)
            or expected_raster_resolution_m <= 0
        ):
            raise ContractError(
                "Corrected corridor imagery must use the declared projected metre grid and positive resolution."
            )
        labels_manifest = args.labels_manifest
        if labels_manifest is None:
            raise ContractError("Corrected dataset construction requires --labels-manifest.")
        labels_payload = load_run_manifest(
            labels_manifest,
            expected_stage="02_validate_candidate_labels",
        )
        if not str(labels_payload.get("contract", "")).startswith("adjudicated-lineage-"):
            raise ContractError("Corrected labels require the adjudicated lineage validation contract.")
        if labels_payload.get("extra", {}).get("validation_status") != "pass":
            raise ContractError("Corrected label validation did not pass.")
        verify_manifest_artifact(labels_payload, args.labels, section="inputs")
        corridor_manifest_sha256 = sha256_file(imagery_manifest)
        if labels_payload.get("extra", {}).get("corridor_manifest_sha256") != corridor_manifest_sha256:
            raise ContractError("Curated labels and corrected imagery do not originate from the same corridor run.")
    else:
        corridor_manifest_sha256 = None

    overlap_policy = args.overlap_policy or ("last" if args.contract == "historical" else "error")
    labels = gpd.read_file(args.labels, layer=args.label_layer)
    if args.class_field not in labels.columns:
        raise ContractError(f"Label field {args.class_field!r} is absent from {args.labels}.")
    if labels.empty:
        raise ContractError("The curated label layer is empty.")
    class_values = labels[args.class_field].dropna().astype(int)
    invalid_classes = sorted(set(class_values).difference(range(6)))
    if invalid_classes:
        raise ContractError(f"Labels contain class IDs outside 0-5: {invalid_classes}.")

    with rasterio.open(args.imagery) as source:
        if source.crs is None:
            raise ContractError("Imagery has no CRS.")
        if args.contract == "corrected":
            assert expected_raster_crs is not None and expected_raster_resolution_m is not None
            validate_open_corrected_corridor_raster(
                source,
                imagery_extra,
                expected_band_names=("R", "G", "B", "N"),
            )
            observed_crs = CRS.from_user_input(source.crs)
            transform = source.transform
            if not observed_crs.equals(expected_raster_crs):
                raise ContractError(
                    f"Corrected imagery CRS {observed_crs} differs from manifest grid {expected_raster_crs}."
                )
            if (
                not math.isclose(abs(transform.a), expected_raster_resolution_m, abs_tol=1e-6)
                or not math.isclose(abs(transform.e), expected_raster_resolution_m, abs_tol=1e-6)
                or not math.isclose(transform.b, 0.0, abs_tol=1e-12)
                or not math.isclose(transform.d, 0.0, abs_tol=1e-12)
            ):
                raise ContractError(
                    "Corrected imagery transform does not match the manifest's north-up projected metre grid."
                )
        if labels.crs is None:
            raise ContractError("Labels have no CRS.")
        labels = labels.to_crs(source.crs)

        if args.contract == "historical":
            band_indexes = (1, 2, 3, 4)
            band_contract = "rgba-alpha"
            normalization = "historical-double-divide"
        else:
            band_contract = args.band_contract or "rgbnir"
            band_indexes = (1, 2, 3) if band_contract == "rgb" else (1, 2, 3, 4)
            normalization = "unit-rgb" if band_contract == "rgb" else "unit-multiband"
        if source.count < len(band_indexes):
            raise ContractError(f"{band_contract} requires {len(band_indexes)} source bands; found {source.count}.")
        color_interpretations = color_interpretation_names(
            tuple(source.colorinterp[index - 1] for index in band_indexes)
        )
        band_names = tuple(str(source.descriptions[index - 1] or "") for index in band_indexes)
        minima = [float("inf")] * len(band_indexes)
        maxima = [float("-inf")] * len(band_indexes)

        windows: list[ChipWindow] = []
        for row, column, read_height, read_width in iter_chip_windows(
            source.width,
            source.height,
            args.chip_size,
            include_partial=args.contract == "corrected",
        ):
            radiometry, valid_pixels, read_window, full_window = _read_padded_chip(
                source,
                band_indexes,
                row=row,
                column=column,
                read_height=read_height,
                read_width=read_width,
                chip_size=args.chip_size,
                window_class=Window,
            )
            transform = window_transform(full_window, source.transform)
            bounds_geometry = box(*window_bounds(read_window, source.transform))
            mask = _mask_for_window(
                labels,
                bounds_geometry=bounds_geometry,
                transform=transform,
                size=args.chip_size,
                class_field=args.class_field,
                overlap_policy=overlap_policy,
            )
            valid_fraction = float(valid_pixels.mean())
            if valid_fraction == 0:
                continue
            for index, values in enumerate(radiometry):
                valid_values = values[valid_pixels]
                finite = valid_values[np.isfinite(valid_values)]
                if finite.size:
                    minima[index] = min(minima[index], float(finite.min()))
                    maxima[index] = max(maxima[index], float(finite.max()))
            mask[~valid_pixels] = 0
            classes = tuple(int(value) for value in np.unique(mask) if value > 0)
            # Preserve the recovered notebook's `<column>_<row>` ID format.
            windows.append(ChipWindow(f"{column}_{row}", row, column, classes, valid_fraction))

        positive_ids = {window.chip_id for window in windows if window.has_positive}
        if not positive_ids:
            raise ContractError("No positive-label chips were found.")
        if any(not math.isfinite(value) for value in (*minima, *maxima)):
            raise ContractError("Could not observe finite radiometry for every declared source band.")
        observed = BandObservation(
            len(band_indexes),
            tuple(minima),
            tuple(maxima),
            color_interpretations,
            band_names,
        )
        validate_band_contract(
            observed,
            band_contract,
            require_identity=args.contract == "corrected",
        )

        if args.contract == "historical":
            selected_ids = positive_ids
        else:
            windows = [
                window
                for window in windows
                if window.has_positive or window.valid_fraction >= args.minimum_negative_valid_fraction
            ]
            positive_ids = {window.chip_id for window in windows if window.has_positive}
            selected_ids = positive_ids | stable_negative_selection(
                windows,
                seed=args.seed,
                negative_ratio=args.negative_ratio,
            )

        selected = [window for window in windows if window.chip_id in selected_ids]
        if args.contract == "historical":
            assignments = {window.chip_id: (window.chip_id, "resubstitution") for window in selected}
        else:
            assignments = build_spatial_partitions(
                (window.chip_id for window in selected),
                seed=args.seed,
                group_pixels=args.spatial_group_pixels,
                train_fraction=args.train_fraction,
                validation_fraction=args.validation_fraction,
            )

        validate_partition_rows(
            (
                {
                    "chip_id": window.chip_id,
                    "spatial_group": assignments[window.chip_id][0],
                    "partition": assignments[window.chip_id][1],
                }
                for window in selected
            ),
            corrected=args.contract == "corrected",
        )
        if args.contract == "corrected":
            required_classes = tuple(args.required_partition_class or (1, 2))
            if any(class_id not in range(1, 6) for class_id in required_classes):
                raise ContractError("required_partition_class values must be in the positive class range 1-5.")
            classes_by_partition = {name: set() for name in ("train", "validation", "test")}
            for window in selected:
                classes_by_partition[assignments[window.chip_id][1]].update(window.classes)
            missing_by_partition = {
                partition: sorted(set(required_classes).difference(classes))
                for partition, classes in classes_by_partition.items()
                if set(required_classes).difference(classes)
            }
            if missing_by_partition:
                raise ContractError(
                    "Spatial split lacks required parking classes in one or more partitions: "
                    f"{missing_by_partition}. Choose a reviewed seed/grouping; do not accept an unscorable test split."
                )

        output_images = args.output_root / "images"
        output_masks = args.output_root / "masks"
        output_valid_masks = args.output_root / "valid_masks" if args.contract == "corrected" else None
        output_images.mkdir(parents=True)
        output_masks.mkdir()
        if output_valid_masks is not None:
            output_valid_masks.mkdir()
        rows: list[dict[str, Any]] = []
        selected_by_id = {window.chip_id: window for window in selected}
        for chip_id in sorted(selected_by_id):
            item = selected_by_id[chip_id]
            read_height = min(args.chip_size, source.height - item.row)
            read_width = min(args.chip_size, source.width - item.column)
            image_bands, valid_pixels, read_window, full_window = _read_padded_chip(
                source,
                band_indexes,
                row=item.row,
                column=item.column,
                read_height=read_height,
                read_width=read_width,
                chip_size=args.chip_size,
                window_class=Window,
            )
            transform = window_transform(full_window, source.transform)
            bounds_geometry = box(*window_bounds(read_window, source.transform))
            mask = _mask_for_window(
                labels,
                bounds_geometry=bounds_geometry,
                transform=transform,
                size=args.chip_size,
                class_field=args.class_field,
                overlap_policy=overlap_policy,
            )
            mask[~valid_pixels] = 0
            image = image_bands.transpose(1, 2, 0)
            image_path = output_images / f"{chip_id}.npy"
            mask_path = output_masks / f"{chip_id}.npy"
            np.save(image_path, image)
            np.save(mask_path, mask)
            valid_mask_path = output_valid_masks / f"{chip_id}.npy" if output_valid_masks is not None else None
            if valid_mask_path is not None:
                np.save(valid_mask_path, valid_pixels)
            spatial_group, partition = assignments[chip_id]
            rows.append(
                {
                    "chip_id": chip_id,
                    "row": item.row,
                    "column": item.column,
                    "spatial_group": spatial_group,
                    "partition": partition,
                    "has_positive": item.has_positive,
                    "valid_fraction": item.valid_fraction,
                    "classes": ",".join(str(value) for value in item.classes),
                    # Keep the split artifact portable across repository moves;
                    # consumers resolve these paths beneath output_root.
                    "image_path": image_path.relative_to(args.output_root).as_posix(),
                    "mask_path": mask_path.relative_to(args.output_root).as_posix(),
                    "valid_mask_path": (
                        valid_mask_path.relative_to(args.output_root).as_posix() if valid_mask_path is not None else ""
                    ),
                    "band_contract": band_contract,
                    "normalization": normalization,
                    "source_crs": str(source.crs),
                    "source_transform": ",".join(str(value) for value in transform),
                }
            )

    validate_partition_rows(rows, corrected=args.contract == "corrected")
    table = pd.DataFrame(rows).sort_values("chip_id")
    split_path = args.output_root / "splits.parquet"
    audit_csv = args.output_root / "splits.csv"
    table.to_parquet(split_path, index=False)
    table.to_csv(audit_csv, index=False)
    manifest_path = args.output_root / "run_manifest.json"
    output_artifacts = [output_images, output_masks, split_path, audit_csv]
    if output_valid_masks is not None:
        output_artifacts.insert(2, output_valid_masks)
    write_run_manifest(
        manifest_path,
        workflow_root=WORKFLOW_ROOT,
        stage="03_build_segmentation_dataset",
        contract=args.contract,
        arguments=vars(args),
        inputs=tuple(
            path for path in (args.imagery, args.labels, imagery_manifest, labels_manifest) if path is not None
        ),
        outputs=tuple(output_artifacts),
        extra={
            "selected_chips": len(rows),
            "positive_chips": sum(bool(row["has_positive"]) for row in rows),
            "band_observation": {
                "channels": observed.channels,
                "minima": observed.minima,
                "maxima": observed.maxima,
                "color_interpretations": observed.color_interpretations,
                "band_names": observed.band_names,
            },
            "historical_metrics_label": "resubstitution" if args.contract == "historical" else None,
            "corridor_manifest_sha256": corridor_manifest_sha256,
            "source_grid_crs": (expected_raster_crs.to_string() if expected_raster_crs is not None else None),
            "source_grid_resolution_m": expected_raster_resolution_m,
            "validity_contract": ("source-validity-plus-finite-radiometry" if args.contract == "corrected" else None),
            "partial_edge_policy": "pad-invalid" if args.contract == "corrected" else "drop-partial",
        },
    )
    print(f"Wrote {len(rows):,} chip pairs and split manifest to {args.output_root}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        build_dataset(args)
    except (ContractError, FileExistsError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
