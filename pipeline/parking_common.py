"""Shared contracts for the recovered parking-opportunity pipeline.

This module intentionally has no geospatial or PyTorch imports so configuration,
split, radiometry, and capacity safeguards remain testable in a light-weight
environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


NOTEBOOK_SHA256 = "c4baeab351199297c505acc7962bbbc46d543501c95b59be8b123b61393186bb"
CLASS_NAMES = (
    "background",
    "paved_parking",
    "dirt_gravel_parking",
    "roof",
    "water",
    "road",
)
EXPECTED_HISTORICAL_LABEL_COUNTS = {
    0: 17_400,
    1: 16,
    2: 50,
    3: 321,
    4: 227,
    5: 203,
}
PARTITIONS = ("train", "validation", "test")
CORRECTED_VALIDITY_CONTRACT = "source-validity-plus-finite-radiometry"
CORRECTED_CORRIDOR_NODATA = -9999.0
CORRECTED_AREA_COMPARISON_DOMAIN = "prediction-valid-domain-plus-study-areas-on-grid"
CORRECTED_MOSAIC_PIXEL_CONTRACT = "all-roi-pixel-centres-valid-all-bands-zero-tolerance"
CORRECTED_DEM_GRID_CONTRACT = "exact-crs-lattice-and-all-request-polygon-pixel-centres-valid"


class ContractError(ValueError):
    """Raised when a run would violate a scientific/data contract."""


def validate_corrected_mosaic_contract(extra: Mapping[str, Any]) -> None:
    """Reject footprint-only/zero-fill mosaics from corrected downstream runs."""

    if not isinstance(extra, Mapping):
        raise ContractError("Corrected mosaic manifest has no structured extra metadata.")
    if extra.get("raster_output_band_names") != ["R", "G", "B", "N"]:
        raise ContractError("Corrected mosaic does not record ordered R/G/B/N bands.")
    try:
        nodata = float(extra.get("mosaic_nodata_value"))
    except (TypeError, ValueError) as exc:
        raise ContractError("Corrected mosaic has no numeric output nodata value.") from exc
    if not math.isclose(nodata, CORRECTED_CORRIDOR_NODATA, rel_tol=0.0, abs_tol=1e-9):
        raise ContractError(f"Corrected mosaic nodata must be {CORRECTED_CORRIDOR_NODATA}.")
    if extra.get("pixel_valid_coverage_contract") != CORRECTED_MOSAIC_PIXEL_CONTRACT:
        raise ContractError("Corrected mosaic lacks the all-ROI-pixel valid-data coverage contract.")
    try:
        total = int(extra.get("roi_pixel_centres_total"))
        valid = int(extra.get("roi_pixel_centres_valid_all_bands"))
        invalid = int(extra.get("roi_pixel_centres_invalid"))
    except (TypeError, ValueError) as exc:
        raise ContractError("Corrected mosaic has malformed ROI valid-pixel counts.") from exc
    if isinstance(extra.get("roi_pixel_centres_total"), bool) or total <= 0 or valid != total or invalid != 0:
        raise ContractError(
            f"Corrected mosaic does not have complete valid data over the ROI ({valid}/{total}, invalid={invalid})."
        )
    if extra.get("shard_family_content_hashes_stable_through_manifest") is not True:
        raise ContractError("Corrected mosaic does not prove stable shard content hashes through manifest commit.")


def validate_corrected_dem_slope_contract(payload: Mapping[str, Any]) -> None:
    """Reject legacy centre-sampled DEM manifests from corrected consumers."""

    arguments = payload.get("arguments")
    extra = payload.get("extra")
    if not isinstance(arguments, Mapping) or not isinstance(extra, Mapping):
        raise ContractError("Corrected terrain manifest lacks structured arguments/extra metadata.")
    if arguments.get("dem_input") not in (None, ""):
        raise ContractError("Corrected terrain must use maintained py3dep acquisition, not an external DEM.")
    canonical_arguments = {
        "min_x": -116.28766600566757,
        "min_y": 43.88355482041268,
        "max_x": -115.491507,
        "max_y": 44.514881,
        "grid_size_deg": 0.02,
        "edge_padding_deg": 0.001,
        "resolution_m": 1.0,
        "metric_crs": "EPSG:26911",
        "slope_threshold_pct": 5.0,
        "resolved_slope_comparison": "le",
        "minimum_polygon_area_m2": 0.0,
    }
    mismatches: dict[str, tuple[Any, Any]] = {}
    for key, expected in canonical_arguments.items():
        observed = arguments.get(key)
        if isinstance(expected, float):
            try:
                matches = not isinstance(observed, bool) and math.isclose(
                    float(observed), expected, rel_tol=0.0, abs_tol=1e-12
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = observed == expected
        if not matches:
            mismatches[key] = (observed, expected)
    if mismatches:
        raise ContractError(f"Corrected terrain arguments differ from the canonical manuscript contract: {mismatches}.")
    if extra.get("dem_grid_validation") != CORRECTED_DEM_GRID_CONTRACT:
        raise ContractError("Corrected terrain lacks all-request-pixel DEM validation.")
    try:
        cells_valid = int(extra.get("request_cells_valid"))
        cells_expected = int(extra.get("request_cells_expected"))
        pixels_valid = int(extra.get("request_polygon_pixel_centres_valid"))
        pixels_expected = int(extra.get("request_polygon_pixel_centres_expected"))
        nodata = float(extra.get("dem_nodata_value"))
        mosaic_nodata = float(extra.get("dem_mosaic_nodata"))
        mosaic_resolution = float(extra.get("dem_mosaic_resolution_m"))
    except (TypeError, ValueError) as exc:
        raise ContractError("Corrected terrain has malformed grid/nodata coverage metadata.") from exc
    if cells_valid != 1280 or cells_expected != 1280:
        raise ContractError(f"Corrected terrain validates {cells_valid}/{cells_expected} request cells, not 1280/1280.")
    if pixels_expected <= 0 or pixels_valid != pixels_expected:
        raise ContractError(
            f"Corrected terrain does not validate every request-polygon pixel ({pixels_valid}/{pixels_expected})."
        )
    if not math.isclose(nodata, -9999.0, rel_tol=0.0, abs_tol=1e-9) or not math.isclose(
        mosaic_nodata, -9999.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ContractError("Corrected terrain DEM/output mosaic nodata must be -9999.")
    if extra.get("dem_valid_elevation_range_m") != [100.0, 10000.0]:
        raise ContractError("Corrected terrain does not record the accepted physical elevation range.")
    if (
        extra.get("chunk_normalization_contract") != "provider-response-to-target-crs-one-metre-aligned-bilinear"
        or extra.get("chunk_normalization_resampling") != "bilinear"
    ):
        raise ContractError(
            "Corrected terrain does not prove that each provider response was normalized "
            "to the signed target lattice before mosaicking."
        )
    if (
        extra.get("dem_mosaic_crs") != "EPSG:26911"
        or not math.isclose(mosaic_resolution, 1.0, rel_tol=0.0, abs_tol=1e-9)
        or extra.get("dem_mosaic_resampling") != "nearest"
        or extra.get("dem_mosaic_target_aligned_pixels") is not True
    ):
        raise ContractError("Corrected terrain mosaic does not use the canonical CRS/lattice/resampling contract.")


def validate_corrected_corridor_grid(
    extra: Mapping[str, Any],
    *,
    expected_band_names: Sequence[str] = ("R", "G", "B", "N"),
) -> tuple[str, float]:
    """Validate the dependency-free portion of the corrected raster contract.

    Callers that open the raster must additionally compare its observed CRS and
    affine transform with the returned manifest values.  Keeping these checks
    here prevents later stages from accepting pre-projection corridor manifests
    or the former ambiguous zero-nodata convention.
    """

    if not isinstance(extra, Mapping):
        raise ContractError("Corrected corridor manifest has no structured extra metadata.")
    if extra.get("input_raster_provenance") != "verified":
        raise ContractError("Corrected corridor imagery is not bound to a verified export-family mosaic.")
    if extra.get("input_mosaic_pixel_valid_contract") != CORRECTED_MOSAIC_PIXEL_CONTRACT:
        raise ContractError("Corrected corridor does not transitively bind the pixel-valid mosaic contract.")
    mosaic_manifest_sha256 = extra.get("input_mosaic_manifest_sha256")
    if not isinstance(mosaic_manifest_sha256, str) or len(mosaic_manifest_sha256) != 64:
        raise ContractError("Corrected corridor lacks the bound pixel-valid mosaic-manifest checksum.")
    if extra.get("raster_output_band_names") != list(expected_band_names):
        raise ContractError(
            "Corrected corridor manifest lacks the required ordered band identity: "
            f"expected {list(expected_band_names)}."
        )
    raster_crs = extra.get("raster_output_crs")
    metric_crs = extra.get("metric_crs")
    if not isinstance(raster_crs, str) or not raster_crs.strip() or raster_crs != metric_crs:
        raise ContractError(
            "Corrected corridor raster CRS must be the same explicit projected metric_crs used for geometry."
        )
    raw_resolution = extra.get("raster_output_resolution_m")
    try:
        resolution_m = float(raw_resolution)
    except (TypeError, ValueError) as exc:
        raise ContractError("Corrected corridor manifest has no numeric raster_output_resolution_m.") from exc
    if isinstance(raw_resolution, bool) or not math.isfinite(resolution_m) or resolution_m <= 0:
        raise ContractError("Corrected corridor raster resolution must be a finite positive number of metres.")
    if extra.get("raster_resampling") != "nearest":
        raise ContractError("Corrected corridor raster must record nearest-neighbour resampling.")
    raw_nodata = extra.get("raster_nodata_value")
    try:
        nodata = float(raw_nodata)
    except (TypeError, ValueError) as exc:
        raise ContractError("Corrected corridor manifest has no numeric raster nodata value.") from exc
    if isinstance(raw_nodata, bool) or not math.isclose(
        nodata,
        CORRECTED_CORRIDOR_NODATA,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ContractError(
            f"Corrected corridor raster nodata must be {CORRECTED_CORRIDOR_NODATA}, not {raw_nodata!r}."
        )
    return raster_crs, resolution_m


def validate_open_corrected_corridor_raster(
    source: Any,
    extra: Mapping[str, Any],
    *,
    expected_band_names: Sequence[str] = ("R", "G", "B", "N"),
) -> tuple[str, float]:
    """Match an opened rasterio dataset to every recorded corridor-grid field."""

    raster_crs, resolution_m = validate_corrected_corridor_grid(
        extra,
        expected_band_names=expected_band_names,
    )
    if source.crs is None or str(source.crs) != raster_crs:
        raise ContractError(f"Corrected raster CRS {source.crs!s} differs from manifest CRS {raster_crs}.")
    if int(source.count) != len(expected_band_names):
        raise ContractError(f"Corrected raster has {source.count} bands; expected exactly {len(expected_band_names)}.")
    transform = source.transform
    if not (
        math.isclose(abs(float(transform.a)), resolution_m, rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(abs(float(transform.e)), resolution_m, rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(float(transform.b), 0.0, abs_tol=1e-12)
        and math.isclose(float(transform.d), 0.0, abs_tol=1e-12)
    ):
        raise ContractError("Corrected raster transform does not match the manifest's north-up metre grid.")
    observed_geotransform = [
        float(transform.c),
        float(transform.a),
        float(transform.b),
        float(transform.f),
        float(transform.d),
        float(transform.e),
    ]
    recorded_geotransform = extra.get("raster_output_geotransform")
    if not isinstance(recorded_geotransform, list) or len(recorded_geotransform) != 6:
        raise ContractError("Corrected corridor manifest has no six-value observed geotransform.")
    try:
        transform_matches = all(
            math.isclose(observed, float(recorded), rel_tol=0.0, abs_tol=1e-9)
            for observed, recorded in zip(observed_geotransform, recorded_geotransform, strict=True)
        )
    except (TypeError, ValueError):
        transform_matches = False
    if not transform_matches:
        raise ContractError("Corrected raster affine transform differs from the observed transform in its manifest.")
    descriptions = [str(value or "") for value in source.descriptions]
    if descriptions != list(expected_band_names):
        raise ContractError(
            f"Corrected raster band descriptions {descriptions!r} differ from {list(expected_band_names)!r}."
        )
    data_types = [str(value).lower() for value in source.dtypes]
    if data_types != ["float32"] * len(expected_band_names):
        raise ContractError(f"Corrected raster bands must all be Float32; found {data_types!r}.")
    nodata_values = list(source.nodatavals)
    if len(nodata_values) != len(expected_band_names) or any(
        value is None or not math.isclose(float(value), CORRECTED_CORRIDOR_NODATA, rel_tol=0.0, abs_tol=1e-9)
        for value in nodata_values
    ):
        raise ContractError(
            f"Corrected raster bands must all declare nodata {CORRECTED_CORRIDOR_NODATA}; found {nodata_values!r}."
        )
    if extra.get("raster_nodata_values_observed") != [CORRECTED_CORRIDOR_NODATA] * len(expected_band_names):
        raise ContractError("Corrected corridor manifest does not record the observed per-band nodata values.")
    if [str(value).lower() for value in extra.get("raster_data_types_observed", [])] != data_types:
        raise ContractError("Corrected corridor manifest does not record the observed Float32 band data types.")
    return raster_crs, resolution_m


@dataclass(frozen=True)
class BandObservation:
    """Small, serialization-friendly summary of an image tensor."""

    channels: int
    minima: tuple[float, ...]
    maxima: tuple[float, ...]
    color_interpretations: tuple[str, ...] = ()
    band_names: tuple[str, ...] = ()

    @classmethod
    def from_array(
        cls,
        array: Any,
        color_interpretations: Sequence[str] = (),
        band_names: Sequence[str] = (),
    ) -> "BandObservation":
        import numpy as np

        values = np.asarray(array)
        if values.ndim != 3:
            raise ContractError(f"Expected a three-dimensional HWC chip, found shape {values.shape}.")
        minima = tuple(float(np.nanmin(values[..., index])) for index in range(values.shape[-1]))
        maxima = tuple(float(np.nanmax(values[..., index])) for index in range(values.shape[-1]))
        return cls(
            values.shape[-1],
            minima,
            maxima,
            tuple(str(value).lower() for value in color_interpretations),
            tuple(str(value).lower().strip() for value in band_names),
        )


def _canonical_band_name(value: str) -> str | None:
    normalized = value.lower().strip().replace("_", " ").replace("-", " ")
    if normalized in {"r", "red"} or normalized.endswith(".red"):
        return "red"
    if normalized in {"g", "green"} or normalized.endswith(".green"):
        return "green"
    if normalized in {"b", "blue"} or normalized.endswith(".blue"):
        return "blue"
    if normalized in {"n", "nir", "near infrared", "nearinfrared"}:
        return "nir"
    if normalized == "alpha" or normalized.endswith(".alpha"):
        return "alpha"
    return None


def _declared_band_order(observation: BandObservation) -> tuple[str | None, ...]:
    if observation.band_names and any(observation.band_names):
        declared = tuple(_canonical_band_name(value) for value in observation.band_names)
    else:
        declared = tuple(_canonical_band_name(value) for value in observation.color_interpretations)
    return declared if any(value is not None for value in declared) else ()


def validate_band_contract(
    observation: BandObservation,
    contract: str,
    *,
    require_identity: bool = False,
) -> tuple[int, ...]:
    """Validate and return zero-based channel indexes for a declared contract."""

    declared = _declared_band_order(observation)

    if contract == "rgb":
        if observation.channels < 3:
            raise ContractError(f"RGB requires at least three channels; found {observation.channels}.")
        if declared and tuple(declared[:3]) != ("red", "green", "blue"):
            raise ContractError(f"RGB band order must be red/green/blue; observed {declared[:3]}.")
        if require_identity and len(declared) < 3:
            raise ContractError("RGB source bands lack ordered identity metadata.")
        return (0, 1, 2)

    if contract == "rgba-alpha":
        if observation.channels != 4:
            raise ContractError(f"Historical RGBA requires exactly four channels; found {observation.channels}.")
        fourth_is_named_alpha = (
            len(observation.color_interpretations) >= 4 and "alpha" in observation.color_interpretations[3]
        )
        fourth_is_constant = abs(observation.maxima[3] - observation.minima[3]) < 1e-12
        if not (fourth_is_named_alpha or fourth_is_constant):
            raise ContractError("The fourth channel is not identifiable as alpha; refuse historical RGBA mode.")
        return (0, 1, 2, 3)

    if contract == "rgbnir":
        if observation.channels < 4:
            raise ContractError(f"RGB+NIR requires at least four channels; found {observation.channels}.")
        if len(observation.color_interpretations) >= 4 and "alpha" in observation.color_interpretations[3]:
            raise ContractError("Band 4 is alpha, not NIR.")
        if declared and tuple(declared[:4]) != ("red", "green", "blue", "nir"):
            raise ContractError(f"RGB+NIR band order must be red/green/blue/NIR; observed {declared[:4]}.")
        if require_identity and len(declared) < 4:
            raise ContractError("RGB+NIR source bands lack ordered identity metadata.")
        if abs(observation.maxima[3] - observation.minima[3]) < 1e-12:
            raise ContractError("Band 4 is constant and cannot be accepted as NIR.")
        return (0, 1, 2, 3)

    raise ContractError(f"Unknown band contract: {contract!r}.")


def normalize_chip(array: Any, strategy: str) -> Any:
    """Apply an explicit radiometric strategy to an HWC array.

    `historical-double-divide` reproduces the notebook defect and must only be
    used for artifact comparison. `unit-rgb` is a generic three-band helper
    retained for contract tests; canonical corrected runs use RGB+NIR through
    the stage-specific `unit-multiband` path.
    """

    import numpy as np

    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 3:
        raise ContractError(f"Expected HWC input, found shape {values.shape}.")

    if strategy == "historical-double-divide":
        observation = BandObservation.from_array(values)
        validate_band_contract(observation, "rgba-alpha")
        if max(observation.maxima[:3]) > 1.01:
            raise ContractError(
                "Historical double-divide expects Earth Engine RGB already scaled to 0-1; "
                "the observed RGB exceeds that range."
            )
        return values / 255.0

    if strategy == "unit-rgb":
        if values.shape[-1] < 3:
            raise ContractError("unit-rgb requires at least three channels.")
        rgb = values[..., :3]
        lower = float(np.nanmin(rgb))
        upper = float(np.nanmax(rgb))
        if lower < 0:
            raise ContractError(f"RGB contains negative values (minimum {lower}).")
        if upper <= 1.01:
            return rgb
        if upper <= 255.0:
            return rgb / 255.0
        raise ContractError(f"RGB maximum {upper} is outside the supported unit/byte range.")

    raise ContractError(f"Unknown normalization strategy: {strategy!r}.")


def parse_chip_position(chip_id: str) -> tuple[int, int]:
    """Parse and return the historical `<column>_<row>` chip identifier."""

    stem = Path(chip_id).stem
    pieces = stem.split("_")
    if len(pieces) != 2:
        raise ContractError(f"Chip ID must have '<column>_<row>' form: {chip_id!r}.")
    try:
        return int(pieces[0]), int(pieces[1])
    except ValueError as exc:
        raise ContractError(f"Chip coordinates must be integers: {chip_id!r}.") from exc


def spatial_group_id(chip_id: str, group_pixels: int) -> str:
    """Group neighboring chips before partitioning to reduce spatial leakage."""

    if group_pixels <= 0:
        raise ContractError("group_pixels must be positive.")
    column, row = parse_chip_position(chip_id)
    return f"c{column // group_pixels:05d}_r{row // group_pixels:05d}"


def partition_for_group(
    group_id: str,
    *,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> str:
    """Assign a stable spatial group to train, validation, or test."""

    if not 0 < train_fraction < 1:
        raise ContractError("train_fraction must be between zero and one.")
    if not 0 <= validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ContractError("validation_fraction must leave a positive test fraction.")
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train_fraction:
        return "train"
    if value < train_fraction + validation_fraction:
        return "validation"
    return "test"


def build_spatial_partitions(
    chip_ids: Iterable[str],
    *,
    seed: int,
    group_pixels: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, tuple[str, str]]:
    """Return chip ID -> (spatial group, partition), deterministically."""

    assignments: dict[str, tuple[str, str]] = {}
    groups: dict[str, str] = {}
    for chip_id in sorted(set(chip_ids)):
        group_id = spatial_group_id(chip_id, group_pixels)
        partition = groups.setdefault(
            group_id,
            partition_for_group(
                group_id,
                seed=seed,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
            ),
        )
        assignments[chip_id] = (group_id, partition)
    return assignments


def validate_partition_rows(rows: Iterable[Mapping[str, Any]], *, corrected: bool) -> None:
    """Assert unique chips and group-level partition disjointness."""

    chip_partitions: dict[str, str] = {}
    group_partitions: dict[str, str] = {}
    seen_partitions: set[str] = set()
    for row in rows:
        chip_id = str(row["chip_id"])
        partition = str(row["partition"])
        group_id = str(row.get("spatial_group", chip_id))
        if chip_id in chip_partitions:
            raise ContractError(f"Duplicate chip in split manifest: {chip_id}.")
        chip_partitions[chip_id] = partition
        previous = group_partitions.setdefault(group_id, partition)
        if previous != partition:
            raise ContractError(f"Spatial group {group_id} crosses {previous}/{partition} partitions.")
        seen_partitions.add(partition)

    if not chip_partitions:
        raise ContractError("Split manifest is empty.")
    if corrected:
        unknown = seen_partitions.difference(PARTITIONS)
        if unknown:
            raise ContractError(f"Corrected split contains unknown partitions: {sorted(unknown)}.")
        missing = set(PARTITIONS).difference(seen_partitions)
        if missing:
            raise ContractError(f"Corrected split is missing partitions: {sorted(missing)}.")
    elif seen_partitions != {"resubstitution"}:
        raise ContractError("Historical reconstruction rows must be labeled only as resubstitution.")


def capacity_spaces(acres: float, spaces_per_acre: float) -> float:
    """Convert area only when the analyst supplies a positive rate."""

    if acres < 0:
        raise ContractError("Parking area cannot be negative.")
    if spaces_per_acre <= 0:
        raise ContractError("spaces_per_acre must be explicitly supplied and positive.")
    return acres * spaces_per_acre


def require_unique_nonempty_names(values: Iterable[Any], field: str) -> list[str]:
    """Reject missing or duplicate reach labels such as historical `area_id`."""

    names = [str(value).strip() for value in values]
    if any(not value or value.lower() == "nan" for value in names):
        raise ContractError(f"{field} contains empty values.")
    duplicates = sorted({value for value in names if names.count(value) > 1})
    if duplicates:
        raise ContractError(f"{field} must be unique; duplicates: {duplicates}.")
    return names


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def is_ignored_metadata_path(path: Path) -> bool:
    """Return true for platform metadata that is not part of an analysis artifact."""

    return any(part == ".DS_Store" or part.startswith("._") for part in path.parts)


def _digest_members(members: Sequence[Path], root: Path) -> tuple[str, int, list[dict[str, Any]]]:
    digest = hashlib.sha256()
    total_size = 0
    records: list[dict[str, Any]] = []
    for member in members:
        relative = member.relative_to(root).as_posix()
        size = member.stat().st_size
        checksum = sha256_file(member)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(checksum.encode("ascii"))
        digest.update(b"\n")
        total_size += size
        records.append({"path": relative, "size_bytes": size, "sha256": checksum})
    return digest.hexdigest(), total_size, records


def describe_artifact(file_path: Path) -> dict[str, Any]:
    """Describe and content-hash a file, shapefile family, or directory tree."""

    resolved = file_path.resolve(strict=False)
    item: dict[str, Any] = {"path": str(resolved), "exists": resolved.exists()}
    if resolved.is_dir():
        members = sorted(
            (
                candidate
                for candidate in resolved.rglob("*")
                if candidate.is_file()
                and not is_ignored_metadata_path(candidate.relative_to(resolved))
            ),
            key=lambda candidate: candidate.relative_to(resolved).as_posix(),
        )
        checksum, total_size, records = _digest_members(members, resolved)
        item.update(
            {
                "artifact_type": "directory-tree",
                "file_count": len(records),
                "size_bytes": total_size,
                "sha256": checksum,
                "members": records,
            }
        )
    elif resolved.is_file() and resolved.suffix.lower() == ".shp":
        stem = resolved.stem.lower()
        accepted_names = {
            f"{stem}.shp",
            f"{stem}.shx",
            f"{stem}.dbf",
            f"{stem}.prj",
            f"{stem}.cpg",
            f"{stem}.qix",
            f"{stem}.sbn",
            f"{stem}.sbx",
            f"{stem}.shp.xml",
        }
        members = sorted(
            (
                candidate
                for candidate in resolved.parent.iterdir()
                if candidate.is_file() and candidate.name.lower() in accepted_names
            ),
            key=lambda candidate: candidate.name.lower(),
        )
        checksum, total_size, records = _digest_members(members, resolved.parent)
        item.update(
            {
                "artifact_type": "shapefile-bundle",
                "file_count": len(records),
                "size_bytes": total_size,
                "sha256": checksum,
                "members": records,
            }
        )
    elif resolved.is_file():
        item.update({"size_bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)})
    return item


def _manifest_workspace_root(workflow_root: Path) -> Path:
    """Return the portable project scope for one workflow checkout.

    In this repository ``analysis/working`` and ``manuscript`` share a project
    root.  Tests and other callers that use a generic workflow root retain that
    root as their portable scope.
    """

    resolved = workflow_root.resolve(strict=False)
    if resolved.name == "working" and resolved.parent.name == "analysis":
        return resolved.parent.parent
    return resolved


def _portable_path_value(value: Path, *, manifest_path: Path, workspace_root: Path) -> tuple[str, str]:
    resolved = value.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(workspace_root)
    except ValueError:
        return str(resolved), "absolute-external"
    return os.path.relpath(resolved, start=manifest_path.parent), "manifest-relative"


def _manifest_artifact_record(
    value: Path,
    *,
    manifest_path: Path,
    workspace_root: Path,
) -> dict[str, Any]:
    record = describe_artifact(value)
    portable, mode = _portable_path_value(
        value,
        manifest_path=manifest_path,
        workspace_root=workspace_root,
    )
    record["path"] = portable
    record["path_mode"] = mode
    return record


def _json_manifest_value(value: Any, *, manifest_path: Path, workspace_root: Path) -> Any:
    """Recursively serialize CLI arguments while preserving portable paths."""

    if isinstance(value, Path):
        return _portable_path_value(
            value,
            manifest_path=manifest_path,
            workspace_root=workspace_root,
        )[0]
    if isinstance(value, Mapping):
        return {
            str(key): _json_manifest_value(
                item,
                manifest_path=manifest_path,
                workspace_root=workspace_root,
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            _json_manifest_value(
                item,
                manifest_path=manifest_path,
                workspace_root=workspace_root,
            )
            for item in value
        ]
    return value


def load_run_manifest(
    path: Path,
    *,
    expected_stage: str | None = None,
    expected_contract: str | None = None,
) -> dict[str, Any]:
    """Load a run manifest and enforce its declared stage/contract."""

    if not path.is_file():
        raise FileNotFoundError(f"Run manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot read run manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"Run manifest must contain a JSON object: {path}")
    if expected_stage is not None and payload.get("stage") != expected_stage:
        raise ContractError(
            f"Run manifest stage mismatch for {path}: expected {expected_stage!r}, found {payload.get('stage')!r}."
        )
    if expected_contract is not None and payload.get("contract") != expected_contract:
        raise ContractError(
            f"Run manifest contract mismatch for {path}: expected {expected_contract!r}, "
            f"found {payload.get('contract')!r}."
        )
    # Runtime-only context lets relative artifact records survive moving the
    # whole repository.  It is intentionally absent from the JSON on disk.
    payload["_manifest_path"] = str(path.expanduser().resolve(strict=False))
    return payload


def resolve_manifest_artifact_path(
    payload: Mapping[str, Any],
    record: Mapping[str, Any],
) -> Path:
    """Resolve an artifact record under schema-1 absolute or schema-2 portable paths."""

    raw = str(record.get("path", "")).strip()
    if not raw:
        raise ContractError("Run-manifest artifact record has no path.")
    mode = record.get("path_mode", "absolute")
    if mode == "manifest-relative":
        manifest_path = payload.get("_manifest_path")
        if not manifest_path:
            raise ContractError("Portable run-manifest records require a payload loaded with load_run_manifest().")
        return (Path(str(manifest_path)).parent / raw).resolve(strict=False)
    if mode in {"absolute", "absolute-external"}:
        return Path(raw).expanduser().resolve(strict=False)
    raise ContractError(f"Unknown run-manifest path mode: {mode!r}.")


def verify_manifest_artifact(
    payload: Mapping[str, Any],
    artifact: Path,
    *,
    section: str = "outputs",
) -> dict[str, Any]:
    """Verify one input/output still matches its content hash in a manifest."""

    if section not in {"inputs", "outputs"}:
        raise ContractError(f"Manifest section must be inputs or outputs, not {section!r}.")
    resolved = artifact.resolve(strict=False)
    records = payload.get(section)
    if not isinstance(records, list):
        raise ContractError(f"Run manifest has no {section!r} artifact list.")
    matches = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        try:
            recorded_path = resolve_manifest_artifact_path(payload, record)
        except ContractError:
            continue
        if recorded_path == resolved:
            matches.append(record)
    if len(matches) != 1 or not matches[0].get("sha256"):
        raise ContractError(
            f"Run manifest does not identify exactly one checksummed {section} artifact for {resolved}."
        )
    observed = describe_artifact(resolved)
    if not observed.get("exists") or not observed.get("sha256"):
        raise ContractError(f"Manifest artifact is missing or unreadable: {resolved}")
    if observed["sha256"] != matches[0]["sha256"]:
        raise ContractError(
            f"Artifact checksum differs from its run manifest for {resolved}: "
            f"expected {matches[0]['sha256']}, observed {observed['sha256']}."
        )
    return dict(matches[0])


def refuse_existing_output(path: Path, *, allow_existing: bool = False) -> None:
    """Prevent a reconstruction run from overwriting historical/curated data."""

    if path.exists() and not allow_existing:
        raise FileExistsError(f"Output already exists; choose a new run path: {path}")


def _git_state(root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout.strip()
        )
        return {"revision": revision, "dirty": dirty}
    except subprocess.TimeoutExpired:
        return {"revision": None, "dirty": None, "error": "git inspection timed out after 20 seconds"}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def write_run_manifest(
    path: Path,
    *,
    workflow_root: Path,
    stage: str,
    contract: str,
    arguments: Mapping[str, Any],
    inputs: Sequence[Path] = (),
    outputs: Sequence[Path] = (),
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write a checksummed record for a completed stage.

    Shapefiles are hashed as multi-file bundles, and directory inputs are
    hashed recursively. This prevents a manifest from integrity-binding only a
    ``.shp`` shell or merely recording a chip-directory pathname.
    """

    path = path.expanduser().resolve(strict=False)
    refuse_existing_output(path)
    workspace_root = _manifest_workspace_root(workflow_root)

    payload = {
        "schema_version": 2,
        "created_utc": datetime.now(UTC).isoformat(),
        "stage": stage,
        "contract": contract,
        "path_contract": {
            "repository_paths": "relative-to-manifest-parent",
            "external_paths": "absolute",
        },
        "arguments": _json_manifest_value(
            arguments,
            manifest_path=path,
            workspace_root=workspace_root,
        ),
        "python": {"version": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "git": _git_state(workflow_root),
        "inputs": [
            _manifest_artifact_record(
                value,
                manifest_path=path,
                workspace_root=workspace_root,
            )
            for value in inputs
        ],
        "outputs": [
            _manifest_artifact_record(
                value,
                manifest_path=path,
                workspace_root=workspace_root,
            )
            for value in outputs
        ],
        "extra": _json_manifest_value(
            dict(extra or {}),
            manifest_path=path,
            workspace_root=workspace_root,
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
