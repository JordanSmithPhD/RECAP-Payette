#!/usr/bin/env python3
"""Validate curated SAM labels, schema, areas, counts, and proposal lineage.

``exact`` and ``subset`` cover simple one-proposal-to-one-label curation.
``adjudicated`` uses a separate CSV/Parquet edge table so merges, splits,
geometry edits, relabels, and manual additions remain auditable.  ``legacy``
can inspect the surviving historical GeoPackage, which lacks proposal IDs and
edit records, but reports that gap instead of inventing provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


WORKING_ROOT = Path(__file__).resolve().parents[1]
if str(WORKING_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKING_ROOT))

from project_paths import CURATED_CANDIDATES, SAM_RAW_PROPOSALS, WORKFLOW_ROOT  # noqa: E402
from pipeline.parking_common import (  # noqa: E402
    CLASS_NAMES,
    ContractError,
    EXPECTED_HISTORICAL_LABEL_COUNTS,
    load_run_manifest,
    refuse_existing_output,
    verify_manifest_artifact,
    write_run_manifest,
)


ADJUDICATED_EDIT_TYPES = frozenset(
    {"unchanged", "relabeled", "merge", "split", "geometry_edit", "manual_addition", "rejected"}
)
ADJUDICATION_DECISIONS = frozenset({"include", "exclude"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate curated candidate-label schema, projected areas, class counts, and raw-proposal lineage."
    )
    parser.add_argument("--labels", type=Path, default=CURATED_CANDIDATES)
    parser.add_argument("--raw-proposals", type=Path, default=SAM_RAW_PROPOSALS)
    parser.add_argument(
        "--raw-manifest",
        type=Path,
        default=None,
        help=(
            "02_generate_sam_candidates run manifest. Non-legacy validation defaults to "
            "<raw-proposals-stem>.run.json and requires a checksum match."
        ),
    )
    parser.add_argument(
        "--lineage-mode",
        choices=("exact", "subset", "adjudicated", "legacy"),
        default="legacy",
        help=(
            "Exact requires identical proposal_id sets and geometry; subset permits reviewed proposal removal; "
            "adjudicated validates a CSV/Parquet edge table for merges, splits, edits, and additions; legacy "
            "allows missing raw proposals/IDs but records the provenance gap. The default audits the surviving "
            "historical artifact; corrected runs should explicitly select exact, subset, or adjudicated."
        ),
    )
    parser.add_argument(
        "--lineage-table",
        type=Path,
        default=None,
        help="CSV or Parquet curation-edge table; required with --lineage-mode adjudicated.",
    )
    parser.add_argument(
        "--count-contract",
        choices=("historical", "schema-only"),
        default="historical",
        help="Historical requires the recovered 18,217-row class distribution; schema-only checks IDs and coverage.",
    )
    parser.add_argument("--class-field", default="class_id")
    parser.add_argument("--proposal-id-field", default="proposal_id")
    parser.add_argument(
        "--curated-id-field",
        default="curated_id",
        help="Unique curated-record ID field in the label layer.",
    )
    parser.add_argument(
        "--lineage-curated-id-field",
        default="curated_id",
        help="Curated-record ID field in the adjudicated lineage table.",
    )
    parser.add_argument(
        "--lineage-parent-id-field",
        default="parent_proposal_id",
        help="Raw proposal parent-ID field in the adjudicated lineage table.",
    )
    parser.add_argument("--lineage-edit-type-field", default="edit_type")
    parser.add_argument("--lineage-reviewer-field", default="reviewer")
    parser.add_argument("--lineage-decision-field", default="decision")
    parser.add_argument("--area-field", default="area_m2")
    parser.add_argument(
        "--metric-crs",
        default="EPSG:32611",
        help="Projected CRS used to independently recalculate square-metre areas.",
    )
    parser.add_argument("--area-relative-tolerance", type=float, default=0.01)
    parser.add_argument("--area-absolute-tolerance-m2", type=float, default=1.0)
    parser.add_argument(
        "--geometry-tolerance-m",
        type=float,
        default=0.01,
        help="Maximum raw/curated geometry-coordinate difference after projected reprojection.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Validation JSON path; defaults beside the curated labels.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Run-manifest path; defaults beside the validation report.",
    )
    return parser


def _add_issue(issues: list[dict[str, Any]], code: str, message: str, **details: Any) -> None:
    issue: dict[str, Any] = {"code": code, "message": message}
    issue.update(details)
    issues.append(issue)


def _class_counts(values: Any) -> dict[int, int]:
    return dict(sorted(Counter(int(value) for value in values).items()))


def _require_complete_raw_sam_manifest(payload: dict[str, Any]) -> None:
    """Reject corrected label lineage derived from an incomplete SAM tile run."""

    extra = payload.get("extra", {})
    raw_count = extra.get("tile_error_count")
    if isinstance(raw_count, bool) or not isinstance(raw_count, int):
        raise ContractError("Raw SAM manifest tile_error_count must be an integer.")
    error_count = raw_count
    if error_count < 0:
        raise ContractError("Raw SAM manifest tile_error_count cannot be negative.")
    if error_count > 0:
        raise ContractError(
            f"Raw SAM proposal run recorded {error_count} tile error(s); incomplete proposals cannot be curated."
        )
    expected_domain = "source-validity-plus-finite-radiometry-intersect-slope-le5-on-source-grid"
    if extra.get("proposal_domain_contract") != expected_domain:
        raise ContractError("Raw SAM manifest lacks the corrected source-validity and slope-domain contract.")
    source_valid = extra.get("source_valid_pixel_count_across_eligible_tiles")
    proposal_domain = extra.get("proposal_domain_pixel_count")
    for label, value in (
        ("source_valid_pixel_count_across_eligible_tiles", source_valid),
        ("proposal_domain_pixel_count", proposal_domain),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ContractError(f"Raw SAM manifest {label} must be a positive integer.")
    if proposal_domain > source_valid:
        raise ContractError("Raw SAM proposal-domain pixels exceed source-valid pixels.")


def _geometry_findings(frame: Any, *, label: str, issues: list[dict[str, Any]]) -> Any:
    missing = frame.geometry.isna()
    empty = frame.geometry.is_empty.fillna(True)
    invalid = (~frame.geometry.is_valid).fillna(True)
    if bool(missing.any()):
        _add_issue(issues, f"{label}_null_geometry", f"{int(missing.sum())} {label} geometries are null.")
    if bool(empty.any()):
        _add_issue(issues, f"{label}_empty_geometry", f"{int(empty.sum())} {label} geometries are empty.")
    if bool(invalid.any()):
        _add_issue(issues, f"{label}_invalid_geometry", f"{int(invalid.sum())} {label} geometries are invalid.")
    return ~(missing | empty | invalid)


def _clean_text(series: Any) -> Any:
    return series.fillna("").astype(str).str.strip()


def _normalize_edit_types(series: Any) -> Any:
    return _clean_text(series).str.lower().str.replace(r"[-\s]+", "_", regex=True)


def _read_lineage_table(path: Path, pandas_module: Any) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pandas_module.read_csv(path, dtype="string", keep_default_na=False)
    if suffix in {".parquet", ".pq"}:
        return pandas_module.read_parquet(path)
    raise ValueError(f"Adjudicated lineage must be a CSV or Parquet table, not {path.suffix or '<no extension>'!r}.")


def _validate_adjudicated_rows(
    labels: Any,
    lineage_table: Any,
    *,
    raw_ids: set[str] | None,
    args: argparse.Namespace,
    errors: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """Validate the edge-table contract and return equality pairs.

    Included records use one row per curated-to-parent edge. A merge therefore
    has multiple rows for one curated ID; a split has one parent connected to
    multiple curated IDs. Manual additions have a blank parent. Every raw
    proposal absent from included edges has one explicit rejected/exclude row
    with a blank curated ID.
    """

    required_table_fields = (
        args.lineage_curated_id_field,
        args.lineage_parent_id_field,
        args.lineage_edit_type_field,
        args.lineage_reviewer_field,
        args.lineage_decision_field,
    )
    missing_table_fields = [field for field in required_table_fields if field not in lineage_table.columns]
    summary: dict[str, Any] = {
        "table_row_count": int(len(lineage_table)),
        "required_fields": list(required_table_fields),
        "allowed_edit_types": sorted(ADJUDICATED_EDIT_TYPES),
        "allowed_decisions": sorted(ADJUDICATION_DECISIONS),
    }
    if args.curated_id_field not in labels.columns:
        _add_issue(
            errors,
            "missing_curated_id_field",
            f"Curated labels lack unique ID field {args.curated_id_field!r}.",
        )
    if missing_table_fields:
        _add_issue(
            errors,
            "missing_lineage_fields",
            f"Adjudicated lineage table lacks required fields: {missing_table_fields}.",
            fields=missing_table_fields,
        )
    if args.curated_id_field not in labels.columns or missing_table_fields:
        return summary, []

    label_ids = _clean_text(labels[args.curated_id_field])
    empty_label_ids = label_ids.eq("")
    if bool(empty_label_ids.any()):
        _add_issue(
            errors,
            "empty_curated_id",
            f"{int(empty_label_ids.sum())} curated label IDs are null or empty.",
        )
    duplicate_label_ids = sorted(label_ids[label_ids.ne("") & label_ids.duplicated(keep=False)].unique().tolist())
    if duplicate_label_ids:
        _add_issue(
            errors,
            "duplicate_curated_id",
            f"Curated IDs must be unique; {len(duplicate_label_ids)} IDs are duplicated.",
            sample=duplicate_label_ids[:10],
        )

    curated_ids = _clean_text(lineage_table[args.lineage_curated_id_field])
    parent_ids = _clean_text(lineage_table[args.lineage_parent_id_field])
    edit_types = _normalize_edit_types(lineage_table[args.lineage_edit_type_field])
    reviewers = _clean_text(lineage_table[args.lineage_reviewer_field])
    decisions = _clean_text(lineage_table[args.lineage_decision_field])

    excluded_rows = decisions.eq("exclude")
    included_rows = decisions.eq("include")
    rejected_rows = edit_types.eq("rejected")
    empty_table_ids = curated_ids.eq("") & ~excluded_rows
    if bool(empty_table_ids.any()):
        _add_issue(
            errors,
            "empty_lineage_curated_id",
            f"{int(empty_table_ids.sum())} lineage rows have no curated ID.",
        )
    unknown_edits = sorted(set(edit_types).difference(ADJUDICATED_EDIT_TYPES))
    if unknown_edits:
        _add_issue(
            errors,
            "unknown_lineage_edit_type",
            f"Lineage edit types are not recognized: {unknown_edits}.",
            values=unknown_edits,
        )
    unknown_decisions = sorted(set(decisions).difference(ADJUDICATION_DECISIONS).difference({""}))
    if unknown_decisions:
        _add_issue(
            errors,
            "unknown_lineage_decision",
            f"Lineage decisions must be include/exclude; found {unknown_decisions}.",
            values=unknown_decisions,
        )
    if bool(reviewers.eq("").any()):
        _add_issue(
            errors,
            "empty_lineage_reviewer",
            f"{int(reviewers.eq('').sum())} lineage rows have no reviewer.",
        )
    if bool(decisions.eq("").any()):
        _add_issue(
            errors,
            "empty_lineage_decision",
            f"{int(decisions.eq('').sum())} lineage rows have no decision.",
        )
    invalid_rejection_contract = excluded_rows ^ rejected_rows
    if bool(invalid_rejection_contract.any()):
        _add_issue(
            errors,
            "invalid_rejection_contract",
            "Every exclude decision must use edit_type=rejected, and rejected rows must use decision=exclude.",
            count=int(invalid_rejection_contract.sum()),
        )
    excluded_with_curated_id = excluded_rows & curated_ids.ne("")
    if bool(excluded_with_curated_id.any()):
        _add_issue(
            errors,
            "rejected_row_has_curated_id",
            f"{int(excluded_with_curated_id.sum())} rejected raw proposals incorrectly identify a curated record.",
        )

    edge_rows = lineage_table.copy()
    edge_rows["__curated_id"] = curated_ids
    edge_rows["__parent_id"] = parent_ids
    edge_rows["__edit_type"] = edit_types
    edge_rows["__reviewer"] = reviewers
    edge_rows["__decision"] = decisions
    duplicate_edges = edge_rows.duplicated(subset=["__curated_id", "__parent_id"], keep=False)
    duplicate_edge_count = int(duplicate_edges.sum())
    if duplicate_edge_count:
        sample = (
            edge_rows.loc[duplicate_edges, ["__curated_id", "__parent_id"]]
            .drop_duplicates()
            .head(10)
            .rename(columns={"__curated_id": "curated_id", "__parent_id": "parent_proposal_id"})
            .to_dict(orient="records")
        )
        _add_issue(
            errors,
            "duplicate_lineage_edge",
            f"{duplicate_edge_count} rows participate in duplicate curated-to-parent lineage edges.",
            sample=sample,
        )

    label_set = set(label_ids[label_ids.ne("")])
    table_set = set(curated_ids[curated_ids.ne("")])
    missing_curated = sorted(label_set.difference(table_set))
    extra_curated = sorted(table_set.difference(label_set))
    if missing_curated:
        _add_issue(
            errors,
            "curated_records_missing_lineage",
            f"{len(missing_curated)} curated records have no lineage rows.",
            sample=missing_curated[:10],
        )
    if extra_curated:
        _add_issue(
            errors,
            "lineage_ids_not_in_curated",
            f"{len(extra_curated)} lineage curated IDs are absent from the label layer.",
            sample=extra_curated[:10],
        )

    manual_rows = edit_types.eq("manual_addition")
    manual_with_parents = manual_rows & parent_ids.ne("")
    nonmanual_without_parents = ~manual_rows & parent_ids.eq("")
    if bool(manual_with_parents.any()):
        _add_issue(
            errors,
            "manual_addition_has_parent",
            f"{int(manual_with_parents.sum())} manual-addition rows incorrectly identify a raw parent.",
        )
    if bool(nonmanual_without_parents.any()):
        _add_issue(
            errors,
            "nonmanual_lineage_missing_parent",
            f"{int(nonmanual_without_parents.sum())} non-manual lineage rows have no raw parent ID.",
        )
    referenced_parents = set(parent_ids[included_rows & (~manual_rows) & parent_ids.ne("")])
    rejected_parents = set(parent_ids[excluded_rows & parent_ids.ne("")])
    missing_parents: list[str] = []
    if raw_ids is not None:
        missing_parents = sorted((referenced_parents | rejected_parents).difference(raw_ids))
        if missing_parents:
            _add_issue(
                errors,
                "lineage_parents_not_in_raw",
                f"{len(missing_parents)} non-manual parent IDs do not exist in the raw proposal layer.",
                sample=missing_parents[:10],
            )
        undisposed_parents = sorted(raw_ids.difference(referenced_parents | rejected_parents))
        conflicting_parents = sorted(referenced_parents & rejected_parents)
        if undisposed_parents:
            _add_issue(
                errors,
                "raw_proposals_without_disposition",
                f"{len(undisposed_parents)} raw proposals have neither an included lineage edge nor an explicit rejection.",
                sample=undisposed_parents[:10],
            )
        if conflicting_parents:
            _add_issue(
                errors,
                "raw_proposals_with_conflicting_disposition",
                f"{len(conflicting_parents)} raw proposals are both included and rejected.",
                sample=conflicting_parents[:10],
            )
    else:
        undisposed_parents = []
        conflicting_parents = []

    for field_name, values, code in (
        (args.lineage_edit_type_field, edit_types, "inconsistent_lineage_edit_type"),
        (args.lineage_reviewer_field, reviewers, "inconsistent_lineage_reviewer"),
        (args.lineage_decision_field, decisions, "inconsistent_lineage_decision"),
    ):
        metadata = edge_rows.assign(__value=values)
        inconsistent = metadata.loc[metadata["__curated_id"].ne("")].groupby("__curated_id")["__value"].nunique()
        inconsistent_ids = sorted(inconsistent[inconsistent > 1].index.astype(str).tolist())
        if inconsistent_ids:
            _add_issue(
                errors,
                code,
                f"{field_name!r} must be consistent across every edge for a curated record.",
                sample=inconsistent_ids[:10],
            )

    equality_pairs: list[tuple[str, str]] = []
    split_parent_to_curated: dict[str, set[str]] = {}
    for curated_id in sorted(table_set):
        rows = edge_rows.loc[edge_rows["__curated_id"].eq(curated_id)]
        record_edit_types = set(rows["__edit_type"])
        if len(record_edit_types) != 1:
            continue
        edit_type = next(iter(record_edit_types))
        parents = sorted(set(rows.loc[rows["__parent_id"].ne(""), "__parent_id"]))
        if edit_type == "manual_addition":
            if parents:
                continue
        elif edit_type == "merge":
            if len(parents) < 2:
                _add_issue(
                    errors,
                    "merge_requires_multiple_parents",
                    f"Merge record {curated_id!r} must identify at least two unique raw parents.",
                )
        elif edit_type in {"unchanged", "relabeled", "geometry_edit", "split"}:
            if len(parents) != 1:
                _add_issue(
                    errors,
                    "edit_requires_one_parent",
                    f"{edit_type} record {curated_id!r} must identify exactly one unique raw parent.",
                )
            elif edit_type in {"unchanged", "relabeled"}:
                equality_pairs.append((curated_id, parents[0]))
            elif edit_type == "split":
                split_parent_to_curated.setdefault(parents[0], set()).add(curated_id)

    invalid_split_parents = sorted(
        parent_id for parent_id, child_ids in split_parent_to_curated.items() if len(child_ids) < 2
    )
    if invalid_split_parents:
        _add_issue(
            errors,
            "split_parent_has_one_child",
            f"{len(invalid_split_parents)} split parents identify fewer than two curated children.",
            sample=invalid_split_parents[:10],
        )

    summary.update(
        {
            "label_curated_id_count": len(label_set),
            "table_curated_id_count": len(table_set),
            "duplicate_edge_row_count": duplicate_edge_count,
            "missing_curated_record_count": len(missing_curated),
            "extra_curated_id_count": len(extra_curated),
            "referenced_raw_parent_count": len(referenced_parents),
            "rejected_raw_parent_count": len(rejected_parents),
            "undisposed_raw_parent_count": len(undisposed_parents),
            "conflicting_raw_parent_count": len(conflicting_parents),
            "missing_raw_parent_count": len(missing_parents),
            "manual_addition_row_count": int(manual_rows.sum()),
            "reviewer_count": len(set(reviewers[reviewers.ne("")])),
            "edit_type_counts": {str(key): int(value) for key, value in edit_types.value_counts().sort_index().items()},
            "decision_counts": {str(key): int(value) for key, value in decisions.value_counts().sort_index().items()},
            "geometry_equality_pair_count": len(equality_pairs),
        }
    )
    return summary, equality_pairs


def run(args: argparse.Namespace) -> int:
    labels_path = args.labels.expanduser().resolve(strict=False)
    raw_path = args.raw_proposals.expanduser().resolve(strict=False)
    raw_manifest_path = (
        args.raw_manifest.expanduser().resolve(strict=False)
        if args.raw_manifest is not None
        else (raw_path.with_name(f"{raw_path.stem}.run.json") if args.lineage_mode != "legacy" else None)
    )
    lineage_table_path = (
        args.lineage_table.expanduser().resolve(strict=False) if args.lineage_table is not None else None
    )
    report_path = (
        args.report.expanduser().resolve(strict=False)
        if args.report is not None
        else labels_path.with_name(f"{labels_path.stem}.validation.json")
    )
    manifest_path = (
        args.manifest.expanduser().resolve(strict=False)
        if args.manifest is not None
        else report_path.with_name(f"{report_path.stem}.run.json")
    )

    def report_relative(value: Path | None) -> str | None:
        if value is None:
            return None
        return os.path.relpath(
            value.resolve(strict=False),
            start=report_path.parent.resolve(strict=False),
        )

    if labels_path == raw_path:
        raise ValueError("Curated labels and raw SAM proposals must be distinct files.")
    input_paths = {labels_path, raw_path}
    if raw_manifest_path is not None:
        input_paths.add(raw_manifest_path)
    if lineage_table_path is not None:
        input_paths.add(lineage_table_path)
    if report_path in input_paths or manifest_path in input_paths | {report_path}:
        raise ValueError(
            "Labels, raw proposals, lineage table, validation report, and manifest must use distinct paths."
        )
    if args.lineage_mode == "adjudicated" and lineage_table_path is None:
        raise ValueError("--lineage-table is required with --lineage-mode adjudicated.")
    if args.lineage_mode != "adjudicated" and lineage_table_path is not None:
        raise ValueError("--lineage-table is only used with --lineage-mode adjudicated.")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Curated label GeoPackage does not exist: {labels_path}")
    if args.lineage_mode != "legacy" and not raw_path.is_file():
        # This is a validation result, not an unhandled I/O error, so it is
        # included in the durable report below.
        raw_required_but_missing = True
    else:
        raw_required_but_missing = False
    raw_manifest_payload: dict[str, Any] | None = None
    if args.lineage_mode != "legacy" and not raw_required_but_missing:
        if raw_manifest_path is None or not raw_manifest_path.is_file():
            raise FileNotFoundError(
                f"Non-legacy curation validation requires the raw-proposal run manifest: {raw_manifest_path}"
            )
        raw_manifest_payload = load_run_manifest(
            raw_manifest_path,
            expected_stage="02_generate_sam_candidates",
        )
        if not str(raw_manifest_payload.get("contract", "")).startswith("corrected-"):
            raise ContractError("Non-legacy curation requires corrected, provenance-bound raw proposals.")
        _require_complete_raw_sam_manifest(raw_manifest_payload)
        verify_manifest_artifact(raw_manifest_payload, raw_path)
        if raw_manifest_payload.get("extra", {}).get("input_provenance") != "verified":
            raise ContractError("Raw SAM proposals are not bound to verified stage-01 inputs.")
        if not raw_manifest_payload.get("extra", {}).get("corridor_manifest_sha256"):
            raise ContractError("Raw SAM manifest lacks its canonical corridor ancestry checksum.")
    lineage_table_required_but_missing = bool(
        args.lineage_mode == "adjudicated" and lineage_table_path is not None and not lineage_table_path.is_file()
    )
    if args.area_relative_tolerance < 0 or args.area_absolute_tolerance_m2 < 0:
        raise ValueError("Area tolerances cannot be negative.")
    if args.geometry_tolerance_m < 0:
        raise ValueError("--geometry-tolerance-m cannot be negative.")
    field_names = (
        args.class_field,
        args.proposal_id_field,
        args.curated_id_field,
        args.lineage_curated_id_field,
        args.lineage_parent_id_field,
        args.lineage_edit_type_field,
        args.lineage_reviewer_field,
        args.lineage_decision_field,
        args.area_field,
    )
    if any(not field for field in field_names):
        raise ValueError("Field names cannot be empty.")
    refuse_existing_output(report_path)
    refuse_existing_output(manifest_path)

    try:
        import geopandas as gpd
        import pandas as pd
        from pyproj import CRS
    except ModuleNotFoundError as exc:
        raise RuntimeError("Candidate-label validation requires GeoPandas, pandas, and pyproj.") from exc

    metric_crs = CRS.from_user_input(args.metric_crs)
    if not metric_crs.is_projected:
        raise ValueError(f"--metric-crs must be projected for area and geometry validation: {args.metric_crs}")
    metric_units = {str(axis.unit_name).lower() for axis in metric_crs.axis_info if axis.unit_name}
    if metric_units and not all("met" in unit for unit in metric_units):
        raise ValueError(f"--metric-crs axes must use metres, not {sorted(metric_units)}: {args.metric_crs}")

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    labels = gpd.read_file(labels_path)
    if labels.empty:
        _add_issue(errors, "empty_labels", "The curated label layer has no records.")
    if labels.crs is None:
        _add_issue(errors, "missing_label_crs", "The curated label layer has no CRS.")
    valid_label_geometry = _geometry_findings(labels, label="label", issues=errors)

    numeric_classes = None
    counts: dict[int, int] = {}
    if args.class_field not in labels.columns:
        _add_issue(errors, "missing_class_field", f"Required field {args.class_field!r} is absent.")
    else:
        numeric_classes = pd.to_numeric(labels[args.class_field], errors="coerce")
        nonnumeric = numeric_classes.isna()
        if bool(nonnumeric.any()):
            _add_issue(
                errors,
                "nonnumeric_class_id",
                f"{int(nonnumeric.sum())} class values are missing or nonnumeric.",
            )
        fractional = numeric_classes.notna() & ((numeric_classes % 1).abs() > 1e-12)
        if bool(fractional.any()):
            _add_issue(errors, "fractional_class_id", f"{int(fractional.sum())} class values are not integers.")
        valid_class_rows = numeric_classes.notna() & ~fractional
        if bool(valid_class_rows.any()):
            class_values = numeric_classes.loc[valid_class_rows].astype(int)
            counts = _class_counts(class_values)
            unknown = sorted(set(counts).difference(range(len(CLASS_NAMES))))
            if unknown:
                _add_issue(errors, "unknown_class_id", f"Class IDs fall outside 0--5: {unknown}.", values=unknown)
            missing_positive = sorted(set(range(1, len(CLASS_NAMES))).difference(counts))
            if missing_positive:
                _add_issue(
                    errors,
                    "missing_positive_classes",
                    f"No curated polygons represent positive classes: {missing_positive}.",
                    values=missing_positive,
                )
            if args.count_contract == "historical" and counts != EXPECTED_HISTORICAL_LABEL_COUNTS:
                _add_issue(
                    errors,
                    "historical_count_mismatch",
                    "Observed class counts do not match the recovered 18,217-row historical label artifact.",
                    expected=EXPECTED_HISTORICAL_LABEL_COUNTS,
                    observed=counts,
                )

    area_summary: dict[str, Any] = {"checked": False}
    if labels.crs is not None and bool(valid_label_geometry.any()):
        projected_labels = labels.loc[valid_label_geometry].to_crs(metric_crs)
        computed_area = projected_labels.geometry.area.astype(float)
        area_summary = {
            "checked": args.area_field in labels.columns,
            "metric_crs": metric_crs.to_string(),
            "computed_total_m2": float(computed_area.sum()),
            "minimum_m2": float(computed_area.min()),
            "maximum_m2": float(computed_area.max()),
        }
        if bool((computed_area <= 0).any()):
            _add_issue(errors, "nonpositive_geometry_area", "One or more valid label geometries have nonpositive area.")
        if args.area_field not in labels.columns:
            _add_issue(
                warnings,
                "missing_area_field",
                f"Optional provenance field {args.area_field!r} is absent; projected areas were independently computed.",
            )
        else:
            stored = pd.to_numeric(labels.loc[valid_label_geometry, args.area_field], errors="coerce")
            missing_area = stored.isna()
            if bool(missing_area.any()):
                _add_issue(
                    errors,
                    "invalid_area_field",
                    f"{int(missing_area.sum())} {args.area_field} values are missing or nonnumeric.",
                )
            comparable = ~missing_area
            if bool(comparable.any()):
                observed = stored.loc[comparable].astype(float)
                expected = computed_area.loc[comparable]
                tolerance = expected.abs() * args.area_relative_tolerance + args.area_absolute_tolerance_m2
                mismatch = (observed - expected).abs() > tolerance
                mismatch_count = int(mismatch.sum())
                area_summary["mismatch_count"] = mismatch_count
                if mismatch_count:
                    _add_issue(
                        errors,
                        "metric_area_mismatch",
                        f"{mismatch_count} stored {args.area_field} values disagree with projected geometry area.",
                        relative_tolerance=args.area_relative_tolerance,
                        absolute_tolerance_m2=args.area_absolute_tolerance_m2,
                    )

    lineage: dict[str, Any] = {
        "mode": args.lineage_mode,
        "raw_exists": raw_path.is_file(),
        "lineage_table": report_relative(lineage_table_path),
        "lineage_table_exists": bool(lineage_table_path is not None and lineage_table_path.is_file()),
        "raw_manifest": report_relative(raw_manifest_path),
        "raw_manifest_verified": raw_manifest_payload is not None,
    }
    raw = None
    valid_raw_geometry = None
    raw_id_values = None
    raw_id_set: set[str] | None = None
    duplicate_raw: list[str] = []
    if raw_required_but_missing:
        _add_issue(errors, "missing_raw_proposals", f"Raw proposal GeoPackage is required but missing: {raw_path}")
    elif args.lineage_mode == "legacy" and not raw_path.is_file():
        _add_issue(
            warnings,
            "legacy_lineage_missing",
            "Raw SAM proposals and proposal IDs are unavailable; the surviving manual labels have no auditable edit lineage.",
        )

    if raw_path.is_file():
        raw = gpd.read_file(raw_path)
        lineage["raw_count"] = int(len(raw))
        if raw.empty:
            _add_issue(errors, "empty_raw_proposals", "The raw proposal layer has no records.")
        if raw.crs is None:
            _add_issue(errors, "missing_raw_crs", "The raw proposal layer has no CRS.")
        valid_raw_geometry = _geometry_findings(raw, label="raw", issues=errors)
        if args.class_field in raw.columns:
            _add_issue(
                warnings,
                "raw_contains_label_field",
                f"Raw proposals contain {args.class_field!r}; raw generation should not create manual labels.",
            )
        if args.proposal_id_field in raw.columns:
            raw_id_values = _clean_text(raw[args.proposal_id_field])
            empty_raw_ids = raw_id_values.eq("")
            if bool(empty_raw_ids.any()):
                _add_issue(
                    errors,
                    "empty_raw_proposal_id",
                    f"{int(empty_raw_ids.sum())} raw proposal IDs are null or empty.",
                )
            duplicate_raw = sorted(
                raw_id_values[raw_id_values.ne("") & raw_id_values.duplicated(keep=False)].unique().tolist()
            )
            if duplicate_raw:
                _add_issue(
                    errors,
                    "duplicate_raw_proposal_id",
                    f"Raw proposal IDs are duplicated ({len(duplicate_raw)} distinct IDs).",
                    sample=duplicate_raw[:10],
                )
            raw_id_set = set(raw_id_values[raw_id_values.ne("")])
        else:
            target = errors if args.lineage_mode != "legacy" else warnings
            _add_issue(
                target,
                "missing_raw_proposal_id",
                f"Raw proposals lack required ID field {args.proposal_id_field!r}.",
            )

    if args.lineage_mode == "adjudicated":
        if lineage_table_required_but_missing:
            _add_issue(
                errors,
                "missing_lineage_table",
                f"Adjudicated lineage table is required but missing: {lineage_table_path}",
            )
        elif lineage_table_path is not None:
            try:
                lineage_table = _read_lineage_table(lineage_table_path, pd)
            except Exception as exc:
                _add_issue(
                    errors,
                    "lineage_table_read_error",
                    f"Cannot read adjudicated lineage table {lineage_table_path}: {exc}",
                )
            else:
                adjudicated_summary, equality_pairs = _validate_adjudicated_rows(
                    labels,
                    lineage_table,
                    raw_ids=raw_id_set,
                    args=args,
                    errors=errors,
                )
                lineage.update(adjudicated_summary)

                can_compare = (
                    raw is not None
                    and raw_id_values is not None
                    and not duplicate_raw
                    and args.curated_id_field in labels.columns
                    and labels.crs is not None
                    and raw.crs is not None
                    and bool(valid_label_geometry.any())
                    and valid_raw_geometry is not None
                    and bool(valid_raw_geometry.any())
                )
                checked = 0
                changed = 0
                if can_compare:
                    label_id_values = _clean_text(labels[args.curated_id_field])
                    duplicate_curated = bool((label_id_values.ne("") & label_id_values.duplicated(keep=False)).any())
                    if not duplicate_curated:
                        label_geometry = labels.loc[valid_label_geometry].copy()
                        raw_geometry = raw.loc[valid_raw_geometry].copy()
                        label_geometry[args.curated_id_field] = label_id_values.loc[valid_label_geometry]
                        raw_geometry[args.proposal_id_field] = raw_id_values.loc[valid_raw_geometry]
                        label_geometry = label_geometry.to_crs(metric_crs).set_index(args.curated_id_field).geometry
                        raw_geometry = raw_geometry.to_crs(metric_crs).set_index(args.proposal_id_field).geometry
                        for curated_id, parent_id in equality_pairs:
                            if curated_id not in label_geometry.index or parent_id not in raw_geometry.index:
                                continue
                            checked += 1
                            if not label_geometry.loc[curated_id].equals_exact(
                                raw_geometry.loc[parent_id], args.geometry_tolerance_m
                            ):
                                changed += 1
                lineage["geometry_equality_checked_count"] = checked
                lineage["unchanged_or_relabeled_geometry_mismatch_count"] = changed
                if changed:
                    _add_issue(
                        errors,
                        "unchanged_or_relabeled_geometry_changed",
                        f"{changed} unchanged/relabeled curated geometries differ from their single raw parent.",
                        tolerance_m=args.geometry_tolerance_m,
                    )
    elif raw is not None:
        label_has_ids = args.proposal_id_field in labels.columns
        raw_has_ids = args.proposal_id_field in raw.columns
        if not label_has_ids or not raw_has_ids:
            target = errors if args.lineage_mode != "legacy" else warnings
            _add_issue(
                target,
                "missing_proposal_id",
                f"Field {args.proposal_id_field!r} must be retained in both raw and curated layers for lineage.",
                labels_have_field=label_has_ids,
                raw_have_field=raw_has_ids,
            )
        else:
            label_ids = _clean_text(labels[args.proposal_id_field])
            empty_label_ids = label_ids.eq("")
            if bool(empty_label_ids.any()):
                _add_issue(
                    errors,
                    "empty_curated_proposal_id",
                    f"{int(empty_label_ids.sum())} curated proposal IDs are null or empty.",
                )
            duplicate_labels = sorted(label_ids[label_ids.ne("") & label_ids.duplicated(keep=False)].unique().tolist())
            if duplicate_labels:
                _add_issue(
                    errors,
                    "duplicate_curated_proposal_id",
                    f"Curated proposal IDs are duplicated ({len(duplicate_labels)} distinct IDs).",
                    sample=duplicate_labels[:10],
                )
            raw_set = raw_id_set or set()
            label_set = set(label_ids[label_ids.ne("")])
            added = sorted(label_set.difference(raw_set))
            removed = sorted(raw_set.difference(label_set))
            lineage.update(
                {
                    "curated_count": int(len(labels)),
                    "added_id_count": len(added),
                    "removed_id_count": len(removed),
                    "added_id_sample": added[:10],
                    "removed_id_sample": removed[:10],
                }
            )
            if added:
                _add_issue(
                    errors,
                    "curated_ids_not_in_raw",
                    f"{len(added)} curated proposal IDs do not exist in the raw layer.",
                    sample=added[:10],
                )
            if args.lineage_mode == "exact" and removed:
                _add_issue(
                    errors,
                    "raw_ids_missing_from_curated",
                    f"Exact lineage requires every raw proposal; {len(removed)} IDs are missing.",
                    sample=removed[:10],
                )

            if (
                not duplicate_labels
                and not duplicate_raw
                and labels.crs is not None
                and raw.crs is not None
                and bool(valid_label_geometry.any())
                and valid_raw_geometry is not None
                and bool(valid_raw_geometry.any())
            ):
                common = label_set.intersection(raw_set)
                label_geometry = labels.loc[valid_label_geometry].copy()
                raw_geometry = raw.loc[valid_raw_geometry].copy()
                label_geometry[args.proposal_id_field] = label_ids.loc[valid_label_geometry]
                raw_geometry[args.proposal_id_field] = raw_id_values.loc[valid_raw_geometry]
                label_geometry = label_geometry.to_crs(metric_crs).set_index(args.proposal_id_field).geometry
                raw_geometry = raw_geometry.to_crs(metric_crs).set_index(args.proposal_id_field).geometry
                changed = 0
                for proposal_id in common.intersection(label_geometry.index).intersection(raw_geometry.index):
                    if not label_geometry.loc[proposal_id].equals_exact(
                        raw_geometry.loc[proposal_id], args.geometry_tolerance_m
                    ):
                        changed += 1
                lineage["changed_geometry_count"] = changed
                if changed:
                    _add_issue(
                        errors,
                        "curated_geometry_changed",
                        f"{changed} curated geometries differ from their raw proposals beyond tolerance.",
                        tolerance_m=args.geometry_tolerance_m,
                    )

    report: dict[str, Any] = {
        "schema_version": 2,
        "path_contract": "relative-to-report-parent",
        "status": "pass" if not errors else "fail",
        "labels": report_relative(labels_path),
        "raw_proposals": report_relative(raw_path),
        "label_count": int(len(labels)),
        "class_names": {str(index): name for index, name in enumerate(CLASS_NAMES)},
        "class_counts": {str(key): value for key, value in counts.items()},
        "count_contract": args.count_contract,
        "expected_historical_counts": {str(key): value for key, value in EXPECTED_HISTORICAL_LABEL_COUNTS.items()},
        "area": area_summary,
        "lineage": lineage,
        "errors": errors,
        "warnings": warnings,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest_inputs = [labels_path, raw_path]
    if raw_manifest_path is not None:
        manifest_inputs.append(raw_manifest_path)
    if lineage_table_path is not None:
        manifest_inputs.append(lineage_table_path)
    write_run_manifest(
        manifest_path,
        workflow_root=WORKFLOW_ROOT,
        stage="02_validate_candidate_labels",
        contract=f"{args.lineage_mode}-lineage-{args.count_contract}-counts-projected-area",
        arguments=vars(args),
        inputs=manifest_inputs,
        outputs=[report_path],
        extra={
            "validation_status": report["status"],
            "error_count": len(errors),
            "warning_count": len(warnings),
            "corridor_manifest_sha256": (
                raw_manifest_payload.get("extra", {}).get("corridor_manifest_sha256")
                if raw_manifest_payload is not None
                else None
            ),
        },
    )
    print(f"Validation {report['status'].upper()}: {len(errors)} error(s), {len(warnings)} warning(s).")
    print(f"Wrote validation report: {report_path}")
    print(f"Wrote run manifest: {manifest_path}")
    for issue in errors:
        print(f"ERROR [{issue['code']}]: {issue['message']}")
    for issue in warnings:
        print(f"WARNING [{issue['code']}]: {issue['message']}")
    return 0 if not errors else 1


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
