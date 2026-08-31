#!/usr/bin/env python3
"""Run the historical final YOLO estimator as a safe, resumable stage 09.

This public implementation preserves the operational model call, target
classes, and confidence rules documented for the manuscript while adding safe
persistence and failure handling:

* inference is written to checksum-bound, atomically committed shards;
* a rerun resumes only shards that pass their sidecar and row-level checks;
* PIL decode failures are retained as bad images, while model/inference errors
  abort the run instead of silently classifying an entire batch as bad; and
* the combined Parquet and JSON run manifest are written atomically only after
  all expected row-count identities have been verified.

Example (write to a new, versioned directory; never directly over the archive):

    python pipeline/09_run_camera_inference.py \
      --manifest data/intermediate/camera/image_manifest_daylight.parquet \
      --root /path/to/camera/archive \
      --weights /path/to/operational_detector.pt \
      --run-dir outputs/runs/camera/stage09 --device cpu --verbose

The canonical daytime manifest includes per-image SHA-256 values.  They are
verified by default.  ``--no-verify-content-hashes`` is available for an
explicitly lower-assurance exploratory run and is recorded in the manifest.

For very large manifests on external drives, ``--defer-existence-check`` skips
the eager ``Path.exists`` pass (including the existence probe in the legacy
path normalizer). Eligible rows are still counted as filtered inputs, and each
one must pass stat, optional size/content-hash, and PIL checks inside its shard.
A missing source therefore aborts rather than becoming a bad image or silently
disappearing from the output.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import random
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from PIL import Image


WORKING_ROOT = Path(__file__).resolve().parents[1]
if str(WORKING_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKING_ROOT))

import project_paths  # noqa: E402


STAGE = "09_run_full_inference_resumable"
SCHEMA_VERSION = 1
VEHICLE_NAMES = frozenset({"car", "bus", "truck", "motorcycle", "vehicle"})
WATERCRAFT_NAMES = frozenset({"boat", "watercraft"})
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def normalize_manifest_path(value: Any, root: str | Path) -> str:
    """Resolve the repository's portable, archive-relative path contract."""

    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    return str(path if path.is_absolute() else Path(root) / path)


def is_valid_image_path(path: Path) -> bool:
    """Reject metadata, flagged files, and unsupported image extensions."""

    name = path.name
    upper = name.upper()
    return bool(
        name
        and not name.startswith(".")
        and not upper.startswith("FLAG_BAD")
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def extract_site_from_path(value: str | Path) -> str:
    """Infer ``camera_type/site`` from a portable archive-relative path."""

    parts = Path(str(value)).parts
    lower = [part.lower() for part in parts]
    for anchor in ("river", "parking"):
        if anchor in lower:
            index = lower.index(anchor)
            return f"{parts[index]}/{parts[index + 1]}" if index + 1 < len(parts) else parts[index]
    return "unknown"


class Stage09Error(RuntimeError):
    """Raised when a stage-09 integrity or execution contract is violated."""


class InferenceError(Stage09Error):
    """Raised for model loading/prediction failures, never decode failures."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run final Payette YOLO inference in atomic, resumable, verified shards."
    )
    parser.add_argument("--manifest", type=Path, default=project_paths.DAYTIME_MANIFEST)
    parser.add_argument("--root", type=Path, default=project_paths.CAMERA_ROOT)
    parser.add_argument("--weights", type=Path, default=project_paths.FINAL_WEIGHTS)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="New or resumable isolated run directory; outputs are written only beneath it.",
    )
    parser.add_argument("--subset", choices=("all", "river", "parking"), default="all")
    parser.add_argument("--max-images", type=int, default=0, help="Apply the historical pre-filter row cap (0=all).")
    parser.add_argument(
        "--require-daylight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter on a manifest daylight column when present (default: enabled).",
    )
    parser.add_argument(
        "--require-readable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter known unreadable rows when a manifest readable column is present (default: enabled).",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--shard-size",
        "--chunk",
        dest="shard_size",
        type=int,
        default=128,
        help="Rows per atomic shard (historical --chunk default: 128).",
    )
    parser.add_argument("--vehicle-conf", type=float, default=0.25)
    parser.add_argument("--watercraft-conf", type=float, default=0.477)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verify-content-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify each selected image against manifest content_sha256 (default: enabled).",
    )
    parser.add_argument(
        "--defer-existence-check",
        action="store_true",
        help=(
            "Skip the all-files existence prepass. Every eligible row remains in filtered_rows and must later "
            "pass stat/size/hash/PIL verification in its shard; a missing file aborts the run."
        ),
    )
    parser.add_argument("--expected-input-rows", type=int, help="Expected total rows in the source manifest.")
    parser.add_argument(
        "--expected-filtered-rows",
        type=int,
        help=(
            "Expected rows after max/subset/name filters and, unless --defer-existence-check is used, "
            "the eager existence filter."
        ),
    )
    parser.add_argument(
        "--expected-final-rows",
        type=int,
        help="Expected inference rows after verified PIL decode failures are removed.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid4().hex}.tmp")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_parent(path: Path) -> None:
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = _atomic_path(path)
    try:
        partial.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _fsync_file(partial)
        os.replace(partial, path)
        _fsync_parent(path)
    finally:
        if partial.exists():
            partial.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = _atomic_path(path)
    try:
        partial.write_text(text, encoding="utf-8")
        _fsync_file(partial)
        os.replace(partial, path)
        _fsync_parent(path)
    finally:
        if partial.exists():
            partial.unlink()


def _import_arrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover - environment preflight
        raise Stage09Error("Stage 09 requires pyarrow for deterministic Parquet shards.") from exc
    return pa, pq


def output_schema(include_datetime: bool) -> Any:
    pa, _ = _import_arrow()
    fields = [
        pa.field("path", pa.large_string(), nullable=False),
        pa.field("filename", pa.large_string(), nullable=False),
        pa.field("site", pa.large_string()),
    ]
    if include_datetime:
        fields.append(pa.field("datetime_from_filename", pa.timestamp("ms")))
    fields.extend(
        [
            pa.field("count_vehicle", pa.int64(), nullable=False),
            pa.field("count_watercraft", pa.int64(), nullable=False),
            pa.field("count_total", pa.int64(), nullable=False),
            pa.field(
                "detections",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("class_id", pa.int64(), nullable=False),
                            pa.field("class_name", pa.string(), nullable=False),
                            pa.field("conf", pa.float64(), nullable=False),
                            # Match the legacy pandas/pyarrow output type.  Row
                            # construction separately enforces exactly four
                            # xyxy values.
                            pa.field("xyxy", pa.list_(pa.float64()), nullable=False),
                        ]
                    )
                ),
                nullable=False,
            ),
        ]
    )
    return pa.schema(fields)


def atomic_write_parquet(rows: Sequence[Mapping[str, Any]], path: Path, schema: Any) -> None:
    pa, pq = _import_arrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = _atomic_path(path)
    try:
        table = pa.Table.from_pylist(list(rows), schema=schema)
        pq.write_table(table, partial, compression="snappy")
        _fsync_file(partial)
        os.replace(partial, path)
        _fsync_parent(path)
    finally:
        if partial.exists():
            partial.unlink()


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise Stage09Error(f"{label} not found: {resolved}")
    return resolved


def _check_positive_settings(args: argparse.Namespace) -> None:
    for name in ("imgsz", "batch", "shard_size"):
        if int(getattr(args, name)) <= 0:
            raise Stage09Error(f"--{name.replace('_', '-')} must be positive.")
    if int(args.max_images) < 0:
        raise Stage09Error("--max-images cannot be negative.")
    for name in ("vehicle_conf", "watercraft_conf"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise Stage09Error(f"--{name.replace('_', '-')} must be in [0, 1].")
    for name in ("expected_input_rows", "expected_filtered_rows", "expected_final_rows"):
        value = getattr(args, name)
        if value is not None and int(value) < 0:
            raise Stage09Error(f"--{name.replace('_', '-')} cannot be negative.")
    if bool(args.defer_existence_check) and not bool(args.verify_content_hashes):
        raise Stage09Error(
            "--defer-existence-check requires --verify-content-hashes so every deferred source is checksum-verified."
        )


def _require_expected(label: str, observed: int, expected: int | None) -> None:
    if expected is not None and observed != int(expected):
        raise Stage09Error(f"{label} row-count mismatch: expected {int(expected)}, observed {observed}.")


def _stable_scalar(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    try:
        missing = pd.isna(value)
        if not hasattr(missing, "__len__") and bool(missing):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (AttributeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _inventory_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for position, row in frame.iterrows():
        record = {"position": int(position)}
        for column in columns:
            if column in frame.columns:
                record[column] = _stable_scalar(row[column])
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def normalize_manifest_path_deferred(value: Any, root: str | Path) -> str:
    """Resolve a portable path without probing the external filesystem."""

    return normalize_manifest_path(value, root)


def prepare_manifest(
    manifest: Path,
    *,
    root: Path,
    subset: str,
    max_images: int,
    require_daylight: bool = True,
    require_readable: bool = True,
    defer_existence_check: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    try:
        frame = pd.read_parquet(manifest)
    except Exception as exc:
        raise Stage09Error(f"Could not read manifest {manifest}: {exc}") from exc
    if "path" not in frame.columns:
        raise Stage09Error("Manifest parquet must contain a 'path' column.")

    counts = {"manifest_rows": int(len(frame))}
    if max_images > 0:
        frame = frame.head(int(max_images)).copy()
    else:
        frame = frame.copy()
    counts["rows_after_max_images"] = int(len(frame))

    def _boolean_mask(column: str) -> pd.Series:
        values = frame[column]
        if pd.api.types.is_bool_dtype(values):
            if values.isna().any():
                raise Stage09Error(f"Manifest {column} contains missing values.")
            return values.astype(bool)
        normalized = values.astype("string").str.strip().str.lower()
        mapped = normalized.map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
        if mapped.isna().any():
            raise Stage09Error(f"Manifest {column} must contain boolean values.")
        return mapped.astype(bool)

    if require_daylight and "daylight" in frame.columns:
        frame = frame.loc[_boolean_mask("daylight")].copy()
    counts["rows_after_daylight_filter"] = int(len(frame))
    if require_readable and "readable" in frame.columns:
        frame = frame.loc[_boolean_mask("readable")].copy()
    counts["rows_after_readable_filter"] = int(len(frame))

    root_text = str(root)
    normalizer = normalize_manifest_path_deferred if defer_existence_check else normalize_manifest_path
    frame["path"] = [normalizer(value, root_text) for value in frame["path"].astype(str).tolist()]
    if subset in {"river", "parking"}:
        if "camera_type" in frame.columns:
            frame = frame.loc[frame["camera_type"].astype(str).str.casefold().eq(subset)].copy()
        else:
            token = "River" if subset == "river" else "Parking"
            prefix = str(root / token).rstrip(os.sep) + os.sep
            frame = frame.loc[frame["path"].astype(str).str.startswith(prefix)].copy()
    counts["rows_after_subset"] = int(len(frame))

    valid_mask = [is_valid_image_path(Path(value)) for value in frame["path"].astype(str).tolist()]
    frame = frame.loc[valid_mask].copy()
    counts["rows_after_name_filter"] = int(len(frame))

    counts["existence_checks_deferred"] = int(defer_existence_check)
    if defer_existence_check:
        counts["rows_checked_for_existence"] = 0
        counts["rows_missing_at_existence_prepass"] = 0
    else:
        exists_mask: list[bool] = []
        for value in frame["path"].astype(str).tolist():
            try:
                # PIL remains authoritative for deciding whether an existing
                # path is a decodable image.
                exists_mask.append(Path(value).exists())
            except OSError:
                exists_mask.append(False)
        counts["rows_checked_for_existence"] = int(len(exists_mask))
        counts["rows_missing_at_existence_prepass"] = int(len(exists_mask) - sum(exists_mask))
        frame = frame.loc[exists_mask].copy()
    counts["filtered_rows"] = int(len(frame))
    if frame.empty:
        qualifier = "eligible" if defer_existence_check else "valid, existing"
        raise Stage09Error(f"0 {qualifier} images remain after historical filtering.")

    if "site" not in frame.columns and "site_id" in frame.columns:
        frame["site"] = frame["site_id"].astype(str)
    if "site" not in frame.columns:
        frame["site"] = [extract_site_from_path(value) for value in frame["path"].astype(str).tolist()]
    frame = frame.reset_index(drop=True)
    return frame, counts


def validate_deferred_source_metadata(frame: pd.DataFrame) -> None:
    """Require the manifest identities needed for fail-closed deferred I/O."""

    missing_columns = {"file_size", "content_sha256"}.difference(frame.columns)
    if missing_columns:
        raise Stage09Error(
            f"--defer-existence-check requires manifest columns: {sorted(missing_columns)}."
        )
    if frame["file_size"].isna().any():
        raise Stage09Error("--defer-existence-check requires file_size for every filtered source row.")
    numeric_sizes = pd.to_numeric(frame["file_size"], errors="coerce")
    if numeric_sizes.isna().any() or (numeric_sizes < 0).any() or (numeric_sizes % 1 != 0).any():
        raise Stage09Error("--defer-existence-check requires nonnegative integer file_size values.")
    if frame["content_sha256"].isna().any():
        raise Stage09Error("--defer-existence-check requires content_sha256 for every filtered source row.")


def _expected_size(row: pd.Series) -> int | None:
    if "file_size" not in row.index or pd.isna(row["file_size"]):
        return None
    try:
        return int(row["file_size"])
    except (TypeError, ValueError) as exc:
        raise Stage09Error(f"Invalid file_size for {row['path']}: {row['file_size']!r}") from exc


def _expected_content_sha256(row: pd.Series) -> str | None:
    if "content_sha256" not in row.index or pd.isna(row["content_sha256"]):
        return None
    value = str(row["content_sha256"]).strip().lower()
    if not SHA256_RE.fullmatch(value):
        raise Stage09Error(f"Invalid content_sha256 for {row['path']}: {value!r}")
    return value


def verify_and_decode(row: pd.Series, *, verify_content_hash: bool) -> str | None:
    """Validate one source file and return a PIL decode error, if any.

    Existence, size, and content-hash changes are integrity failures.  Only an
    exception raised while PIL decodes verified bytes is a bad-image result.
    """

    path = Path(str(row["path"]))
    try:
        observed_size = path.stat().st_size
    except OSError as exc:
        raise Stage09Error(f"Selected image became unavailable: {path}: {exc}") from exc
    expected_size = _expected_size(row)
    if expected_size is not None and observed_size != expected_size:
        raise Stage09Error(
            f"Image size mismatch for {path}: manifest={expected_size}, observed={observed_size}."
        )

    expected_hash = _expected_content_sha256(row)
    if verify_content_hash and expected_hash is None:
        raise Stage09Error(f"Content-hash verification requested, but content_sha256 is missing for {path}.")

    try:
        if verify_content_hash:
            image_bytes = path.read_bytes()
            observed_hash = hashlib.sha256(image_bytes).hexdigest()
            if observed_hash != expected_hash:
                raise Stage09Error(
                    f"Image SHA-256 mismatch for {path}: manifest={expected_hash}, observed={observed_hash}."
                )
            source: Any = io.BytesIO(image_bytes)
        else:
            source = path
        with Image.open(source) as image:
            image.load()
    except Stage09Error:
        raise
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, Mapping):
        try:
            return {int(class_id): str(name) for class_id, name in names.items()}
        except (TypeError, ValueError) as exc:
            raise InferenceError(f"Model class mapping is invalid: {names!r}") from exc
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return {index: str(name) for index, name in enumerate(names)}
    raise InferenceError(f"Model does not expose a usable class-name mapping: {names!r}")


def _load_model(weights: Path, model_factory: Callable[[str], Any] | None) -> Any:
    try:
        if model_factory is not None:
            return model_factory(str(weights))
        from ultralytics import YOLO

        return YOLO(str(weights))
    except Exception as exc:
        raise InferenceError(f"Could not load YOLO weights {weights}: {exc}") from exc


def _model_contract(model: Any) -> tuple[dict[int, str], set[int], set[int]]:
    names = _normalize_names(getattr(model, "names", None))
    vehicle_ids = {class_id for class_id, name in names.items() if name in VEHICLE_NAMES}
    watercraft_ids = {class_id for class_id, name in names.items() if name in WATERCRAFT_NAMES}
    if not vehicle_ids:
        raise InferenceError(f"Model has no vehicle target class; classes={names}")
    if not watercraft_ids:
        raise InferenceError(f"Model has no watercraft target class; classes={names}")
    return names, vehicle_ids, watercraft_ids


def _tensor_array(value: Any) -> Any:
    return value.detach().cpu().numpy()


def result_row(
    source_row: pd.Series,
    result: Any,
    *,
    names: Mapping[int, str],
    vehicle_ids: set[int],
    watercraft_ids: set[int],
    vehicle_conf: float,
    watercraft_conf: float,
    include_datetime: bool,
) -> dict[str, Any]:
    """Translate one Ultralytics result with the historical threshold rules."""

    path = Path(str(source_row["path"]))
    output: dict[str, Any] = {
        "path": str(path),
        "filename": path.name,
        "site": None if pd.isna(source_row["site"]) else str(source_row["site"]),
    }
    if include_datetime:
        value = source_row["datetime_from_filename"]
        output["datetime_from_filename"] = None if pd.isna(value) else value

    detections: list[dict[str, Any]] = []
    count_vehicle = 0
    count_watercraft = 0
    boxes = getattr(result, "boxes", None)
    if boxes is not None and getattr(boxes, "xyxy", None) is not None:
        xyxy = _tensor_array(boxes.xyxy)
        classes = _tensor_array(boxes.cls)
        confidences = _tensor_array(boxes.conf)
        if not (len(xyxy) == len(classes) == len(confidences)):
            raise InferenceError("YOLO returned inconsistent box/class/confidence array lengths.")
        for bounds, raw_class, raw_confidence in zip(xyxy, classes, confidences):
            class_id = int(raw_class)
            confidence = float(raw_confidence)
            if class_id in watercraft_ids:
                if confidence < float(watercraft_conf):
                    continue
                count_watercraft += 1
            elif class_id in vehicle_ids:
                if confidence < float(vehicle_conf):
                    continue
                count_vehicle += 1
            else:
                continue
            coordinates = [float(value) for value in bounds.tolist()]
            if len(coordinates) != 4:
                raise InferenceError(f"YOLO returned a non-xyxy box with {len(coordinates)} coordinates.")
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": str(names.get(class_id, class_id)),
                    "conf": confidence,
                    "xyxy": coordinates,
                }
            )
    output.update(
        {
            "count_vehicle": int(count_vehicle),
            "count_watercraft": int(count_watercraft),
            "count_total": int(count_vehicle + count_watercraft),
            "detections": detections,
        }
    )
    return output


def predict_rows(
    model: Any,
    source: pd.DataFrame,
    *,
    args: argparse.Namespace,
    names: Mapping[int, str],
    vehicle_ids: set[int],
    watercraft_ids: set[int],
    include_datetime: bool,
    shard_index: int,
) -> list[dict[str, Any]]:
    paths = source["path"].astype(str).tolist()
    try:
        results = model.predict(
            source=paths,
            imgsz=int(args.imgsz),
            conf=float(min(float(args.vehicle_conf), float(args.watercraft_conf), 0.25)),
            iou=0.7,
            device=str(args.device),
            stream=False,
            verbose=False,
            max_det=300,
            batch=int(args.batch),
        )
        results = list(results)
    except Exception as exc:
        raise InferenceError(
            f"Inference failed in shard {shard_index:06d} for {len(paths)} PIL-decodable images; "
            "no image in this batch was classified as bad."
        ) from exc
    if len(results) != len(source):
        raise InferenceError(
            f"Inference returned {len(results)} results for {len(source)} inputs in shard {shard_index:06d}."
        )
    return [
        result_row(
            row,
            result,
            names=names,
            vehicle_ids=vehicle_ids,
            watercraft_ids=watercraft_ids,
            vehicle_conf=float(args.vehicle_conf),
            watercraft_conf=float(args.watercraft_conf),
            include_datetime=include_datetime,
        )
        for (_, row), result in zip(source.iterrows(), results)
    ]


def _path_hash(paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        encoded = path.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _detection_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    class_counts: dict[str, int] = defaultdict(int)
    vehicle = watercraft = total = with_vehicle = with_watercraft = 0
    for row in rows:
        row_vehicle = int(row["count_vehicle"])
        row_watercraft = int(row["count_watercraft"])
        row_total = int(row["count_total"])
        if row_total != row_vehicle + row_watercraft:
            raise Stage09Error(f"Detection count identity failed for {row['path']}.")
        vehicle += row_vehicle
        watercraft += row_watercraft
        total += row_total
        with_vehicle += int(row_vehicle > 0)
        with_watercraft += int(row_watercraft > 0)
        for detection in row["detections"]:
            class_counts[str(detection["class_name"])] += 1
    if sum(class_counts.values()) != total:
        raise Stage09Error("Detection-list count does not equal count_total across output rows.")
    return {
        "image_rows": len(rows),
        "images_with_vehicle": with_vehicle,
        "images_with_watercraft": with_watercraft,
        "vehicle_detections": vehicle,
        "watercraft_detections": watercraft,
        "total_detections": total,
        "detections_by_class_name": dict(sorted(class_counts.items())),
    }


def _site_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, int]] = defaultdict(
        lambda: {"image_rows": 0, "vehicle_detections": 0, "watercraft_detections": 0, "total_detections": 0}
    )
    for row in rows:
        site = "<missing>" if row.get("site") is None else str(row["site"])
        summary = grouped[site]
        summary["image_rows"] += 1
        summary["vehicle_detections"] += int(row["count_vehicle"])
        summary["watercraft_detections"] += int(row["count_watercraft"])
        summary["total_detections"] += int(row["count_total"])
    return [{"site": site, **grouped[site]} for site in sorted(grouped)]


def _rows_from_table(path: Path) -> tuple[list[dict[str, Any]], Any]:
    _, pq = _import_arrow()
    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise Stage09Error(f"Could not read shard {path}: {exc}") from exc
    return table.to_pylist(), table.schema.remove_metadata()


def _merge_site_summaries(items: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, int]] = defaultdict(
        lambda: {"image_rows": 0, "vehicle_detections": 0, "watercraft_detections": 0, "total_detections": 0}
    )
    for group in items:
        for item in group:
            target = merged[str(item["site"])]
            for key in target:
                target[key] += int(item[key])
    return [{"site": site, **merged[site]} for site in sorted(merged)]


def _merge_detection_summaries(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "image_rows",
        "images_with_vehicle",
        "images_with_watercraft",
        "vehicle_detections",
        "watercraft_detections",
        "total_detections",
    )
    merged = {key: sum(int(item[key]) for item in items) for key in keys}
    class_counts: dict[str, int] = defaultdict(int)
    for item in items:
        for name, count in item["detections_by_class_name"].items():
            class_counts[str(name)] += int(count)
    merged["detections_by_class_name"] = dict(sorted(class_counts.items()))
    if merged["vehicle_detections"] + merged["watercraft_detections"] != merged["total_detections"]:
        raise Stage09Error("Merged detection count identity failed.")
    if sum(class_counts.values()) != merged["total_detections"]:
        raise Stage09Error("Merged class count identity failed.")
    return merged


def _validate_known_run_directory(run_dir: Path, plan_exists: bool) -> None:
    if not run_dir.exists():
        return
    if not run_dir.is_dir():
        raise Stage09Error(f"--run-dir is not a directory: {run_dir}")
    entries = {path.name for path in run_dir.iterdir()}
    if not entries:
        return
    if not plan_exists:
        raise Stage09Error(
            f"Refusing non-empty run directory without a stage-09 plan: {run_dir}. Choose a new --run-dir."
        )
    allowed = {"plan.json", "shards", "detections.parquet", "bad_images.txt", "run_manifest.json"}
    unknown = sorted(name for name in entries if name not in allowed and not name.startswith("."))
    if unknown:
        raise Stage09Error(f"Run directory contains unrecognized artifacts: {unknown}")


def _prepare_plan(
    args: argparse.Namespace,
    *,
    manifest: Path,
    weights: Path,
    root: Path,
    frame: pd.DataFrame,
    counts: Mapping[str, int],
) -> tuple[dict[str, Any], str]:
    configuration = {
        "selection": {
            "subset": args.subset,
            "max_images": int(args.max_images),
            "require_daylight": bool(args.require_daylight),
            "require_readable": bool(args.require_readable),
        },
        "inference": {
            "device": str(args.device),
            "imgsz": int(args.imgsz),
            "batch": int(args.batch),
            "shard_size": int(args.shard_size),
            "vehicle_conf": float(args.vehicle_conf),
            "watercraft_conf": float(args.watercraft_conf),
            "predict_conf": float(min(float(args.vehicle_conf), float(args.watercraft_conf), 0.25)),
            "iou": 0.7,
            "stream": False,
            "verbose": False,
            "max_det": 300,
            "seed": int(args.seed),
        },
        "verification": {
            "verify_content_hashes": bool(args.verify_content_hashes),
            "existence_check_mode": "deferred-per-shard" if args.defer_existence_check else "eager-prepass",
            "missing_source_policy": "abort",
        },
        "expected_counts": {
            "input_rows": args.expected_input_rows,
            "filtered_rows": args.expected_filtered_rows,
            "final_rows": args.expected_final_rows,
        },
    }
    config_sha256 = _sha256_json(configuration)
    inventory_columns = ("path", "content_sha256", "file_size", "site", "datetime_from_filename")
    contract = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": {
            "manifest": {"path": str(manifest), "size_bytes": manifest.stat().st_size, "sha256": sha256_file(manifest)},
            "weights": {"path": str(weights), "size_bytes": weights.stat().st_size, "sha256": sha256_file(weights)},
            "camera_root": str(root),
            "filtered_inventory_sha256": _inventory_hash(frame, inventory_columns),
        },
        "configuration": configuration,
        "configuration_sha256": config_sha256,
        "filter_counts": dict(counts),
        "include_datetime": "datetime_from_filename" in frame.columns,
    }
    return contract, _sha256_json(contract)


def _load_or_create_plan(run_dir: Path, contract: Mapping[str, Any], fingerprint: str) -> dict[str, Any]:
    plan_path = run_dir / "plan.json"
    _validate_known_run_directory(run_dir, plan_path.is_file())
    run_dir.mkdir(parents=True, exist_ok=True)
    if plan_path.is_file():
        try:
            existing = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Stage09Error(f"Could not read existing plan {plan_path}: {exc}") from exc
        if existing.get("plan_fingerprint") != fingerprint or existing.get("contract") != contract:
            raise Stage09Error(
                f"Existing run plan does not match requested inputs/configuration: {plan_path}. Choose a new --run-dir."
            )
        return existing
    payload = {"created_utc": _utc_now(), "plan_fingerprint": fingerprint, "contract": contract}
    atomic_write_json(plan_path, payload)
    return payload


def _shard_paths(run_dir: Path, index: int) -> tuple[Path, Path]:
    stem = f"shard-{index:06d}"
    return run_dir / "shards" / f"{stem}.parquet", run_dir / "shards" / f"{stem}.json"


def _expected_shard_inventory(frame: pd.DataFrame, start: int, end: int) -> str:
    return _inventory_hash(frame.iloc[start:end].reset_index(drop=True), ("path", "content_sha256", "file_size"))


def _validate_shard(
    parquet_path: Path,
    sidecar_path: Path,
    *,
    expected_frame: pd.DataFrame,
    expected_schema: Any,
    plan_fingerprint: str,
    shard_index: int,
    start: int,
    end: int,
    verify_content_hashes: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage09Error(f"Could not read completed shard sidecar {sidecar_path}: {exc}") from exc
    exact = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "plan_fingerprint": plan_fingerprint,
        "shard_index": shard_index,
        "start_row": start,
        "end_row_exclusive": end,
        "source_row_count": end - start,
        "source_inventory_sha256": _expected_shard_inventory(expected_frame, start, end),
    }
    for key, expected in exact.items():
        if sidecar.get(key) != expected:
            raise Stage09Error(
                f"Completed shard {shard_index:06d} failed sidecar check {key}: "
                f"expected {expected!r}, observed {sidecar.get(key)!r}."
            )
    expected_hash = sidecar.get("output", {}).get("sha256")
    if not isinstance(expected_hash, str) or sha256_file(parquet_path) != expected_hash:
        raise Stage09Error(f"Completed shard {shard_index:06d} failed Parquet SHA-256 verification.")
    expected_size = sidecar.get("output", {}).get("size_bytes")
    if expected_size != parquet_path.stat().st_size:
        raise Stage09Error(f"Completed shard {shard_index:06d} failed Parquet size verification.")

    rows, observed_schema = _rows_from_table(parquet_path)
    if not observed_schema.equals(expected_schema):
        raise Stage09Error(f"Completed shard {shard_index:06d} has an unexpected Parquet schema.")
    bad_rows = sidecar.get("decode_failures")
    if not isinstance(bad_rows, list):
        raise Stage09Error(f"Completed shard {shard_index:06d} has no valid decode_failures list.")
    try:
        bad_positions = [int(item["source_position"]) for item in bad_rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage09Error(f"Completed shard {shard_index:06d} has malformed decode-failure metadata.") from exc
    if len(bad_positions) != len(set(bad_positions)):
        raise Stage09Error(f"Completed shard {shard_index:06d} repeats a bad source position.")
    if any(position < start or position >= end for position in bad_positions):
        raise Stage09Error(f"Completed shard {shard_index:06d} records an out-of-range bad source position.")
    for item, position in zip(bad_rows, bad_positions):
        expected_bad_path = str(expected_frame.iloc[position]["path"])
        if item.get("path") != expected_bad_path or not isinstance(item.get("error"), str) or not item["error"]:
            raise Stage09Error(f"Completed shard {shard_index:06d} has invalid decode-failure metadata.")

    expected_paths = [
        str(expected_frame.iloc[position]["path"])
        for position in range(start, end)
        if position not in set(bad_positions)
    ]
    observed_paths = [str(row["path"]) for row in rows]
    if observed_paths != expected_paths or _path_hash(observed_paths) != sidecar.get("output_paths_sha256"):
        raise Stage09Error(f"Completed shard {shard_index:06d} failed output path/order verification.")
    if len(rows) + len(bad_rows) != end - start:
        raise Stage09Error(f"Completed shard {shard_index:06d} failed source/output/bad row accounting.")
    if len(rows) != sidecar.get("output", {}).get("row_count"):
        raise Stage09Error(f"Completed shard {shard_index:06d} failed output row-count verification.")
    if _detection_summary(rows) != sidecar.get("detection_summary"):
        raise Stage09Error(f"Completed shard {shard_index:06d} failed detection-summary verification.")
    if _site_summary(rows) != sidecar.get("site_summary"):
        raise Stage09Error(f"Completed shard {shard_index:06d} failed site-summary verification.")

    if verify_content_hashes:
        bad_position_set = set(bad_positions)
        for position in range(start, end):
            # Completed shards are resumable only while their external image
            # bytes still match the checksum-bound manifest and PIL gives the
            # same good/bad classification. Error message text is deliberately
            # not compared because Pillow patch releases may word it differently.
            decode_error = verify_and_decode(expected_frame.iloc[position], verify_content_hash=True)
            if (decode_error is not None) != (position in bad_position_set):
                raise Stage09Error(
                    f"Completed shard {shard_index:06d} changed PIL decode status at source row {position}."
                )
    return sidecar, rows


def _process_shard(
    model: Any,
    frame: pd.DataFrame,
    *,
    args: argparse.Namespace,
    expected_schema: Any,
    plan_fingerprint: str,
    shard_index: int,
    start: int,
    end: int,
    names: Mapping[int, str],
    vehicle_ids: set[int],
    watercraft_ids: set[int],
    include_datetime: bool,
    parquet_path: Path,
    sidecar_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = frame.iloc[start:end]
    good_positions: list[int] = []
    decode_failures: list[dict[str, Any]] = []
    for position, (_, row) in zip(range(start, end), source.iterrows()):
        error = verify_and_decode(row, verify_content_hash=bool(args.verify_content_hashes))
        if error is None:
            good_positions.append(position)
        else:
            decode_failures.append(
                {"source_position": position, "path": str(row["path"]), "error": error}
            )
    good = frame.iloc[good_positions]
    rows = (
        predict_rows(
            model,
            good,
            args=args,
            names=names,
            vehicle_ids=vehicle_ids,
            watercraft_ids=watercraft_ids,
            include_datetime=include_datetime,
            shard_index=shard_index,
        )
        if good_positions
        else []
    )
    if len(rows) + len(decode_failures) != end - start:
        raise Stage09Error(f"New shard {shard_index:06d} failed source/output/bad row accounting.")

    atomic_write_parquet(rows, parquet_path, expected_schema)
    output_hash = sha256_file(parquet_path)
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "plan_fingerprint": plan_fingerprint,
        "shard_index": shard_index,
        "start_row": start,
        "end_row_exclusive": end,
        "source_row_count": end - start,
        "source_inventory_sha256": _expected_shard_inventory(frame, start, end),
        "output_paths_sha256": _path_hash([str(row["path"]) for row in rows]),
        "decode_failures": decode_failures,
        "model_classes": {str(class_id): name for class_id, name in sorted(names.items())},
        "target_class_ids": {
            "vehicle": sorted(vehicle_ids),
            "watercraft": sorted(watercraft_ids),
        },
        "output": {
            "path": str(parquet_path),
            "size_bytes": parquet_path.stat().st_size,
            "sha256": output_hash,
            "row_count": len(rows),
        },
        "detection_summary": _detection_summary(rows),
        "site_summary": _site_summary(rows),
        "completed_utc": _utc_now(),
    }
    atomic_write_json(sidecar_path, sidecar)
    return sidecar, rows


def _atomic_combine_shards(shard_paths: Sequence[Path], output: Path, schema: Any) -> int:
    _, pq = _import_arrow()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = _atomic_path(output)
    row_count = 0
    writer = None
    try:
        writer = pq.ParquetWriter(partial, schema, compression="snappy")
        for shard in shard_paths:
            table = pq.read_table(shard)
            if not table.schema.remove_metadata().equals(schema):
                raise Stage09Error(f"Shard schema changed before combination: {shard}")
            writer.write_table(table.cast(schema))
            row_count += table.num_rows
        writer.close()
        writer = None
        _fsync_file(partial)
        os.replace(partial, output)
        _fsync_parent(output)
    finally:
        if writer is not None:
            writer.close()
        if partial.exists():
            partial.unlink()
    return row_count


def _software_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
    }
    try:
        import PIL

        versions["pillow"] = PIL.__version__
    except (ImportError, AttributeError):  # pragma: no cover
        pass
    for module_name in ("pyarrow", "torch", "ultralytics"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[module_name] = "unavailable"
    return versions


def execute(
    args: argparse.Namespace,
    *,
    model_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    _check_positive_settings(args)
    manifest = _require_file(Path(args.manifest), "Input manifest")
    weights = _require_file(Path(args.weights), "YOLO weights")
    root = Path(args.root).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise Stage09Error(f"Camera root not found: {root}")
    run_dir = Path(args.run_dir).expanduser().resolve(strict=False)

    frame, filter_counts = prepare_manifest(
        manifest,
        root=root,
        subset=str(args.subset),
        max_images=int(args.max_images),
        require_daylight=bool(args.require_daylight),
        require_readable=bool(args.require_readable),
        defer_existence_check=bool(args.defer_existence_check),
    )
    if args.defer_existence_check:
        validate_deferred_source_metadata(frame)
    _require_expected("Input manifest", filter_counts["manifest_rows"], args.expected_input_rows)
    _require_expected("Filtered input", filter_counts["filtered_rows"], args.expected_filtered_rows)

    contract, plan_fingerprint = _prepare_plan(
        args,
        manifest=manifest,
        weights=weights,
        root=root,
        frame=frame,
        counts=filter_counts,
    )
    plan = _load_or_create_plan(run_dir, contract, plan_fingerprint)
    shards_dir = run_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    include_datetime = bool(contract["include_datetime"])
    schema = output_schema(include_datetime)
    shard_count = (len(frame) + int(args.shard_size) - 1) // int(args.shard_size)
    expected_names = {
        _shard_paths(run_dir, index)[offset].name
        for index in range(shard_count)
        for offset in (0, 1)
    }
    unexpected = sorted(
        path.name for path in shards_dir.glob("shard-*.*") if path.name not in expected_names
    )
    if unexpected:
        raise Stage09Error(f"Shard directory contains artifacts outside the current plan: {unexpected}")

    random.seed(int(args.seed))
    try:
        import numpy as np

        np.random.seed(int(args.seed))
    except ModuleNotFoundError:  # pragma: no cover - ultralytics requires numpy
        pass
    try:
        import torch

        torch.manual_seed(int(args.seed))
    except Exception:
        pass

    model: Any | None = None
    model_contract: tuple[dict[int, str], set[int], set[int]] | None = None
    shard_sidecars: list[dict[str, Any]] = []
    shard_paths: list[Path] = []
    bad_rows: list[dict[str, Any]] = []
    expected_model_classes: dict[str, str] | None = None

    for shard_index in range(shard_count):
        start = shard_index * int(args.shard_size)
        end = min(start + int(args.shard_size), len(frame))
        parquet_path, sidecar_path = _shard_paths(run_dir, shard_index)
        if parquet_path.is_file() and sidecar_path.is_file():
            sidecar, _ = _validate_shard(
                parquet_path,
                sidecar_path,
                expected_frame=frame,
                expected_schema=schema,
                plan_fingerprint=plan_fingerprint,
                shard_index=shard_index,
                start=start,
                end=end,
                verify_content_hashes=bool(args.verify_content_hashes),
            )
            if args.verbose:
                print(f"resume: verified shard {shard_index + 1}/{shard_count}")
        else:
            if model is None:
                model = _load_model(weights, model_factory)
                model_contract = _model_contract(model)
            assert model_contract is not None
            names, vehicle_ids, watercraft_ids = model_contract
            sidecar, _ = _process_shard(
                model,
                frame,
                args=args,
                expected_schema=schema,
                plan_fingerprint=plan_fingerprint,
                shard_index=shard_index,
                start=start,
                end=end,
                names=names,
                vehicle_ids=vehicle_ids,
                watercraft_ids=watercraft_ids,
                include_datetime=include_datetime,
                parquet_path=parquet_path,
                sidecar_path=sidecar_path,
            )
            if args.verbose:
                print(f"wrote: shard {shard_index + 1}/{shard_count}")

        observed_classes = sidecar.get("model_classes")
        if not isinstance(observed_classes, dict):
            raise Stage09Error(f"Shard {shard_index:06d} lacks a model class contract.")
        if expected_model_classes is None:
            expected_model_classes = observed_classes
        elif observed_classes != expected_model_classes:
            raise Stage09Error("Model class mapping differs across completed shards.")
        if model_contract is not None:
            live_classes = {str(class_id): name for class_id, name in sorted(model_contract[0].items())}
            if observed_classes != live_classes:
                raise Stage09Error("Resumed shard model classes differ from the currently loaded checkpoint.")
        shard_sidecars.append(sidecar)
        shard_paths.append(parquet_path)
        bad_rows.extend(sidecar["decode_failures"])

    bad_positions = [int(item["source_position"]) for item in bad_rows]
    if len(bad_positions) != len(set(bad_positions)):
        raise Stage09Error("A source row was classified as a decode failure in multiple shards.")
    final_expected = len(frame) - len(bad_rows)
    _require_expected("Final output", final_expected, args.expected_final_rows)
    if sum(int(item["output"]["row_count"]) for item in shard_sidecars) != final_expected:
        raise Stage09Error("Shard output counts do not equal filtered rows minus PIL decode failures.")

    unique_bad_paths = sorted({str(item["path"]) for item in bad_rows})
    bad_path = run_dir / "bad_images.txt"
    atomic_write_text(bad_path, "".join(f"{path}\n" for path in unique_bad_paths))

    output = run_dir / "detections.parquet"
    combined_rows = _atomic_combine_shards(shard_paths, output, schema)
    if combined_rows != final_expected:
        raise Stage09Error(
            f"Combined output row-count mismatch: expected {final_expected}, observed {combined_rows}."
        )
    _, pq = _import_arrow()
    combined_file = pq.ParquetFile(output)
    if combined_file.metadata.num_rows != final_expected:
        raise Stage09Error("Combined Parquet metadata row count does not match the verified final count.")
    if not combined_file.schema_arrow.remove_metadata().equals(schema):
        raise Stage09Error("Combined Parquet schema does not match the stage-09 output contract.")

    detection_summary = _merge_detection_summaries(
        [item["detection_summary"] for item in shard_sidecars]
    )
    site_summary = _merge_site_summaries([item["site_summary"] for item in shard_sidecars])
    if detection_summary["image_rows"] != final_expected:
        raise Stage09Error("Detection summary row count does not match combined output.")
    if sum(item["image_rows"] for item in site_summary) != final_expected:
        raise Stage09Error("Site summary row count does not match combined output.")

    completed_utc = _utc_now()
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "status": "complete",
        "created_utc": plan["created_utc"],
        "completed_utc": completed_utc,
        "plan_fingerprint": plan_fingerprint,
        "configuration": contract["configuration"],
        "hashes": {
            "implementation_sha256": contract["implementation_sha256"],
            "input_manifest_sha256": contract["inputs"]["manifest"]["sha256"],
            "weights_sha256": contract["inputs"]["weights"]["sha256"],
            "filtered_inventory_sha256": contract["inputs"]["filtered_inventory_sha256"],
            "configuration_sha256": contract["configuration_sha256"],
            "combined_output_sha256": sha256_file(output),
            "bad_images_sha256": sha256_file(bad_path),
        },
        "inputs": contract["inputs"],
        "row_counts": {
            **filter_counts,
            "pil_decode_failure_rows": len(bad_rows),
            "pil_decode_failure_unique_paths": len(unique_bad_paths),
            "inference_input_rows": final_expected,
            "final_rows": combined_rows,
        },
        "model_classes": expected_model_classes or {},
        "outputs": {
            "detections": {
                "path": str(output),
                "size_bytes": output.stat().st_size,
                "sha256": sha256_file(output),
                "row_count": combined_rows,
            },
            "bad_images": {
                "path": str(bad_path),
                "size_bytes": bad_path.stat().st_size,
                "sha256": sha256_file(bad_path),
                "row_count": len(bad_rows),
                "unique_path_count": len(unique_bad_paths),
            },
            "shards": [item["output"] for item in shard_sidecars],
        },
        "detection_summary": detection_summary,
        "site_summary": site_summary,
        "decode_failures": bad_rows,
        "software": _software_versions(),
    }
    manifest_path = run_dir / "run_manifest.json"
    atomic_write_json(manifest_path, run_manifest)
    if args.verbose:
        print(f"Wrote detections: {output} (n={combined_rows})")
        print(f"Wrote run manifest: {manifest_path}")
        print(f"PIL decode failures: {len(bad_rows)} rows / {len(unique_bad_paths)} unique paths")
    return run_manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        execute(args)
    except Stage09Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
