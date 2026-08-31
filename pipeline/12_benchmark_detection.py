#!/usr/bin/env python3
"""Run the paper-specific, post-hoc detector architecture benchmark.

The benchmark starts both detectors from public COCO weights and uses one
frozen two-class training/validation inventory.  The validation split is
evaluated only after the fixed final epoch; it is never used for early stopping
or checkpoint selection.  The output is therefore a reviewer-requested,
common-validation architecture check rather than an independent test-set
result or a replacement for the operational detector.

Models
------
* ``yolo26n`` (Ultralytics, COCO initialized)
* ``fcos_resnet50_fpn`` (torchvision, COCO initialized)

The script is intentionally import-light.  ``--dry-run`` performs the complete
data/leakage audit and materializes immutable manifests without importing a
deep-learning framework.  ``--smoke`` performs a one-epoch wiring check on a
small deterministic subset and marks every result as smoke-only.  The portable
default uses each base training image once.  Reproducing the paper's effective
hard-sampled training list also requires ``--historical-hard-list``; its
validation rows are audited and excluded before training.

Example
-------
python pipeline/12_benchmark_detection.py \
  --dataset-root data/external_training/detection_benchmark \
  --output-dir outputs/results/reviewer_benchmarks/detection \
  --dry-run
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import project_paths  # noqa: E402

CLASS_NAMES = ("vehicle", "watercraft")
MODEL_NAMES = (
    "yolo26n",
    "fcos_resnet50_fpn",
)
PAPER_SPLIT_COUNTS = {"train": 1395, "val": 100}
IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
JPEG_SUFFIXES = {".jpg", ".jpeg"}


class ContractError(RuntimeError):
    """Raised when a scientific or artifact contract is violated."""


@dataclass(frozen=True)
class ImageRecord:
    split: str
    image: Path
    label: Path
    image_sha256: str
    label_sha256: str
    width: int
    height: int
    n_vehicle: int
    n_watercraft: int

    @property
    def image_id(self) -> str:
        return self.image.name


@dataclass(frozen=True)
class Detection:
    image_id: str
    class_id: int
    score: float
    box_xyxy: tuple[float, float, float, float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_paths.TRAINING_ROOT / "detection_benchmark",
        help="YOLO-format dataset root with images/{train,val} and labels/{train,val}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_paths.RESULTS_ROOT / "reviewer_benchmarks" / "detection",
    )
    parser.add_argument(
        "--expected-train-images",
        type=int,
        default=PAPER_SPLIT_COUNTS["train"],
        help="Required training-image count (paper inventory: 1395).",
    )
    parser.add_argument(
        "--expected-val-images",
        type=int,
        default=PAPER_SPLIT_COUNTS["val"],
        help="Required validation-image count (paper inventory: 100).",
    )
    parser.add_argument(
        "--historical-hard-list",
        type=Path,
        default=None,
        help=(
            "Optional historical oversampled train list. If supplied, recover the paper's legitimate train-only "
            "hard examples while rejecting leaked validation rows; otherwise use each base training image once."
        ),
    )
    parser.add_argument(
        "--expected-hard-train-images",
        type=int,
        default=14,
        help="Expected recoverable train-only hard examples when --historical-hard-list is supplied (paper: 14).",
    )
    parser.add_argument(
        "--hard-example-extra-repeats",
        type=int,
        default=5,
        help="Extra occurrences per recovered hard example (paper: 5).",
    )
    parser.add_argument(
        "--al-batches-root",
        type=Path,
        default=None,
        help="Annotation-batch root; required only with --yolo-cumulative-stages.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
        help="Model subset to run.",
    )
    parser.add_argument(
        "--yolo-cumulative-stages",
        action="store_true",
        help=(
            "Also train fresh YOLO26n models on reconstructed cumulative annotation stages. "
            "These are descriptive annotation-stage runs, not causal evidence for active learning."
        ),
    )
    parser.add_argument(
        "--yolo-weights",
        default="yolo26n.pt",
        help="Local COCO-initialized YOLO26n weights or an Ultralytics model identifier.",
    )
    parser.add_argument(
        "--torch-home",
        type=Path,
        default=None,
        help=(
            "Writable torchvision weight cache. Defaults to an ignored cache beside the configured dataset root."
        ),
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Common optimizer-update batch size for both architectures (default: 8).",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--horizontal-flip", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one epoch on deterministic small subsets; smoke metrics must not be reported as benchmark results.",
    )
    parser.add_argument("--smoke-train-images", type=int, default=24)
    parser.add_argument("--smoke-val-images", type=int, default=12)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only an output directory whose immutable configuration exactly matches this invocation.",
    )
    return parser


def _metadata_path(path: Path) -> bool:
    return path.name.startswith(".") or path.name.startswith("._") or path.name == ".DS_Store"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_yolo_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read and strictly validate one normalized two-class YOLO label file."""

    boxes: list[tuple[int, float, float, float, float]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ContractError(f"{path}:{line_number}: expected five YOLO fields, found {len(parts)}.")
        try:
            class_float, xc, yc, width, height = (float(value) for value in parts)
        except ValueError as exc:
            raise ContractError(f"{path}:{line_number}: non-numeric YOLO label.") from exc
        class_id = int(class_float)
        if class_float != class_id or class_id not in (0, 1):
            raise ContractError(f"{path}:{line_number}: class must be exactly 0 or 1, found {parts[0]!r}.")
        if not all(math.isfinite(value) for value in (xc, yc, width, height)):
            raise ContractError(f"{path}:{line_number}: box coordinates must be finite.")
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            raise ContractError(f"{path}:{line_number}: center lies outside [0,1].")
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ContractError(f"{path}:{line_number}: width and height must lie in (0,1].")
        # The historical exporter independently clamped centers and sizes, so
        # a few otherwise valid edge boxes extend slightly past the frame.
        # Both benchmark loaders use the same labels; xyxy conversion below
        # clips those extents to the image rather than silently dropping them.
        boxes.append((class_id, xc, yc, width, height))
    return boxes


def _indexed_files(
    directory: Path,
    suffixes: set[str],
    kind: str,
    *,
    cache_suffixes: set[str] | None = None,
) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing {kind} directory: {directory}")
    files = [path for path in directory.iterdir() if path.is_file() and not _metadata_path(path)]
    cache_suffixes = cache_suffixes or set()
    unsupported = [path.name for path in files if path.suffix.lower() not in suffixes | cache_suffixes]
    if unsupported:
        preview = ", ".join(sorted(unsupported)[:8])
        raise ContractError(f"Unexpected files in {directory}: {preview}")
    index: dict[str, Path] = {}
    for path in sorted(path for path in files if path.suffix.lower() in suffixes):
        if path.stem in index:
            raise ContractError(f"Duplicate {kind} stem {path.stem!r} in {directory}.")
        index[path.stem] = path.resolve()
    orphaned_cache = sorted(
        path.name for path in files if path.suffix.lower() in cache_suffixes and path.stem not in index
    )
    if orphaned_cache:
        raise ContractError(f"Orphaned framework cache files in {directory}: {orphaned_cache[:8]}")
    return index


def inspect_split(dataset_root: Path, split: str, expected_count: int) -> list[ImageRecord]:
    """Validate image/label pairs, JPEG decoding, boxes, and hashes for one split."""

    # Ultralytics ``cache=disk`` leaves one same-stem .npy beside each JPEG.
    # Treat those as derived caches, never as source images.
    image_index = _indexed_files(
        dataset_root / "images" / split,
        JPEG_SUFFIXES,
        "image",
        cache_suffixes={".npy"},
    )
    label_index = _indexed_files(dataset_root / "labels" / split, {".txt"}, "label")
    if len(image_index) != expected_count:
        raise ContractError(
            f"Expected exactly {expected_count} base {split} JPEGs after ignoring metadata sidecars; "
            f"found {len(image_index)}."
        )
    if image_index.keys() != label_index.keys():
        images_only = sorted(image_index.keys() - label_index.keys())
        labels_only = sorted(label_index.keys() - image_index.keys())
        raise ContractError(
            f"{split} image/label pairing failed (images-only={images_only[:8]}, labels-only={labels_only[:8]})."
        )

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover - environment failure
        raise RuntimeError("Pillow is required for detector data validation.") from exc

    records: list[ImageRecord] = []
    for stem, image_path in sorted(image_index.items()):
        label_path = label_index[stem]
        labels = parse_yolo_label(label_path)
        try:
            with Image.open(image_path) as image:
                # These camera JPEGs contain an auxiliary MPF directory, so
                # Pillow identifies the valid .jpg container as MPO.
                if image.format not in {"JPEG", "MPO"}:
                    raise ContractError(f"Expected JPEG/MPO encoding, found {image.format!r}: {image_path}")
                image.load()
                width, height = image.size
        except ContractError:
            raise
        except Exception as exc:
            raise ContractError(f"Unreadable JPEG: {image_path}: {exc}") from exc
        if width <= 0 or height <= 0:
            raise ContractError(f"Invalid image dimensions {width}x{height}: {image_path}")
        counts = Counter(label[0] for label in labels)
        records.append(
            ImageRecord(
                split=split,
                image=image_path,
                label=label_path,
                image_sha256=sha256_file(image_path),
                label_sha256=sha256_file(label_path),
                width=width,
                height=height,
                n_vehicle=counts[0],
                n_watercraft=counts[1],
            )
        )
    return records


def validate_dataset(
    dataset_root: Path,
    expected_train_images: int = PAPER_SPLIT_COUNTS["train"],
    expected_val_images: int = PAPER_SPLIT_COUNTS["val"],
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Audit a frozen YOLO export against explicit split-size contracts."""

    dataset_root = dataset_root.expanduser().resolve()
    train = inspect_split(dataset_root, "train", expected_train_images)
    val = inspect_split(dataset_root, "val", expected_val_images)

    train_names = {record.image.name for record in train}
    val_names = {record.image.name for record in val}
    basename_overlap = sorted(train_names & val_names)
    if basename_overlap:
        raise ContractError(f"Train/validation image basename overlap: {basename_overlap[:12]}")
    train_stems = {record.image.stem for record in train}
    val_stems = {record.image.stem for record in val}
    if train_stems & val_stems:
        raise ContractError(f"Train/validation image stem overlap: {sorted(train_stems & val_stems)[:12]}")

    hashes_train: dict[str, list[str]] = defaultdict(list)
    hashes_val: dict[str, list[str]] = defaultdict(list)
    for record in train:
        hashes_train[record.image_sha256].append(record.image.name)
    for record in val:
        hashes_val[record.image_sha256].append(record.image.name)
    digest_overlap = sorted(hashes_train.keys() & hashes_val.keys())
    if digest_overlap:
        examples = [f"{hashes_train[digest][0]} == {hashes_val[digest][0]}" for digest in digest_overlap[:8]]
        raise ContractError(f"Train/validation SHA-256 image overlap: {examples}")
    return train, val


def parse_camera_timestamp(image_id: str) -> tuple[str, datetime]:
    """Parse the uniform ``camera_camera_YYYY_MM_DD_HH_MM_SS[_dupN]`` schema."""

    parts = Path(image_id).stem.split("_")
    if len(parts) < 8:
        raise ContractError(f"Image basename does not match the camera timestamp schema: {image_id}")
    if any(not token.startswith("dup") for token in parts[8:]):
        raise ContractError(f"Unexpected suffix in camera timestamp basename: {image_id}")
    try:
        timestamp = datetime(*(int(value) for value in parts[2:8]))
    except (TypeError, ValueError) as exc:
        raise ContractError(f"Invalid timestamp fields in image basename: {image_id}") from exc
    return f"{parts[0]}_{parts[1]}", timestamp


def audit_same_camera_temporal_proximity(
    train_records: Sequence[ImageRecord], val_records: Sequence[ImageRecord]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Quantify post-split temporal dependence without changing the frozen split."""

    train_by_camera: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
    for record in train_records:
        camera, timestamp = parse_camera_timestamp(record.image_id)
        train_by_camera[camera].append((timestamp, record.image_id))
    for values in train_by_camera.values():
        values.sort()

    rows: list[dict[str, Any]] = []
    finite_deltas: list[float] = []
    for record in val_records:
        camera, timestamp = parse_camera_timestamp(record.image_id)
        candidates = train_by_camera.get(camera, [])
        nearest_name: str | None = None
        nearest_seconds: float | None = None
        if candidates:
            times = [item[0] for item in candidates]
            position = bisect_left(times, timestamp)
            nearby = candidates[max(0, position - 1) : min(len(candidates), position + 1)]
            nearest_time, nearest_name = min(
                nearby,
                key=lambda item: (abs((item[0] - timestamp).total_seconds()), item[1]),
            )
            nearest_seconds = abs((nearest_time - timestamp).total_seconds())
            finite_deltas.append(nearest_seconds)
        rows.append(
            {
                "validation_image": record.image_id,
                "camera_id": camera,
                "nearest_train_image": nearest_name,
                "nearest_train_seconds": nearest_seconds,
            }
        )
    summary = {
        "filename_schema": "camera_camera_YYYY_MM_DD_HH_MM_SS with optional _dupN suffix",
        "validation_images": len(val_records),
        "validation_images_with_same_camera_train_image": len(finite_deltas),
        "same_camera_nearest_within_1_hour": sum(value <= 3600 for value in finite_deltas),
        "same_camera_nearest_within_24_hours": sum(value <= 86400 for value in finite_deltas),
        "minimum_nearest_seconds": min(finite_deltas) if finite_deltas else None,
        "median_nearest_seconds": median(finite_deltas) if finite_deltas else None,
        "interpretation": (
            "Descriptive dependence audit only. A zero basename/SHA overlap does not imply camera- or "
            "time-independent validation when nearby frames from the same fixed camera occur in train."
        ),
    }
    return summary, rows


def write_temporal_proximity_audit(path: Path, rows: Sequence[Mapping[str, Any]], *, resume: bool) -> None:
    import io

    fieldnames = ["validation_image", "camera_id", "nearest_train_image", "nearest_train_seconds"]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _write_text(path, stream.getvalue(), resume=resume)


def recover_hard_training_names(
    historical_list: Path,
    train_records: Sequence[ImageRecord],
    val_records: Sequence[ImageRecord],
    *,
    expected_train_names: int = 14,
) -> tuple[list[str], list[str]]:
    """Recover train-only mined examples while explicitly excluding historical val leakage.

    The historical file contains the 1,395-image base train list followed by
    five extra occurrences per mined example.  Four mined validation images
    entered that old list.  They are returned for provenance but never enter
    the corrected benchmark list.
    """

    historical_list = historical_list.expanduser().resolve()
    if not historical_list.is_file():
        raise FileNotFoundError(f"Historical oversampling list not found: {historical_list}")
    names = [Path(line.strip()).name for line in historical_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = Counter(names)
    train_names = {record.image.name for record in train_records}
    val_names = {record.image.name for record in val_records}

    # In-train mined rows occur once in the base list plus five extra times.
    recovered_train = sorted(name for name, count in counts.items() if name in train_names and count == 6)
    leaked_val = sorted(name for name, count in counts.items() if name in val_names and count == 5)
    malformed_train = sorted((name, count) for name, count in counts.items() if name in train_names and count not in (1, 6))
    if malformed_train:
        raise ContractError(f"Unexpected historical repeat counts for train images: {malformed_train[:8]}")
    if len(recovered_train) != expected_train_names:
        raise ContractError(
            f"Expected exactly {expected_train_names} recoverable in-train mined images; found {len(recovered_train)}."
        )
    if any(name in val_names for name in recovered_train):  # defensive; the sets must already be disjoint
        raise ContractError("A validation basename entered the corrected hard-sampling list.")
    return recovered_train, leaked_val


def build_hard_sampled_records(
    train_records: Sequence[ImageRecord], hard_names: Sequence[str], extra_repeats: int = 5
) -> list[ImageRecord]:
    by_name = {record.image.name: record for record in train_records}
    unknown = sorted(set(hard_names) - by_name.keys())
    if unknown:
        raise ContractError(f"Hard-sampling names outside train: {unknown[:8]}")
    output = list(train_records)
    for name in sorted(hard_names):
        output.extend([by_name[name]] * extra_repeats)
    expected = len(train_records) + extra_repeats * len(hard_names)
    if len(output) != expected:
        raise AssertionError("Hard-sampling list length invariant failed.")
    return output


def reconstruct_annotation_stages(
    train_records: Sequence[ImageRecord], val_records: Sequence[ImageRecord], al_batches_root: Path
) -> tuple[dict[str, list[ImageRecord]], dict[str, Any]]:
    """Reconstruct cumulative labeled-data stages or fail rather than guess.

    The two retained batch manifests jointly cover the present export and have
    a small overlap.  Stage 1 is batch 001 intersected with frozen train; stage
    2 is the de-duplicated union intersected with frozen train.  Validation
    names are explicitly excluded.  No pre-batch baseline is recoverable.
    """

    batch_paths = (
        al_batches_root / "batch_001" / "paths.txt",
        al_batches_root / "batch_002_river" / "paths.txt",
    )
    if not all(path.is_file() for path in batch_paths):
        missing = [str(path) for path in batch_paths if not path.is_file()]
        raise ContractError(f"Cannot reconstruct cumulative annotation stages; missing: {missing}")

    batches: list[list[str]] = []
    for path in batch_paths:
        values = [Path(line.strip()).name for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(values) != len(set(values)):
            raise ContractError(f"Duplicate basename within annotation batch: {path}")
        batches.append(sorted(values))
    by_name = {record.image.name: record for record in train_records}
    val_names = {record.image.name for record in val_records}
    train_names = set(by_name)
    batch_1, batch_2 = set(batches[0]), set(batches[1])
    requested = batch_1 | batch_2
    current_inventory = train_names | val_names
    unknown = sorted(requested - current_inventory)
    missing = sorted(current_inventory - requested)
    if unknown or missing:
        raise ContractError(
            "Retained annotation batches do not exactly reconstruct the current train+validation inventory "
            f"(batch-only={unknown[:8]}, inventory-only={missing[:8]})."
        )

    stage_1_names = sorted(batch_1 & train_names)
    stage_2_names = sorted(requested & train_names)
    stage_1 = [by_name[name] for name in stage_1_names]
    stage_2 = [by_name[name] for name in stage_2_names]
    if {record.image.name for record in stage_2} != set(by_name):
        raise AssertionError("Final cumulative annotation stage does not reconstruct base train.")
    overlap = batch_1 & batch_2
    metadata = {
        "pre_batch_baseline_recoverable": False,
        "batch_001_manifest_names": len(batch_1),
        "batch_002_manifest_names": len(batch_2),
        "batch_overlap_names": sorted(overlap),
        "batch_overlap_train_count": len(overlap & train_names),
        "batch_overlap_validation_count": len(overlap & val_names),
        "stage_1_excluded_validation_names": sorted(batch_1 & val_names),
        "stage_2_excluded_validation_names": sorted(requested & val_names),
        "interpretation": (
            "Descriptive cumulative annotation-volume trajectory only. The stages are not randomized and do not "
            "isolate a causal active-learning effect."
        ),
    }
    return {"annotation_stage_1": stage_1, "annotation_stage_2": stage_2}, metadata


def normalized_box_to_xyxy(
    label: tuple[int, float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    _, xc, yc, box_width, box_height = label
    return (
        max(0.0, (xc - box_width / 2.0) * width),
        max(0.0, (yc - box_height / 2.0) * height),
        min(float(width), (xc + box_width / 2.0) * width),
        min(float(height), (yc + box_height / 2.0) * height),
    )


def ground_truth_detections(records: Sequence[ImageRecord]) -> list[Detection]:
    detections: list[Detection] = []
    for record in records:
        for label in parse_yolo_label(record.label):
            detections.append(
                Detection(
                    image_id=record.image_id,
                    class_id=label[0],
                    score=1.0,
                    box_xyxy=normalized_box_to_xyxy(label, record.width, record.height),
                )
            )
    return detections


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def coco_101_ap(
    ground_truth: Sequence[Detection],
    predictions: Sequence[Detection],
    image_ids: Sequence[str],
    class_id: int,
    iou_threshold: float,
    max_detections: int = 100,
) -> float | None:
    """Compute COCO-style 101-point interpolated AP without pycocotools.

    This implements class-wise, all-area AP with greedy score-ordered matching
    and a per-image max-detection cap.  Crowd/ignore annotations are absent from
    this two-class YOLO export, so those COCO branches do not apply.

    ``image_ids`` may contain duplicates for an image bootstrap.  Each sampled
    occurrence is treated as an independent copy of that image and its labels.
    """

    gt_source: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    pred_source: dict[str, list[Detection]] = defaultdict(list)
    for item in ground_truth:
        if item.class_id == class_id:
            gt_source[item.image_id].append(item.box_xyxy)
    for item in predictions:
        if item.class_id == class_id:
            pred_source[item.image_id].append(item)

    copied_gt: dict[tuple[int, str], list[tuple[float, float, float, float]]] = {}
    ranked: list[tuple[float, int, int, str, tuple[float, float, float, float]]] = []
    n_ground_truth = 0
    for occurrence, image_id in enumerate(image_ids):
        key = (occurrence, image_id)
        boxes = list(gt_source.get(image_id, ()))
        copied_gt[key] = boxes
        n_ground_truth += len(boxes)
        image_predictions = sorted(
            pred_source.get(image_id, ()),
            key=lambda item: (-item.score, item.box_xyxy),
        )[:max_detections]
        for local_order, item in enumerate(image_predictions):
            ranked.append((item.score, occurrence, local_order, image_id, item.box_xyxy))
    if n_ground_truth == 0:
        return None

    ranked.sort(key=lambda row: (-row[0], row[3], row[1], row[2]))
    matched = {key: [False] * len(boxes) for key, boxes in copied_gt.items()}
    true_positives: list[int] = []
    false_positives: list[int] = []
    for _, occurrence, _, image_id, predicted_box in ranked:
        key = (occurrence, image_id)
        best_index = -1
        best_iou = iou_threshold
        for gt_index, gt_box in enumerate(copied_gt[key]):
            if matched[key][gt_index]:
                continue
            overlap = box_iou(predicted_box, gt_box)
            if overlap >= best_iou:
                best_iou = overlap
                best_index = gt_index
        if best_index >= 0:
            matched[key][best_index] = True
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)

    cumulative_tp = 0
    cumulative_fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for tp, fp in zip(true_positives, false_positives):
        cumulative_tp += tp
        cumulative_fp += fp
        recalls.append(cumulative_tp / n_ground_truth)
        precisions.append(cumulative_tp / max(1, cumulative_tp + cumulative_fp))

    # No predictions produces a well-defined AP of zero when GT exists.
    sampled_precision: list[float] = []
    for recall_level in (index / 100.0 for index in range(101)):
        candidates = [precision for recall, precision in zip(recalls, precisions) if recall >= recall_level]
        sampled_precision.append(max(candidates, default=0.0))
    return sum(sampled_precision) / 101.0


def evaluate_predictions(
    ground_truth: Sequence[Detection],
    predictions: Sequence[Detection],
    image_ids: Sequence[str],
    max_detections: int = 100,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        values = [
            coco_101_ap(
                ground_truth,
                predictions,
                image_ids,
                class_id,
                threshold,
                max_detections=max_detections,
            )
            for threshold in IOU_THRESHOLDS
        ]
        metrics[f"{class_name}_ap50"] = values[0]
        valid = [value for value in values if value is not None]
        metrics[f"{class_name}_ap50_95"] = sum(valid) / len(valid) if valid else None
    return metrics


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def bootstrap_confidence_intervals(
    ground_truth: Sequence[Detection],
    predictions: Sequence[Detection],
    image_ids: Sequence[str],
    replicates: int,
    seed: int,
    max_detections: int = 100,
) -> dict[str, dict[str, float | int | None]]:
    if replicates < 0:
        raise ContractError("--bootstrap-replicates cannot be negative.")
    keys = [f"{class_name}_{metric}" for class_name in CLASS_NAMES for metric in ("ap50", "ap50_95")]
    samples: dict[str, list[float]] = {key: [] for key in keys}
    if not image_ids or replicates == 0:
        return {key: {"low": None, "high": None, "valid_replicates": 0} for key in keys}
    generator = random.Random(seed)
    for _ in range(replicates):
        selected = [generator.choice(image_ids) for _ in image_ids]
        values = evaluate_predictions(ground_truth, predictions, selected, max_detections=max_detections)
        for key, value in values.items():
            if value is not None:
                samples[key].append(float(value))
    return {
        key: {
            "low": _percentile(values, 2.5) if values else None,
            "high": _percentile(values, 97.5) if values else None,
            "valid_replicates": len(values),
        }
        for key, values in samples.items()
    }


def _write_text(path: Path, text: str, *, resume: bool = False, allow_managed_update: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == text:
            return
        if not (resume and allow_managed_update):
            raise FileExistsError(f"Refusing to overwrite non-identical artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Any, *, resume: bool = False, allow_managed_update: bool = False) -> None:
    _write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        resume=resume,
        allow_managed_update=allow_managed_update,
    )


def prepare_output(output_dir: Path, resume: bool) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"Output directory is nonempty: {output_dir}. Choose a new directory or pass --resume for a matching run."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def preserve_immutable_provenance(
    path: Path, payload: Mapping[str, Any], *, resume: bool, run_config_sha256: str
) -> dict[str, Any]:
    """Write provenance once or verify and preserve it during an explicit resume."""

    if resume and path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("run_config_sha256") != run_config_sha256:
            raise ContractError("Existing provenance does not match the immutable run configuration.")
        return existing
    _write_json(path, dict(payload))
    return dict(payload)


def write_path_list(path: Path, records: Sequence[ImageRecord], *, resume: bool = False) -> None:
    _write_text(path, "".join(f"{record.image}\n" for record in records), resume=resume)


def write_data_yaml(path: Path, train_list: Path, val_list: Path, *, resume: bool = False) -> None:
    contents = (
        "# Generated by pipeline/12_benchmark_detection.py\n"
        "path: /\n"
        f"train: {train_list.resolve()}\n"
        f"val: {val_list.resolve()}\n"
        "names:\n"
        "  0: vehicle\n"
        "  1: watercraft\n"
    )
    _write_text(path, contents, resume=resume)


def write_dataset_manifest(
    path: Path, records: Sequence[ImageRecord], hard_names: set[str], *, resume: bool = False
) -> None:
    rows: list[list[Any]] = []
    for record in records:
        rows.append(
            [
                record.split,
                record.image_id,
                str(record.image),
                str(record.label),
                record.image_sha256,
                record.label_sha256,
                record.width,
                record.height,
                record.n_vehicle,
                record.n_watercraft,
                int(record.image_id in hard_names),
            ]
        )
    header = [
        "split",
        "image_id",
        "image_path",
        "label_path",
        "image_sha256",
        "label_sha256",
        "width",
        "height",
        "n_vehicle",
        "n_watercraft",
        "hard_sampled_train_image",
    ]
    import io

    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    _write_text(path, stream.getvalue(), resume=resume)


def write_predictions(
    path: Path,
    detections: Sequence[Detection],
    image_ids: Sequence[str] = (),
    *,
    resume: bool = False,
) -> None:
    by_image: dict[str, list[Detection]] = defaultdict(list)
    for image_id in image_ids:
        by_image[image_id] = []
    for detection in detections:
        by_image[detection.image_id].append(detection)
    lines: list[str] = []
    for image_id in sorted(by_image):
        rows = [
            {
                "class_id": detection.class_id,
                "class_name": CLASS_NAMES[detection.class_id],
                "score": detection.score,
                "box_xyxy": list(detection.box_xyxy),
            }
            for detection in sorted(by_image[image_id], key=lambda item: (-item.score, item.class_id, item.box_xyxy))
        ]
        lines.append(json.dumps({"image_id": image_id, "detections": rows}, sort_keys=True))
    _write_text(path, "\n".join(lines) + ("\n" if lines else ""), resume=resume)


def load_predictions(path: Path) -> list[Detection]:
    output: list[Detection] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        payload = json.loads(raw)
        for row in payload["detections"]:
            output.append(
                Detection(
                    image_id=str(payload["image_id"]),
                    class_id=int(row["class_id"]),
                    score=float(row["score"]),
                    box_xyxy=tuple(float(value) for value in row["box_xyxy"]),
                )
            )
    return output


def choose_device(requested: str) -> str:
    import torch

    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable.")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def yolo_inference_device(training_device: str) -> str:
    """Keep Ultralytics NMS off MPS while preserving the requested training device."""

    return "cpu" if training_device == "mps" else training_device


def yolo_benchmark_trainer_class() -> type[Any]:
    """Return a detector trainer that delegates all validation to this script."""

    from ultralytics.models.yolo.detect import DetectionTrainer

    class SharedEvaluatorDetectionTrainer(DetectionTrainer):
        def validate(self) -> tuple[dict[str, float], float]:
            return {}, 0.0

        def final_eval(self) -> None:
            return None

    return SharedEvaluatorDetectionTrainer


def torchvision_label_contract(model_name: str) -> tuple[int, int]:
    """Return ``(target/prediction offset, output class count)`` for a detector."""

    if model_name == "fcos_resnet50_fpn":
        return 0, len(CLASS_NAMES)
    raise ValueError(f"Not a torchvision benchmark model: {model_name}")


def torchvision_resize_bounds(imgsz: int) -> tuple[int, int]:
    """Use the same long-edge content resolution as a square YOLO letterbox."""

    return imgsz, imgsz


def torchvision_initializer_artifact(model_name: str) -> dict[str, str]:
    """Resolve and hash the exact torchvision COCO checkpoint loaded by a model."""

    from urllib.parse import urlparse

    import torch
    from torchvision.models.detection import FCOS_ResNet50_FPN_Weights

    if model_name == "fcos_resnet50_fpn":
        weights = FCOS_ResNet50_FPN_Weights.DEFAULT
    else:
        raise ValueError(model_name)
    filename = Path(urlparse(weights.url).path).name
    artifact = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not artifact.is_file():
        raise FileNotFoundError(f"Torchvision initializer cache artifact not found after model construction: {artifact}")
    return {
        "weights_enum": str(weights),
        "url": weights.url,
        "path": str(artifact.resolve()),
        "sha256": sha256_file(artifact),
    }


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # older torch without warn_only
        torch.use_deterministic_algorithms(True)


def _replace_fcos_classification_logits(model: Any, output_classes: int) -> Any:
    """Retain the COCO-trained FCOS tower while replacing its class logits."""

    import torch.nn as nn

    old_head = model.head.classification_head
    replacement_logits = nn.Conv2d(
        old_head.cls_logits.in_channels,
        old_head.num_anchors * output_classes,
        kernel_size=old_head.cls_logits.kernel_size,
        stride=old_head.cls_logits.stride,
        padding=old_head.cls_logits.padding,
    )
    nn.init.normal_(replacement_logits.weight, std=0.01)
    nn.init.constant_(replacement_logits.bias, -math.log((1.0 - 0.01) / 0.01))
    old_head.cls_logits = replacement_logits
    old_head.num_classes = output_classes
    return model


def _torchvision_model(model_name: str, imgsz: int, nms_iou: float) -> Any:
    """Build a COCO-initialized torchvision detector with the correct label semantics.

    FCOS uses a sigmoid focal classifier with no explicit background logit and
    therefore has two logits/targets numbered 0--1.
    """

    from torchvision.models.detection import (
        FCOS_ResNet50_FPN_Weights,
        fcos_resnet50_fpn,
    )

    # With min_size=max_size=imgsz, torchvision caps the long image edge at
    # imgsz.  For these 4:3 camera frames this yields 640x480 content, matching
    # the content resolution inside YOLO's 640x640 letterbox.
    min_size, max_size = torchvision_resize_bounds(imgsz)
    if model_name == "fcos_resnet50_fpn":
        model = fcos_resnet50_fpn(
            weights=FCOS_ResNet50_FPN_Weights.DEFAULT,
            min_size=min_size,
            max_size=max_size,
        )
        _, output_classes = torchvision_label_contract(model_name)
        _replace_fcos_classification_logits(model, output_classes)
        model.score_thresh = 0.001
        model.nms_thresh = nms_iou
        model.detections_per_img = 300
        return model
    raise ValueError(model_name)


def _make_torch_dataset(records: Sequence[ImageRecord], flip_probability: float, label_offset: int) -> Any:
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision.transforms import functional as F

    class YoloExportDataset(Dataset):
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
            record = records[index]
            with Image.open(record.image) as source:
                image = source.convert("RGB")
            image_tensor = F.convert_image_dtype(F.pil_to_tensor(image), torch.float32)
            labels = parse_yolo_label(record.label)
            boxes = [normalized_box_to_xyxy(label, record.width, record.height) for label in labels]
            box_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            class_tensor = torch.as_tensor([label[0] + label_offset for label in labels], dtype=torch.int64)
            if flip_probability > 0.0 and torch.rand(1).item() < flip_probability:
                image_tensor = F.hflip(image_tensor)
                if box_tensor.numel():
                    old = box_tensor.clone()
                    box_tensor[:, 0] = record.width - old[:, 2]
                    box_tensor[:, 2] = record.width - old[:, 0]
            target = {
                "boxes": box_tensor,
                "labels": class_tensor,
                "image_id": torch.tensor([index], dtype=torch.int64),
                "area": (
                    (box_tensor[:, 2] - box_tensor[:, 0]) * (box_tensor[:, 3] - box_tensor[:, 1])
                    if box_tensor.numel()
                    else torch.empty((0,), dtype=torch.float32)
                ),
                "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
            }
            return image_tensor, target

    return YoloExportDataset()


def train_torchvision(
    model_name: str,
    train_records: Sequence[ImageRecord],
    model_dir: Path,
    args: argparse.Namespace,
    device_name: str,
) -> tuple[Any, Path, float, int, list[dict[str, float | int]], dict[str, str]]:
    import torch
    from torch.utils.data import DataLoader

    set_reproducible_seed(args.seed)
    model = _torchvision_model(model_name, args.imgsz, args.nms_iou)
    initializer = torchvision_initializer_artifact(model_name)
    model.to(torch.device(device_name))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    label_offset, _ = torchvision_label_contract(model_name)
    dataset = _make_torch_dataset(train_records, args.horizontal_flip, label_offset)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=lambda batch: tuple(zip(*batch)),
        generator=generator,
    )
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest = checkpoint_dir / "latest.pt"
    start_epoch = 0
    history: list[dict[str, float | int]] = []
    if args.resume and latest.is_file():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        expected_config = getattr(args, "run_config_sha256", None)
        if saved.get("run_config_sha256") != expected_config:
            raise ContractError(
                "Torchvision checkpoint configuration hash does not match this run; refusing unsafe resume."
            )
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        if "data_loader_generator_state" not in saved or "torch_rng_state" not in saved:
            raise ContractError("Checkpoint lacks RNG state required for a reproducible resume.")
        generator.set_state(saved["data_loader_generator_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        random.setstate(saved["python_rng_state"])
        if torch.cuda.is_available() and saved.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state_all"])
        if device_name == "mps" and saved.get("mps_rng_state") is not None and hasattr(torch.mps, "set_rng_state"):
            torch.mps.set_rng_state(saved["mps_rng_state"])
        start_epoch = int(saved["epoch"])
        history = list(saved.get("history", []))

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        batches = 0
        for images, targets in loader:
            images_device = [image.to(device_name) for image in images]
            targets_device = [{key: value.to(device_name) for key, value in target.items()} for target in targets]
            losses = model(images_device, targets_device)
            loss = sum(component for component in losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite {model_name} loss at epoch {epoch + 1}: {float(loss)}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            batches += 1
        history.append({"epoch": epoch + 1, "mean_training_loss": running_loss / max(1, batches)})
        payload = {
            "epoch": epoch + 1,
            "model_name": model_name,
            "run_config_sha256": getattr(args, "run_config_sha256", None),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "data_loader_generator_state": generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "mps_rng_state": (
                torch.mps.get_rng_state()
                if device_name == "mps" and hasattr(torch.mps, "get_rng_state")
                else None
            ),
            "history": history,
        }
        temporary = latest.with_name(".latest.pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(latest)
    train_seconds = time.perf_counter() - started
    # ``latest.pt`` is the final-epoch checkpoint after this fixed loop.  Keep
    # one copy because each torchvision optimizer checkpoint is several
    # hundred MB and may live on a synchronized filesystem.
    final_checkpoint = latest
    return model, final_checkpoint, train_seconds, parameter_count, history, initializer


def predict_torchvision(
    model: Any, val_records: Sequence[ImageRecord], device_name: str, model_name: str
) -> tuple[list[Detection], float]:
    import torch
    from PIL import Image
    from torchvision.transforms import functional as F

    model.eval()
    output: list[Detection] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for record in val_records:
            with Image.open(record.image) as source:
                tensor = F.convert_image_dtype(F.pil_to_tensor(source.convert("RGB")), torch.float32)
            result = model([tensor.to(device_name)])[0]
            for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
                label_offset, _ = torchvision_label_contract(model_name)
                class_id = int(label.detach().cpu()) - label_offset
                if class_id not in (0, 1):
                    continue
                output.append(
                    Detection(
                        image_id=record.image_id,
                        class_id=class_id,
                        score=float(score.detach().cpu()),
                        box_xyxy=tuple(float(value) for value in box.detach().cpu().tolist()),
                    )
                )
    return output, time.perf_counter() - started


def train_and_predict_yolo(
    run_name: str,
    data_yaml: Path,
    val_records: Sequence[ImageRecord],
    model_dir: Path,
    args: argparse.Namespace,
    device_name: str,
) -> tuple[list[Detection], Path, float, float, int, list[dict[str, Any]]]:
    from ultralytics import YOLO

    set_reproducible_seed(args.seed)
    trainer_class = yolo_benchmark_trainer_class()
    training_root = model_dir / "ultralytics"
    run_dir = training_root / "train"
    last_checkpoint = run_dir / "weights" / "last.pt"
    started = time.perf_counter()
    if args.resume and last_checkpoint.is_file() and not (model_dir / "metrics.json").is_file():
        model = YOLO(str(last_checkpoint))
        import torch

        checkpoint_payload = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        saved_epoch = int(checkpoint_payload.get("epoch", -1))
        # Ultralytics strips completed checkpoints and records epoch=-1.  In
        # that case training already reached the fixed endpoint and only the
        # script-owned prediction/metric phase needs to resume.
        if saved_epoch != -1 and saved_epoch + 1 < args.epochs:
            model.train(trainer=trainer_class, resume=True)
    else:
        model = YOLO(args.yolo_weights)
        names = getattr(model, "names", {})
        if len(names) < 80 or "car" not in set(names.values()) or "boat" not in set(names.values()):
            raise ContractError("--yolo-weights do not appear to be an 80-class COCO-initialized checkpoint.")
        model.train(
            trainer=trainer_class,
            data=str(data_yaml),
            model=args.yolo_weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch_size,
            device=device_name,
            workers=args.workers,
            seed=args.seed,
            deterministic=True,
            amp=False,
            optimizer="SGD",
            nbs=args.batch_size,
            lr0=args.learning_rate,
            lrf=1.0,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            warmup_epochs=0.0,
            patience=args.epochs + 1,
            val=False,
            save=True,
            plots=False,
            project=str(training_root),
            name="train",
            exist_ok=args.resume,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=args.horizontal_flip,
            bgr=0.0,
            auto_augment=None,
            mosaic=0.0,
            mixup=0.0,
            cutmix=0.0,
            copy_paste=0.0,
            erasing=0.0,
            close_mosaic=0,
            cos_lr=False,
        )
    train_seconds = time.perf_counter() - started
    if not last_checkpoint.is_file():
        raise FileNotFoundError(f"YOLO final-epoch checkpoint missing: {last_checkpoint}")

    final_model = YOLO(str(last_checkpoint))
    parameter_count = sum(parameter.numel() for parameter in final_model.model.parameters())
    image_paths = [str(record.image) for record in val_records]
    started = time.perf_counter()
    prediction_device = yolo_inference_device(device_name)
    results = final_model.predict(
        source=image_paths,
        imgsz=args.imgsz,
        conf=0.001,
        iou=args.nms_iou,
        max_det=300,
        device=prediction_device,
        stream=False,
        verbose=False,
    )
    inference_seconds = time.perf_counter() - started
    if len(results) != len(val_records):
        raise RuntimeError(f"YOLO returned {len(results)} results for {len(val_records)} validation images.")
    detections: list[Detection] = []
    for record, result in zip(val_records, results):
        boxes = result.boxes
        if boxes is None:
            continue
        for xyxy, confidence, class_value in zip(boxes.xyxy, boxes.conf, boxes.cls):
            class_id = int(class_value.detach().cpu())
            if class_id not in (0, 1):
                raise ContractError(f"YOLO benchmark emitted unexpected class ID {class_id}.")
            detections.append(
                Detection(
                    image_id=record.image_id,
                    class_id=class_id,
                    score=float(confidence.detach().cpu()),
                    box_xyxy=tuple(float(value) for value in xyxy.detach().cpu().tolist()),
                )
            )
    history: list[dict[str, Any]] = []
    results_csv = run_dir / "results.csv"
    if results_csv.is_file():
        with results_csv.open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
    return detections, last_checkpoint, train_seconds, inference_seconds, parameter_count, history


def benchmark_one(
    model_name: str,
    train_records: Sequence[ImageRecord],
    val_records: Sequence[ImageRecord],
    data_yaml: Path,
    output_dir: Path,
    args: argparse.Namespace,
    device_name: str,
    *,
    result_label: str | None = None,
) -> dict[str, Any]:
    label = result_label or model_name
    model_dir = output_dir / "models" / label
    metrics_path = model_dir / "metrics.json"
    predictions_path = model_dir / "predictions.jsonl"
    if args.resume and metrics_path.is_file() and predictions_path.is_file():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_name == "yolo26n":
        predictions, checkpoint, train_seconds, inference_seconds, parameter_count, history = train_and_predict_yolo(
            label, data_yaml, val_records, model_dir, args, device_name
        )
        family = "one-stage anchor-free detector"
        initialization = "Ultralytics YOLO26n COCO weights"
        training_device = device_name
        inference_device = yolo_inference_device(device_name)
        yolo_path = Path(args.yolo_weights).expanduser()
        initializer = {
            "identifier": args.yolo_weights,
            "path": str(yolo_path.resolve()) if yolo_path.is_file() else None,
            "sha256": sha256_file(yolo_path.resolve()) if yolo_path.is_file() else None,
        }
    else:
        model, checkpoint, train_seconds, parameter_count, history, initializer = train_torchvision(
            model_name, train_records, model_dir, args, device_name
        )
        predictions, inference_seconds = predict_torchvision(model, val_records, device_name, model_name)
        family = "one-stage anchor-free detector"
        initialization = f"torchvision {model_name} COCO DEFAULT weights"
        training_device = device_name
        inference_device = device_name

    write_predictions(
        predictions_path,
        predictions,
        [record.image_id for record in val_records],
        resume=args.resume,
    )
    ground_truth = ground_truth_detections(val_records)
    image_ids = [record.image_id for record in val_records]
    point = evaluate_predictions(ground_truth, predictions, image_ids, max_detections=args.max_detections)
    intervals = bootstrap_confidence_intervals(
        ground_truth,
        predictions,
        image_ids,
        replicates=args.bootstrap_replicates,
        seed=args.seed + 1009,
        max_detections=args.max_detections,
    )
    metrics: dict[str, Any] = {
        "model": label,
        "architecture": model_name,
        "family": family,
        "initialization": initialization,
        "training_device": training_device,
        "inference_device": inference_device,
        "initializer_artifact": initializer,
        "scientific_status": "smoke_only" if args.smoke else "post_hoc_common_validation_benchmark",
        "validation_is_independent_test": False,
        "checkpoint_selection": "fixed final epoch; validation not used for early stopping or model selection",
        "epochs": args.epochs,
        "train_list_entries": len(train_records),
        "train_unique_images": len({record.image_id for record in train_records}),
        "validation_images": len(val_records),
        "parameters": parameter_count,
        "training_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "predictions": str(predictions_path.resolve()),
        "point_estimates": point,
        "image_bootstrap_95_ci": intervals,
        "bootstrap_replicates": args.bootstrap_replicates,
        "coco_style_contract": {
            "interpolation": "101 recall points",
            "iou_thresholds": list(IOU_THRESHOLDS),
            "area_range": "all",
            "max_detections_per_image": args.max_detections,
            "crowd_or_ignore": "not present in source annotations",
        },
    }
    _write_json(model_dir / "training_history.json", history, resume=args.resume, allow_managed_update=True)
    _write_json(metrics_path, metrics, resume=args.resume, allow_managed_update=True)
    return metrics


def flatten_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        key: metrics.get(key)
        for key in (
            "model",
            "architecture",
            "family",
            "scientific_status",
            "epochs",
            "train_list_entries",
            "train_unique_images",
            "validation_images",
            "parameters",
            "training_seconds",
            "inference_seconds",
        )
    }
    point = metrics["point_estimates"]
    intervals = metrics["image_bootstrap_95_ci"]
    for class_name in CLASS_NAMES:
        for metric_name in ("ap50", "ap50_95"):
            key = f"{class_name}_{metric_name}"
            row[key] = point[key]
            row[f"{key}_ci_low"] = intervals[key]["low"]
            row[f"{key}_ci_high"] = intervals[key]["high"]
    return row


def write_summary(output_dir: Path, metrics: Sequence[Mapping[str, Any]], *, resume: bool) -> None:
    rows = [flatten_summary(item) for item in metrics]
    _write_json(output_dir / "summary.json", rows, resume=resume, allow_managed_update=True)
    if not rows:
        return
    import io

    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _write_text(output_dir / "summary.csv", stream.getvalue(), resume=resume, allow_managed_update=True)


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def config_payload(args: argparse.Namespace, dataset_root: Path, output_dir: Path) -> dict[str, Any]:
    epochs = 1 if args.smoke else args.epochs
    yolo_weights_path = Path(args.yolo_weights).expanduser()
    local_yolo_weights = yolo_weights_path.resolve() if yolo_weights_path.is_file() else None
    return {
        "schema_version": 2,
        "benchmark_scope": "paper-specific post-hoc common-validation architecture comparison",
        "entrypoint": str(Path(__file__).resolve()),
        "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "expected_split_counts": {
            "train": args.expected_train_images,
            "val": args.expected_val_images,
        },
        "paper_split_counts": dict(PAPER_SPLIT_COUNTS),
        "historical_hard_list": (
            str(args.historical_hard_list.expanduser().resolve())
            if args.historical_hard_list is not None
            else None
        ),
        "expected_hard_train_images": args.expected_hard_train_images,
        "hard_example_extra_repeats": args.hard_example_extra_repeats,
        "paper_hard_sampling_reproduced": args.historical_hard_list is not None,
        "al_batches_root": (
            str(args.al_batches_root.expanduser().resolve()) if args.al_batches_root is not None else None
        ),
        "models": list(dict.fromkeys(args.models)),
        "yolo_cumulative_stages": args.yolo_cumulative_stages,
        "yolo_weights": str(local_yolo_weights) if local_yolo_weights else args.yolo_weights,
        "yolo_weights_sha256": sha256_file(local_yolo_weights) if local_yolo_weights else None,
        "torch_home": str(args.torch_home.expanduser().resolve()),
        "torchvision_initializers": {
            "fcos_resnet50_fpn": "FCOS_ResNet50_FPN_Weights.DEFAULT (COCO_V1)",
        },
        "epochs": epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "imgsz": args.imgsz,
        "spatial_transform": (
            "longest image edge capped at imgsz; YOLO letterboxes to imgsz square and torchvision uses "
            "min_size=max_size=imgsz"
        ),
        "requested_device": args.device,
        "training_device": getattr(args, "resolved_training_device", None),
        "inference_devices": (
            {
                "yolo26n": yolo_inference_device(args.resolved_training_device),
                "fcos_resnet50_fpn": args.resolved_training_device,
            }
            if getattr(args, "resolved_training_device", None) is not None
            else None
        ),
        "seed": args.seed,
        "optimizer": "SGD",
        "effective_update_batch_size": args.batch_size,
        "precision": "float32 (Ultralytics AMP disabled)",
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "scheduler": "constant",
        "horizontal_flip_probability": args.horizontal_flip,
        "other_training_augmentation": "disabled",
        "recipe_scope": (
            "Common data, content resolution, update batch, SGD hyperparameters, epochs, and horizontal flip; "
            "architecture-native losses and parameter grouping remain framework-specific."
        ),
        "nms_iou": args.nms_iou,
        "bootstrap_replicates": args.bootstrap_replicates,
        "max_detections": args.max_detections,
        "dry_run": args.dry_run,
        "smoke": args.smoke,
        "smoke_train_images": args.smoke_train_images,
        "smoke_val_images": args.smoke_val_images,
        "validation_usage": "final-epoch comparison only; no early stopping or checkpoint selection",
        "package_versions": _package_versions(("torch", "torchvision", "ultralytics", "Pillow", "PyYAML")),
    }


def write_checksums(output_dir: Path, source_paths: Sequence[Path], *, resume: bool) -> None:
    generated = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "checksums.sha256" and not path.name.startswith(".")
    ]
    paths = sorted({path.resolve() for path in list(source_paths) + generated}, key=str)
    lines = [f"{sha256_file(path)}  {path}\n" for path in paths]
    _write_text(output_dir / "checksums.sha256", "".join(lines), resume=resume, allow_managed_update=True)


def validate_cli(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.imgsz <= 0:
        raise ContractError("Epochs, batch size, and image size must be positive.")
    if args.workers < 0 or args.smoke_train_images <= 0 or args.smoke_val_images <= 0:
        raise ContractError("Worker count cannot be negative and smoke subset sizes must be positive.")
    if not (0.0 <= args.horizontal_flip <= 1.0):
        raise ContractError("--horizontal-flip must lie in [0,1].")
    if not (0.0 < args.nms_iou <= 1.0):
        raise ContractError("--nms-iou must lie in (0,1].")
    if args.max_detections <= 0:
        raise ContractError("--max-detections must be positive.")
    if args.bootstrap_replicates < 0:
        raise ContractError("--bootstrap-replicates cannot be negative.")
    if args.expected_train_images <= 0 or args.expected_val_images <= 0:
        raise ContractError("--expected-train-images and --expected-val-images must be positive.")
    if args.expected_hard_train_images < 0 or args.hard_example_extra_repeats < 0:
        raise ContractError("Hard-example counts and repeats cannot be negative.")
    if args.workers != 0 and any(name != "yolo26n" for name in args.models):
        raise ContractError(
            "Torchvision benchmark models require --workers 0: the local deterministic dataset/transform closure "
            "is deliberately not serialized into spawned worker processes."
        )
    if args.dry_run and args.smoke:
        raise ContractError("Choose either --dry-run or --smoke, not both.")
    if args.yolo_cumulative_stages and args.al_batches_root is None:
        raise ContractError("--yolo-cumulative-stages requires an explicit --al-batches-root.")
    if args.yolo_cumulative_stages and "yolo26n" not in args.models:
        raise ContractError("--yolo-cumulative-stages requires yolo26n in --models.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_cli(args)
        if args.smoke:
            args.epochs = 1
            args.bootstrap_replicates = min(args.bootstrap_replicates, 20)
        dataset_root = args.dataset_root.expanduser().resolve()
        args.torch_home = (
            args.torch_home.expanduser().resolve()
            if args.torch_home is not None
            else (dataset_root.parent / ".cache" / "payette_detector_benchmark").resolve()
        )
        output_dir = prepare_output(args.output_dir, args.resume)

        device_name: str | None = None
        if not args.dry_run:
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            args.torch_home.mkdir(parents=True, exist_ok=True)
            os.environ["TORCH_HOME"] = str(args.torch_home)
            try:
                device_name = choose_device(args.device)
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Training requires torch, torchvision, and ultralytics. "
                    "The data audit can be run without them via --dry-run."
                ) from exc
        args.resolved_training_device = device_name

        config = config_payload(args, dataset_root, output_dir)
        args.run_config_sha256 = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        config_path = output_dir / "config.json"
        _write_json(config_path, config, resume=args.resume)
        train, val = validate_dataset(
            dataset_root,
            expected_train_images=args.expected_train_images,
            expected_val_images=args.expected_val_images,
        )
        temporal_proximity, temporal_rows = audit_same_camera_temporal_proximity(train, val)
        if args.historical_hard_list is not None:
            hard_names, rejected_val_names = recover_hard_training_names(
                args.historical_hard_list,
                train,
                val,
                expected_train_names=args.expected_hard_train_images,
            )
        else:
            hard_names, rejected_val_names = [], []
        effective_train = build_hard_sampled_records(
            train,
            hard_names,
            extra_repeats=args.hard_example_extra_repeats,
        )

        lists_dir = output_dir / "lists"
        base_train_list = lists_dir / "train_base.txt"
        effective_train_list = lists_dir / "train_effective.txt"
        val_list = lists_dir / "val.txt"
        data_yaml = lists_dir / "data.yaml"
        write_path_list(base_train_list, train, resume=args.resume)
        write_path_list(effective_train_list, effective_train, resume=args.resume)
        write_path_list(val_list, val, resume=args.resume)
        write_data_yaml(data_yaml, effective_train_list, val_list, resume=args.resume)
        write_dataset_manifest(output_dir / "dataset_manifest.csv", [*train, *val], set(hard_names), resume=args.resume)
        write_temporal_proximity_audit(
            output_dir / "validation_temporal_proximity.csv", temporal_rows, resume=args.resume
        )

        stage_lists: dict[str, tuple[list[ImageRecord], Path]] = {}
        if args.yolo_cumulative_stages:
            assert args.al_batches_root is not None
            stages, stage_metadata = reconstruct_annotation_stages(
                train, val, args.al_batches_root.expanduser().resolve()
            )
            stage_metadata["stages"] = {}
            for stage_name, records in stages.items():
                list_path = lists_dir / f"{stage_name}.txt"
                yaml_path = lists_dir / f"{stage_name}.yaml"
                write_path_list(list_path, records, resume=args.resume)
                write_data_yaml(yaml_path, list_path, val_list, resume=args.resume)
                stage_lists[stage_name] = (records, yaml_path)
                stage_metadata["stages"][stage_name] = {
                    "images": len(records),
                    "vehicle_instances": sum(record.n_vehicle for record in records),
                    "watercraft_instances": sum(record.n_watercraft for record in records),
                }
            _write_json(output_dir / "annotation_stages.json", stage_metadata, resume=args.resume)

        provenance_path = output_dir / "provenance.json"
        candidate_provenance = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "git_revision": _git_revision(),
                "run_config_sha256": args.run_config_sha256,
                "entrypoint_sha256": config["entrypoint_sha256"],
                "yolo_initializer_sha256": config["yolo_weights_sha256"],
                "python": sys.version,
                "platform": platform.platform(),
                "packages": config["package_versions"],
                "devices": {
                    "requested": config["requested_device"],
                    "training": config["training_device"],
                    "inference": config["inference_devices"],
                },
                "dataset_contract": {
                    "train_images": len(train),
                    "validation_images": len(val),
                    "train_validation_basename_overlap": 0,
                    "train_validation_sha256_overlap": 0,
                    "classes": list(CLASS_NAMES),
                    "vehicle_train_instances": sum(record.n_vehicle for record in train),
                    "watercraft_train_instances": sum(record.n_watercraft for record in train),
                    "vehicle_validation_instances": sum(record.n_vehicle for record in val),
                    "watercraft_validation_instances": sum(record.n_watercraft for record in val),
                    "same_camera_temporal_proximity": temporal_proximity,
                },
                "hard_sampling": {
                    "enabled": args.historical_hard_list is not None,
                    "paper_benchmark_reproduction_requires_this_input": True,
                    "base_train_images": len(train),
                    "recoverable_train_only_mined_images": hard_names,
                    "extra_repeats_per_image": args.hard_example_extra_repeats,
                    "effective_train_list_entries": len(effective_train),
                    "historical_validation_names_rejected": rejected_val_names,
                },
                "scientific_scope": (
                    "Paper-specific post-hoc common-validation architecture benchmark. Validation was frozen before "
                    "these fresh runs and is not used for early stopping or checkpoint selection; it is not an "
                    "independent test set. Without --historical-hard-list, the run is a portable base-inventory "
                    "adaptation rather than an exact reproduction of the reported hard-sampled benchmark."
                ),
            }
        preserve_immutable_provenance(
            provenance_path,
            candidate_provenance,
            resume=args.resume,
            run_config_sha256=args.run_config_sha256,
        )

        source_paths = [
            *[record.image for record in train],
            *[record.label for record in train],
            *[record.image for record in val],
            *[record.label for record in val],
        ]
        if args.historical_hard_list is not None:
            source_paths.append(args.historical_hard_list.expanduser().resolve())
        if args.yolo_cumulative_stages:
            assert args.al_batches_root is not None
            source_paths.extend(
                [
                    (args.al_batches_root / "batch_001" / "paths.txt").expanduser().resolve(),
                    (args.al_batches_root / "batch_002_river" / "paths.txt").expanduser().resolve(),
                ]
            )
        if args.dry_run:
            write_checksums(output_dir, source_paths, resume=args.resume)
            print(json.dumps({"status": "dry_run_complete", "output_dir": str(output_dir)}, indent=2))
            return 0

        assert device_name is not None

        selected_train = effective_train
        selected_val = val
        if args.smoke:
            selected_train = effective_train[: min(len(effective_train), args.smoke_train_images)]
            selected_val = val[: min(len(val), args.smoke_val_images)]
            smoke_train_list = lists_dir / "smoke_train.txt"
            smoke_val_list = lists_dir / "smoke_val.txt"
            smoke_yaml = lists_dir / "smoke_data.yaml"
            write_path_list(smoke_train_list, selected_train, resume=args.resume)
            write_path_list(smoke_val_list, selected_val, resume=args.resume)
            write_data_yaml(smoke_yaml, smoke_train_list, smoke_val_list, resume=args.resume)
            data_yaml = smoke_yaml

        all_metrics: list[dict[str, Any]] = []
        for model_name in dict.fromkeys(args.models):
            all_metrics.append(
                benchmark_one(
                    model_name,
                    selected_train,
                    selected_val,
                    data_yaml,
                    output_dir,
                    args,
                    device_name,
                )
            )
        if args.yolo_cumulative_stages and "yolo26n" in args.models:
            for stage_name, (stage_records, stage_yaml) in stage_lists.items():
                if args.smoke:
                    stage_records = stage_records[: min(len(stage_records), args.smoke_train_images)]
                    # The stage YAML points to the full stage.  Materialize a smoke-specific stage list.
                    smoke_stage_list = lists_dir / f"{stage_name}_smoke.txt"
                    smoke_stage_yaml = lists_dir / f"{stage_name}_smoke.yaml"
                    write_path_list(smoke_stage_list, stage_records, resume=args.resume)
                    write_data_yaml(smoke_stage_yaml, smoke_stage_list, lists_dir / "smoke_val.txt", resume=args.resume)
                    stage_yaml = smoke_stage_yaml
                all_metrics.append(
                    benchmark_one(
                        "yolo26n",
                        stage_records,
                        selected_val,
                        stage_yaml,
                        output_dir,
                        args,
                        device_name,
                        result_label=f"yolo26n_{stage_name}",
                    )
                )
        write_summary(output_dir, all_metrics, resume=args.resume)
        for metric_payload in all_metrics:
            initializer_path = metric_payload.get("initializer_artifact", {}).get("path")
            if initializer_path:
                source_paths.append(Path(initializer_path))
        write_checksums(output_dir, source_paths, resume=args.resume)
        print(json.dumps({"status": "complete", "output_dir": str(output_dir), "models": len(all_metrics)}, indent=2))
        return 0
    except (ContractError, FileExistsError, FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
