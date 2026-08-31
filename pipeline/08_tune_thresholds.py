#!/usr/bin/env python3
"""Choose a class-specific confidence threshold on a frozen validation set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def is_valid_image_path(path: Path) -> bool:
    name = path.name
    return not name.startswith(".") and path.suffix.lower() in {".jpg", ".jpeg", ".png"}


def can_open_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def load_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing validation label: {path}")
    labels: list[tuple[int, float, float, float, float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected five YOLO fields")
        class_id, x, y, width, height = fields
        labels.append((int(float(class_id)), float(x), float(y), float(width), float(height)))
    return labels


def xywhn_to_xyxy(
    x: float,
    y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    return np.array(
        [
            (x - width / 2) * image_width,
            (y - height / 2) * image_height,
            (x + width / 2) * image_width,
            (y + height / 2) * image_height,
        ],
        dtype=float,
    )


def iou_xyxy(left: np.ndarray, right: np.ndarray) -> float:
    x_min = max(left[0], right[0])
    y_min = max(left[1], right[1])
    x_max = min(left[2], right[2])
    y_max = min(left[3], right[3])
    intersection = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return float(intersection / max(1e-9, left_area + right_area - intersection))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="YOLO export root containing images/ and labels/")
    parser.add_argument("--weights", type=Path, required=True, help="Detector checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cls", type=int, choices=(0, 1), default=1, help="0=vehicle, 1=watercraft")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--conf-min", type=float, default=0.01)
    parser.add_argument("--conf-max", type=float, default=0.60)
    parser.add_argument("--conf-steps", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser


def _match_counts(
    ground_truth: list[np.ndarray],
    predictions: list[tuple[float, np.ndarray]],
    threshold: float,
    match_iou: float,
) -> tuple[int, int, int]:
    retained = [box for confidence, box in predictions if confidence >= threshold]
    matched: set[int] = set()
    true_positive = false_positive = 0
    for prediction in retained:
        best_index: int | None = None
        best_iou = 0.0
        for index, reference in enumerate(ground_truth):
            if index in matched:
                continue
            overlap = iou_xyxy(prediction, reference)
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index is not None and best_iou >= match_iou:
            matched.add(best_index)
            true_positive += 1
        else:
            false_positive += 1
    return true_positive, false_positive, len(ground_truth) - len(matched)


def run(args: argparse.Namespace) -> int:
    if not (0 < args.match_iou <= 1 and 0 < args.nms_iou <= 1):
        raise ValueError("IoU thresholds must fall within (0, 1]")
    if not (0 <= args.conf_min <= args.conf_max <= 1) or args.conf_steps < 2:
        raise ValueError("confidence bounds must fall within [0, 1], and --conf-steps must be at least two")
    data_root = args.data_root.expanduser().resolve(strict=True)
    weights = args.weights.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing nonempty output directory {output_dir}")

    image_dir = data_root / "images" / "val"
    label_dir = data_root / "labels" / "val"
    images_all = sorted(path for path in image_dir.glob("*") if is_valid_image_path(path))
    images = [path for path in images_all if can_open_image(path)]
    if not images:
        raise RuntimeError(f"No readable validation images found in {image_dir}")
    if len(images) != len(images_all):
        raise RuntimeError("Unreadable validation images must be resolved before threshold selection")

    inventory: list[dict[str, Any]] = []
    inventory_digest = hashlib.sha256()
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing validation label: {label}")
        row = {
            "image_name": image.name,
            "image_sha256": sha256_file(image),
            "label_name": label.name,
            "label_sha256": sha256_file(label),
        }
        inventory.append(row)
        inventory_digest.update(json.dumps(row, sort_keys=True).encode("utf-8"))

    from ultralytics import YOLO, __version__ as ultralytics_version

    model = YOLO(str(weights))
    results = list(
        model.predict(
            source=[str(path) for path in images],
            imgsz=int(args.imgsz),
            conf=float(args.conf_min),
            iou=float(args.nms_iou),
            device=str(args.device),
            batch=int(args.batch),
            stream=True,
            verbose=False,
        )
    )
    if len(results) != len(images):
        raise RuntimeError("Detector returned a different number of results than validation images")

    cached: list[tuple[list[np.ndarray], list[tuple[float, np.ndarray]]]] = []
    for image_path, result in zip(images, results, strict=True):
        height, width = result.orig_shape
        references = [
            xywhn_to_xyxy(x, y, box_width, box_height, width, height)
            for class_id, x, y, box_width, box_height in load_yolo_labels(label_dir / f"{image_path.stem}.txt")
            if class_id == args.cls
        ]
        predictions: list[tuple[float, np.ndarray]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and getattr(boxes, "xyxy", None) is not None:
            bounds = boxes.xyxy.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
            predictions = [
                (float(confidence), box.astype(float))
                for box, class_id, confidence in zip(bounds, classes, confidences, strict=True)
                if int(class_id) == args.cls
            ]
        cached.append((references, predictions))

    rows: list[dict[str, Any]] = []
    for threshold in np.linspace(args.conf_min, args.conf_max, args.conf_steps):
        true_positive = false_positive = false_negative = 0
        for references, predictions in cached:
            counts = _match_counts(references, predictions, float(threshold), float(args.match_iou))
            true_positive += counts[0]
            false_positive += counts[1]
            false_negative += counts[2]
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        rows.append(
            {
                "confidence": float(threshold),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    best = max(rows, key=lambda row: (row["f1"], row["recall"], row["confidence"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (output_dir / "threshold_metrics.csv").write_text(stream.getvalue(), encoding="utf-8")
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(inventory[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(inventory)
    (output_dir / "validation_inventory.csv").write_text(stream.getvalue(), encoding="utf-8")
    manifest = {
        "stage": "08_tune_thresholds",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "class_id": args.cls,
        "class_name": "vehicle" if args.cls == 0 else "watercraft",
        "match_iou": args.match_iou,
        "nms_iou": args.nms_iou,
        "confidence_grid": [args.conf_min, args.conf_max, args.conf_steps],
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "validation_images": len(images),
        "validation_inventory_sha256": inventory_digest.hexdigest(),
        "weights_sha256": sha256_file(weights),
        "python": platform.python_version(),
        "ultralytics": ultralytics_version,
        "selection_rule": "maximum F1; ties resolved by recall then higher confidence",
        "selected": best,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "selected": best}, indent=2, sort_keys=True))
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
