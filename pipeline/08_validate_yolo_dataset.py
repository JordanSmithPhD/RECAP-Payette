#!/usr/bin/env python3
"""Validate a two-class YOLO dataset before training or benchmarking."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=("train", "val", "test"))
    parser.add_argument("--allow-missing-split", action="store_true")
    parser.add_argument("--summary", type=Path, default=None)
    return parser


def validate_label(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return counts
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected five YOLO fields")
        class_id = int(fields[0])
        if class_id not in {0, 1}:
            raise ValueError(f"{path}:{line_number}: class must be 0 (vehicle) or 1 (watercraft)")
        x_center, y_center, width, height = (float(value) for value in fields[1:])
        if not all(0 <= value <= 1 for value in (x_center, y_center, width, height)):
            raise ValueError(f"{path}:{line_number}: normalized coordinates must fall within [0, 1]")
        if width <= 0 or height <= 0:
            raise ValueError(f"{path}:{line_number}: width and height must be positive")
        if x_center - width / 2 < 0 or x_center + width / 2 > 1:
            raise ValueError(f"{path}:{line_number}: box crosses the horizontal image boundary")
        if y_center - height / 2 < 0 or y_center + height / 2 > 1:
            raise ValueError(f"{path}:{line_number}: box crosses the vertical image boundary")
        counts[class_id] += 1
    return counts


def run(args: argparse.Namespace) -> int:
    root = args.dataset_root.expanduser().resolve(strict=True)
    content_splits: dict[str, str] = {}
    split_summaries: dict[str, dict[str, int]] = {}
    for split in args.splits:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            if args.allow_missing_split:
                continue
            raise FileNotFoundError(f"Missing images/{split} or labels/{split}")
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            if args.allow_missing_split:
                continue
            raise ValueError(f"No supported images in {image_dir}")
        label_index = {path.stem: path for path in label_dir.glob("*.txt")}
        image_stems = [path.stem for path in images]
        if len(image_stems) != len(set(image_stems)):
            raise ValueError(f"Duplicate image stems in {image_dir}")
        missing_labels = sorted(set(image_stems).difference(label_index))
        extra_labels = sorted(set(label_index).difference(image_stems))
        if missing_labels or extra_labels:
            raise ValueError(
                f"Image/label mismatch in {split}: missing={missing_labels[:5]}, extra={extra_labels[:5]}"
            )
        class_counts: Counter[int] = Counter()
        for image in images:
            with Image.open(image) as opened:
                opened.verify()
            digest = sha256_file(image)
            previous = content_splits.setdefault(digest, split)
            if previous != split:
                raise ValueError(f"Byte-identical image appears in both {previous} and {split}: {image.name}")
            class_counts.update(validate_label(label_index[image.stem]))
        split_summaries[split] = {
            "images": len(images),
            "vehicle_boxes": class_counts[0],
            "watercraft_boxes": class_counts[1],
            "empty_images": sum(not label_index[stem].read_text(encoding="utf-8").strip() for stem in image_stems),
        }

    summary = {
        "dataset_root_recorded": False,
        "class_names": {"0": "vehicle", "1": "watercraft"},
        "unique_image_hashes": len(content_splits),
        "splits": split_summaries,
    }
    if args.summary:
        path = args.summary.expanduser().resolve(strict=False)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
