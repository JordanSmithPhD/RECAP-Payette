"""Export reviewed FiftyOne detections to Ultralytics YOLO format.

What it exports
---------------
- Images and labels are copied to ``train``, ``val``, and optionally ``test``
  subdirectories.
- data.yaml written to:
    OUT_DIR/data.yaml

Label assumptions
-----------------
- FiftyOne bounding boxes are normalized [x, y, w, h] with (x, y) = top-left.
- YOLO expects normalized [class_id, x_center, y_center, width, height].

Class list / mapping
--------------------
This exporter writes a *two-class* YOLO dataset:
  - vehicle (car, bus, truck, or already-coarsened vehicle)
  - watercraft (boat or already-coarsened watercraft)

`LABEL_MAP` controls how your FiftyOne `ground_truth.detections.label` values
are mapped into the two-class taxonomy. Unknown labels abort the export rather
than being silently omitted.

Notes
-----
- If an image is a true zero, its label file should be empty.
- If the label field is missing/None on a sample, we export an empty label file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from project_paths import YOLO_AL_V1_ROOT


# -----------------------------------------------------------------------------
# Two-class objective (your training target)
# -----------------------------------------------------------------------------
# You want to train 2 classes:
#   - vehicle:    car, bus, truck
#   - watercraft: boat
#
# We export exactly these two classes in a fixed order to guarantee stable IDs:
#   0 = vehicle
#   1 = watercraft
CLASSES_OVERRIDE = ["vehicle", "watercraft"]

# Map your ground-truth fine-grained labels -> the two-class taxonomy
LABEL_MAP = {
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "boat": "watercraft",
    "vehicle": "vehicle",
    "watercraft": "watercraft",
}


def map_source_label(label: object) -> str:
    """Map one source label to the fixed two-class taxonomy.

    Raises:
        ValueError: If ``label`` is missing or outside the explicit contract.
    """
    mapped = LABEL_MAP.get(label) if isinstance(label, str) else None
    if mapped is None:
        allowed = ", ".join(sorted(LABEL_MAP))
        raise ValueError(f"Unsupported source label {label!r}; expected one of: {allowed}")
    return mapped


def validate_source_labels(ds: object, label_field: str, split_field: str, splits: tuple[str, ...]) -> None:
    """Fail before writing if an exported sample has an unknown label."""
    unknown: set[str] = set()
    for split_value in splits:
        view = ds.match({split_field: split_value})
        for sample in view.select_fields([label_field]):
            gt = getattr(sample, label_field, None)
            dets = getattr(gt, "detections", None) if gt is not None else None
            for detection in dets or []:
                label = getattr(detection, "label", None)
                try:
                    map_source_label(label)
                except ValueError:
                    unknown.add(repr(label))

    if unknown:
        allowed = ", ".join(sorted(LABEL_MAP))
        found = ", ".join(sorted(unknown))
        raise ValueError(
            f"Unsupported labels in exported samples: {found}. "
            f"Expected one of: {allowed}. No export files were written."
        )


def prepare_output_directory(out_dir: Path, *, clean: bool) -> None:
    """Create an empty export destination or explicitly clean managed outputs."""
    if out_dir.exists() and not out_dir.is_dir():
        raise NotADirectoryError(f"YOLO export destination is not a directory: {out_dir}")

    if out_dir.exists() and any(out_dir.iterdir()) and not clean:
        raise FileExistsError(
            f"YOLO export destination is not empty: {out_dir}. "
            "Choose a new --out directory or pass --clean to replace managed export artifacts."
        )

    if clean and out_dir.exists():
        for name in ("images", "labels"):
            managed = out_dir / name
            if managed.is_symlink() or managed.is_file():
                managed.unlink()
            elif managed.is_dir():
                shutil.rmtree(managed)
        data_yaml = out_dir / "data.yaml"
        if data_yaml.exists() or data_yaml.is_symlink():
            data_yaml.unlink()

    out_dir.mkdir(parents=True, exist_ok=True)


def fo_to_yolo_xywh(bb):
    """Convert FiftyOne [x, y, w, h] (top-left) to YOLO [xc, yc, w, h]."""
    x, y, w, h = bb
    xc = x + w / 2.0
    yc = y + h / 2.0
    return xc, yc, w, h




def export_split(
    ds: object,
    split_value: str,
    out_dir: Path,
    label_field: str,
    split_field: str,
    cls_to_id: dict[str, int],
    verbose: bool = False,
) -> int:
    """Export one partition and return its sample count."""
    view = ds.match({split_field: split_value})

    img_dir = out_dir / "images" / split_value
    lbl_dir = out_dir / "labels" / split_value
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n = view.count()
    if verbose:
        print(f"Exporting split='{split_value}' with {n} samples")

    # Pull only the label field to reduce DB payload
    for s in view.select_fields([label_field]):
        src = Path(s.filepath)
        dst = img_dir / src.name

        # Copy image (skip if already copied)
        if not dst.exists():
            shutil.copy2(src, dst)

        # Build YOLO label lines
        lines: list[str] = []

        gt = getattr(s, label_field, None)
        dets = getattr(gt, "detections", None) if gt is not None else None

        if dets:
            for d in dets:
                label = getattr(d, "label", None)
                bb = getattr(d, "bounding_box", None)
                if bb is None:
                    continue
                # Map fine-grained label -> two-class taxonomy
                mapped = map_source_label(label)
                if mapped not in cls_to_id:
                    raise RuntimeError(f"Mapped label {mapped!r} has no configured class ID")

                xc, yc, w, h = fo_to_yolo_xywh(bb)

                # Defensive clamp (keeps files valid even if a few boxes are slightly out of bounds)
                # YOLO expects normalized [0,1] values.
                xc = min(max(xc, 0.0), 1.0)
                yc = min(max(yc, 0.0), 1.0)
                w = min(max(w, 0.0), 1.0)
                h = min(max(h, 0.0), 1.0)

                lines.append(f"{cls_to_id[mapped]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        # True zero => empty file
        (lbl_dir / f"{src.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    return n


def write_data_yaml(out_dir: Path, classes: list[str], include_test: bool) -> None:
    """Write Ultralytics-style data.yaml."""
    yaml = f"path: {out_dir}\ntrain: images/train\nval: images/val\n"
    if include_test:
        yaml += "test: images/test\n"
    yaml += "names:\n"
    for i, c in enumerate(classes):
        yaml += f"  {i}: {c}\n"

    (out_dir / "data.yaml").write_text(yaml)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a FiftyOne dataset to YOLO format")
    parser.add_argument("--dataset", default="recap_camera_annotations", help="FiftyOne dataset name")
    parser.add_argument(
        "--out",
        default=str(YOLO_AL_V1_ROOT),
        help="Output directory for YOLO export",
    )
    parser.add_argument("--label-field", default="ground_truth", help="Label field (e.g., ground_truth)")
    parser.add_argument("--split-field", default="split", help="Split field name (default: split)")
    parser.add_argument(
        "--include-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export a test partition when present (default: enabled).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Replace this export's managed images/, labels/, and data.yaml in a nonempty --out directory.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    args = parser.parse_args()

    import fiftyone as fo

    out_dir = Path(args.out).expanduser().resolve()

    ds = fo.load_dataset(args.dataset)

    # Determine class list (fixed two-class order)
    classes = list(CLASSES_OVERRIDE)
    if classes != ["vehicle", "watercraft"]:
        raise RuntimeError("CLASSES_OVERRIDE must be ['vehicle', 'watercraft'] for two-class export")

    cls_to_id = {c: i for i, c in enumerate(classes)}

    # Validate the complete exported label contract before copying or writing any
    # files, then enforce a deliberate choice when an old export is present.
    splits = ("train", "val", "test") if args.include_test else ("train", "val")
    validate_source_labels(ds, args.label_field, args.split_field, splits)
    prepare_output_directory(out_dir, clean=args.clean)

    if args.verbose:
        print(f"Dataset: {args.dataset}")
        print(f"Output:  {out_dir}")
        print(f"Label field: {args.label_field}")
        print(f"Split field: {args.split_field}")
        print(f"Classes ({len(classes)}): {classes}")
        print(f"Label map: {LABEL_MAP}")

    counts = {
        split: export_split(
            ds,
            split_value=split,
            out_dir=out_dir,
            label_field=args.label_field,
            split_field=args.split_field,
            cls_to_id=cls_to_id,
            verbose=args.verbose,
        )
        for split in splits
    }

    # data.yaml
    write_data_yaml(out_dir, classes, include_test=args.include_test and counts.get("test", 0) > 0)

    print("Export complete: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    print("Wrote:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
