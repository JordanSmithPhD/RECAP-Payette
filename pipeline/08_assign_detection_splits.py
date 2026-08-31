#!/usr/bin/env python3
"""Assign deterministic, duplicate-safe train/validation/test splits in FiftyOne.

The recommended workflow keeps byte-identical images in one partition and
reserves an independent test set.  The manuscript's historical 1,395/100 split
had no test set and is replayed from its frozen labels rather than regenerated
with this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def partition_for(key: str, seed: int, validation_fraction: float, test_fraction: float) -> str:
    value = int(hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()[:16], 16) / float(16**16)
    if value < test_fraction:
        return "test"
    if value < test_fraction + validation_fraction:
        return "val"
    return "train"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="FiftyOne dataset name")
    parser.add_argument("--split-field", default="split")
    parser.add_argument("--hash-field", default="content_sha256")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=9401)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary", type=Path, default=None)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.validation_fraction < 0 or args.test_fraction < 0:
        raise ValueError("split fractions cannot be negative")
    if args.validation_fraction + args.test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than one")

    import fiftyone as fo

    dataset = fo.load_dataset(args.dataset)
    if args.split_field in dataset.get_field_schema() and not args.overwrite:
        existing = [value for value in dataset.values(args.split_field) if value not in (None, "")]
        if existing:
            raise RuntimeError(f"{args.split_field!r} already contains values; pass --overwrite deliberately")

    assignments: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for sample in dataset:
        path = Path(sample.filepath).expanduser().resolve(strict=True)
        content_hash = sample.get(args.hash_field, None) or sha256_file(path)
        split = assignments.setdefault(
            str(content_hash),
            partition_for(str(content_hash), args.seed, args.validation_fraction, args.test_fraction),
        )
        sample[args.hash_field] = str(content_hash)
        sample[args.split_field] = split
        sample.save()
        hashes[str(sample.id)] = str(content_hash)

    dataset.persistent = True
    values = dataset.values(args.split_field)
    counts = Counter(str(value) for value in values)
    if len(hashes) != len(dataset):
        raise RuntimeError("not every sample received a content hash")
    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "samples": len(dataset),
        "unique_content_hashes": len(assignments),
        "counts": dict(sorted(counts.items())),
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
