#!/usr/bin/env python3
"""Train a two-class Ultralytics detector and retain run provenance.

This entry point is intended for new/adapted studies. Exact manuscript replay
starts from the archived operational checkpoint because the complete original
training provenance is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Ultralytics data.yaml")
    parser.add_argument("--model", default="yolo26n.pt", help="Public initialization or local checkpoint")
    parser.add_argument("--run-dir", type=Path, required=True, help="New output directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=9401)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cache", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    data = args.data.expanduser().resolve(strict=True)
    run_dir = args.run_dir.expanduser().resolve(strict=False)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite {run_dir}")
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0:
        raise ValueError("epochs, imgsz, and batch must be positive")

    from ultralytics import YOLO, __version__ as ultralytics_version

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    result = model.train(
        data=str(data),
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=False,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        workers=args.workers,
        patience=args.patience,
        cache=args.cache,
    )
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Training completed without expected checkpoint {best}")
    payload = {
        "stage": "08_train_detector",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": str(data),
        "data_sha256": sha256_file(data),
        "initial_model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "seed": args.seed,
        "workers": args.workers,
        "patience": args.patience,
        "cache": args.cache,
        "python": platform.python_version(),
        "ultralytics": ultralytics_version,
        "best_checkpoint": str(best),
        "best_checkpoint_sha256": sha256_file(best),
        "result_type": type(result).__name__,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"checkpoint": str(best), "sha256": payload["best_checkpoint_sha256"]}, indent=2))
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
