from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


PUBLIC_ROOT = Path(__file__).resolve().parents[1]


def _load_stage():
    path = PUBLIC_ROOT / "pipeline" / "09_run_camera_inference.py"
    spec = importlib.util.spec_from_file_location("stage09_resumable_inference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeTensor:
    def __init__(self, values: Any):
        self.values = np.asarray(values)

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeBoxes:
    def __init__(self, xyxy: Any, classes: Any, confidences: Any):
        self.xyxy = FakeTensor(xyxy)
        self.cls = FakeTensor(classes)
        self.conf = FakeTensor(confidences)


class FakeResult:
    def __init__(self, boxes: FakeBoxes | None = None):
        self.boxes = boxes


class FakeModel:
    names = {0: "car", 1: "boat", 2: "person"}

    def __init__(self, results: list[FakeResult] | None = None, error: Exception | None = None):
        self.results = results
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> list[FakeResult]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.results is not None:
            return self.results
        return [FakeResult() for _ in kwargs["source"]]


class Stage09ResumableInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stage = _load_stage()

    @staticmethod
    def _write_jpeg(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 8), color=color).save(path, format="JPEG")

    @staticmethod
    def _manifest_row(path: Path, site: str = "River/Test") -> dict[str, Any]:
        content = path.read_bytes()
        return {
            "path": str(path),
            "filename": path.name,
            "file_size": len(content),
            "content_sha256": hashlib.sha256(content).hexdigest(),
            "site": site,
            "datetime_from_filename": pd.Timestamp("2024-07-01T12:00:00"),
        }

    def _args(
        self,
        *,
        root: Path,
        manifest: Path,
        weights: Path,
        run_dir: Path,
        extra: list[str] | None = None,
    ) -> argparse.Namespace:
        # argparse is deliberately reached through the production parser so
        # defaults remain part of the tested scientific contract.
        values = [
            "--root",
            str(root),
            "--manifest",
            str(manifest),
            "--weights",
            str(weights),
            "--run-dir",
            str(run_dir),
            "--device",
            "cpu",
        ]
        values.extend(extra or [])
        return self.stage.build_parser().parse_args(values)

    def test_historical_predict_settings_and_per_class_thresholds_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "River" / "Test" / "one.jpg"
            self._write_jpeg(image)
            manifest = root / "manifest.parquet"
            pd.DataFrame([self._manifest_row(image)]).to_parquet(manifest, index=False)
            weights = root / "best.pt"
            weights.write_bytes(b"test checkpoint")
            run_dir = root / "run"
            boxes = FakeBoxes(
                xyxy=[
                    [0, 0, 1, 1],
                    [1, 1, 2, 2],
                    [2, 2, 3, 3],
                    [3, 3, 4, 4],
                    [4, 4, 5, 5],
                ],
                classes=[0, 0, 1, 1, 2],
                confidences=[0.249, 0.25, 0.476, 0.477, 0.99],
            )
            model = FakeModel([FakeResult(boxes)])
            args = self._args(root=root, manifest=manifest, weights=weights, run_dir=run_dir)

            result = self.stage.execute(args, model_factory=lambda _weights: model)

            self.assertEqual(len(model.calls), 1)
            call = model.calls[0]
            self.assertEqual(call["source"], [str(image)])
            self.assertEqual(
                {key: call[key] for key in ("imgsz", "conf", "iou", "device", "stream", "verbose", "max_det", "batch")},
                {
                    "imgsz": 960,
                    "conf": 0.25,
                    "iou": 0.7,
                    "device": "cpu",
                    "stream": False,
                    "verbose": False,
                    "max_det": 300,
                    "batch": 8,
                },
            )
            output = pd.read_parquet(run_dir / "detections.parquet")
            self.assertEqual(int(output.loc[0, "count_vehicle"]), 1)
            self.assertEqual(int(output.loc[0, "count_watercraft"]), 1)
            self.assertEqual(int(output.loc[0, "count_total"]), 2)
            self.assertEqual(result["detection_summary"]["detections_by_class_name"], {"boat": 1, "car": 1})

    def test_decode_failures_are_retained_and_do_not_enter_inference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "River" / "Test" / "good.jpg"
            bad = root / "River" / "Test" / "bad.jpg"
            self._write_jpeg(good)
            bad.parent.mkdir(parents=True, exist_ok=True)
            bad.write_bytes(b"not an image")
            manifest = root / "manifest.parquet"
            pd.DataFrame([self._manifest_row(good), self._manifest_row(bad)]).to_parquet(manifest, index=False)
            weights = root / "best.pt"
            weights.write_bytes(b"test checkpoint")
            run_dir = root / "run"
            model = FakeModel()
            args = self._args(
                root=root,
                manifest=manifest,
                weights=weights,
                run_dir=run_dir,
                extra=["--expected-input-rows", "2", "--expected-filtered-rows", "2", "--expected-final-rows", "1"],
            )

            result = self.stage.execute(args, model_factory=lambda _weights: model)

            self.assertEqual(model.calls[0]["source"], [str(good)])
            self.assertEqual(result["row_counts"]["pil_decode_failure_rows"], 1)
            self.assertEqual(result["row_counts"]["final_rows"], 1)
            self.assertEqual((run_dir / "bad_images.txt").read_text(encoding="utf-8"), f"{bad}\n")
            self.assertIn("UnidentifiedImageError", result["decode_failures"][0]["error"])

    def test_inference_failure_aborts_without_marking_decodable_images_bad(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "River" / "Test" / "good.jpg"
            self._write_jpeg(image)
            manifest = root / "manifest.parquet"
            pd.DataFrame([self._manifest_row(image)]).to_parquet(manifest, index=False)
            weights = root / "best.pt"
            weights.write_bytes(b"test checkpoint")
            run_dir = root / "run"
            model = FakeModel(error=RuntimeError("accelerator stopped"))
            args = self._args(root=root, manifest=manifest, weights=weights, run_dir=run_dir)

            with self.assertRaisesRegex(self.stage.InferenceError, "no image in this batch was classified as bad"):
                self.stage.execute(args, model_factory=lambda _weights: model)

            self.assertFalse((run_dir / "shards" / "shard-000000.json").exists())
            self.assertFalse((run_dir / "bad_images.txt").exists())
            self.assertFalse((run_dir / "detections.parquet").exists())

    def test_second_execution_resumes_hash_verified_shards_without_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = [root / "River" / "Test" / f"{index}.jpg" for index in range(2)]
            for index, image in enumerate(images):
                self._write_jpeg(image, color=(10 + index, 20, 30))
            manifest = root / "manifest.parquet"
            pd.DataFrame([self._manifest_row(image) for image in images]).to_parquet(manifest, index=False)
            weights = root / "best.pt"
            weights.write_bytes(b"test checkpoint")
            run_dir = root / "run"
            args = self._args(
                root=root,
                manifest=manifest,
                weights=weights,
                run_dir=run_dir,
                extra=["--shard-size", "1", "--expected-final-rows", "2"],
            )
            first_model = FakeModel()
            first = self.stage.execute(args, model_factory=lambda _weights: first_model)
            self.assertEqual(len(first_model.calls), 2)

            def fail_if_loaded(_weights: str) -> Any:
                raise AssertionError("A fully resumed run must not reload the model")

            second = self.stage.execute(args, model_factory=fail_if_loaded)

            self.assertEqual(second["hashes"]["combined_output_sha256"], first["hashes"]["combined_output_sha256"])
            self.assertEqual(second["row_counts"]["final_rows"], 2)
            self.assertEqual(len(list((run_dir / "shards").glob("*.parquet"))), 2)
            sidecar = json.loads((run_dir / "shards" / "shard-000000.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["output"]["sha256"], self.stage.sha256_file(run_dir / "shards" / "shard-000000.parquet"))

    def test_tampered_completed_shard_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "River" / "Test" / "one.jpg"
            self._write_jpeg(image)
            manifest = root / "manifest.parquet"
            pd.DataFrame([self._manifest_row(image)]).to_parquet(manifest, index=False)
            weights = root / "best.pt"
            weights.write_bytes(b"test checkpoint")
            run_dir = root / "run"
            args = self._args(root=root, manifest=manifest, weights=weights, run_dir=run_dir)
            self.stage.execute(args, model_factory=lambda _weights: FakeModel())
            shard = run_dir / "shards" / "shard-000000.parquet"
            with shard.open("ab") as handle:
                handle.write(b"tampered")

            with self.assertRaisesRegex(self.stage.Stage09Error, "SHA-256 verification"):
                self.stage.execute(args, model_factory=lambda _weights: FakeModel())

    def test_deferred_existence_mode_records_auditable_counts_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "River" / "Test" / "one.jpg"
            self._write_jpeg(image)
            manifest = root / "manifest.parquet"
            pd.DataFrame([self._manifest_row(image)]).to_parquet(manifest, index=False)
            weights = root / "best.pt"
            weights.write_bytes(b"test checkpoint")
            run_dir = root / "run"
            args = self._args(
                root=root,
                manifest=manifest,
                weights=weights,
                run_dir=run_dir,
                extra=["--defer-existence-check", "--expected-filtered-rows", "1"],
            )

            result = self.stage.execute(args, model_factory=lambda _weights: FakeModel())

            self.assertEqual(result["configuration"]["verification"]["existence_check_mode"], "deferred-per-shard")
            self.assertEqual(result["configuration"]["verification"]["missing_source_policy"], "abort")
            self.assertEqual(result["row_counts"]["existence_checks_deferred"], 1)
            self.assertEqual(result["row_counts"]["rows_checked_for_existence"], 0)
            self.assertEqual(result["row_counts"]["filtered_rows"], 1)

    def test_deferred_missing_source_remains_filtered_then_aborts_in_its_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "River" / "Test" / "missing.jpg"
            manifest = root / "manifest.parquet"
            pd.DataFrame(
                [
                    {
                        "path": str(missing),
                        "filename": "missing.jpg",
                        "file_size": 123,
                        "content_sha256": "0" * 64,
                        "site": "River/Test",
                    }
                ]
            ).to_parquet(manifest, index=False)
            weights = root / "best.pt"
            weights.write_bytes(b"test checkpoint")
            run_dir = root / "run"
            model = FakeModel()
            args = self._args(
                root=root,
                manifest=manifest,
                weights=weights,
                run_dir=run_dir,
                extra=["--defer-existence-check", "--expected-filtered-rows", "1"],
            )

            with self.assertRaisesRegex(self.stage.Stage09Error, "Selected image became unavailable"):
                self.stage.execute(args, model_factory=lambda _weights: model)

            self.assertEqual(model.calls, [])
            plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["contract"]["filter_counts"]["filtered_rows"], 1)
            self.assertEqual(
                plan["contract"]["configuration"]["verification"]["existence_check_mode"],
                "deferred-per-shard",
            )
            self.assertEqual(
                plan["contract"]["inputs"]["camera_root"],
                str(root.resolve()),
            )
            self.assertFalse(missing.exists())
            self.assertFalse((run_dir / "bad_images.txt").exists())

    def test_deferred_existence_cannot_disable_content_hash_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "River" / "Test" / "one.jpg"
            self._write_jpeg(image)
            manifest = root / "manifest.parquet"
            pd.DataFrame([self._manifest_row(image)]).to_parquet(manifest, index=False)
            weights = root / "best.pt"
            weights.write_bytes(b"test checkpoint")
            args = self._args(
                root=root,
                manifest=manifest,
                weights=weights,
                run_dir=root / "run",
                extra=["--defer-existence-check", "--no-verify-content-hashes"],
            )

            with self.assertRaisesRegex(self.stage.Stage09Error, "requires --verify-content-hashes"):
                self.stage.execute(args, model_factory=lambda _weights: FakeModel())


if __name__ == "__main__":
    unittest.main()
