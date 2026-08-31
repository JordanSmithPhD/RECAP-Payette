from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "pipeline" / "12_benchmark_detection.py"


def load_module():
    spec = importlib.util.spec_from_file_location("detection_benchmark_stage_for_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 10), color=color).save(path, format="JPEG")


def make_pair(root: Path, split: str, name: str, color: tuple[int, int, int], label: str = "") -> None:
    make_image(root / "images" / split / name, color)
    label_path = root / "labels" / split / f"{Path(name).stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(label, encoding="utf-8")


class DatasetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_split_ignores_appledouble_and_same_stem_npy_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_pair(root, "train", "a.jpg", (1, 2, 3), "0 0.5 0.5 0.2 0.4\n")
            (root / "images" / "train" / "a.npy").write_bytes(b"cache")
            (root / "images" / "train" / "._a.jpg").write_bytes(b"AppleDouble")
            (root / "labels" / "train" / "._a.txt").write_bytes(b"AppleDouble")

            records = self.module.inspect_split(root, "train", 1)

            self.assertEqual([record.image_id for record in records], ["a.jpg"])
            self.assertEqual(records[0].n_vehicle, 1)

    def test_sha_overlap_is_rejected_even_when_basenames_differ(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_pair(root, "train", "a.jpg", (10, 20, 30))
            make_pair(root, "val", "b.jpg", (40, 50, 60))
            # Exact image duplicate under a distinct validation basename.
            (root / "images" / "val" / "b.jpg").write_bytes((root / "images" / "train" / "a.jpg").read_bytes())
            with self.assertRaisesRegex(self.module.ContractError, "SHA-256 image overlap"):
                self.module.validate_dataset(root, expected_train_images=1, expected_val_images=1)

    def test_adapted_split_counts_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_pair(root, "train", "1_01_2025_01_01_00_00_00.jpg", (1, 2, 3))
            make_pair(root, "val", "2_01_2025_01_01_00_00_00.jpg", (4, 5, 6))

            train, val = self.module.validate_dataset(
                root,
                expected_train_images=1,
                expected_val_images=1,
            )

            self.assertEqual(len(train), 1)
            self.assertEqual(len(val), 1)

    def test_historical_val_rows_are_excluded_from_corrected_hard_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_pair(root, "train", "train.jpg", (1, 2, 3))
            make_pair(root, "val", "val.jpg", (4, 5, 6))
            train = self.module.inspect_split(root, "train", 1)
            val = self.module.inspect_split(root, "val", 1)
            history = root / "old_train.txt"
            history.write_text(
                "\n".join([str(train[0].image)] * 6 + [str(val[0].image)] * 5) + "\n",
                encoding="utf-8",
            )

            hard, rejected = self.module.recover_hard_training_names(
                history, train, val, expected_train_names=1
            )
            sampled = self.module.build_hard_sampled_records(train, hard, extra_repeats=5)

            self.assertEqual(hard, ["train.jpg"])
            self.assertEqual(rejected, ["val.jpg"])
            self.assertEqual(len(sampled), 6)
            self.assertNotIn("val.jpg", {record.image_id for record in sampled})

    def test_same_camera_temporal_proximity_is_audited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_pair(root, "train", "1_01_2025_01_01_00_00_00.jpg", (1, 2, 3))
            make_pair(root, "train", "2_01_2025_01_10_00_00_00.jpg", (4, 5, 6))
            make_pair(root, "val", "1_01_2025_01_01_00_30_00_dup1.jpg", (7, 8, 9))
            train = self.module.inspect_split(root, "train", 2)
            val = self.module.inspect_split(root, "val", 1)

            summary, rows = self.module.audit_same_camera_temporal_proximity(train, val)

            self.assertEqual(summary["validation_images_with_same_camera_train_image"], 1)
            self.assertEqual(summary["same_camera_nearest_within_1_hour"], 1)
            self.assertEqual(summary["same_camera_nearest_within_24_hours"], 1)
            self.assertEqual(summary["minimum_nearest_seconds"], 1800.0)
            self.assertEqual(rows[0]["nearest_train_image"], "1_01_2025_01_01_00_00_00.jpg")


class MetricContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def detection(self, image: str, class_id: int, score: float, box: tuple[float, float, float, float]):
        return self.module.Detection(image, class_id, score, box)

    def test_perfect_predictions_have_unit_coco_101_ap(self) -> None:
        ground_truth = [self.detection("a.jpg", 0, 1.0, (0, 0, 10, 10))]
        predictions = [self.detection("a.jpg", 0, 0.9, (0, 0, 10, 10))]

        value = self.module.coco_101_ap(ground_truth, predictions, ["a.jpg"], 0, 0.5)

        self.assertAlmostEqual(value, 1.0)

    def test_high_scored_false_positive_reduces_ap(self) -> None:
        ground_truth = [self.detection("a.jpg", 0, 1.0, (0, 0, 10, 10))]
        predictions = [
            self.detection("a.jpg", 0, 0.95, (10, 10, 20, 20)),
            self.detection("a.jpg", 0, 0.90, (0, 0, 10, 10)),
        ]

        value = self.module.coco_101_ap(ground_truth, predictions, ["a.jpg"], 0, 0.5)

        self.assertAlmostEqual(value, 0.5)

    def test_bootstrap_is_seed_reproducible_and_preserves_duplicate_occurrences(self) -> None:
        ground_truth = [
            self.detection("a.jpg", 0, 1.0, (0, 0, 10, 10)),
            self.detection("b.jpg", 0, 1.0, (0, 0, 10, 10)),
        ]
        predictions = [self.detection("a.jpg", 0, 0.9, (0, 0, 10, 10))]

        first = self.module.bootstrap_confidence_intervals(
            ground_truth, predictions, ["a.jpg", "b.jpg"], replicates=30, seed=17
        )
        second = self.module.bootstrap_confidence_intervals(
            ground_truth, predictions, ["a.jpg", "b.jpg"], replicates=30, seed=17
        )

        self.assertEqual(first, second)
        self.assertEqual(first["vehicle_ap50"]["valid_replicates"], 30)


class ArchitectureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_torchvision_label_semantics_are_architecture_specific(self) -> None:
        self.assertEqual(self.module.torchvision_label_contract("fcos_resnet50_fpn"), (0, 2))

    def test_only_paper_reported_architectures_are_exposed(self) -> None:
        self.assertEqual(self.module.MODEL_NAMES, ("yolo26n", "fcos_resnet50_fpn"))

    def test_fcos_replacement_preserves_pretrained_towers_and_uses_two_logits(self) -> None:
        try:
            from torchvision.models.detection import fcos_resnet50_fpn
        except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
            self.skipTest(f"torchvision detection models unavailable: {exc}")

        model = fcos_resnet50_fpn(weights=None, weights_backbone=None)
        classification_tower = model.head.classification_head.conv
        regression_head = model.head.regression_head
        self.module._replace_fcos_classification_logits(model, 2)

        self.assertIs(model.head.classification_head.conv, classification_tower)
        self.assertIs(model.head.regression_head, regression_head)
        self.assertEqual(model.head.classification_head.num_classes, 2)
        self.assertEqual(model.head.classification_head.num_anchors, 1)
        self.assertEqual(model.head.classification_head.cls_logits.out_channels, 2)

    def test_torchvision_long_edge_matches_yolo_content_resolution(self) -> None:
        self.assertEqual(self.module.torchvision_resize_bounds(640), (640, 640))

    def test_torchvision_workers_must_remain_zero(self) -> None:
        args = self.module.build_parser().parse_args(
            ["--models", "fcos_resnet50_fpn", "--workers", "1"]
        )
        with self.assertRaisesRegex(self.module.ContractError, "require --workers 0"):
            self.module.validate_cli(args)

    def test_yolo_mps_inference_is_routed_to_cpu(self) -> None:
        self.assertEqual(self.module.yolo_inference_device("mps"), "cpu")
        self.assertEqual(self.module.yolo_inference_device("cpu"), "cpu")
        self.assertEqual(self.module.yolo_inference_device("cuda"), "cuda")

    def test_config_records_resolved_training_and_inference_devices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.module.build_parser().parse_args([])
            args.torch_home = root / "torch-cache"
            args.resolved_training_device = "mps"
            payload = self.module.config_payload(args, root / "data", root / "output")

        self.assertEqual(payload["training_device"], "mps")
        self.assertEqual(payload["inference_devices"]["yolo26n"], "cpu")
        self.assertEqual(payload["inference_devices"]["fcos_resnet50_fpn"], "mps")
        self.assertIsNone(payload["historical_hard_list"])
        self.assertIsNone(payload["al_batches_root"])

    def test_yolo_trainer_delegates_validation_to_shared_evaluator(self) -> None:
        try:
            trainer_class = self.module.yolo_benchmark_trainer_class()
        except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
            self.skipTest(f"Ultralytics unavailable: {exc}")
        sentinel = object()
        self.assertEqual(trainer_class.validate(sentinel), ({}, 0.0))
        self.assertIsNone(trainer_class.final_eval(sentinel))

    def test_cumulative_stages_deduplicate_overlap_and_exclude_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, name in enumerate(("a.jpg", "b.jpg", "c.jpg")):
                make_pair(root, "train", name, (index + 1, 2, 3))
            make_pair(root, "val", "v.jpg", (10, 11, 12))
            train = self.module.inspect_split(root, "train", 3)
            val = self.module.inspect_split(root, "val", 1)
            batches = root / "batches"
            (batches / "batch_001").mkdir(parents=True)
            (batches / "batch_002_river").mkdir(parents=True)
            (batches / "batch_001" / "paths.txt").write_text("a.jpg\nb.jpg\nv.jpg\n", encoding="utf-8")
            (batches / "batch_002_river" / "paths.txt").write_text("b.jpg\nc.jpg\nv.jpg\n", encoding="utf-8")

            stages, metadata = self.module.reconstruct_annotation_stages(train, val, batches)

            self.assertEqual([row.image_id for row in stages["annotation_stage_1"]], ["a.jpg", "b.jpg"])
            self.assertEqual(
                [row.image_id for row in stages["annotation_stage_2"]], ["a.jpg", "b.jpg", "c.jpg"]
            )
            self.assertEqual(metadata["batch_overlap_train_count"], 1)
            self.assertEqual(metadata["batch_overlap_validation_count"], 1)
            self.assertEqual(metadata["stage_2_excluded_validation_names"], ["v.jpg"])

    def test_cumulative_stages_require_explicit_batch_root(self) -> None:
        args = self.module.build_parser().parse_args(["--yolo-cumulative-stages"])

        with self.assertRaisesRegex(self.module.ContractError, "explicit --al-batches-root"):
            self.module.validate_cli(args)


class PortableDryRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_default_dry_run_needs_no_hard_list_or_annotation_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            output = root / "output"
            make_pair(dataset, "train", "1_01_2025_01_01_00_00_00.jpg", (1, 2, 3))
            make_pair(dataset, "val", "2_01_2025_01_01_00_00_00.jpg", (4, 5, 6))

            status = self.module.main(
                [
                    "--dataset-root",
                    str(dataset),
                    "--output-dir",
                    str(output),
                    "--expected-train-images",
                    "1",
                    "--expected-val-images",
                    "1",
                    "--dry-run",
                ]
            )

            self.assertEqual(status, 0)
            effective = (output / "lists" / "train_effective.txt").read_text(encoding="utf-8").splitlines()
            provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(len(effective), 1)
            self.assertFalse(provenance["hard_sampling"]["enabled"])
            self.assertEqual(provenance["hard_sampling"]["effective_train_list_entries"], 1)


class OutputSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_nonempty_output_requires_explicit_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "keep.txt").write_text("user data", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "nonempty"):
                self.module.prepare_output(root, resume=False)

            self.assertEqual(self.module.prepare_output(root, resume=True), root.resolve())
            self.assertEqual((root / "keep.txt").read_text(encoding="utf-8"), "user data")

    def test_resume_preserves_original_provenance_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "provenance.json"
            first = {"created_utc": "first", "run_config_sha256": "abc"}
            second = {"created_utc": "second", "run_config_sha256": "abc"}
            self.module.preserve_immutable_provenance(
                path, first, resume=False, run_config_sha256="abc"
            )

            loaded = self.module.preserve_immutable_provenance(
                path, second, resume=True, run_config_sha256="abc"
            )

            self.assertEqual(loaded["created_utc"], "first")
            self.assertEqual(path.read_text(encoding="utf-8").count("first"), 1)

    def test_resume_rejects_provenance_from_another_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "provenance.json"
            self.module.preserve_immutable_provenance(
                path,
                {"created_utc": "first", "run_config_sha256": "abc"},
                resume=False,
                run_config_sha256="abc",
            )
            with self.assertRaisesRegex(self.module.ContractError, "does not match"):
                self.module.preserve_immutable_provenance(
                    path,
                    {"created_utc": "second", "run_config_sha256": "different"},
                    resume=True,
                    run_config_sha256="different",
                )


if __name__ == "__main__":
    unittest.main()
