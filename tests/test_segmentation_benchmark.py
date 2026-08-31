from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.parking_common import ContractError


WORKING_ROOT = Path(__file__).resolve().parents[1]


def _load_stage():
    path = WORKING_ROOT / "pipeline" / "12_benchmark_segmentation.py"
    spec = importlib.util.spec_from_file_location("segmentation_reviewer_benchmark_stage", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SegmentationBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stage = _load_stage()

    def test_reviewer_defaults_are_frozen(self) -> None:
        args = self.stage.build_parser().parse_args([])
        self.assertEqual(args.expected_chip_count, 561)
        self.assertEqual(args.split_seed, 9401)
        self.assertEqual(args.spatial_group_pixels, 4096)
        self.assertEqual(args.seed, 20261244)
        self.assertEqual(args.epochs, 15)
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.learning_rate, 1e-4)
        self.assertEqual(args.models, list(self.stage.MODEL_BUILDERS))
        self.assertEqual(self.stage.CLASS_WEIGHTS, (1.0, 50.0, 50.0, 10.0, 10.0, 10.0))

    def test_alpha_is_stripped_without_rescaling_and_decodes_validity(self) -> None:
        image = np.zeros((2, 3, 4), dtype=np.float32)
        image[..., :3] = np.array([0.25, 0.5, 0.75], dtype=np.float32)
        image[..., 3] = np.array([[255, 0, 255], [0, 255, 255]], dtype=np.float32)
        image[0, 1, :3] = np.nan  # Invalid RGB need not carry meaningful radiometry.

        rgb, valid = self.stage.legacy_unit_rgb(image, chip_id="0_0")

        self.assertEqual(rgb.shape, (2, 3, 3))
        np.testing.assert_array_equal(valid, np.array([[True, False, True], [False, True, True]]))
        np.testing.assert_allclose(rgb[valid], np.array([[0.25, 0.5, 0.75]] * 4))

    def test_nonbinary_alpha_and_byte_scaled_valid_rgb_are_rejected(self) -> None:
        image = np.zeros((2, 2, 4), dtype=np.float32)
        image[..., 3] = 128
        with self.assertRaisesRegex(ContractError, "binary alpha"):
            self.stage.legacy_unit_rgb(image)

        image[..., 3] = 255
        image[..., 0] = 128
        with self.assertRaisesRegex(ContractError, "already be unit-scaled"):
            self.stage.legacy_unit_rgb(image)

    def test_deterministic_augmentation_keeps_image_mask_and_valid_aligned(self) -> None:
        mask = np.arange(12, dtype=np.uint8).reshape(3, 4)
        image = np.repeat(mask[..., None], 3, axis=2).astype(np.float32)
        valid = mask % 2 == 0

        first = self.stage.deterministic_augment(
            image,
            mask,
            valid,
            chip_id="4096_8192",
            epoch=3,
            seed=20261244,
        )
        second = self.stage.deterministic_augment(
            image,
            mask,
            valid,
            chip_id="4096_8192",
            epoch=3,
            seed=20261244,
        )

        for left, right in zip(first, second, strict=True):
            np.testing.assert_array_equal(left, right)
        transformed_image, transformed_mask, transformed_valid = first
        np.testing.assert_array_equal(transformed_image[..., 0], transformed_mask)
        np.testing.assert_array_equal(transformed_valid, transformed_mask % 2 == 0)

    def test_metrics_use_parking_class_mean_for_selection(self) -> None:
        confusion = np.diag([10, 8, 6, 4, 2, 1]).astype(np.int64)
        confusion[1, 0] = 2  # paved IoU = 8 / 10
        confusion[2, 0] = 6  # dirt IoU = 6 / 12

        metrics = self.stage.metrics_from_confusion(confusion)

        self.assertAlmostEqual(metrics["per_class_iou"]["paved_parking"], 0.8)
        self.assertAlmostEqual(metrics["per_class_iou"]["dirt_gravel_parking"], 0.5)
        self.assertAlmostEqual(metrics["parking_mean_iou"], 0.65)

    def test_discovery_refuses_partial_or_nonfrozen_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            (root / "images" / "0_0.npy").write_bytes(b"image")
            (root / "images" / "._0_0.npy").write_bytes(b"sidecar")
            with self.assertRaisesRegex(ContractError, "inventories differ"):
                self.stage.discover_inventory(root, expected_count=1)
            (root / "masks" / "0_0.npy").write_bytes(b"mask")
            self.assertEqual(len(self.stage.discover_inventory(root, expected_count=1)), 1)
            with self.assertRaisesRegex(ContractError, "Expected the 2-chip"):
                self.stage.discover_inventory(root, expected_count=2)

    def test_resume_requires_identical_config_and_split_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark"
            config = {"schema_version": 1, "seed": 9401}
            split = "chip_id,partition\n0_0,train\n"
            first_hash = self.stage._initialize_or_validate_run(
                output_dir=output,
                resume=False,
                config=config,
                split_text=split,
            )
            self.assertEqual(first_hash, self.stage.sha256_file(output / "config.json"))
            self.assertEqual(json.loads((output / "config.json").read_text()), config)
            resumed_hash = self.stage._initialize_or_validate_run(
                output_dir=output,
                resume=True,
                config=config,
                split_text=split,
            )
            self.assertEqual(resumed_hash, first_hash)
            with self.assertRaisesRegex(ContractError, "configuration"):
                self.stage._initialize_or_validate_run(
                    output_dir=output,
                    resume=True,
                    config={"schema_version": 1, "seed": 9402},
                    split_text=split,
                )
            with self.assertRaisesRegex(ContractError, "split.csv bytes"):
                self.stage._initialize_or_validate_run(
                    output_dir=output,
                    resume=True,
                    config=config,
                    split_text=split + "4096_0,test\n",
                )


if __name__ == "__main__":
    unittest.main()
