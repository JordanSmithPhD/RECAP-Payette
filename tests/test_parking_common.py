from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pipeline.parking_common import (
    BandObservation,
    ContractError,
    build_spatial_partitions,
    capacity_spaces,
    load_run_manifest,
    normalize_chip,
    partition_for_group,
    refuse_existing_output,
    require_unique_nonempty_names,
    validate_band_contract,
    validate_corrected_corridor_grid,
    validate_corrected_dem_slope_contract,
    validate_corrected_mosaic_contract,
    validate_open_corrected_corridor_raster,
    validate_partition_rows,
    verify_manifest_artifact,
    write_run_manifest,
)


def _load_dataset_module():
    path = Path(__file__).resolve().parents[1] / "pipeline" / "03_build_segmentation_dataset.py"
    spec = importlib.util.spec_from_file_location("parking_dataset_stage", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_road_proximity_module():
    path = Path(__file__).resolve().parents[1] / "pipeline" / "05_test_road_proximity.py"
    spec = importlib.util.spec_from_file_location("parking_road_proximity_stage", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BandContractTests(unittest.TestCase):
    def test_alpha_is_rejected_as_nir(self) -> None:
        observation = BandObservation(4, (0, 0, 0, 255), (1, 1, 1, 255), ("red", "green", "blue", "alpha"))
        with self.assertRaisesRegex(ContractError, "alpha, not NIR"):
            validate_band_contract(observation, "rgbnir")

    def test_named_bgr_source_is_rejected(self) -> None:
        observation = BandObservation(
            3,
            (0, 0, 0),
            (1, 1, 1),
            band_names=("B", "G", "R"),
        )
        with self.assertRaisesRegex(ContractError, "red/green/blue"):
            validate_band_contract(observation, "rgb", require_identity=True)

    def test_arbitrary_varying_fourth_band_is_not_nir(self) -> None:
        observation = BandObservation(
            4,
            (0, 0, 0, 0),
            (1, 1, 1, 1),
            band_names=("R", "G", "B", "quality"),
        )
        with self.assertRaisesRegex(ContractError, "blue/NIR"):
            validate_band_contract(observation, "rgbnir", require_identity=True)

    def test_historical_double_divide_is_explicit(self) -> None:
        chip = np.zeros((2, 2, 4), dtype=np.float32)
        chip[..., :3] = 0.5
        chip[..., 3] = 255
        normalized = normalize_chip(chip, "historical-double-divide")
        self.assertAlmostEqual(float(normalized[0, 0, 0]), 0.5 / 255)
        self.assertEqual(float(normalized[0, 0, 3]), 1.0)

    def test_corrected_rgb_drops_alpha_without_double_scaling(self) -> None:
        chip = np.zeros((2, 2, 4), dtype=np.float32)
        chip[..., :3] = 0.5
        chip[..., 3] = 255
        normalized = normalize_chip(chip, "unit-rgb")
        self.assertEqual(normalized.shape, (2, 2, 3))
        self.assertEqual(float(normalized.max()), 0.5)

    def test_corrected_corridor_grid_requires_projection_and_unambiguous_nodata(self) -> None:
        extra = {
            "input_raster_provenance": "verified",
            "input_mosaic_pixel_valid_contract": "all-roi-pixel-centres-valid-all-bands-zero-tolerance",
            "input_mosaic_manifest_sha256": "a" * 64,
            "raster_output_band_names": ["R", "G", "B", "N"],
            "raster_output_crs": "EPSG:32611",
            "metric_crs": "EPSG:32611",
            "raster_output_resolution_m": 1.0,
            "raster_resampling": "nearest",
            "raster_nodata_value": -9999.0,
        }
        self.assertEqual(validate_corrected_corridor_grid(extra), ("EPSG:32611", 1.0))
        for key, value in (
            ("raster_output_crs", "EPSG:4326"),
            ("raster_output_resolution_m", 0),
            ("raster_resampling", "bilinear"),
            ("raster_nodata_value", 0),
        ):
            with self.subTest(key=key, value=value):
                with self.assertRaises(ContractError):
                    validate_corrected_corridor_grid({**extra, key: value})

    def test_open_corridor_raster_must_match_observed_manifest_grid(self) -> None:
        extra = {
            "input_raster_provenance": "verified",
            "input_mosaic_pixel_valid_contract": "all-roi-pixel-centres-valid-all-bands-zero-tolerance",
            "input_mosaic_manifest_sha256": "a" * 64,
            "raster_output_band_names": ["R", "G", "B", "N"],
            "raster_output_crs": "EPSG:32611",
            "metric_crs": "EPSG:32611",
            "raster_output_resolution_m": 1.0,
            "raster_resampling": "nearest",
            "raster_nodata_value": -9999.0,
            "raster_output_geotransform": [0.0, 1.0, 0.0, 10.0, 0.0, -1.0],
            "raster_nodata_values_observed": [-9999.0] * 4,
            "raster_data_types_observed": ["Float32"] * 4,
        }
        source = SimpleNamespace(
            crs="EPSG:32611",
            count=4,
            transform=SimpleNamespace(a=1.0, b=0.0, c=0.0, d=0.0, e=-1.0, f=10.0),
            descriptions=("R", "G", "B", "N"),
            dtypes=("float32",) * 4,
            nodatavals=(-9999.0,) * 4,
        )
        self.assertEqual(validate_open_corrected_corridor_raster(source, extra), ("EPSG:32611", 1.0))
        with self.assertRaisesRegex(ContractError, "nodata"):
            validate_open_corrected_corridor_raster(
                SimpleNamespace(**{**vars(source), "nodatavals": (None,) * 4}),
                extra,
            )

    def test_mosaic_and_dem_contracts_reject_legacy_weak_coverage(self) -> None:
        mosaic = {
            "raster_output_band_names": ["R", "G", "B", "N"],
            "mosaic_nodata_value": -9999.0,
            "pixel_valid_coverage_contract": "all-roi-pixel-centres-valid-all-bands-zero-tolerance",
            "roi_pixel_centres_total": 10,
            "roi_pixel_centres_valid_all_bands": 10,
            "roi_pixel_centres_invalid": 0,
            "shard_family_content_hashes_stable_through_manifest": True,
        }
        validate_corrected_mosaic_contract(mosaic)
        with self.assertRaisesRegex(ContractError, "valid-data coverage"):
            validate_corrected_mosaic_contract(
                {key: value for key, value in mosaic.items() if key != "pixel_valid_coverage_contract"}
            )

        terrain_arguments = {
            "dem_input": None,
            "min_x": -116.28766600566757,
            "min_y": 43.88355482041268,
            "max_x": -115.491507,
            "max_y": 44.514881,
            "grid_size_deg": 0.02,
            "edge_padding_deg": 0.001,
            "resolution_m": 1.0,
            "metric_crs": "EPSG:26911",
            "slope_threshold_pct": 5.0,
            "resolved_slope_comparison": "le",
            "minimum_polygon_area_m2": 0.0,
        }
        terrain_extra = {
            "dem_grid_validation": "exact-crs-lattice-and-all-request-polygon-pixel-centres-valid",
            "request_cells_valid": 1280,
            "request_cells_expected": 1280,
            "request_polygon_pixel_centres_valid": 100,
            "request_polygon_pixel_centres_expected": 100,
            "dem_nodata_value": -9999.0,
            "dem_valid_elevation_range_m": [100.0, 10000.0],
            "dem_mosaic_crs": "EPSG:26911",
            "dem_mosaic_resolution_m": 1.0,
            "dem_mosaic_resampling": "nearest",
            "dem_mosaic_nodata": -9999.0,
            "dem_mosaic_target_aligned_pixels": True,
            "chunk_normalization_contract": "provider-response-to-target-crs-one-metre-aligned-bilinear",
            "chunk_normalization_resampling": "bilinear",
        }
        validate_corrected_dem_slope_contract({"arguments": terrain_arguments, "extra": terrain_extra})
        with self.assertRaisesRegex(ContractError, "all-request-pixel"):
            validate_corrected_dem_slope_contract(
                {
                    "arguments": terrain_arguments,
                    "extra": {**terrain_extra, "dem_grid_validation": "complete-extent-resolution-and-cell-centres"},
                }
            )
        with self.assertRaisesRegex(ContractError, "provider response"):
            validate_corrected_dem_slope_contract(
                {
                    "arguments": terrain_arguments,
                    "extra": {**terrain_extra, "chunk_normalization_contract": "provider-native-grid"},
                }
            )


class SplitContractTests(unittest.TestCase):
    def test_spatial_partition_is_stable_and_keeps_neighbors_together(self) -> None:
        chip_ids = ["0_0", "512_512", "4096_0", "8192_8192"]
        first = build_spatial_partitions(chip_ids, seed=7, group_pixels=4096)
        second = build_spatial_partitions(reversed(chip_ids), seed=7, group_pixels=4096)
        self.assertEqual(first, second)
        self.assertEqual(first["0_0"], first["512_512"])

    def test_partition_assignment_is_deterministic(self) -> None:
        self.assertEqual(
            partition_for_group("r00001_c00002", seed=10),
            partition_for_group("r00001_c00002", seed=10),
        )

    def test_group_cannot_cross_partitions(self) -> None:
        rows = [
            {"chip_id": "0_0", "spatial_group": "g", "partition": "train"},
            {"chip_id": "512_0", "spatial_group": "g", "partition": "test"},
        ]
        with self.assertRaisesRegex(ContractError, "crosses"):
            validate_partition_rows(rows, corrected=True)

    def test_historical_rows_must_say_resubstitution(self) -> None:
        with self.assertRaisesRegex(ContractError, "resubstitution"):
            validate_partition_rows([{"chip_id": "0_0", "partition": "test"}], corrected=False)


class CapacityAndSafetyTests(unittest.TestCase):
    def test_capacity_rate_is_required_and_positive(self) -> None:
        self.assertEqual(capacity_spaces(2.0, 50.0), 100.0)
        with self.assertRaises(ContractError):
            capacity_spaces(2.0, 0.0)

    def test_reach_names_must_be_unique(self) -> None:
        with self.assertRaisesRegex(ContractError, "duplicates"):
            require_unique_nonempty_names(["Cabarton", "Cabarton"], "area_name")

    def test_existing_output_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                refuse_existing_output(Path(directory))

    def test_negative_selection_is_order_independent(self) -> None:
        module = _load_dataset_module()
        windows = [
            module.ChipWindow("0_0", 0, 0, (1,)),
            module.ChipWindow("0_512", 0, 512, ()),
            module.ChipWindow("0_1024", 0, 1024, ()),
        ]
        first = module.stable_negative_selection(windows, seed=4, negative_ratio=1)
        second = module.stable_negative_selection(list(reversed(windows)), seed=4, negative_ratio=1)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)

    def test_road_proximity_uses_area_fraction(self) -> None:
        module = _load_road_proximity_module()
        result = module.proximity_statistics(200.0, 8.0)
        self.assertAlmostEqual(result["near_road_fraction"], 0.04)
        self.assertAlmostEqual(result["near_road_percent"], 4.0)

    def test_run_manifest_hashes_directory_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "chips"
            inputs.mkdir()
            (inputs / "a.npy").write_bytes(b"first")
            (inputs / "b.npy").write_bytes(b"second")
            (inputs / "._a.npy").write_bytes(b"appledouble")
            (inputs / ".DS_Store").write_bytes(b"finder")
            manifest = root / "run.json"
            write_run_manifest(
                manifest,
                workflow_root=root,
                stage="fixture",
                contract="test",
                arguments={},
                inputs=(inputs,),
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        artifact = payload["inputs"][0]
        self.assertEqual(artifact["artifact_type"], "directory-tree")
        self.assertEqual(artifact["file_count"], 2)
        self.assertEqual({item["path"] for item in artifact["members"]}, {"a.npy", "b.npy"})

    def test_run_manifest_hashes_shapefile_sidecars_as_one_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = root / "roads.shp"
            shape.write_bytes(b"geometry")
            (root / "roads.dbf").write_bytes(b"attributes")
            (root / "roads.shx").write_bytes(b"index")
            manifest = root / "run.json"
            write_run_manifest(
                manifest,
                workflow_root=root,
                stage="fixture",
                contract="test",
                arguments={},
                inputs=(shape,),
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        artifact = payload["inputs"][0]
        self.assertEqual(artifact["artifact_type"], "shapefile-bundle")
        self.assertEqual(artifact["file_count"], 3)

    def test_manifest_artifact_verification_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "result.csv"
            artifact.write_text("value\n1\n", encoding="utf-8")
            manifest = root / "run.json"
            write_run_manifest(
                manifest,
                workflow_root=root,
                stage="fixture",
                contract="test",
                arguments={},
                outputs=(artifact,),
            )
            payload = load_run_manifest(
                manifest,
                expected_stage="fixture",
                expected_contract="test",
            )
            verify_manifest_artifact(payload, artifact)
            artifact.write_text("value\n2\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "checksum differs"):
                verify_manifest_artifact(payload, artifact)

    def test_manifest_repository_paths_survive_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            original = temporary / "original"
            workflow = original / "analysis" / "working"
            artifact = original / "analysis" / "data_intermediate" / "result.csv"
            manifest = original / "analysis" / "results" / "run.json"
            workflow.mkdir(parents=True)
            artifact.parent.mkdir(parents=True)
            artifact.write_text("value\n1\n", encoding="utf-8")
            write_run_manifest(
                manifest,
                workflow_root=workflow,
                stage="fixture",
                contract="test",
                arguments={"inputs": [artifact]},
                outputs=(artifact,),
            )
            on_disk = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["outputs"][0]["path_mode"], "manifest-relative")
            self.assertNotIn(str(original), on_disk["outputs"][0]["path"])
            self.assertIsInstance(on_disk["arguments"]["inputs"][0], str)

            moved = temporary / "moved"
            original.rename(moved)
            moved_manifest = moved / "analysis" / "results" / "run.json"
            moved_artifact = moved / "analysis" / "data_intermediate" / "result.csv"
            payload = load_run_manifest(moved_manifest)
            verify_manifest_artifact(payload, moved_artifact)


if __name__ == "__main__":
    unittest.main()
