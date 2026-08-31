from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "pipeline" / "08_tune_thresholds.py"


def load_module():
    spec = importlib.util.spec_from_file_location("threshold_selection_for_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_match_counts_perfect_prediction() -> None:
    module = load_module()
    reference = np.array([0.0, 0.0, 10.0, 10.0])
    assert module._match_counts([reference], [(0.9, reference.copy())], 0.5, 0.5) == (1, 0, 0)


def test_match_counts_respects_confidence_and_false_positives() -> None:
    module = load_module()
    reference = np.array([0.0, 0.0, 10.0, 10.0])
    far_box = np.array([20.0, 20.0, 30.0, 30.0])
    assert module._match_counts([reference], [(0.4, reference.copy()), (0.9, far_box)], 0.5, 0.5) == (0, 1, 1)
