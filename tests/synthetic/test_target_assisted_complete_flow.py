from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from crfid.adaptation.target_assisted import execute_target_informed_ncm
from crfid.evaluation.metrics import classification_metrics


ROOT = Path(__file__).resolve().parents[2]


class TargetAssistedSyntheticFlowTests(unittest.TestCase):
    def test_deterministic_repeated_execution(self) -> None:
        source = np.asarray([[2.0, 0.0], [0.0, 3.0]])
        labels = np.asarray([0, 1])
        target = np.asarray([[1.0, 0.1], [0.1, 1.0]])
        first = execute_target_informed_ncm(source, labels, target, class_order=(0, 1))
        second = execute_target_informed_ncm(source, labels, target, class_order=(0, 1))
        np.testing.assert_array_equal(first.predictions, second.predictions)
        np.testing.assert_array_equal(first.distances, second.distances)

    def test_synthetic_selection_and_metrics(self) -> None:
        result = execute_target_informed_ncm(
            np.eye(2), np.asarray([0, 1]), np.eye(2), class_order=(0, 1)
        )
        metrics = classification_metrics(np.asarray([0, 1]), result.predictions, class_count=2)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)

    def test_real_reproduction_artifacts(self) -> None:
        final = ROOT / "outputs" / "target_assisted_p4" / "final"
        if not (final / "aggregate_metrics.json").is_file() or not (final / "TARGET_ASSISTED_INTEGRITY.json").is_file():
            self.skipTest("governed target-assisted reproduction artifacts are external")
        aggregate = json.loads((final / "aggregate_metrics.json").read_text(encoding="utf-8"))
        integrity = json.loads((final / "TARGET_ASSISTED_INTEGRITY.json").read_text(encoding="utf-8"))
        self.assertEqual(aggregate["accuracy_mean"], 0.5771428571428572)
        self.assertEqual(aggregate["macro_f1_mean"], 0.5835693346352661)
        self.assertTrue(integrity["metric_match"])
        self.assertFalse(integrity["independent_target_evaluation"])


if __name__ == "__main__":
    unittest.main()
