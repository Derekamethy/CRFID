from __future__ import annotations

import unittest

import numpy as np

from crfid.evaluation.metrics import classification_metrics, condition_block_metrics


class MetricTests(unittest.TestCase):
    def test_known_macro_f1(self) -> None:
        metrics = classification_metrics(np.asarray([0, 0, 1, 1]), np.asarray([0, 1, 1, 1]), class_count=2)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 0.8) / 2)

    def test_condition_block_mean_logits(self) -> None:
        truth = np.asarray([0, 0, 1, 1])
        logits = np.asarray([[2, 0], [0, 1], [0, 2], [1, 0]], dtype=np.float32)
        metrics = condition_block_metrics(truth, logits, ["a", "a", "b", "b"], class_count=2)
        self.assertEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
