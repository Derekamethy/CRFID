from __future__ import annotations

import unittest

import numpy as np

from crfid.evaluation.metrics import classification_metrics
from crfid.openems.confirmation import ConfirmationEvidence, assess_confirmation
from crfid.preprocessing.transforms import fit_transform


class SyntheticPipelineTests(unittest.TestCase):
    def test_preprocess_and_metric_pipeline(self) -> None:
        signals = np.arange(8 * 9, dtype=np.float64).reshape(8, 9)
        _, transformed = fit_transform(signals, "first_difference")
        self.assertEqual(transformed.shape, (8, 1, 8))
        metrics = classification_metrics(np.asarray([0, 1]), np.asarray([0, 1]), class_count=2)
        self.assertEqual(metrics["macro_f1"], 1.0)

    def test_physical_redesign_fails_when_global_gain_is_too_small(self) -> None:
        evidence = ConfirmationEvidence(3.0, 3.02, "a-b", "a-c", 1.0, 20.0, True)
        result = assess_confirmation(
            evidence,
            minimum_improvement_fraction=0.05,
            convergence_rms_db_max=2.0,
            peak_shift_mhz_max=150.0,
        )
        self.assertEqual(result["status"], "NOT_CONFIRMED")
        self.assertTrue(result["bottleneck_moved"])


if __name__ == "__main__":
    unittest.main()
