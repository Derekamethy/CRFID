from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows/11_p4_factor_aware_few_shot"
sys.path.insert(0, str(WORKFLOW))

from p4_factor_aware.bootstrap import (
    paired_bootstrap_effects,
    seed_resampling_sensitivity,
    stratified_block_bootstrap_indices,
)
from p4_factor_aware.metrics import metrics_from_predictions


class StatisticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.truth = np.repeat(np.arange(7), 9)
        self.indices = stratified_block_bootstrap_indices(self.truth, replicates=300, seed=7)

    def test_26_block_level_macro_f1(self) -> None:
        metric = metrics_from_predictions(self.truth, self.truth)
        self.assertEqual(metric["macro_f1"], 1.0)

    def test_27_tagid_stratified_paired_bootstrap(self) -> None:
        sampled_truth = self.truth[self.indices]
        self.assertTrue(np.all(np.stack([(sampled_truth == c).sum(axis=1) for c in range(7)], axis=1) == 9))

    def test_28_known_positive_factor_coverage_benefit(self) -> None:
        baseline = np.tile(np.roll(self.truth, 1), (25, 1))
        comparison = np.tile(self.truth, (25, 1))
        effects = paired_bootstrap_effects(
            truth=self.truth,
            baseline_predictions=baseline,
            comparison_predictions=comparison,
            sampled_indices=self.indices,
            metric="macro_f1",
        )
        self.assertGreater(float(effects.mean()), 0)

    def test_29_known_no_factor_coverage_benefit(self) -> None:
        predictions = np.tile(self.truth, (25, 1))
        effects = paired_bootstrap_effects(
            truth=self.truth,
            baseline_predictions=predictions,
            comparison_predictions=predictions,
            sampled_indices=self.indices,
            metric="macro_f1",
        )
        self.assertTrue(np.array_equal(effects, np.zeros_like(effects)))

    def test_30_known_factor_coverage_harm(self) -> None:
        baseline = np.tile(self.truth, (25, 1))
        comparison = np.tile(np.roll(self.truth, 1), (25, 1))
        effects = paired_bootstrap_effects(
            truth=self.truth,
            baseline_predictions=baseline,
            comparison_predictions=comparison,
            sampled_indices=self.indices,
            metric="macro_f1",
        )
        self.assertLess(float(effects.mean()), 0)

    def test_31_seed_resampling_sensitivity_preserves_pairing(self) -> None:
        effects = np.ones((300, 25), dtype=np.float64)
        sensitivity = seed_resampling_sensitivity(effects, seed=11)
        self.assertEqual(set(sensitivity), {"TAGID_STRATIFIED_BLOCK", "BLOCK_PLUS_SOURCE_SEED", "BLOCK_PLUS_SUPPORT_SEED", "BLOCK_PLUS_BOTH_SEEDS"})
        self.assertTrue(all(np.array_equal(value, np.ones(300)) for value in sensitivity.values()))


if __name__ == "__main__":
    unittest.main()
