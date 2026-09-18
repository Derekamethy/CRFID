from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows/12_p4_large_calibration_curve"
FACTOR_WORKFLOW = ROOT / "workflows/11_p4_factor_aware_few_shot"
sys.path.insert(0, str(WORKFLOW))
sys.path.insert(0, str(FACTOR_WORKFLOW))

from p4_factor_aware.data import GovernedP4
from p4_factor_aware.metrics import confusion_matrix
from p4_factor_aware.protocol import QueryLabelBoundaryError, QueryLabelSeal
from p4_large_calibration.constants import POSITIVE_BUDGETS, SUPPORT_SEEDS, TOTAL_BUDGETS
from p4_large_calibration.sampling import (
    build_nested_plan,
    signal_digest,
    validate_plan,
)
from p4_large_calibration.statistics import (
    aggregate_metric_from_histograms,
    extended_metrics,
    hierarchical_bootstrap,
)
from p4_large_calibration.study import _nested_prototypes


def synthetic_p4() -> GovernedP4:
    labels = []
    er = []
    surface = []
    source_row = []
    signals = []
    for surface_index in range(3):
        row_index = 0
        for class_index in range(7):
            for er_index in range(3):
                for repeat in range(50):
                    labels.append(class_index)
                    er.append(er_index)
                    surface.append(surface_index)
                    source_row.append(row_index)
                    base = class_index * 1000 + er_index * 100 + surface_index * 10 + repeat
                    signals.append(np.linspace(base, base + 1, 281, dtype=np.float64))
                    row_index += 1
    return GovernedP4(
        signals=np.asarray(signals),
        labels=np.asarray(labels),
        er=np.asarray(er),
        surface=np.asarray(surface),
        source_row=np.asarray(source_row),
        source_hashes={"synthetic": "0" * 64},
    )


class LargerCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = synthetic_p4()
        # Synthetic floating ramps can collide after large-value rounding; use
        # explicit unique administrative digests for the split-isolation test.
        cls.digests = tuple(f"{index:064x}" for index in range(len(cls.data.signals)))

    def test_01_budget_registry_and_shot_anchors(self) -> None:
        self.assertEqual(TOTAL_BUDGETS, (0, 7, 10, 20, 21, 35, 50, 100, 200, 500))
        self.assertEqual(POSITIVE_BUDGETS[:5], (7, 10, 20, 21, 35))

    def test_02_sampling_is_deterministic(self) -> None:
        left = build_nested_plan(self.data, fold=0, support_seed=SUPPORT_SEEDS[0])
        right = build_nested_plan(self.data, fold=0, support_seed=SUPPORT_SEEDS[0])
        np.testing.assert_array_equal(left.ordered_indices, right.ordered_indices)

    def test_03_seed_changes_sampling(self) -> None:
        left = build_nested_plan(self.data, fold=0, support_seed=SUPPORT_SEEDS[0])
        right = build_nested_plan(self.data, fold=0, support_seed=SUPPORT_SEEDS[1])
        self.assertFalse(np.array_equal(left.ordered_indices[:500], right.ordered_indices[:500]))

    def test_04_exact_budget_accounting_and_no_row_reuse(self) -> None:
        plan = build_nested_plan(self.data, fold=1, support_seed=SUPPORT_SEEDS[2])
        for budget in POSITIVE_BUDGETS:
            indices = plan.indices_for_budget(budget)
            self.assertEqual(len(indices), budget)
            self.assertEqual(len(set(indices.tolist())), budget)

    def test_05_class_balance(self) -> None:
        plan = build_nested_plan(self.data, fold=2, support_seed=SUPPORT_SEEDS[3])
        for budget in POSITIVE_BUDGETS:
            counts = np.bincount(self.data._labels[plan.indices_for_budget(budget)], minlength=7)
            self.assertLessEqual(int(counts.max() - counts.min()), 1)
        for budget, per_class in ((7, 1), (21, 3), (35, 5)):
            counts = np.bincount(self.data._labels[plan.indices_for_budget(budget)], minlength=7)
            self.assertTrue(np.all(counts == per_class))

    def test_06_supports_are_nested_prefixes(self) -> None:
        plan = build_nested_plan(self.data, fold=0, support_seed=SUPPORT_SEEDS[4])
        previous: set[int] = set()
        for budget in POSITIVE_BUDGETS:
            current = set(plan.indices_for_budget(budget).tolist())
            self.assertTrue(previous.issubset(current))
            previous = current

    def test_07_condition_coverage_before_repetition(self) -> None:
        plan = build_nested_plan(self.data, fold=1, support_seed=SUPPORT_SEEDS[5])
        for budget in POSITIVE_BUDGETS:
            indices = plan.indices_for_budget(budget)
            blocks = {self.data._block_ids[index] for index in indices}
            self.assertEqual(len(blocks), min(budget, 42))

    def test_08_support_query_row_block_and_signal_disjoint(self) -> None:
        plan = build_nested_plan(self.data, fold=2, support_seed=SUPPORT_SEEDS[6])
        rows = validate_plan(self.data, plan, signal_digests=self.digests)
        self.assertTrue(all(bool(row["passed"]) for row in rows))
        self.assertEqual({row["support_query_row_overlap"] for row in rows}, {0})
        self.assertEqual({row["support_query_block_overlap"] for row in rows}, {0})
        self.assertEqual({row["support_query_exact_signal_overlap"] for row in rows}, {0})

    def test_09_query_is_fixed_across_budgets(self) -> None:
        query = self.data.query_view(0)
        self.assertEqual(len(query.indices), 1050)
        self.assertEqual(len(set(query.block_ids)), 21)

    def test_10_query_labels_fail_closed(self) -> None:
        seal = QueryLabelSeal(np.asarray([0, 1, 2]))
        with self.assertRaises(QueryLabelBoundaryError):
            seal.open_labels()
        seal.freeze_predictions(np.asarray([0, 1, 2]))
        np.testing.assert_array_equal(seal.open_labels(), np.asarray([0, 1, 2]))
        seal.mark_metrics_computed()

    def test_11_flexible_prototypes_cover_every_class_without_mutation(self) -> None:
        plan = build_nested_plan(self.data, fold=0, support_seed=SUPPORT_SEEDS[7])
        indices = plan.indices_for_budget(500)
        generator = torch.Generator().manual_seed(123)
        embeddings = torch.randn((500, 256), generator=generator)
        before = embeddings.clone()
        prototypes = _nested_prototypes(embeddings, self.data._labels[indices])
        self.assertEqual(set(prototypes), set(POSITIVE_BUDGETS))
        self.assertEqual(tuple(prototypes[500].shape), (7, 256))
        torch.testing.assert_close(torch.linalg.vector_norm(prototypes[500], dim=1), torch.ones(7, dtype=torch.float64))
        torch.testing.assert_close(embeddings, before)

    def test_12_metric_correctness(self) -> None:
        truth = np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
        prediction = truth.copy()
        metrics = extended_metrics(truth, prediction)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        np.testing.assert_array_equal(confusion_matrix(truth, prediction), np.eye(7, dtype=np.int64) * 2)

    def test_13_aggregate_histogram_metrics(self) -> None:
        histograms = np.zeros((len(TOTAL_BUDGETS), 5, 20, 63, 7), dtype=np.int16)
        for block, class_index in enumerate(np.repeat(np.arange(7), 9)):
            histograms[..., block, class_index] = 50
        macro, accuracy = aggregate_metric_from_histograms(histograms)
        self.assertTrue(np.all(macro == 1.0))
        self.assertTrue(np.all(accuracy == 1.0))

    def test_14_bootstrap_reproducibility(self) -> None:
        histograms = np.zeros((len(TOTAL_BUDGETS), 5, 20, 63, 7), dtype=np.int16)
        truth = np.repeat(np.arange(7), 9).astype(np.int64)
        for block, class_index in enumerate(truth):
            histograms[..., block, class_index] = 50
        left = hierarchical_bootstrap(histograms, truth, replicates=16, seed=20260809)
        right = hierarchical_bootstrap(histograms, truth, replicates=16, seed=20260809)
        np.testing.assert_array_equal(left.macro_f1, right.macro_f1)
        np.testing.assert_array_equal(left.accuracy, right.accuracy)
        np.testing.assert_array_equal(left.block_weights_sha256_payload, right.block_weights_sha256_payload)

    def test_15_bootstrap_changes_with_rng_seed(self) -> None:
        histograms = np.zeros((len(TOTAL_BUDGETS), 5, 20, 63, 7), dtype=np.int16)
        truth = np.repeat(np.arange(7), 9).astype(np.int64)
        for block, class_index in enumerate(truth):
            histograms[..., block, class_index] = 50
        left = hierarchical_bootstrap(histograms, truth, replicates=8, seed=1)
        right = hierarchical_bootstrap(histograms, truth, replicates=8, seed=2)
        self.assertFalse(np.array_equal(left.block_weights_sha256_payload, right.block_weights_sha256_payload))

    def test_16_result_schema_budget_axis_is_unique(self) -> None:
        self.assertEqual(len(TOTAL_BUDGETS), len(set(TOTAL_BUDGETS)))
        self.assertEqual(len(SUPPORT_SEEDS), 20)


if __name__ == "__main__":
    unittest.main()
