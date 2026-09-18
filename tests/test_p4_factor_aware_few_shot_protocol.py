from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows/11_p4_factor_aware_few_shot"
sys.path.insert(0, str(WORKFLOW))

from p4_factor_aware.constants import (
    BUDGET_STRATEGIES,
    ER_BALANCED,
    JOINT_FACTOR_COVERAGE,
    QUERY_FOLDS,
    RANDOM_BLOCK,
    SURFACE_BALANCED,
)
from p4_factor_aware.data import GovernedP4
from p4_factor_aware.metrics import assert_no_row_level_pseudoreplication, majority_vote
from p4_factor_aware.protocol import (
    ProtocolViolation,
    QueryLabelBoundaryError,
    QueryLabelSeal,
    assert_no_query_informed_selection,
    outer_fold_manifest_rows,
    validate_complete_oof_blocks,
    validate_outer_folds,
)
from p4_factor_aware.selection import select_support_cells


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
                    signals.append(np.full(281, class_index + er_index / 10 + surface_index / 100))
                    row_index += 1
    return GovernedP4(
        signals=np.asarray(signals),
        labels=np.asarray(labels),
        er=np.asarray(er),
        surface=np.asarray(surface),
        source_row=np.asarray(source_row),
        source_hashes={"synthetic": "0" * 64},
    )


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = synthetic_p4()

    def test_01_latin_square_outer_partition(self) -> None:
        validate_outer_folds()
        self.assertEqual(set().union(*(set(value) for value in QUERY_FOLDS.values())), set((e, s) for e in range(3) for s in range(3)))

    def test_02_complete_63_block_oof_coverage(self) -> None:
        rows = outer_fold_manifest_rows()
        validate_complete_oof_blocks(rows)
        self.assertEqual(sum(row["partition"] == "QUERY" for row in rows), 63)

    def test_03_support_query_condition_block_disjointness(self) -> None:
        with self.assertRaises(ProtocolViolation):
            self.data.support_set(fold=0, selected_cells=(QUERY_FOLDS[0][0],), support_seed=1)

    def test_04_one_labelled_row_per_support_block(self) -> None:
        cells = select_support_cells(fold=0, budget=3, strategy=JOINT_FACTOR_COVERAGE, support_seed=104729)
        support = self.data.support_set(fold=0, selected_cells=cells, support_seed=104729)
        self.assertEqual(len(support.indices), 21)
        self.assertEqual({row["labelled_rows_from_block"] for row in support.manifest_rows}, {1})

    def test_05_equal_label_budgets(self) -> None:
        for budget, strategies in BUDGET_STRATEGIES.items():
            sizes = []
            for strategy in strategies:
                cells = select_support_cells(fold=1, budget=budget, strategy=strategy, support_seed=130363)
                sizes.append(len(self.data.support_set(fold=1, selected_cells=cells, support_seed=130363).indices))
            self.assertEqual(set(sizes), {budget * 7})

    def test_06_random_support_selection_without_replacement(self) -> None:
        cells = select_support_cells(fold=2, budget=5, strategy=RANDOM_BLOCK, support_seed=155921)
        self.assertEqual(len(cells), len(set(cells)))
        self.assertFalse(set(cells) & set(QUERY_FOLDS[2]))

    def test_07_er_balanced_selection(self) -> None:
        cells = select_support_cells(fold=0, budget=3, strategy=ER_BALANCED, support_seed=181081)
        self.assertEqual({cell[0] for cell in cells}, {0, 1, 2})

    def test_08_surface_balanced_selection(self) -> None:
        cells = select_support_cells(fold=0, budget=3, strategy=SURFACE_BALANCED, support_seed=181081)
        self.assertEqual({cell[1] for cell in cells}, {0, 1, 2})

    def test_09_joint_factor_coverage_selection(self) -> None:
        for budget in (3, 5):
            cells = select_support_cells(fold=1, budget=budget, strategy=JOINT_FACTOR_COVERAGE, support_seed=205439)
            self.assertEqual({cell[0] for cell in cells}, {0, 1, 2})
            self.assertEqual({cell[1] for cell in cells}, {0, 1, 2})

    def test_10_deterministic_support_row_selection(self) -> None:
        cells = select_support_cells(fold=2, budget=3, strategy=JOINT_FACTOR_COVERAGE, support_seed=104729)
        left = self.data.support_set(fold=2, selected_cells=cells, support_seed=104729)
        right = self.data.support_set(fold=2, selected_cells=cells, support_seed=104729)
        np.testing.assert_array_equal(left.indices, right.indices)

    def test_11_query_label_seal_blocks_premature_access(self) -> None:
        seal = QueryLabelSeal(np.asarray([0, 1, 2]))
        with self.assertRaisesRegex(QueryLabelBoundaryError, "FAIL_QUERY_LABEL_BOUNDARY"):
            seal.open_labels()

    def test_12_query_label_seal_freeze_then_open_once(self) -> None:
        seal = QueryLabelSeal(np.asarray([0, 1, 2]))
        seal.freeze_predictions(np.asarray([0, 1, 2]))
        np.testing.assert_array_equal(seal.open_labels(), np.asarray([0, 1, 2]))
        seal.mark_metrics_computed()
        with self.assertRaises(QueryLabelBoundaryError):
            seal.open_labels()

    def test_13_prohibited_query_informed_selection(self) -> None:
        with self.assertRaises(ProtocolViolation):
            assert_no_query_informed_selection(query_embeddings=np.zeros((3, 2)))
        with self.assertRaises(ProtocolViolation):
            select_support_cells(fold=0, budget=1, strategy=RANDOM_BLOCK, support_seed=1, query_metrics={})

    def test_14_majority_vote_uses_lowest_class_tie_break(self) -> None:
        self.assertEqual(majority_vote(np.asarray([2, 2, 1, 1])), 1)

    def test_15_all_query_repetitions_remain_grouped(self) -> None:
        query = self.data.query_view(0)
        counts = np.unique(query.block_ids, return_counts=True)[1]
        self.assertEqual(set(counts.tolist()), {50})
        self.assertEqual(len(counts), 21)

    def test_16_missing_condition_detection(self) -> None:
        with self.assertRaisesRegex(ProtocolViolation, "BLOCKED_P4_STRUCTURE_MISMATCH"):
            GovernedP4(
                signals=self.data.signals[:-1],
                labels=self.data._labels[:-1],
                er=self.data.er[:-1],
                surface=self.data.surface[:-1],
                source_row=self.data.source_row[:-1],
                source_hashes={},
            )

    def test_17_duplicate_block_detection(self) -> None:
        labels = self.data._labels.copy()
        labels[0] = 1
        with self.assertRaisesRegex(ProtocolViolation, "BLOCKED_P4_STRUCTURE_MISMATCH"):
            GovernedP4(
                signals=self.data.signals,
                labels=labels,
                er=self.data.er,
                surface=self.data.surface,
                source_row=self.data.source_row,
                source_hashes={},
            )

    def test_18_no_row_level_pseudoreplication(self) -> None:
        assert_no_row_level_pseudoreplication(inference_unit="condition_block", block_count=63, row_count=3150)
        with self.assertRaises(ProtocolViolation):
            assert_no_row_level_pseudoreplication(inference_unit="row", block_count=63, row_count=3150)


if __name__ == "__main__":
    unittest.main()
