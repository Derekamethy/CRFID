"""Migration-validation tests for the frozen External Data1 branch.

These tests never load a model, fit an estimator, run inference, regenerate a
prediction or open raw external signal data. They rehash frozen artifacts and
recompute reported decisions from persisted records.

When the frozen branch has not been materialised (the output tree is excluded
from the source distribution) every test skips rather than fails, so a fresh
checkout stays green while a corrupted branch still fails closed.
"""

from __future__ import annotations

import json
import unittest
from fractions import Fraction
from pathlib import Path

from crfid.config import load_config
from crfid.external_data1 import frozen_manifests
from crfid.external_data1.paths import (
    FrozenBranch,
    frozen_branch_root,
    is_materialised,
    provenance_path,
)

ROOT = Path(__file__).resolve().parents[2]
ANCHORS_PATH = ROOT / "configs" / "external_data1" / "frozen_branch.yaml"


class FrozenBranchMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not is_materialised():
            raise unittest.SkipTest("Frozen External Data1 branch is not materialised in this tree")
        cls.anchors = load_config(ANCHORS_PATH, environment={})
        cls.branch = FrozenBranch.resolve()

    # ------------------------------------------------------------------ integrity

    def test_01_every_stage_aggregate_matches_the_repository_anchor(self) -> None:
        report = frozen_manifests.verify_branch(
            expected_aggregates=self.anchors["frozen_branch"]["stage_aggregates"],
            expected_file_count=self.anchors["frozen_branch"]["file_count"],
            expected_total_bytes=self.anchors["frozen_branch"]["total_bytes"],
            verify_every_file=False,
        )
        self.assertEqual([c.check_id for c in report.checks if not c.passed], [])

    def test_02_every_manifest_row_is_byte_identical(self) -> None:
        report = frozen_manifests.verify_branch(
            expected_aggregates=self.anchors["frozen_branch"]["stage_aggregates"],
            expected_file_count=self.anchors["frozen_branch"]["file_count"],
            expected_total_bytes=self.anchors["frozen_branch"]["total_bytes"],
            verify_every_file=True,
        )
        self.assertEqual(report.file_mismatches, [])
        self.assertEqual(report.unclaimed_files, [])
        self.assertTrue(report.passed)

    def test_03_stage_entry_counts_match_the_anchor(self) -> None:
        expected = self.anchors["frozen_branch"]["stage_entry_counts"]
        for stage, manifest_name in frozen_manifests.STAGE_MANIFESTS:
            blob = json.loads(self.branch.read_text(manifest_name))
            self.assertEqual(len(blob["entries"]), expected[stage], stage)

    def test_04_migration_provenance_records_a_byte_identical_copy(self) -> None:
        provenance = json.loads(provenance_path().read_text(encoding="utf-8"))
        migration = provenance["migration"]
        self.assertTrue(migration["byte_identical_to_source"])
        self.assertFalse(migration["source_modified_by_migration"])
        self.assertEqual(migration["artifacts_regenerated"], 0)
        self.assertEqual(migration["experiments_executed"], 0)
        self.assertEqual(migration["scientific_values_changed"], 0)
        self.assertEqual(migration["migrated_file_count"], self.anchors["frozen_branch"]["file_count"])
        self.assertEqual(migration["migrated_total_bytes"], self.anchors["frozen_branch"]["total_bytes"])
        self.assertTrue(provenance["sealed_identity"]["all_stage_seals_reverified_at_destination"])

    # ------------------------------------------------------------------ protocol

    def test_05_majority_tie_rule_recomputes_from_the_frozen_counts(self) -> None:
        checks = frozen_manifests.verify_majority_tie_rule(
            expected=self.anchors["majority_tie_rule"]
        )
        self.assertEqual([c.check_id for c in checks if not c.passed], [])

    def test_06_tie_rule_selection_is_not_the_weakest_constant_baseline(self) -> None:
        """The frozen order must not have been used to manufacture an easy baseline."""

        counterfactual = self.anchors["majority_tie_rule"]["counterfactual_constant_baseline_accuracy"]
        selected = str(self.anchors["majority_tie_rule"]["selected_class"])
        self.assertEqual(
            counterfactual[selected], max(counterfactual.values()),
            "the selected constant class must be a strongest-available baseline, not a weakened one",
        )

    def test_07_split_anchors_recompute_from_the_frozen_coverage_audit(self) -> None:
        rows = [
            line.split(",")
            for line in self.branch.read_text(frozen_manifests.CLASS_COVERAGE_PATH).splitlines()[1:]
            if line.strip()
        ]
        by_partition = {row[1]: row for row in rows}
        anchors = self.anchors["split_anchors"]
        train = by_partition["train"]
        test = by_partition["test"]
        self.assertEqual(int(train[2]), anchors["train_sample_count"])
        self.assertEqual([int(train[i]) for i in range(4, 8)], anchors["train_class_counts"])
        self.assertEqual(int(test[2]), anchors["test_sample_count"])
        self.assertEqual([int(test[i]) for i in range(4, 8)], anchors["test_class_counts"])

    def test_08_blocked_stage_produced_no_performance_information(self) -> None:
        checks = frozen_manifests.verify_blocked_stage_had_no_performance_access()
        self.assertEqual([c.check_id for c in checks if not c.passed], [])

    def test_09_stage_chronology_is_coherent(self) -> None:
        checks = frozen_manifests.verify_stage_chronology()
        self.assertEqual([c.check_id for c in checks if not c.passed], [])

    def test_10_held_out_scoring_is_one_access_per_run_after_the_freeze(self) -> None:
        events = [
            json.loads(line)
            for line in self.branch.read_text(frozen_manifests.FINAL_TEST_ACCESS_LEDGER).splitlines()
            if line.strip()
        ]
        self.assertEqual(len(events), self.anchors["execution_anchors"]["held_out_scoring_accesses"])
        self.assertEqual(len({event["run_id"] for event in events}), len(events))
        for event in events:
            self.assertTrue(event["access_after_frozen_fit_or_checkpoint"])
            self.assertTrue(event["single_final_access_for_run"])
            self.assertFalse(event["selection_influence"])

    # ------------------------------------------------------------------ results

    def test_11_frozen_conclusions_match_the_repository_anchor(self) -> None:
        checks = frozen_manifests.verify_conclusions(expected=self.anchors["conclusion_anchors"])
        self.assertEqual([c.check_id for c in checks if not c.passed], [])

    def test_12_metric_anchors_recompute_from_the_frozen_prediction_tables(self) -> None:
        """Recount every prediction file and rebuild each score from integer counts."""

        run_paths = {
            "EV3R_R1_REF_SPLIT00":
                "05_external_execution/ev3r_r1_runs/reference_pca_logreg/predictions.csv",
            "EV3R_R1_MAJORITY_BASELINE":
                "06_internal_portability_benchmark/ev3r_r1/EV3R_R1_MAJORITY_BASELINE_PREDICTIONS.csv",
        }
        for seed in self.anchors["random_seeds"]:
            run_paths[f"EV3R_R1_RAW_SPLIT00_SEED{seed}"] = (
                f"05_external_execution/ev3r_r1_runs/raw_1dcnn/seed_{seed}/predictions.csv"
            )
            run_paths[f"EV3R_R1_DIFF_SPLIT00_SEED{seed}"] = (
                f"05_external_execution/ev3r_r1_runs/first_difference_1dcnn/seed_{seed}/predictions.csv"
            )

        order = self.anchors["class_order"]
        expected_n = self.anchors["execution_anchors"]["samples_per_run"]
        self.assertEqual(len(run_paths), self.anchors["execution_anchors"]["run_count"])

        for run_id, relative in run_paths.items():
            lines = self.branch.read_text(relative).splitlines()
            header = lines[0].split(",")
            true_index = header.index("true_label")
            pred_index = header.index("predicted_label")
            matrix = {(t, p): 0 for t in order for p in order}
            rows = 0
            for line in lines[1:]:
                if not line.strip():
                    continue
                parts = line.split(",")
                matrix[(int(parts[true_index]), int(parts[pred_index]))] += 1
                rows += 1
            self.assertEqual(rows, expected_n, run_id)

            correct = sum(matrix[(c, c)] for c in order)
            accuracy = float(Fraction(correct, rows))
            f1_total = Fraction(0)
            for c in order:
                tp = matrix[(c, c)]
                fp = sum(matrix[(t, c)] for t in order) - tp
                fn = sum(matrix[(c, p)] for p in order) - tp
                precision = Fraction(tp, tp + fp) if tp + fp else Fraction(0)
                recall = Fraction(tp, tp + fn) if tp + fn else Fraction(0)
                if precision + recall:
                    f1_total += 2 * precision * recall / (precision + recall)
            macro_f1 = float(f1_total / len(order))

            anchor = self.anchors["metric_anchors"][run_id]
            self.assertEqual(accuracy, anchor["accuracy"], f"{run_id} accuracy")
            self.assertAlmostEqual(macro_f1, anchor["macro_f1"], delta=2e-16, msg=f"{run_id} macro_f1")

    def test_13_majority_baseline_is_constant_at_the_selected_class(self) -> None:
        relative = "06_internal_portability_benchmark/ev3r_r1/EV3R_R1_MAJORITY_BASELINE_PREDICTIONS.csv"
        lines = self.branch.read_text(relative).splitlines()
        index = lines[0].split(",").index("predicted_label")
        predicted = {int(line.split(",")[index]) for line in lines[1:] if line.strip()}
        self.assertEqual(predicted, {self.anchors["majority_tie_rule"]["selected_class"]})

    # ------------------------------------------------------------------ boundaries

    def test_14_no_source_model_or_adjacent_branch_state_was_used(self) -> None:
        status = json.loads(self.branch.read_text("EV3R_R1_STATUS.json"))
        self.assertFalse(status["source_checkpoint_used"])
        self.assertFalse(status["few_shot_checkpoint_used"])
        self.assertFalse(status["P4_numerical_access"])
        self.assertFalse(status["test_metric_during_training"])
        self.assertFalse(status["test_selected_checkpoint"])
        self.assertEqual(status["classification"], self.anchors["execution_anchors"]["classification"])

    def test_15_verification_runs_from_the_repository_without_the_source_branch(self) -> None:
        report = frozen_manifests.verify_branch(verify_every_file=False)
        reported_root = report.as_dict()["frozen_branch_root"]
        self.assertFalse(Path(reported_root).is_absolute())
        self.assertEqual(reported_root, "outputs/external_data1/frozen_branch")
        self.assertTrue((ROOT / reported_root).is_dir())
        self.assertEqual(frozen_branch_root(), ROOT / "outputs" / "external_data1" / "frozen_branch")

    def test_16_no_verification_module_carries_an_absolute_path(self) -> None:
        targets = [
            ROOT / "src" / "crfid" / "external_data1" / "paths.py",
            ROOT / "src" / "crfid" / "external_data1" / "frozen_manifests.py",
            ROOT / "workflows" / "05_external_data1_validation" / "verify_frozen_external_data1_branch.py",
            ANCHORS_PATH,
        ]
        for path in targets:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(":\\", text, path.name)
            self.assertNotIn("/home/", text, path.name)


if __name__ == "__main__":
    unittest.main()
