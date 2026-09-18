from __future__ import annotations

import csv
import importlib.util
import os
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = Path(
    os.environ.get("CRFID_SOURCE_SELECTION_ARCHIVE", "__missing_external_archive__")
).expanduser()
RESULT_ROOT = REPO_ROOT / "results/canonical_metrics/source_selection_regret"
MODULE_PATH = REPO_ROOT / "workflows/13_source_selection_regret/source_selection_regret.py"
SPEC = importlib.util.spec_from_file_location("source_selection_regret_tested", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load source-selection-regret module")
SSR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SSR
SPEC.loader.exec_module(SSR)


def csv_rows(name: str) -> list[dict[str, str]]:
    with (RESULT_ROOT / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def valid_published_grid() -> list[dict[str, str]]:
    rows = []
    required = set(SSR.REPORTED_METRIC_MAP) | {
        "outer_development_sample_accuracy",
        "outer_development_sample_macro_f1",
        "inner_selected_sample_macro_f1",
        "selected_inner_epoch",
    }
    for candidate in SSR.CANDIDATES:
        for fold in SSR.FOLDS:
            for seed in SSR.SEEDS:
                row = {
                    "candidate_id": candidate,
                    "fold_id": fold,
                    "seed": str(seed),
                    "execution_status": "NEWLY_TRAINED_FROZEN_CANDIDATE",
                }
                row.update({name: "0" for name in required})
                rows.append(row)
    return rows


class SourceSelectionRegretProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not ARCHIVE_ROOT.is_dir():
            raise unittest.SkipTest(
                "Requires the external historical source-selection archive. "
                "Set CRFID_SOURCE_SELECTION_ARCHIVE to run these data-bound tests."
            )
        cls.study = SSR.run_study(
            REPO_ROOT,
            ARCHIVE_ROOT,
            RESULT_ROOT,
            bootstrap_replicates=200,
        )

    def test_01_p4_denylist(self) -> None:
        tokens = {token for _, _, group in SSR.P4_DENY_RULES for token in group}
        for required in ("p4", "target", "strict_dg_final_results", "final_p4"):
            self.assertIn(required, tokens)
        self.assertIsNotNone(SSR.classify_denied_path("x/FINAL_P4_RESULTS.csv"))

    def test_02_p4_access_rejection(self) -> None:
        ledger = SSR.AccessLedger(ARCHIVE_ROOT)
        with self.assertRaises(SSR.P4SealViolation):
            ledger.read_bytes(
                "outputs/strict_dg/strict_dg_final_results/FINAL_P4_RESULTS.csv",
                "synthetic accidental P4 request",
            )
        self.assertEqual(ledger.entries[-1]["action"], "REJECTED_BEFORE_OPEN")

    def test_03_c0_c3_lineage_binding(self) -> None:
        candidates = self.study["lineage"]["candidates"]
        self.assertEqual([row["candidate_id"] for row in candidates], list(SSR.CANDIDATES))
        for row in candidates:
            self.assertTrue(row["configuration"])
            self.assertEqual(row["folds"], list(SSR.FOLDS))
            self.assertEqual(row["seeds"], list(SSR.SEEDS))
            self.assertEqual(row["released_evidence"]["prediction_bundle_count"], 15)
            self.assertIn("checkpoint_binding_status", row["checkpoint_lineage"])
        self.assertEqual(
            candidates[-1]["checkpoint_lineage"]["checkpoint_binding_status"],
            "NOT_RETAINED_BUT_15_PREDICTION_OUTPUTS_HASH_BOUND",
        )

    def test_04_exact_three_events(self) -> None:
        self.assertEqual(len(SSR.EVENTS), 3)
        self.assertEqual({row["held_position"] for row in SSR.EVENTS}, set(SSR.POSITIONS))
        self.assertEqual(len(self.study["event_manifest"]), 3)

    def test_05_allowed_positions_exclude_held(self) -> None:
        for event in SSR.EVENTS:
            self.assertEqual(len(event["allowed_positions"]), 2)
            self.assertNotIn(event["held_position"], event["allowed_positions"])
            self.assertEqual(set(event["allowed_positions"]) | {event["held_position"]}, set(SSR.POSITIONS))

    def test_06_held_metric_sealing(self) -> None:
        vault = SSR.EvidenceVault([])
        with self.assertRaises(SSR.HeldMetricSealError):
            vault.held_rows(SSR.EVENTS[0])

    def test_07_selection_freeze_precedes_held_open(self) -> None:
        global_actions = self.study["event_results"]["vault"].sequence_log
        first_held = next(
            index
            for index, row in enumerate(global_actions)
            if row["action"] == "HELD_ROWS_OPENED_AFTER_FREEZE"
        )
        self.assertEqual(
            sum(row["action"] == "SELECTION_FROZEN" for row in global_actions[:first_held]),
            3,
        )
        for event in SSR.EVENTS:
            actions = [row["action"] for row in global_actions if row["event_id"] == event["event_id"]]
            self.assertEqual(actions, ["SELECTION_ROWS_OPENED", "SELECTION_FROZEN", "HELD_ROWS_OPENED_AFTER_FREEZE"])
            manifest = next(row for row in self.study["event_manifest"] if row["event_id"] == event["event_id"])
            self.assertEqual(len(manifest["decision_sha256"]), 64)

    def test_08_complete_tie_break_and_all_tie_synthetic(self) -> None:
        tied = []
        for candidate in reversed(SSR.CANDIDATES):
            tied.append({
                "candidate_id": candidate,
                "worst_outer_held_position_macro_f1": 0.5,
                "mean_outer_held_position_macro_f1": 0.5,
                "worst_class_recall": 0.5,
                "zero_recall_class_frequency": 0.0,
                "condition_block_macro_f1": 0.5,
                "unique_signal_weighted_macro_f1": 0.5,
                "across_seed_standard_deviation": 0.0,
            })
        self.assertEqual(SSR.rank_selector(tied)[0]["candidate_id"], SSR.CANDIDATES[0])

    def test_09_exact_regret_oracle_and_worst_synthetic(self) -> None:
        selected = 0.70
        held = [0.70, 0.75, 0.60, 0.72]
        self.assertAlmostEqual(max(held) - selected, 0.05)
        statistics_rows = []
        for index, candidate in enumerate(SSR.CANDIDATES):
            statistics_rows.append({
                "candidate_id": candidate,
                "worst_outer_held_position_macro_f1": [0.4, 0.6, 0.5, 0.3][index],
                "mean_outer_held_position_macro_f1": 0.9,
                "worst_class_recall": 0.9,
                "zero_recall_class_frequency": 0.0,
                "condition_block_macro_f1": 0.9,
                "unique_signal_weighted_macro_f1": 0.9,
                "across_seed_standard_deviation": 0.0,
            })
        self.assertEqual(SSR.rank_selector(statistics_rows)[0]["candidate_id"], SSR.CANDIDATES[1])

    def test_10_regret_nonnegative_with_numerical_tolerance(self) -> None:
        for row in self.study["event_results"]["primary"]:
            if row["row_type"] in {"seed", "event"}:
                self.assertGreaterEqual(float(row["regret"]), -SSR.NUMERICAL_TOLERANCE)

    def test_11_top1_recovery(self) -> None:
        correlations = self.study["event_results"]["correlations"]
        self.assertEqual(sum(bool(row["top1_recovered"]) for row in correlations), 2)

    def test_12_top2_recovery(self) -> None:
        correlations = self.study["event_results"]["correlations"]
        self.assertEqual(sum(bool(row["selected_in_held_top2"]) for row in correlations), 2)
        self.assertEqual(sum(bool(row["held_oracle_in_selector_top2"]) for row in correlations), 2)

    def test_13_spearman_and_rank_reversal_synthetic(self) -> None:
        self.assertEqual(SSR.spearman([4, 3, 2, 1], [4, 3, 2, 1]), 1.0)
        self.assertEqual(SSR.spearman([4, 3, 2, 1], [1, 2, 3, 4]), -1.0)
        reversal = next(row for row in self.study["event_results"]["correlations"] if row["event_id"] == "HOLD_P2")
        self.assertEqual(reversal["spearman_rho"], -1.0)

    def test_14_kendall_tau_b(self) -> None:
        self.assertEqual(SSR.kendall_tau_b([4, 3, 2, 1], [4, 3, 2, 1]), 1.0)
        self.assertEqual(SSR.kendall_tau_b([4, 3, 2, 1], [1, 2, 3, 4]), -1.0)

    def test_15_seed_paired_bootstrap(self) -> None:
        metadata = self.study["bootstrap_metadata"]
        self.assertEqual(metadata["resampling_unit"], "seed")
        self.assertTrue(metadata["paired_across_candidates_and_positions"])
        self.assertEqual(metadata["replicates"], 200)
        self.assertTrue(all(len(sample) == len(SSR.SEEDS) for sample in metadata["first_three_sampled_seed_tuples"]))
        manifest = json.loads(
            (
                REPO_ROOT
                / "manifests/source_selection_regret/SCIENTIFIC_INPUT_MANIFEST.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["bootstrap"]["replicates"], 10000)

    def test_16_leave_one_seed_out_stability(self) -> None:
        rows = [row for row in self.study["stability"] if row["analysis_type"] == "leave_one_seed_out"]
        self.assertEqual(len(rows), 15)
        self.assertEqual({int(row["seed_or_omitted_seed"]) for row in rows}, set(SSR.SEEDS))

    def test_17_symmetric_margin_perturbation(self) -> None:
        rows = self.study["sensitivity"]
        self.assertEqual(len(rows), 3 * 3 * 4)
        for event in SSR.EVENTS:
            for delta in (0.001, 0.0025, 0.005):
                selected = [row for row in rows if row["event_id"] == event["event_id"] and row["delta_macro_f1_equivalent"] == delta]
                self.assertEqual(sum(row["winner_scenario_count"] for row in selected), 16)
                self.assertTrue(all("not probabilistic" in row["interpretation"] for row in selected))
                self.assertTrue(all(float(row["nominal_relative_margin"]) >= 0 for row in selected))

    def test_18_required_selector_baselines(self) -> None:
        observed = {row["selector"] for row in self.study["baselines"]}
        self.assertEqual(observed, {
            "CANONICAL_HISTORICAL_SELECTOR",
            "ALWAYS_C0",
            "ALWAYS_C1",
            "TWO_POSITION_MEAN_MACRO_F1",
            "TWO_POSITION_WORST_CASE_MACRO_F1",
            "TWO_POSITION_MEAN_MINUS_VARIABILITY",
        })

    def test_19_complete_inventory(self) -> None:
        rows = self.study["inventory"]
        self.assertEqual(len(rows), 60)
        keys = {(row["candidate_id"], row["held_source_position"], int(row["seed"])) for row in rows}
        self.assertEqual(len(keys), 60)
        self.assertTrue(all(row["metric_replay_match"] for row in rows))
        required = {
            "00_EXECUTIVE_SUMMARY.md",
            "01_INPUT_BINDING.json",
            "02_P4_ACCESS_DENYLIST_AND_LOG.csv",
            "03_CANDIDATE_LINEAGE_BINDING.json",
            "04_SOURCE_EVIDENCE_INVENTORY.csv",
            "05_EXECUTION_DECISION.md",
            "06_FROZEN_SELECTION_RULE.md",
            "07_SELECTION_EVENT_MANIFEST.csv",
            "08_SELECTION_DECISIONS.csv",
            "09_HELD_POSITION_ORACLE_RESULTS.csv",
            "10_PRIMARY_SELECTION_REGRET.csv",
            "11_ACCURACY_SELECTION_REGRET.csv",
            "12_RANK_CORRELATION_RESULTS.csv",
            "13_SEED_WINNER_STABILITY.csv",
            "14_SELECTOR_BASELINE_COMPARISON.csv",
            "15_SELECTION_MARGIN_SENSITIVITY.csv",
            "16_BOOTSTRAP_RESULTS.csv",
            "17_CANDIDATE_RANKING_CONSISTENCY.csv",
            "18_SCIENTIFIC_INTERPRETATION.md",
            "19_LIMITATIONS_AND_NONCLAIMS.md",
            "STATUS.md",
        }
        self.assertEqual({path.name for path in RESULT_ROOT.iterdir() if path.is_file()}, required)

    def test_20_missing_evidence_detection(self) -> None:
        rows = valid_published_grid()
        rows.pop()
        with self.assertRaises(SSR.ProtocolError):
            SSR.validate_inventory_grid(rows)

    def test_21_duplicate_evidence_detection(self) -> None:
        rows = valid_published_grid()
        rows.append(dict(rows[0]))
        with self.assertRaises(SSR.ProtocolError):
            SSR.validate_inventory_grid(rows)

    def test_22_no_row_pseudoreplication(self) -> None:
        metadata = self.study["bootstrap_metadata"]
        self.assertFalse(metadata["row_pseudoreplication"])
        manifest = json.loads((REPO_ROOT / "manifests/source_selection_regret/SCIENTIFIC_INPUT_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["bootstrap"]["resampling_unit"], "seed")

    def test_23_prediction_metric_replay(self) -> None:
        self.assertEqual(self.study["metric_replay_count"], 60)
        self.assertEqual(
            sum(
                row["action"] == "OPENED_FOR_SELECTION_BEFORE_GLOBAL_HELD_RELEASE"
                for row in self.study["ledger"].entries
            ),
            120,
        )
        self.assertEqual(
            sum(
                row["action"] == "OPENED_HELD_AFTER_ALL_SELECTIONS_FROZEN"
                for row in self.study["ledger"].entries
            ),
            60,
        )
        parsed_table_index = next(
            index
            for index, row in enumerate(self.study["ledger"].entries)
            if row["path_or_pattern"].endswith("per_run_metrics.csv")
            and row["action"] == "OPENED"
        )
        last_held_index = max(
            index
            for index, row in enumerate(self.study["ledger"].entries)
            if row["action"] == "OPENED_HELD_AFTER_ALL_SELECTIONS_FROZEN"
        )
        self.assertGreater(parsed_table_index, last_held_index)


if __name__ == "__main__":
    unittest.main()
