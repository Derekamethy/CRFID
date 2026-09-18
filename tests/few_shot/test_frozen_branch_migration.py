"""Migration-validation tests for the frozen Few-Shot P4 adaptation branch.

These tests never load a model, extract an embedding, build a prototype or an
adapted head, run an optimiser, or regenerate a prediction. They rehash frozen
artifacts and recompute reported statistics from persisted per-unit records.

When the frozen branch has not been materialised (the output tree is excluded
from the source distribution) every test skips rather than fails, so a fresh
checkout stays green while a corrupted branch still fails closed.
"""

from __future__ import annotations

import json
import statistics
import unittest
from pathlib import Path

from crfid.config import load_config
from crfid.few_shot import manifests, protocol, results
from crfid.few_shot.paths import frozen_branch_root

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "few_shot" / "canonical.yaml"
PLACES = 12


def _branch_available() -> bool:
    root = frozen_branch_root()
    return (root / "06_final_comparative_audit_and_archive" / "FINAL_FEW_SHOT_LINE_SEAL.json").is_file()


class FrozenBranchMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not _branch_available():
            raise unittest.SkipTest("Frozen Few-Shot branch is not materialised in this tree")
        cls.config = load_config(CONFIG_PATH)
        cls.protocol = protocol.load_protocol()
        cls.fs3 = results.load_unit_results("FS3")
        cls.fs5 = results.load_unit_results("FS5")
        cls.canonical = results.load_canonical_results()

    def test_01_every_stage_aggregate_matches_the_repository_anchor(self) -> None:
        report = manifests.verify_branch(
            expected_aggregates=self.config["canonical_lineage_aggregates"],
            expected_inventory=self.config["frozen_payload_inventory"],
            verify_every_file=False,
        )
        failed = [check.check_id for check in report.checks if not check.passed]
        self.assertEqual(failed, [])

    def test_02_every_manifest_row_is_byte_identical(self) -> None:
        report = manifests.verify_branch(
            expected_aggregates=self.config["canonical_lineage_aggregates"],
            expected_inventory=self.config["frozen_payload_inventory"],
            verify_every_file=True,
        )
        self.assertEqual(report.file_mismatches, [])
        self.assertTrue(report.passed)

    def test_03_preaudit_baseline_bytes_survive_migration(self) -> None:
        count, mismatches, absent_bytecode = manifests.verify_preaudit_baseline()
        self.assertEqual(mismatches, [])
        self.assertGreater(count, 5000)
        for relative_path in absent_bytecode:
            self.assertTrue(manifests.is_bytecode_cache(relative_path))

    def test_04_protocol_matches_the_declared_configuration(self) -> None:
        self.assertEqual(self.protocol.episode_count, self.config["episode_count"])
        self.assertEqual(
            self.protocol.query_samples_per_episode, self.config["query_samples_per_episode"]
        )
        self.assertEqual(list(self.protocol.checkpoint_seeds), self.config["random_seeds"])
        self.assertEqual(list(self.protocol.shot_counts), self.config["evaluated_shot_conditions"])
        self.assertEqual(
            {str(k): v for k, v in self.protocol.support_sizes.items()},
            self.config["support_sizes"],
        )

    def test_04a_release_inventory_has_no_missing_or_extra_files(self) -> None:
        inventory = manifests.verify_release_inventory()
        self.assertEqual(inventory.row_count, 5807)
        self.assertEqual(inventory.total_bytes, 268302053)
        self.assertEqual(inventory.missing, [])
        self.assertEqual(inventory.extra, [])
        self.assertEqual(inventory.mismatches, [])

    def test_05_formal_unit_counts_reconcile(self) -> None:
        for rows in (self.fs3, self.fs5):
            self.assertEqual(len(rows), self.config["formal_units_per_method_line"])
            self.assertEqual(
                len([row for row in rows if int(row["shot_count"]) == 0]),
                self.config["zero_shot_units_per_method_line"],
            )
            self.assertEqual(
                len([row for row in rows if int(row["shot_count"]) != 0]),
                self.config["adaptation_units_per_method_line"],
            )
            self.assertEqual(len({row["unit_id"] for row in rows}), len(rows))

    def test_06_every_episode_seed_shot_cell_appears_exactly_once(self) -> None:
        expected = {
            (episode, seed, shot)
            for episode in range(self.protocol.episode_count)
            for seed in self.protocol.checkpoint_seeds
            for shot in self.protocol.shot_counts
        }
        for rows in (self.fs3, self.fs5):
            observed = [
                (int(row["episode_id"]), int(row["checkpoint_seed"]), int(row["shot_count"]))
                for row in rows
            ]
            self.assertEqual(len(observed), len(set(observed)))
            self.assertEqual(set(observed), expected)

    def test_07_class_order_is_canonical_in_every_per_class_table(self) -> None:
        expected_names = self.config["class_names"]
        for stage in ("FS3", "FS5"):
            rows = results.load_per_class_results(stage)
            mapping = {int(row["class_index"]): row["class_name"] for row in rows}
            self.assertEqual(sorted(mapping), self.config["class_order"])
            self.assertEqual([mapping[index] for index in sorted(mapping)], expected_names)

    def test_08_canonical_table_recomputes_from_unit_records(self) -> None:
        sources = {"PROTOTYPES": self.fs3, "SOURCE_ANCHORED": self.fs5}
        checked = 0
        for fragment, rows in sources.items():
            grouped = self._episode_means(rows, "sample_macro_f1")
            accuracy = self._episode_means(rows, "sample_accuracy")
            for output in [row for row in self.canonical if fragment in row["method"]]:
                shot = int(output["shot_count"])
                f1_values = [grouped[(episode, shot)] for episode in range(18)]
                acc_values = [accuracy[(episode, shot)] for episode in range(18)]
                self.assertAlmostEqual(
                    statistics.fmean(f1_values), float(output["sample_macro_f1_mean"]), places=PLACES
                )
                self.assertAlmostEqual(
                    statistics.pstdev(f1_values),
                    float(output["sample_macro_f1_population_sd"]),
                    places=PLACES,
                )
                self.assertAlmostEqual(
                    statistics.fmean(acc_values), float(output["sample_accuracy_mean"]), places=PLACES
                )
                checked += 1
        self.assertEqual(checked, 6)

    def test_09_zero_shot_rows_are_shared_not_duplicated_results(self) -> None:
        fs3_zero = {row["unit_id"]: row for row in self.fs3 if row["shot_count"] == "0"}
        fs5_zero = {row["unit_id"]: row for row in self.fs5 if row["shot_count"] == "0"}
        self.assertEqual(fs3_zero, fs5_zero)
        self.assertEqual(len(fs3_zero), self.config["zero_shot_units_per_method_line"])

    def test_10_shot_conditions_are_not_mixed_between_curves(self) -> None:
        for row in self.canonical:
            shot = int(row["shot_count"])
            self.assertIn(shot, self.config["evaluated_shot_conditions"])
            self.assertEqual(
                int(row["labelled_support_per_episode"]), self.config["support_sizes"][str(shot)]
            )
            self.assertEqual(
                int(row["query_samples_per_episode"]), self.config["query_samples_per_episode"]
            )
            if shot == 0:
                self.assertEqual(row["method"], "FROZEN_SOURCE_HEAD")
            else:
                self.assertNotEqual(row["method"], "FROZEN_SOURCE_HEAD")

    def test_11_paired_deltas_recompute(self) -> None:
        deltas = results.load_paired_episode_deltas()
        source = {
            "PROTOTYPE": self._episode_means(self.fs3, "sample_macro_f1"),
            "HEAD": self._episode_means(self.fs5, "sample_macro_f1"),
        }
        self.assertEqual(len(deltas), 15 * self.protocol.episode_count)
        for row in deltas:
            episode = int(row["episode_id"])
            expected = source[row["to_method"]][(episode, int(row["to_shot"]))] - source[
                row["from_method"]
            ][(episode, int(row["from_shot"]))]
            self.assertAlmostEqual(expected, float(row["macro_f1_delta"]), places=PLACES)

    def test_12_no_dependency_on_strict_dg_finalisation_outputs(self) -> None:
        forbidden = ("Strict_DG_Final", "outputs/strict_dg", "strict_runtime")
        for path in sorted((ROOT / "src" / "crfid" / "few_shot").rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for term in forbidden:
                self.assertNotIn(term, text, f"{path.name} references {term}")
        verifier = ROOT / "workflows" / "05_few_shot_p4_adaptation" / "verify_frozen_few_shot_branch.py"
        text = verifier.read_text(encoding="utf-8")
        for term in forbidden:
            self.assertNotIn(term, text)

    def test_13_verification_runs_from_the_repository_without_the_source_branch(self) -> None:
        report = json.loads(
            json.dumps(
                manifests.verify_branch(
                    expected_aggregates=self.config["canonical_lineage_aggregates"],
                    expected_inventory=self.config["frozen_payload_inventory"],
                    verify_every_file=False,
                ).as_dict()
            )
        )
        self.assertTrue(report["all_checks_passed"])
        reported_root = report["frozen_branch_root"]
        self.assertFalse(Path(reported_root).is_absolute())
        self.assertEqual(reported_root, "outputs/few_shot/frozen_branch")
        self.assertTrue((ROOT / reported_root).is_dir())

    @staticmethod
    def _episode_means(rows: list[dict[str, str]], field: str) -> dict[tuple[int, int], float]:
        grouped: dict[tuple[int, int], list[float]] = {}
        for row in rows:
            grouped.setdefault((int(row["episode_id"]), int(row["shot_count"])), []).append(
                float(row[field])
            )
        return {key: statistics.fmean(values) for key, values in grouped.items()}


if __name__ == "__main__":
    unittest.main()
