"""Support/query separation and query-label access-boundary tests.

The sealed query-label file is never opened by these tests. Support and query
membership is recomputed from the adaptation-visible manifests only.
"""

from __future__ import annotations

import ast
import csv
import json
import unittest
from collections import defaultdict
from pathlib import Path

from crfid.few_shot.paths import FrozenBranch, frozen_branch_root

ROOT = Path(__file__).resolve().parents[2]

SUPPORT_MANIFEST = "02_support_query_protocol/SUPPORT_SAMPLE_MANIFEST.csv"
QUERY_MANIFEST = "02_support_query_protocol/QUERY_PUBLIC_MANIFEST.csv"
QUERY_LABEL_SEAL = "02_support_query_protocol/QUERY_LABEL_SEAL.json"
LINEAGE = "06_final_comparative_audit_and_archive/FINAL_LINEAGE_MANIFEST.json"

#: Identity columns on which support and query must be disjoint.
DISJOINT_KEYS = (
    "sample_id",
    "raw_condition_id",
    "repeat_group_id",
    "exact_signal_sha256",
    "source_row_id",
)

#: Stage-A entry points must never reach evaluation code or sealed labels.
STAGE_A_ENTRY_POINTS = (
    "scripts/execute_fs3_predictions.py",
    "scripts/execute_fs5_predictions.py",
)
EVALUATOR_MODULE = "sealed_evaluator"


def _branch_available() -> bool:
    root = frozen_branch_root()
    return (root / SUPPORT_MANIFEST).is_file()


def _read(branch: FrozenBranch, relative_path: str) -> list[dict[str, str]]:
    with branch.path(relative_path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class LeakageBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not _branch_available():
            raise unittest.SkipTest("Frozen Few-Shot branch is not materialised in this tree")
        cls.branch = FrozenBranch.resolve()
        cls.support = _read(cls.branch, SUPPORT_MANIFEST)
        cls.query = _read(cls.branch, QUERY_MANIFEST)
        cls.support_by_episode: dict[int, list[dict[str, str]]] = defaultdict(list)
        cls.query_by_episode: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in cls.support:
            cls.support_by_episode[int(row["episode_id"])].append(row)
        for row in cls.query:
            cls.query_by_episode[int(row["episode_id"])].append(row)

    def test_01_query_manifest_carries_no_label_column(self) -> None:
        columns = {name.lower() for name in self.query[0]}
        for forbidden in ("label", "label_index", "tag_id", "class_index", "y", "target"):
            self.assertNotIn(forbidden, columns)

    def test_02_support_manifest_carries_support_labels(self) -> None:
        columns = set(self.support[0])
        self.assertIn("label_index", columns)
        self.assertIn("tag_id", columns)

    def test_03_support_and_query_are_disjoint_on_every_identity(self) -> None:
        for episode in sorted(self.support_by_episode):
            support_rows = self.support_by_episode[episode]
            query_rows = self.query_by_episode[episode]
            for key in DISJOINT_KEYS:
                overlap = {row[key] for row in support_rows} & {row[key] for row in query_rows}
                self.assertEqual(overlap, set(), f"episode {episode} overlaps on {key}")

    def test_04_unused_support_pool_blocks_are_also_excluded_from_query(self) -> None:
        for episode in sorted(self.support_by_episode):
            pool = self.support_by_episode[episode]
            unused = [row for row in pool if row["included_in_5_shot"] != "True"]
            query_conditions = {row["raw_condition_id"] for row in self.query_by_episode[episode]}
            self.assertEqual(
                {row["raw_condition_id"] for row in unused} & query_conditions, set()
            )

    def test_05_support_sizes_are_class_balanced_and_nested(self) -> None:
        for episode in sorted(self.support_by_episode):
            rows = self.support_by_episode[episode]
            members = {
                shot: [row for row in rows if row[f"included_in_{shot}_shot"] == "True"]
                for shot in (1, 3, 5)
            }
            for shot, selected in members.items():
                self.assertEqual(len(selected), shot * 7)
                counts: dict[int, int] = defaultdict(int)
                for row in selected:
                    counts[int(row["label_index"])] += 1
                self.assertEqual(sorted(counts), list(range(7)))
                self.assertEqual(set(counts.values()), {shot})
            one = {row["sample_id"] for row in members[1]}
            three = {row["sample_id"] for row in members[3]}
            five = {row["sample_id"] for row in members[5]}
            self.assertTrue(one <= three <= five)

    def test_06_query_population_is_constant_across_shots_and_episodes(self) -> None:
        for episode in sorted(self.query_by_episode):
            rows = self.query_by_episode[episode]
            self.assertEqual(len(rows), 1400)
            self.assertEqual(len({row["sample_id"] for row in rows}), 1400)

    def test_07_stage_a_entry_points_never_import_the_sealed_evaluator(self) -> None:
        for relative_path in STAGE_A_ENTRY_POINTS:
            source = self.branch.read_text(relative_path)
            tree = ast.parse(source)
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.add(node.module or "")
                    imported.update(alias.name for alias in node.names)
            self.assertFalse(
                any(EVALUATOR_MODULE in name for name in imported),
                f"{relative_path} imports the sealed evaluator",
            )

    def test_08_stage_a_recorded_zero_sealed_label_opens(self) -> None:
        for relative_path in (
            "04_prototype_adaptation/fs3_execution/FS3_P4_ACCESS_AUDIT.json",
            "05_head_finetuning/fs5_p4_execution/FS5_P4_ACCESS_AUDIT.json",
        ):
            audit = json.loads(self.branch.read_text(relative_path))
            if "stage_a_sealed_query_label_open_count" in audit:
                self.assertEqual(audit["stage_a_sealed_query_label_open_count"], 0)
                self.assertFalse(audit["query_labels_used_for_prediction_or_selection"])
                self.assertFalse(audit["predictions_regenerated_after_label_access"])
            else:
                self.assertTrue(audit["label_access_permanently_closed"])
                self.assertFalse(audit["prediction_regeneration_during_fs5r"])
                self.assertFalse(audit["model_or_head_change_during_fs5r"])

    def test_09_head_recipe_selection_used_no_target_information(self) -> None:
        audit = json.loads(
            self.branch.read_text("05_head_finetuning/fs4_source_recipe_selection/FS4_ACCESS_AUDIT.json")
        )
        self.assertTrue(audit["source_only_selector"])
        self.assertEqual(audit["p4_raw_signal_open_count"], 0)
        self.assertEqual(audit["p4_support_content_open_count"], 0)
        self.assertEqual(audit["sealed_p4_query_label_open_count"], 0)
        self.assertFalse(audit["p4_metrics_used_for_selection"])
        self.assertFalse(audit["fs3_metrics_used_for_selection"])

    def test_10_sealed_label_custody_is_intact_without_reading_values(self) -> None:
        seal = json.loads(self.branch.read_text(QUERY_LABEL_SEAL))
        self.assertFalse(seal["adaptation_visible_manifests_contain_query_labels"])
        for entry in (seal["query_labels"], seal["query_class_mapping"]):
            path = self.branch.path(entry["relative_path"])
            self.assertTrue(path.is_file())
            self.assertEqual(path.stat().st_size, entry["size_bytes"])
        self.assertEqual(seal["query_labels"]["row_count"], 25200)

    def test_11_recorded_isolation_checks_cover_every_episode_and_shot(self) -> None:
        lineage = json.loads(self.branch.read_text(LINEAGE))
        audit = lineage["protocol_audit"]
        self.assertTrue(audit["all_isolation_checks_passed"])
        self.assertEqual(audit["formal_leakage_check_count"], 54)
        for key in (
            "sample_isolation",
            "condition_block_isolation",
            "repeat_group_isolation",
            "exact_signal_isolation",
            "source_row_isolation",
        ):
            self.assertTrue(audit[key])

    def test_12_no_transductive_query_use_is_declared(self) -> None:
        config = json.loads((ROOT / "configs" / "few_shot" / "canonical.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["transductive_operations"], [])
        self.assertFalse(config["query_labels_during_adaptation"])
        self.assertFalse(config["query_labels_during_model_selection"])
        self.assertFalse(config["query_labels_during_hyperparameter_selection"])


if __name__ == "__main__":
    unittest.main()
