"""Static hygiene boundary for the migrated Few-Shot branch.

The repository-wide guards in ``tests/static`` require that no text file carries
a machine-specific absolute path or a legacy project name. The frozen Few-Shot
artifacts cannot satisfy that rule: they are immutable sealed evidence whose
bytes may not be edited, and several sealed status records legitimately embed
the execution paths of the original run.

This test pins the resulting exception to a single directory. Every file the
migration authored must be clean, and every unavoidable occurrence must sit
inside the frozen replica.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from crfid.few_shot import manifests

ROOT = Path(__file__).resolve().parents[2]
FROZEN_PREFIX = "outputs/few_shot/frozen_branch/"
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".toml", ".cff", ".txt", ".json"}

#: Files and directories authored by the Few-Shot migration itself.
AUTHORED_PATHS = (
    "src/crfid/few_shot",
    "workflows/05_few_shot_p4_adaptation",
    "configs/few_shot",
    "tests/few_shot",
)
AUTHORED_GLOBS = ("docs/few_shot_*.md",)

ABSOLUTE_PATH_PATTERN = re.compile(
    r"[A-Za-z]:[\\/](?:" + "Users|Documents and Settings" + r")[\\/]" + "|" + "/" + "home/" + r"[^/]+/"
)


def prohibited_terms() -> tuple[str, ...]:
    """Mirror the repository-wide prohibited-name list without embedding it."""

    return (
        "whole" + " code",
        "last" + " code",
        "Freezing" + " Five",
        "new" + "code",
        "Dataset" + "-2-Compare",
        "Co" + "dex",
        "One" + "Drive",
    )


def _authored_files() -> list[Path]:
    files: list[Path] = []
    for relative in AUTHORED_PATHS:
        target = ROOT / relative
        if target.is_dir():
            files.extend(path for path in target.rglob("*") if path.is_file())
        elif target.is_file():
            files.append(target)
    for pattern in AUTHORED_GLOBS:
        files.extend(ROOT.glob(pattern))
    return [
        path
        for path in files
        if path.suffix.lower() in TEXT_SUFFIXES and "__pycache__" not in path.parts
    ]


class FewShotStaticHygieneTests(unittest.TestCase):
    def test_01_authored_files_exist(self) -> None:
        self.assertGreater(len(_authored_files()), 8)

    def test_02_authored_files_carry_no_absolute_machine_paths(self) -> None:
        violations = [
            path.relative_to(ROOT).as_posix()
            for path in _authored_files()
            if ABSOLUTE_PATH_PATTERN.search(path.read_text(encoding="utf-8", errors="ignore"))
        ]
        self.assertEqual(violations, [])

    def test_03_authored_files_carry_no_prohibited_project_names(self) -> None:
        terms = prohibited_terms()
        violations = []
        for path in _authored_files():
            text = path.read_text(encoding="utf-8", errors="ignore")
            relative = path.relative_to(ROOT).as_posix()
            for term in terms:
                if term in text or term in relative:
                    violations.append((relative, term))
        self.assertEqual(violations, [])

    def test_04_authored_files_resolve_paths_relative_to_the_repository(self) -> None:
        package = ROOT / "src" / "crfid" / "few_shot"
        if not package.is_dir():
            self.skipTest("Few-Shot package is not present")
        for path in sorted(package.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("C:" + "\\", text)
            self.assertNotIn("C:" + "/", text)

    def test_05_unavoidable_occurrences_are_confined_to_the_frozen_replica(self) -> None:
        frozen_root = ROOT / "outputs" / "few_shot" / "frozen_branch"
        if not frozen_root.is_dir():
            self.skipTest("Frozen Few-Shot branch is not materialised in this tree")
        branch_root = ROOT / "outputs" / "few_shot"
        stray = []
        for path in branch_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative.startswith(FROZEN_PREFIX):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if ABSOLUTE_PATH_PATTERN.search(text) or any(term in text for term in prohibited_terms()):
                stray.append(relative)
        self.assertEqual(stray, [])

    def test_06_frozen_exception_is_bound_to_the_exact_release_inventory(self) -> None:
        frozen_root = ROOT / "outputs" / "few_shot" / "frozen_branch"
        if not frozen_root.is_dir():
            self.skipTest("Frozen Few-Shot branch is not materialised in this tree")
        inventory = manifests.verify_release_inventory(verify_hashes=False)
        self.assertEqual(inventory.missing, [])
        self.assertEqual(inventory.extra, [])
        self.assertEqual(inventory.mismatches, [])


if __name__ == "__main__":
    unittest.main()
