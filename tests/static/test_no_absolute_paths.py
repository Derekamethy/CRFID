from __future__ import annotations

import csv
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".toml", ".cff", ".txt", ".json"}
MACHINE_ENVIRONMENT_FILES = {
    "AGENTS.md",
    "environment/ENVIRONMENT_SETUP.md",
    "tools/assert_environment.py",
}
FROZEN_PREFIX = "outputs/few_shot/frozen_branch/"
FROZEN_MANIFEST = ROOT / "configs" / "few_shot" / "frozen_payload_manifest.csv"


def frozen_allowlist() -> set[str]:
    if not FROZEN_MANIFEST.is_file():
        return set()
    with FROZEN_MANIFEST.open(newline="", encoding="utf-8") as handle:
        return {FROZEN_PREFIX + row["relative_path"] for row in csv.DictReader(handle)}


class AbsolutePathTests(unittest.TestCase):
    def test_no_local_absolute_paths(self) -> None:
        windows_root = r"[A-Za-z]:[\\/](?:" + "Users|Documents and Settings" + r")[\\/]"
        unix_root = "/" + "home/" + r"[^/]+/"
        pattern = re.compile(windows_root + "|" + unix_root)
        violations = []
        allowed_frozen = frozen_allowlist()
        for path in ROOT.rglob("*"):
            relative = path.relative_to(ROOT).as_posix()
            if (
                path.is_file()
                and path.suffix.lower() in TEXT_SUFFIXES
                and relative not in MACHINE_ENVIRONMENT_FILES
                and relative not in allowed_frozen
            ):
                if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                    violations.append(relative)
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
