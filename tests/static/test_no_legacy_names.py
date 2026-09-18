from __future__ import annotations

import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_ROOTS = ("src", "workflows", "configs", "scripts")
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".toml", ".cff", ".txt", ".json"}
MACHINE_LOCK_PATH_FILES = {"environment/CRFID_ENVIRONMENT_LOCK.json"}
FROZEN_PREFIX = "outputs/few_shot/frozen_branch/"
FROZEN_MANIFEST = ROOT / "configs" / "few_shot" / "frozen_payload_manifest.csv"


def frozen_allowlist() -> set[str]:
    if not FROZEN_MANIFEST.is_file():
        return set()
    with FROZEN_MANIFEST.open(newline="", encoding="utf-8") as handle:
        return {FROZEN_PREFIX + row["relative_path"] for row in csv.DictReader(handle)}


def prohibited_terms() -> tuple[str, ...]:
    return (
        "whole" + " code",
        "last" + " code",
        "Freezing" + " Five",
        "Final " + "whole" + " code",
        "new" + "code",
        "Dataset" + "-2-Compare",
        "One" + "Drive",
        "Stage" + " 17",
        "Round" + " 6",
        "Round" + " 7",
        "R" + "6",
        "R" + "7",
    )


class NamingTests(unittest.TestCase):
    def test_prohibited_terms_absent(self) -> None:
        violations = []
        allowed_frozen = frozen_allowlist()
        for active_root in ACTIVE_ROOTS:
            for path in (ROOT / active_root).rglob("*"):
                relative = path.relative_to(ROOT).as_posix()
                if (
                    path.is_file()
                    and path.suffix.lower() in TEXT_SUFFIXES
                    and relative not in allowed_frozen
                ):
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    for term in prohibited_terms():
                        if relative in MACHINE_LOCK_PATH_FILES and term == "One" + "Drive":
                            continue
                        if term in text or term in relative:
                            violations.append((relative, term))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
