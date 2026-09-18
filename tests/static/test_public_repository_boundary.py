from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class PublicBoundaryTests(unittest.TestCase):
    def test_private_ledger_is_outside_public_root(self) -> None:
        private_name = "CRFID_reconstruction_" + "audit_private"
        self.assertFalse((ROOT / private_name).exists())

    def test_no_source_mapping_tables(self) -> None:
        disallowed = {"OLD_TO_NEW_FUNCTION_MAP.csv", "SOURCE_EVIDENCE_INDEX.csv", "SEMANTIC_DIFFERENCES.csv"}
        present = {path.name for path in ROOT.rglob("*") if path.is_file()}
        self.assertFalse(disallowed.intersection(present))


if __name__ == "__main__":
    unittest.main()
