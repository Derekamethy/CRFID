from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys
import unittest
from pathlib import Path

import crfid


ROOT = Path(__file__).resolve().parents[2]


class ImportSafetyTests(unittest.TestCase):
    def test_all_public_modules_import(self) -> None:
        names = sorted(module.name for module in pkgutil.walk_packages(crfid.__path__, crfid.__name__ + "."))
        for name in names:
            importlib.import_module(name)

    def test_workflow_cli_help(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "workflows" / "02_strict_source_only_dg" / "run.py"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--config", result.stdout)


if __name__ == "__main__":
    unittest.main()
