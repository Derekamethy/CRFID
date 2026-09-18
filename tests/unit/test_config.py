from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from crfid.config import config_digest, load_config


ROOT = Path(__file__).resolve().parents[2]


class ConfigTests(unittest.TestCase):
    def test_json_compatible_yaml_loads_without_optional_parser(self) -> None:
        config = load_config(ROOT / "configs" / "strict_dg" / "canonical.yaml", environment={})
        self.assertEqual(config["workflow"], "strict_dg")

    def test_digest_is_order_independent(self) -> None:
        self.assertEqual(config_digest({"a": 1, "b": 2}), config_digest({"b": 2, "a": 1}))

    def test_environment_substitution_preserves_windows_paths(self) -> None:
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                """{
                    "schema_version": 1,
                    "workflow": "baseline",
                    "status": "TEST",
                    "random_seeds": [42],
                    "output_path": "${ROOT}\\\\outputs",
                    "input_path": "${ROOT}\\\\input"
                }""",
                encoding="utf-8",
            )
            root = r"workspace\research\data"
            config = load_config(config_path, environment={"ROOT": root})
            self.assertEqual(config["input_path"], root + r"\input")
            self.assertEqual(config["output_path"], root + r"\outputs")


if __name__ == "__main__":
    unittest.main()
