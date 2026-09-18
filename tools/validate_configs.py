from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crfid.config import load_config


ENV_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)(?::-([^}]*))?\}")
REQUIRED_WORKFLOW_FIELDS = {"schema_version", "workflow", "status", "random_seeds", "output_path"}


def main() -> int:
    paths = sorted(ROOT.glob("configs/**/*.yaml"))
    results = []
    failed = False
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
            parsed = yaml.safe_load(text)
            if not isinstance(parsed, dict):
                raise ValueError("top-level configuration is not a mapping")
            if not REQUIRED_WORKFLOW_FIELDS.issubset(parsed):
                results.append(
                    {
                        "path": path.relative_to(ROOT).as_posix(),
                        "status": "PASS_FRAGMENT",
                        "role": "component configuration merged by a canonical workflow",
                    }
                )
                continue
            environment = {
                match.group(1): f"<EXTERNAL:{match.group(1)}>"
                for match in ENV_PATTERN.finditer(text)
                if match.group(2) is None
            }
            config = load_config(path, environment=environment)
            results.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "workflow": config["workflow"],
                    "status": "PASS",
                    "external_environment_variables": sorted(environment),
                }
            )
        except Exception as exc:
            failed = True
            results.append({"path": path.relative_to(ROOT).as_posix(), "status": "FAIL", "error": str(exc)})
    print(json.dumps({"configuration_count": len(paths), "failed": failed, "results": results}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
