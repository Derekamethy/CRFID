from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crfid.config import load_config
from crfid.workflows.pre_dg import run


def main() -> int:
    parser = argparse.ArgumentParser(description="CRFID baseline workflow")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if config["workflow"] != "baseline":
        parser.error(f"Expected workflow baseline, found {config['workflow']}")
    result = run(config, execute=arguments.execute)
    payload = result.as_dict() if hasattr(result, "as_dict") else result
    print(json.dumps(payload, indent=2, default=lambda value: value.tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
