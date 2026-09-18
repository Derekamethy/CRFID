"""Preflight or execute the frozen paired angle-distance diagnostic."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from crfid.angle_distance_factorial_contrast import DEFAULT_CONFIG, preflight_factorial, run_factorial_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/angle_distance_factorial")
    parser.add_argument("--execute", action="store_true", help="Run the fixed 60-run study; otherwise preflight only")
    arguments = parser.parse_args()
    operation = run_factorial_analysis if arguments.execute else preflight_factorial
    try:
        result = operation(raw_root=arguments.raw_root, config_path=arguments.config, output_root=arguments.output_root)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
