"""Run the preregistered fixed-position grouped-validity benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crfid.fixed_position_grouped_validity import (
    environment_paths,
    execute_benchmark,
    reaggregate_from_runtime,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="audit governed inputs, persist exact manifests, and run all pre-training gates",
    )
    parser.add_argument(
        "--reaggregate-only",
        action="store_true",
        help="recompute compact aggregates from the 60 saved checkpoints without training",
    )
    args = parser.parse_args()
    raw_root, strict_root, runtime_root = environment_paths()
    if args.prepare_only and args.reaggregate_only:
        parser.error("--prepare-only and --reaggregate-only are mutually exclusive")
    if args.reaggregate_only:
        result = reaggregate_from_runtime(
            raw_root=raw_root,
            strict_root=strict_root,
            runtime_root=runtime_root,
        )
    else:
        result = execute_benchmark(
            raw_root=raw_root,
            strict_root=strict_root,
            runtime_root=runtime_root,
            prepare_only=args.prepare_only,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
