from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKFLOW_ROOT))

from p4_factor_aware.study import run_study


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered P4 factor-aware block-disjoint Few-Shot patch."
    )
    parser.add_argument(
        "--archive-repository",
        default=os.environ.get("CRFID_FORENSIC_ARCHIVE", ""),
        help="Read-only forensic archive carrying the frozen Few-Shot payload.",
    )
    parser.add_argument(
        "--data-directory",
        default=os.environ.get("CRFID_GOVERNED_P4_DIRECTORY", ""),
        help="Directory containing the governed A1/A2/A3 P4 CSV files.",
    )
    parser.add_argument(
        "--output-directory",
        default=str(ROOT / "results/canonical_metrics/p4_factor_aware_few_shot"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if not arguments.archive_repository or not arguments.data_directory:
        parser.error(
            "provide --archive-repository and --data-directory (or the corresponding CRFID environment variables)"
        )
    result = run_study(
        repository_root=ROOT,
        archive_repository=arguments.archive_repository,
        data_directory=arguments.data_directory,
        output_directory=arguments.output_directory,
        dry_run=arguments.dry_run,
        bootstrap_replicates=arguments.bootstrap_replicates,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
