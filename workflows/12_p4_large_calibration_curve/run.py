from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKFLOW_ROOT))

from p4_large_calibration.study import finalize_artifacts, run_study


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the preregistered larger P4 labelled-target calibration curve."
    )
    parser.add_argument(
        "--archive-repository",
        default=os.environ.get("CRFID_FORENSIC_ARCHIVE", ""),
        help="Read-only canonical V2 forensic repository containing frozen Few-Shot artifacts.",
    )
    parser.add_argument(
        "--data-directory",
        default=os.environ.get("CRFID_GOVERNED_P4_DIRECTORY", ""),
        help="Directory containing governed A1_P4.csv, A2_P4.csv, and A3_P4.csv.",
    )
    parser.add_argument(
        "--output-directory",
        default=str(ROOT / "results/canonical_metrics/p4_large_calibration_curve"),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    arguments = parser.parse_args()

    if arguments.finalize_only:
        result = finalize_artifacts(arguments.output_directory)
    else:
        if not arguments.archive_repository or not arguments.data_directory:
            parser.error(
                "provide --archive-repository and --data-directory (or their CRFID environment variables)"
            )
        selected_mode = "dry-run" if arguments.dry_run else "smoke" if arguments.smoke else "full"
        result = run_study(
            repository_root=ROOT,
            archive_repository=arguments.archive_repository,
            data_directory=arguments.data_directory,
            output_directory=arguments.output_directory,
            mode=selected_mode,
            bootstrap_replicates=arguments.bootstrap_replicates,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
