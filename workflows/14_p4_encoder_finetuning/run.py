from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "workflows/11_p4_factor_aware_few_shot"))
sys.path.insert(0, str(ROOT / "workflows/12_p4_large_calibration_curve"))
sys.path.insert(0, str(ROOT / "workflows/13_p4_trainable_linear_readout"))
sys.path.insert(0, str(WORKFLOW_ROOT))
sys.path.insert(0, str(ROOT / "src"))

from p4_encoder_ft.source_selection import run_source_selection
from p4_encoder_ft.study import finalize_artifacts, recover_analysis, run_study
from p4_factor_aware.protocol import ProtocolViolation


def main() -> int:
    parser = argparse.ArgumentParser(description="Historical P4 fine-tuning implementation: requires omitted hash-bound parent assets as well as governed measurements.")
    parser.add_argument("--archive-repository", default=os.environ.get("CRFID_FORENSIC_ARCHIVE", ""))
    parser.add_argument("--data-directory", default=os.environ.get("CRFID_GOVERNED_P4_DIRECTORY", ""))
    parser.add_argument("--output-directory", default=str(ROOT / "outputs/p4_encoder_finetuning"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source-select", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--recover-analysis", action="store_true")
    mode.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    arguments = parser.parse_args()
    if arguments.finalize_only:
        result = finalize_artifacts(arguments.output_directory)
    elif arguments.source_select:
        if not arguments.archive_repository:
            parser.error("--source-select requires --archive-repository")
        result = run_source_selection(archive_repository=arguments.archive_repository, output_directory=arguments.output_directory)
    elif arguments.recover_analysis:
        if not arguments.archive_repository or not arguments.data_directory:
            parser.error("--recover-analysis requires --archive-repository and --data-directory")
        result = recover_analysis(
            repository_root=ROOT,
            archive_repository=arguments.archive_repository,
            data_directory=arguments.data_directory,
            output_directory=arguments.output_directory,
            bootstrap_replicates=arguments.bootstrap_replicates,
        )
    else:
        if not arguments.archive_repository or not arguments.data_directory:
            parser.error("HISTORICAL_ARCHIVE_REQUIRED: --archive-repository must contain the original parent predictions/manifests and outputs/few_shot/frozen_branch checkpoint/preprocessing assets; --data-directory must supply governed P4 CSVs. Measurements alone are insufficient. See REPRODUCIBILITY.md.")
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
    try:
        raise SystemExit(main())
    except (ProtocolViolation, FileNotFoundError) as exc:
        print(f"Historical scientific inputs unavailable or inconsistent: {exc}. See REPRODUCIBILITY.md.", file=sys.stderr)
        raise SystemExit(2)
