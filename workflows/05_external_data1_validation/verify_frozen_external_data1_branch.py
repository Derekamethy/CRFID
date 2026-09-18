"""Branch-local verification of the migrated frozen External Data1 branch.

This entry point is read-only. It loads no model, fits no estimator, runs no
inference, regenerates no prediction and opens no raw external signal file. It
rehashes frozen bytes, recomputes the declared stage aggregates, recomputes the
majority tie-break decision from the frozen training class counts, and checks
the recorded stage chronology and claim boundaries against the repository
anchors in ``configs/external_data1/frozen_branch.yaml``.

Usage::

    python workflows/05_external_data1_validation/verify_frozen_external_data1_branch.py
    python workflows/05_external_data1_validation/verify_frozen_external_data1_branch.py \
        --root <frozen-branch-root> --report <output.json> --skip-file-hashes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crfid.external_data1 import frozen_manifests  # noqa: E402
from crfid.external_data1.paths import is_materialised  # noqa: E402

DEFAULT_ANCHORS = ROOT / "configs" / "external_data1" / "frozen_branch.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", type=Path, default=None,
        help="Frozen branch root (default: outputs/external_data1/frozen_branch)",
    )
    parser.add_argument("--report", type=Path, default=None, help="Write the JSON report here")
    parser.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument(
        "--skip-file-hashes", action="store_true",
        help="Compare sizes only; skip the per-file rehash of every manifest row",
    )
    arguments = parser.parse_args(argv)

    if not is_materialised(arguments.root):
        print(
            json.dumps(
                {
                    "verification": "FROZEN_EXTERNAL_DATA1_BRANCH_VERIFICATION",
                    "status": "SKIPPED_BRANCH_NOT_MATERIALISED",
                    "detail": (
                        "The output tree is excluded from the source distribution. "
                        "Migrate the frozen branch before verification."
                    ),
                },
                indent=2,
            )
        )
        return 0

    anchors: dict[str, Any] = {}
    if arguments.anchors.is_file():
        anchors = json.loads(arguments.anchors.read_text(encoding="utf-8"))
    frozen_anchor = anchors.get("frozen_branch", {})

    report = frozen_manifests.verify_branch(
        arguments.root,
        expected_aggregates=frozen_anchor.get("stage_aggregates"),
        expected_file_count=frozen_anchor.get("file_count"),
        expected_total_bytes=frozen_anchor.get("total_bytes"),
        verify_every_file=not arguments.skip_file_hashes,
    )

    checks = [check.as_dict() for check in report.checks]
    checks += [c.as_dict() for c in frozen_manifests.verify_stage_chronology(arguments.root)]
    checks += [
        c.as_dict()
        for c in frozen_manifests.verify_blocked_stage_had_no_performance_access(arguments.root)
    ]
    checks += [
        c.as_dict()
        for c in frozen_manifests.verify_majority_tie_rule(
            arguments.root, expected=anchors.get("majority_tie_rule")
        )
    ]
    checks += [
        c.as_dict()
        for c in frozen_manifests.verify_conclusions(
            arguments.root, expected=anchors.get("conclusion_anchors")
        )
    ]

    failed = [check for check in checks if not check["passed"]]
    payload: dict[str, Any] = {
        "verification": "FROZEN_EXTERNAL_DATA1_BRANCH_VERIFICATION",
        "execution_performed": {
            "model_loaded": False,
            "estimator_fitted": False,
            "inference_run": False,
            "prediction_regenerated": False,
            "raw_external_signal_data_read": False,
        },
        "frozen_branch_root": report.frozen_branch_root,
        "file_count": report.file_count,
        "total_bytes": report.total_bytes,
        "per_file_rehash_performed": not arguments.skip_file_hashes,
        "file_mismatches": report.file_mismatches,
        "unclaimed_files": report.unclaimed_files,
        "stage_aggregates": report.stage_aggregates,
        "checks": checks,
        "failed_check_ids": [check["check_id"] for check in failed],
        "all_checks_passed": not failed and not report.file_mismatches,
    }
    payload["status"] = (
        "PASS_FROZEN_EXTERNAL_DATA1_BRANCH_VERIFIED"
        if payload["all_checks_passed"]
        else "FAIL_FROZEN_EXTERNAL_DATA1_BRANCH_VERIFICATION"
    )

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(text, encoding="utf-8")
    print(text)
    return 0 if payload["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
