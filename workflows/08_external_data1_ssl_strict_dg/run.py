"""Executable stages of the external-Data1 SSL strict-DG branch.

    python workflows/08_external_data1_ssl_strict_dg/run.py <stage> [options]

Stages run in order and each writes its own evidence under
``outputs/external_data1_ssl_strict_dg``:

    gate-a       data/architecture readiness, manifests, leakage checks
    screen       source-only SSL objective screen (Gate C input)
    select       freeze the generic and physics-aware objectives (Gate C)
    treatment    one treatment's source-side sweep (Gate B for T0)
    gate-t5      evaluate the preregistered T5 promotion gate
    finalize     final all-source models, checkpoint freeze, Gate-D preregistration
    evaluate-p4  Gate E: single sealed target evaluation
    analyse      hypotheses, registers, figures
    package      final status, file manifest, checksums
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crfid.external_data1_ssl_strict_dg import stages  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    subparsers.add_parser("gate-a")
    subparsers.add_parser("screen").add_argument("--objective", action="append", default=None)
    subparsers.add_parser("select")
    treatment = subparsers.add_parser("treatment")
    treatment.add_argument("--treatment-id", required=True)
    subparsers.add_parser("gate-t5")
    subparsers.add_parser("finalize")
    subparsers.add_parser("rehearse-p4")
    subparsers.add_parser("evaluate-p4")
    subparsers.add_parser("analyse")
    subparsers.add_parser("package")

    arguments = parser.parse_args()
    handler = {
        "gate-a": lambda: stages.run_gate_a(),
        "screen": lambda: stages.run_objective_screen(arguments.objective),
        "select": lambda: stages.run_objective_selection(),
        "treatment": lambda: stages.run_treatment(arguments.treatment_id),
        "gate-t5": lambda: stages.run_t5_gate(),
        "finalize": lambda: stages.run_finalize(),
        "rehearse-p4": lambda: stages.run_target_rehearsal(),
        "evaluate-p4": lambda: stages.run_sealed_target_evaluation(),
        "analyse": lambda: stages.run_analysis(),
        "package": lambda: stages.run_package(),
    }[arguments.stage]
    report = handler()
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
