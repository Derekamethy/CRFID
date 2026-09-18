"""Run the self-contained release tests with branch-local import isolation.

The historical DANN v1, DANN v2, and position-intervention closures import
their workflow modules under the same top-level names (for example ``model``
and ``protocol``).  They are therefore tested in separate Python processes.
Tests that explicitly require the immutable parent Git object store, external
raw measurements, or unsanitized source-text hashes are not self-contained and
are excluded here; their frozen results are covered by custody evidence and
the metric verifier.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(label: str, arguments: list[str]) -> bool:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONPATH"] = str(ROOT / "src")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        *arguments,
    ]
    print(f"\n[{label}] {' '.join(arguments)}", flush=True)
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    print(f"[{label}] exit={completed.returncode}", flush=True)
    return completed.returncode == 0


def main() -> int:
    core = [
        "tests",
        "--ignore-glob=tests/test_dann*.py",
        "--ignore=tests/test_representation_signal_diagnostic_protocol.py",
        "--ignore=tests/test_source_selection_regret_protocol.py",
        "--deselect=tests/test_angle_distance_factorial_results.py::test_changed_paths_since_the_canonical_release_match_the_committed_allowlists",
        "--deselect=tests/test_groupdro_p4_persistence_repair.py::test_source_only_results_and_failed_execution_are_preserved",
        "--deselect=tests/test_p4_trainable_linear_readout.py::test_parent_artifacts_are_physically_frozen",
    ]
    groups = [
        ("portable_core", core),
        (
            "dann_v1",
            [
                "tests/test_dann_v1_model.py",
                "tests/test_dann_v1_protocol.py",
                "tests/test_dann_v1_evaluation.py",
            ],
        ),
        (
            "dann_v2",
            [
                "tests/test_dann_v2_model.py",
                "tests/test_dann_v2_protocol.py",
                "tests/test_dann_v2_evaluation.py",
            ],
        ),
        (
            "dann_position_intervention",
            [
                "tests/test_dann_position_intervention_model.py",
                "tests/test_dann_position_intervention_protocol.py",
                "tests/test_dann_position_intervention_evaluation.py",
            ],
        ),
    ]
    passed = [run(label, arguments) for label, arguments in groups]
    if all(passed):
        print("\nPASS isolated self-contained release test suite")
        return 0
    print("\nFAIL isolated self-contained release test suite")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
