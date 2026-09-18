"""Branch-local paths and a hard write guard.

Every artefact this workflow produces must land under
``outputs/external_data1_ssl_strict_dg``. :func:`ensure_branch_output` refuses any
other destination, keeping other workflow outputs and the retained Strict-DG
evidence untouched.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import BRANCH_ID


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = Path(os.environ.get("CRFID_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs"))).expanduser()
STRICT_DG_ARTIFACT_ROOT = Path(
    os.environ.get("CRFID_STRICT_DG_ARTIFACT_ROOT", str(OUTPUT_ROOT / "strict_dg"))
).expanduser()
BRANCH_OUTPUT_ROOT = OUTPUT_ROOT / BRANCH_ID
BRANCH_CONFIG_ROOT = PROJECT_ROOT / "configs" / BRANCH_ID

# Read-only canonical Strict-DG inputs. These are P1-P3 only.
STRICT_DG_SOURCE_INPUTS = STRICT_DG_ARTIFACT_ROOT / "source_inputs"
STRICT_DG_SOURCE_SIGNALS = STRICT_DG_SOURCE_INPUTS / "source_signals_float64.npy"
STRICT_DG_SOURCE_LABELS = STRICT_DG_SOURCE_INPUTS / "source_labels_int64.npy"
STRICT_DG_SOURCE_REGISTRY = STRICT_DG_SOURCE_INPUTS / "CANONICAL_SOURCE_REGISTRY.csv"
STRICT_DG_LOPO_SPLITS = STRICT_DG_SOURCE_INPUTS / "SOURCE_ONLY_LOPO_SPLITS.csv"

# Read-only canonical Strict-DG anchor identities (no target metric is read here).
STRICT_DG_FROZEN_RELEASE = STRICT_DG_ARTIFACT_ROOT / "frozen_release"
STRICT_DG_SOURCE_SELECTION = STRICT_DG_ARTIFACT_ROOT / "source_selection"

# Sealed target. Opened only by the Gate-E evaluator, only after preregistration.
SEALED_P4_FILE_NAMES = ("A1_P4.csv", "A2_P4.csv", "A3_P4.csv")

OUTPUT_SUBDIRECTORIES = (
    "00_discovery",
    "01_manifests",
    "02_source_only_objective_screen",
    "03_source_only_treatment_selection",
    "04_frozen_final_configs",
    "05_frozen_checkpoints",
    "06_p4_predictions",
    "07_p4_metrics",
    "08_analysis",
    "09_figures",
    "10_logs",
    "11_final_package",
)


class BranchWriteViolation(RuntimeError):
    """Raised when a write is attempted outside this branch's output tree."""


def ensure_branch_output(path: str | Path) -> Path:
    """Return ``path`` resolved, or raise if it escapes the branch output root."""

    resolved = Path(path).resolve()
    root = BRANCH_OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise BranchWriteViolation(
            f"Refusing to write outside the branch output root: {resolved}"
        )
    return resolved


def branch_output(*parts: str) -> Path:
    """Resolve a branch output path and create its parent directory."""

    path = ensure_branch_output(BRANCH_OUTPUT_ROOT.joinpath(*parts))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _declared_root(variable: str) -> Path:
    declared = os.environ.get(variable)
    if not declared:
        raise RuntimeError(
            f"{variable} is not set; this branch never guesses a data location"
        )
    path = Path(declared)
    if not path.is_dir():
        raise RuntimeError(f"{variable} does not point at a directory: {path}")
    return path


def data1_root() -> Path:
    """Directory holding ``set_1.csv`` .. ``set_9.csv`` (unlabelled corpus source)."""

    return _declared_root("CRFID_SSL_DG_DATA1_ROOT")


def sealed_target_root() -> Path:
    """Directory holding the sealed ``A*_P4.csv`` files.

    Resolving this path is not itself target access; only the Gate-E evaluator
    reads the files, and only with a validated preregistration token.
    """

    return _declared_root("CRFID_SSL_DG_TYNDALL_ROOT")
