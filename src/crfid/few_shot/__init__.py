"""Access and verification layer for the frozen Few-Shot P4 adaptation branch.

This package does not implement few-shot adaptation. The Few-Shot line was
executed once, audited and permanently sealed before migration; regenerating any
prediction, prototype, adapted head or metric is prohibited by the frozen phase
authorization. The modules here therefore only resolve, verify and read the
migrated frozen artifacts.
"""

from .manifests import (
    CANONICAL_STAGES,
    aggregate_rows,
    canonical_payload_sha256,
    verify_branch,
)
from .paths import FrozenBranch, frozen_branch_root, repository_root
from .protocol import FrozenProtocol, load_protocol
from .results import (
    load_canonical_results,
    load_method_comparison,
    load_unit_results,
)

__all__ = [
    "CANONICAL_STAGES",
    "FrozenBranch",
    "FrozenProtocol",
    "aggregate_rows",
    "canonical_payload_sha256",
    "frozen_branch_root",
    "load_canonical_results",
    "load_method_comparison",
    "load_protocol",
    "load_unit_results",
    "repository_root",
    "verify_branch",
]
