"""Repository-relative resolution of the migrated frozen Few-Shot branch.

No absolute path is stored anywhere in this package. The frozen branch root is
resolved from the repository root, optionally redirected by ``CRFID_OUTPUT_ROOT``
so the same code works from a relocated output tree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BRANCH_NAME = "few_shot"
FROZEN_BRANCH_DIRECTORY = "frozen_branch"
PROVENANCE_FILE = "MIGRATION_PROVENANCE.json"


def repository_root() -> Path:
    """Return the repository root that contains ``src``, ``configs`` and ``workflows``."""

    return Path(__file__).resolve().parents[3]


def output_root() -> Path:
    """Return the output root, honouring the ``CRFID_OUTPUT_ROOT`` redirection."""

    configured = os.environ.get("CRFID_OUTPUT_ROOT", "").strip()
    if configured:
        candidate = Path(configured)
        return candidate if candidate.is_absolute() else repository_root() / candidate
    return repository_root() / "outputs"


def branch_root() -> Path:
    """Return the migrated Few-Shot output branch root."""

    return output_root() / BRANCH_NAME


def frozen_branch_root() -> Path:
    """Return the byte-identical frozen replica of the authoritative branch."""

    return branch_root() / FROZEN_BRANCH_DIRECTORY


@dataclass(frozen=True)
class FrozenBranch:
    """Fail-closed handle on the migrated frozen Few-Shot branch."""

    root: Path

    @classmethod
    def resolve(cls, root: str | Path | None = None) -> "FrozenBranch":
        target = Path(root) if root is not None else frozen_branch_root()
        target = target.resolve()
        if not target.is_dir():
            raise FileNotFoundError(
                f"Frozen Few-Shot branch is not present at {target}; "
                "the branch must be migrated before verification"
            )
        marker = target / "06_final_comparative_audit_and_archive" / "FINAL_FEW_SHOT_LINE_SEAL.json"
        if not marker.is_file():
            raise FileNotFoundError(
                f"{target} does not contain the sealed Few-Shot archive record"
            )
        return cls(root=target)

    def path(self, relative_path: str) -> Path:
        """Resolve a branch-relative path, rejecting traversal outside the branch."""

        candidate = (self.root / relative_path).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"Path escapes the frozen branch: {relative_path}")
        return candidate

    def read_bytes(self, relative_path: str) -> bytes:
        return self.path(relative_path).read_bytes()

    def read_text(self, relative_path: str) -> str:
        return self.path(relative_path).read_text(encoding="utf-8")
