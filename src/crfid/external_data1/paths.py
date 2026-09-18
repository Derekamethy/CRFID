"""Repository-relative resolution of the migrated frozen External Data1 branch.

No absolute path is stored anywhere in this package. The frozen branch root is
resolved from the repository root, optionally redirected by ``CRFID_OUTPUT_ROOT``
so the same code works from a relocated output tree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BRANCH_NAME = "external_data1"
FROZEN_BRANCH_DIRECTORY = "frozen_branch"
PROVENANCE_FILE = "MIGRATION_PROVENANCE.json"

#: Presence of this file is what distinguishes a materialised frozen branch from
#: an empty or partially copied directory.
FINAL_STAGE_STATUS = "EV3R_R1_STATUS.json"
FINAL_STAGE_MANIFEST = "EV3R_R1_ARTIFACT_HASHES.json"


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
    """Return the External Data1 output branch root."""

    return output_root() / BRANCH_NAME


def frozen_branch_root() -> Path:
    """Return the byte-identical frozen replica of the authoritative branch."""

    return branch_root() / FROZEN_BRANCH_DIRECTORY


def provenance_path() -> Path:
    """Return the migration provenance record for the branch."""

    return branch_root() / PROVENANCE_FILE


def is_materialised(root: str | Path | None = None) -> bool:
    """Report whether the frozen branch is present in this working tree.

    The output tree is excluded from the source distribution, so a fresh
    checkout legitimately has no frozen branch. Callers use this to skip rather
    than fail in that situation, while a present-but-corrupt branch still fails.
    """

    target = Path(root) if root is not None else frozen_branch_root()
    return (target / FINAL_STAGE_STATUS).is_file() and (target / FINAL_STAGE_MANIFEST).is_file()


@dataclass(frozen=True)
class FrozenBranch:
    """Fail-closed handle on the migrated frozen External Data1 branch."""

    root: Path

    @classmethod
    def resolve(cls, root: str | Path | None = None) -> "FrozenBranch":
        target = Path(root) if root is not None else frozen_branch_root()
        target = target.resolve()
        if not target.is_dir():
            raise FileNotFoundError(
                f"Frozen External Data1 branch is not present at {target}; "
                "the branch must be migrated before verification"
            )
        if not (target / FINAL_STAGE_STATUS).is_file():
            raise FileNotFoundError(
                f"{target} does not contain the final External Data1 stage status record"
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
