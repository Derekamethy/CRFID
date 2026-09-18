"""Frozen Few-Shot protocol constants, read from the migrated artifacts.

Nothing here is a redefinition. Every value is loaded from the sealed protocol
freeze and audited lineage manifest and then checked against the declared
expectations, so a drifted artifact fails closed instead of being silently
accepted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import FrozenBranch

PROTOCOL_FREEZE = "02_support_query_protocol/PROTOCOL_FREEZE.json"
LINEAGE_MANIFEST = "06_final_comparative_audit_and_archive/FINAL_LINEAGE_MANIFEST.json"
SHOT_DEFINITION = "02_support_query_protocol/SHOT_DEFINITION.md"

CLASS_COUNT = 7
EMBEDDING_DIMENSION = 256
INPUT_LENGTH = 280
EPISODE_COUNT = 18
QUERY_SAMPLES_PER_EPISODE = 1400
CHECKPOINT_SEEDS = (42, 43, 44, 45, 46)
SHOT_COUNTS = (0, 1, 3, 5)
SUPPORT_SIZES = {0: 0, 1: 7, 3: 21, 5: 35}
FORMAL_UNITS_PER_METHOD_LINE = 360
ZERO_SHOT_UNITS_PER_METHOD_LINE = 90
ADAPTED_UNITS_PER_METHOD_LINE = 270
METHOD_LINES = (
    "TARGET_ONLY_SQUARED_EUCLIDEAN_PROTOTYPES",
    "FROZEN_ENCODER_SOURCE_ANCHORED_LINEAR_HEAD_ADAPTATION",
)
ZERO_SHOT_METHOD = "FROZEN_SOURCE_HEAD"


@dataclass(frozen=True)
class FrozenProtocol:
    episode_count: int
    query_samples_per_episode: int
    checkpoint_seeds: tuple[int, ...]
    shot_counts: tuple[int, ...]
    support_sizes: dict[int, int]
    nested_support_sets: bool
    constant_query_identity_across_shots: bool
    all_isolation_checks_passed: bool
    formal_leakage_check_count: int

    def formal_unit_count(self) -> int:
        return self.episode_count * len(self.checkpoint_seeds) * len(self.shot_counts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_count": self.episode_count,
            "query_samples_per_episode": self.query_samples_per_episode,
            "checkpoint_seeds": list(self.checkpoint_seeds),
            "shot_counts": list(self.shot_counts),
            "support_sizes": {str(k): v for k, v in self.support_sizes.items()},
            "nested_support_sets": self.nested_support_sets,
            "constant_query_identity_across_shots": self.constant_query_identity_across_shots,
            "all_isolation_checks_passed": self.all_isolation_checks_passed,
            "formal_leakage_check_count": self.formal_leakage_check_count,
            "formal_unit_count_per_method_line": self.formal_unit_count(),
        }


def load_protocol(root: str | Path | None = None) -> FrozenProtocol:
    """Load and fail-closed validate the frozen protocol."""

    branch = FrozenBranch.resolve(root)
    lineage = json.loads(branch.read_text(LINEAGE_MANIFEST))
    audit = lineage["protocol_audit"]
    protocol = FrozenProtocol(
        episode_count=int(audit["episode_count"]),
        query_samples_per_episode=int(audit["query_size_per_episode"]),
        checkpoint_seeds=tuple(int(seed) for seed in audit["checkpoint_seeds"]),
        shot_counts=tuple(int(shot) for shot in audit["shot_counts"]),
        support_sizes={int(k): int(v) for k, v in audit["support_sizes"].items()},
        nested_support_sets=bool(audit["nested_support_sets"]),
        constant_query_identity_across_shots=bool(audit["constant_query_identity_across_shots"]),
        all_isolation_checks_passed=bool(audit["all_isolation_checks_passed"]),
        formal_leakage_check_count=int(audit["formal_leakage_check_count"]),
    )
    _assert(protocol.episode_count == EPISODE_COUNT, "episode count")
    _assert(protocol.query_samples_per_episode == QUERY_SAMPLES_PER_EPISODE, "query size")
    _assert(protocol.checkpoint_seeds == CHECKPOINT_SEEDS, "checkpoint seeds")
    _assert(protocol.shot_counts == SHOT_COUNTS, "shot counts")
    _assert(protocol.support_sizes == SUPPORT_SIZES, "support sizes")
    _assert(protocol.nested_support_sets, "nested support sets")
    _assert(protocol.constant_query_identity_across_shots, "constant query identity")
    _assert(protocol.all_isolation_checks_passed, "isolation checks")
    _assert(protocol.formal_unit_count() == FORMAL_UNITS_PER_METHOD_LINE, "formal unit count")
    return protocol


def _assert(condition: bool, subject: str) -> None:
    if not condition:
        raise ValueError(f"Frozen Few-Shot protocol drifted from its declared {subject}")
