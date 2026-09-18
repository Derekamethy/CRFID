"""Consume and audit the parent fold and nested-support manifests exactly."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from p4_factor_aware.protocol import ProtocolViolation
from p4_large_calibration.sampling import signal_digest

from .constants import PARENT_ARTIFACT_HASHES, PARENT_RESULTS_RELATIVE, POSITIVE_BUDGETS, SUPPORT_SEEDS


@dataclass(frozen=True)
class InheritedPlan:
    fold: int
    support_seed: int
    indices: np.ndarray

    def indices_for_budget(self, budget: int) -> np.ndarray:
        if budget not in POSITIVE_BUDGETS:
            raise ValueError("unknown positive budget")
        return self.indices[:budget].copy()


def load_inherited_plans(
    *, repository_root: Path, data: object, sha256_file: object
) -> tuple[dict[tuple[int, int], InheritedPlan], list[dict[str, object]]]:
    parent = repository_root / PARENT_RESULTS_RELATIVE
    fold_path = parent / "06_OUTER_QUERY_FOLD_MANIFEST.csv"
    support_path = parent / "07_SUPPORT_SELECTION_MANIFEST.csv"
    if sha256_file(fold_path) != PARENT_ARTIFACT_HASHES[fold_path.name]:
        raise ProtocolViolation("INHERITED_FOLD_MANIFEST_HASH_MISMATCH")
    if sha256_file(support_path) != PARENT_ARTIFACT_HASHES[support_path.name]:
        raise ProtocolViolation("INHERITED_SUPPORT_MANIFEST_HASH_MISMATCH")

    with fold_path.open(encoding="utf-8", newline="") as handle:
        fold_rows = list(csv.DictReader(handle))
    if len(fold_rows) != 189:
        raise ProtocolViolation("INHERITED_FOLD_MANIFEST_STRUCTURE_MISMATCH")
    for fold in range(3):
        selected = [row for row in fold_rows if int(row["fold"]) == fold]
        if len(selected) != 63:
            raise ProtocolViolation("INHERITED_FOLD_MANIFEST_STRUCTURE_MISMATCH")
        if sum(row["partition"] == "QUERY" for row in selected) != 21:
            raise ProtocolViolation("INHERITED_FOLD_MANIFEST_STRUCTURE_MISMATCH")
        if len(data.query_view(fold).indices) != 1050:
            raise ProtocolViolation("INHERITED_FOLD_QUERY_MISMATCH")

    with support_path.open(encoding="utf-8", newline="") as handle:
        support_rows = list(csv.DictReader(handle))
    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in support_rows:
        grouped[(int(row["fold"]), int(row["support_seed"]))].append(row)
    expected_keys = {(fold, seed) for fold in range(3) for seed in SUPPORT_SEEDS}
    if set(grouped) != expected_keys or len(support_rows) != 30_000:
        raise ProtocolViolation("INHERITED_SUPPORT_MANIFEST_STRUCTURE_MISMATCH")

    signal_digests = tuple(signal_digest(row) for row in data.signals)
    plans: dict[tuple[int, int], InheritedPlan] = {}
    audit_rows: list[dict[str, object]] = []
    for key in sorted(grouped):
        fold, support_seed = key
        rows = sorted(grouped[key], key=lambda row: int(row["selection_rank"]))
        if [int(row["selection_rank"]) for row in rows] != list(range(1, 501)):
            raise ProtocolViolation("INHERITED_SUPPORT_RANK_MISMATCH")
        indices = np.asarray([int(row["dataset_index"]) for row in rows], dtype=np.int64)
        if len(np.unique(indices)) != 500:
            raise ProtocolViolation("INHERITED_SUPPORT_ROW_REUSE")
        query = data.query_view(fold)
        query_index_set = set(query.indices.tolist())
        query_block_set = set(query.block_ids)
        query_signal_set = {signal_digests[index] for index in query.indices}
        if set(indices.tolist()) & query_index_set:
            raise ProtocolViolation("INHERITED_SUPPORT_QUERY_ROW_OVERLAP")
        for row, index in zip(rows, indices, strict=True):
            if (
                int(row["class_index"]) != int(data._labels[index])
                or int(row["ER"]) != int(data.er[index])
                or int(row["surface_index"]) != int(data.surface[index])
                or row["condition_block_id"] != data._block_ids[index]
                or row["exact_signal_sha256"] != signal_digests[index]
            ):
                raise ProtocolViolation("INHERITED_SUPPORT_ROW_IDENTITY_MISMATCH")
        plans[key] = InheritedPlan(fold=fold, support_seed=support_seed, indices=indices)
        for budget in POSITIVE_BUDGETS:
            prefix = indices[:budget]
            counts = np.bincount(data._labels[prefix], minlength=7)
            blocks = {data._block_ids[index] for index in prefix}
            signals = {signal_digests[index] for index in prefix}
            if np.any(counts == 0) or int(counts.max() - counts.min()) > 1:
                raise ProtocolViolation("INHERITED_SUPPORT_CLASS_BALANCE_MISMATCH")
            if blocks & query_block_set or signals & query_signal_set:
                raise ProtocolViolation("INHERITED_SUPPORT_QUERY_GROUP_OVERLAP")
            audit_rows.append(
                {
                    "fold": fold,
                    "support_seed": support_seed,
                    "budget": budget,
                    "support_rows": len(prefix),
                    "class_count_min": int(counts.min()),
                    "class_count_max": int(counts.max()),
                    "unique_condition_blocks": len(blocks),
                    "unique_exact_signals": len(signals),
                    "row_disjoint": True,
                    "block_disjoint": True,
                    "exact_signal_disjoint": True,
                    "manifest_identity_verified": True,
                }
            )
    return plans, audit_rows
