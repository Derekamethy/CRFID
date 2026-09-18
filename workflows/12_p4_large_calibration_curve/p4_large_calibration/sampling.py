"""Deterministic nested, block-coverage-first P4 support sampling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from p4_factor_aware.constants import ALL_CELLS, CLASS_ORDER, QUERY_FOLDS, ROWS_PER_BLOCK, cell_id
from p4_factor_aware.data import GovernedP4
from p4_factor_aware.protocol import ProtocolViolation

from .constants import MAXIMUM_BUDGET, POSITIVE_BUDGETS


def derived_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def signal_digest(signal: np.ndarray) -> str:
    values = np.ascontiguousarray(signal, dtype="<f8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class NestedSupportPlan:
    fold: int
    support_seed: int
    ordered_indices: np.ndarray
    ordered_cells: tuple[tuple[int, int], ...]
    class_order: tuple[int, ...]

    def indices_for_budget(self, budget: int) -> np.ndarray:
        if budget not in POSITIVE_BUDGETS:
            raise ValueError("budget is outside the preregistered positive budgets")
        return self.ordered_indices[:budget].copy()


def build_nested_plan(data: GovernedP4, *, fold: int, support_seed: int) -> NestedSupportPlan:
    if fold not in QUERY_FOLDS:
        raise ValueError("unknown outer fold")
    pool = tuple(sorted(set(ALL_CELLS) - set(QUERY_FOLDS[fold])))
    if len(pool) != 6:
        raise ProtocolViolation("support-candidate pool must contain six cells")

    cell_rng = np.random.default_rng(derived_seed("calibration-cell-order", support_seed, fold))
    cell_order = tuple(pool[int(index)] for index in cell_rng.permutation(len(pool)))
    class_offset = derived_seed("calibration-class-offset", support_seed, fold) % len(CLASS_ORDER)
    class_order = tuple(CLASS_ORDER[(class_offset + index) % len(CLASS_ORDER)] for index in CLASS_ORDER)

    row_orders: dict[tuple[int, int, int], np.ndarray] = {}
    for class_index in CLASS_ORDER:
        for er, surface in cell_order:
            group = data._groups[(class_index, er, surface)]
            if len(group) != ROWS_PER_BLOCK:
                raise ProtocolViolation("support block does not contain all 50 rows")
            rng = np.random.default_rng(
                derived_seed(
                    "calibration-row-order",
                    support_seed,
                    fold,
                    class_index,
                    er,
                    surface,
                )
            )
            row_orders[(class_index, er, surface)] = group[rng.permutation(ROWS_PER_BLOCK)]

    ordered: list[int] = []
    for repetition_ordinal in range(ROWS_PER_BLOCK):
        for er, surface in cell_order:
            for class_index in class_order:
                ordered.append(
                    int(row_orders[(class_index, er, surface)][repetition_ordinal])
                )
    values = np.asarray(ordered, dtype=np.int64)
    if len(values) != 2100 or len(set(values.tolist())) != 2100:
        raise ProtocolViolation("nested support plan is not a 2,100-row permutation")
    if len(values) < MAXIMUM_BUDGET:
        raise ProtocolViolation("maximum preregistered budget is infeasible")
    return NestedSupportPlan(
        fold=fold,
        support_seed=support_seed,
        ordered_indices=values,
        ordered_cells=cell_order,
        class_order=class_order,
    )


def validate_plan(
    data: GovernedP4,
    plan: NestedSupportPlan,
    *,
    signal_digests: tuple[str, ...],
) -> list[dict[str, object]]:
    query = data.query_view(plan.fold)
    query_indices = set(query.indices.tolist())
    query_blocks = {data._block_ids[index] for index in query.indices}
    query_digests = {signal_digests[index] for index in query.indices}
    rows: list[dict[str, object]] = []
    previous: set[int] = set()
    for budget in POSITIVE_BUDGETS:
        indices = plan.indices_for_budget(budget)
        current = set(indices.tolist())
        labels = data._labels[indices]
        blocks = {data._block_ids[index] for index in indices}
        digests = {signal_digests[index] for index in indices}
        counts = np.bincount(labels, minlength=7)
        passed = bool(
            len(indices) == budget
            and len(current) == budget
            and previous.issubset(current)
            and not (current & query_indices)
            and not (blocks & query_blocks)
            and not (digests & query_digests)
            and int(counts.max() - counts.min()) <= 1
            and len(blocks) == min(budget, 42)
        )
        rows.append(
            {
                "fold": plan.fold,
                "support_seed": plan.support_seed,
                "budget": budget,
                "passed": passed,
                "row_count": len(indices),
                "unique_rows": len(current),
                "unique_condition_blocks": len(blocks),
                "unique_exact_signals": len(digests),
                "minimum_class_count": int(counts.min()),
                "maximum_class_count": int(counts.max()),
                "support_query_row_overlap": len(current & query_indices),
                "support_query_block_overlap": len(blocks & query_blocks),
                "support_query_exact_signal_overlap": len(digests & query_digests),
                "nested_prefix": previous.issubset(current),
            }
        )
        if not passed:
            raise ProtocolViolation("INVALID_NESTED_SUPPORT_PLAN")
        previous = current
    return rows


def plan_manifest_rows(
    data: GovernedP4,
    plan: NestedSupportPlan,
    *,
    signal_digests: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank, index in enumerate(plan.ordered_indices[:MAXIMUM_BUDGET], start=1):
        class_index = int(data._labels[index])
        er = int(data.er[index])
        surface = int(data.surface[index])
        group = data._groups[(class_index, er, surface)]
        source_ordinal = int(np.flatnonzero(group == index)[0])
        first_budget = next(budget for budget in POSITIVE_BUDGETS if rank <= budget)
        rows.append(
            {
                "fold": plan.fold,
                "support_seed": plan.support_seed,
                "selection_rank": rank,
                "first_included_budget": first_budget,
                "dataset_index": int(index),
                "sample_id": f"P4_S{surface}_R{int(data.source_row[index]):04d}",
                "class_index": class_index,
                "TagID": class_index + 1,
                "ER": er,
                "surface_index": surface,
                "factor_cell": cell_id((er, surface)),
                "condition_block_id": data._block_ids[index],
                "row_ordinal_within_block": source_ordinal,
                "exact_signal_sha256": signal_digests[index],
            }
        )
    return rows


def coverage_rows(
    data: GovernedP4,
    plan: NestedSupportPlan,
    *,
    signal_digests: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for budget in POSITIVE_BUDGETS:
        indices = plan.indices_for_budget(budget)
        labels = data._labels[indices]
        blocks = [data._block_ids[index] for index in indices]
        block_counts = np.asarray(list({block: blocks.count(block) for block in set(blocks)}.values()))
        class_counts = np.bincount(labels, minlength=7)
        rows.append(
            {
                "fold": plan.fold,
                "support_seed": plan.support_seed,
                "total_labelled_measurements": budget,
                "unique_TagIDs": len(set(labels.tolist())),
                "unique_ER_levels": len(set(data.er[indices].tolist())),
                "unique_surfaces": len(set(data.surface[indices].tolist())),
                "unique_condition_blocks": len(set(blocks)),
                "unique_exact_signals": len({signal_digests[index] for index in indices}),
                "repeated_measurements_beyond_first_per_block": budget - len(set(blocks)),
                "minimum_labels_per_TagID": int(class_counts.min()),
                "maximum_labels_per_TagID": int(class_counts.max()),
                "minimum_labels_per_covered_block": int(block_counts.min()),
                "maximum_labels_per_covered_block": int(block_counts.max()),
                "selected_factor_cell_order": ";".join(cell_id(cell) for cell in plan.ordered_cells),
            }
        )
    return rows
