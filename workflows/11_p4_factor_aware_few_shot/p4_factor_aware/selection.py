"""Matched-budget support-cell and support-row selection."""

from __future__ import annotations

import hashlib
import itertools
from collections import Counter

import numpy as np

from .constants import (
    ALL_CELLS,
    BUDGET_STRATEGIES,
    ER_BALANCED,
    JOINT_FACTOR_COVERAGE,
    QUERY_FOLDS,
    RANDOM_BLOCK,
    SURFACE_BALANCED,
    cell_id,
)
from .protocol import ProtocolViolation, assert_no_query_informed_selection


def _derived_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def factor_count_imbalance(cells: tuple[tuple[int, int], ...], axis: int) -> int:
    counts = Counter(cell[axis] for cell in cells)
    values = [counts[index] for index in range(3)]
    return max(values) - min(values)


def pairwise_dispersion(cells: tuple[tuple[int, int], ...]) -> int:
    return sum(
        abs(left[0] - right[0]) + abs(left[1] - right[1])
        for left, right in itertools.combinations(cells, 2)
    )


def _choose_tie(
    candidates: list[tuple[tuple[int, int], ...]],
    *,
    support_seed: int,
    fold: int,
    strategy: str,
    budget: int,
) -> tuple[tuple[int, int], ...]:
    ordered = sorted(candidates)
    rng = np.random.default_rng(
        _derived_seed("support-cell", support_seed, fold, strategy, budget)
    )
    return ordered[int(rng.integers(0, len(ordered)))]


def select_support_cells(
    *,
    fold: int,
    budget: int,
    strategy: str,
    support_seed: int,
    query_labels: object | None = None,
    query_predictions: object | None = None,
    query_embeddings: object | None = None,
    query_metrics: object | None = None,
) -> tuple[tuple[int, int], ...]:
    """Select cells without access to any query outcome or representation."""

    assert_no_query_informed_selection(
        query_labels=query_labels,
        query_predictions=query_predictions,
        query_embeddings=query_embeddings,
        query_metrics=query_metrics,
    )
    if fold not in QUERY_FOLDS or strategy not in BUDGET_STRATEGIES.get(budget, ()):
        raise ValueError("strategy is outside the preregistered budget scope")
    pool = tuple(sorted(set(ALL_CELLS) - set(QUERY_FOLDS[fold])))
    if len(pool) != 6:
        raise ProtocolViolation("support-candidate pool must contain six cells")

    if strategy == RANDOM_BLOCK:
        rng = np.random.default_rng(
            _derived_seed("support-cell", support_seed, fold, strategy, budget)
        )
        selected = tuple(sorted(pool[index] for index in rng.choice(6, budget, replace=False)))
    else:
        candidates = [tuple(combo) for combo in itertools.combinations(pool, budget)]
        if strategy == ER_BALANCED:
            candidates = [combo for combo in candidates if len({c[0] for c in combo}) == 3]
        elif strategy == SURFACE_BALANCED:
            candidates = [combo for combo in candidates if len({c[1] for c in combo}) == 3]
        elif strategy == JOINT_FACTOR_COVERAGE and budget == 3:
            candidates = [
                combo
                for combo in candidates
                if len({c[0] for c in combo}) == 3 and len({c[1] for c in combo}) == 3
            ]
        elif strategy == JOINT_FACTOR_COVERAGE and budget == 5:
            candidates = [
                combo
                for combo in candidates
                if len({c[0] for c in combo}) == 3 and len({c[1] for c in combo}) == 3
            ]
            best_key = min(
                (
                    factor_count_imbalance(combo, 0),
                    factor_count_imbalance(combo, 1),
                    -pairwise_dispersion(combo),
                )
                for combo in candidates
            )
            candidates = [
                combo
                for combo in candidates
                if (
                    factor_count_imbalance(combo, 0),
                    factor_count_imbalance(combo, 1),
                    -pairwise_dispersion(combo),
                )
                == best_key
            ]
        else:
            raise ValueError("unsupported factor-aware strategy")
        if not candidates:
            raise ProtocolViolation("factor-coverage strategy has no feasible support set")
        selected = _choose_tie(
            candidates,
            support_seed=support_seed,
            fold=fold,
            strategy=strategy,
            budget=budget,
        )

    if len(selected) != budget or len(set(selected)) != budget:
        raise ProtocolViolation("support strategy violated the matched block-shot budget")
    if set(selected) & set(QUERY_FOLDS[fold]):
        raise ProtocolViolation("support/query condition-block overlap")
    return selected


def deterministic_row_offset(
    *, support_seed: int, fold: int, class_index: int, cell: tuple[int, int], row_count: int
) -> int:
    if row_count <= 0:
        raise ValueError("support block has no rows")
    return _derived_seed(
        "support-row", support_seed, fold, class_index, cell_id(cell)
    ) % row_count


def coverage_metadata(
    support_cells: tuple[tuple[int, int], ...],
    query_cells: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    unique_er = len({cell[0] for cell in support_cells})
    unique_surface = len({cell[1] for cell in support_cells})
    dispersion = pairwise_dispersion(support_cells)
    pairs = max(1, len(support_cells) * (len(support_cells) - 1) // 2)
    mean_pairwise = dispersion / pairs if len(support_cells) > 1 else 0.0
    minimum_distances = [
        min(abs(query[0] - support[0]) + abs(query[1] - support[1]) for support in support_cells)
        for query in query_cells
    ]
    return {
        "unique_er_levels": unique_er,
        "unique_surfaces": unique_surface,
        "unique_factor_cells": len(set(support_cells)),
        "er_count_imbalance": factor_count_imbalance(support_cells, 0),
        "surface_count_imbalance": factor_count_imbalance(support_cells, 1),
        "pairwise_factor_grid_dispersion": dispersion,
        "mean_pairwise_factor_grid_distance": mean_pairwise,
        "minimum_query_to_support_factor_distance": min(minimum_distances),
        "mean_query_to_support_factor_distance": float(np.mean(minimum_distances)),
        "every_query_er_represented": {cell[0] for cell in query_cells}.issubset(
            {cell[0] for cell in support_cells}
        ),
        "every_query_surface_represented": {cell[1] for cell in query_cells}.issubset(
            {cell[1] for cell in support_cells}
        ),
        "support_coverage_score": 0.5 * (unique_er / 3 + unique_surface / 3) / 2
        + 0.5 * (mean_pairwise / 4.0),
    }
