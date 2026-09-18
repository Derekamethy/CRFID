from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows" / "19_dann_v2"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(WORKFLOW))

for _workflow_module in ("data", "metrics", "model", "protocol", "target", "training"):
    sys.modules.pop(_workflow_module, None)

from data import SourceData, assert_held_position_isolated, assert_no_block_leakage
from protocol import ARM_A2, FOLDS, LAMBDAS, SEEDS, derive_final_epochs, select_lambda


def _selection_rows(*, strong: bool = True) -> list[dict]:
    rows = []
    for index, value in enumerate(LAMBDAS):
        position = -0.01 * (index + 1)
        upper = position - 0.001 if strong else 0.01
        rows.append(
            {
                "lambda_max": value,
                "run_count": 15,
                "familiar_validation_macro_f1_mean": 0.80 - index * 0.001,
                "familiar_validation_macro_f1_difference": -0.005,
                "position_probe_macro_f1_mean": 0.7 + position,
                "position_probe_macro_f1_difference": position,
                "position_probe_difference_interval_95": [position - 0.02, upper],
                "tagid_probe_macro_f1_mean": 0.90 - index * 0.001,
                "tagid_probe_macro_f1_difference": -0.01,
                "p4_used": False,
                "outer_held_lopo_used_for_selection": False,
            }
        )
    return rows


def test_source_only_selection_prefers_largest_strong_position_reduction() -> None:
    result = select_lambda(_selection_rows())
    assert result["status"] == "PASS_DANN_V2_SOURCE_ONLY_LAMBDA_SELECTION"
    assert result["mechanism_candidate_class"] == "STRONG_POSITION_REDUCTION_CANDIDATES"
    assert result["selected_lambda"] == 1.0
    assert not result["p4_used"]


def test_preregistered_weak_candidate_fallback() -> None:
    result = select_lambda(_selection_rows(strong=False))
    assert result["mechanism_candidate_class"] == "WEAK_POSITION_REDUCTION_CANDIDATES"
    assert result["selected_lambda"] == 1.0


def test_no_candidate_when_retention_or_position_gate_fails() -> None:
    rows = _selection_rows()
    for row in rows:
        row["position_probe_macro_f1_difference"] = 0.01
        row["position_probe_difference_interval_95"] = [-0.01, 0.03]
    result = select_lambda(rows)
    assert result["status"] == "NO_DANN_V2_CANDIDATE_MEETS_SOURCE_MECHANISM_GATE"
    assert result["selected_lambda"] is None


def test_selection_rejects_target_or_outer_held_ranking() -> None:
    rows = _selection_rows()
    rows[0]["p4_used"] = True
    with pytest.raises(RuntimeError):
        select_lambda(rows)
    rows = _selection_rows()
    rows[0]["outer_held_lopo_used_for_selection"] = True
    with pytest.raises(RuntimeError):
        select_lambda(rows)


def test_final_epochs_are_seedwise_three_fold_medians() -> None:
    records = []
    for seed in SEEDS:
        for fold, epoch in zip(FOLDS, (5, 9, 7), strict=True):
            records.append(
                {
                    "arm": ARM_A2,
                    "lambda_max": 0.3,
                    "seed": seed,
                    "fold": fold,
                    "selected_epoch": epoch,
                }
            )
    assert derive_final_epochs(records, 0.3) == {str(seed): 7 for seed in SEEDS}


def test_split_integrity_checks_reject_leakage() -> None:
    rows = (
        {"raw_condition_id": "shared", "position": "P1"},
        {"raw_condition_id": "shared", "position": "P2"},
    )
    with pytest.raises(RuntimeError):
        assert_no_block_leakage(
            rows,
            {
                "inner_train": np.asarray([0]),
                "inner_validation": np.asarray([1]),
                "outer_held": np.asarray([], dtype=int),
            },
        )
    with pytest.raises(RuntimeError):
        assert_held_position_isolated(
            rows,
            {
                "inner_train": np.asarray([0]),
                "inner_validation": np.asarray([], dtype=int),
                "outer_held": np.asarray([0]),
            },
            "P1",
        )

