from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
PRE_DG = ROOT / "outputs" / "pre_dg"


def _require_output(path: Path) -> Path:
    if not path.is_file():
        pytest.skip("Complete Pre-DG real-data output is not present")
    return path


def test_pre_dg_real_custody_identity() -> None:
    payload = json.loads(
        _require_output(PRE_DG / "custody" / "dataset_identity.json").read_text()
    )
    assert payload["paper3_tyndall"]["row_count"] == 9600
    assert payload["paper4_depolarizing"]["row_count"] == 12600


def test_pre_dg_real_split_count_and_leakage() -> None:
    payload = json.loads(
        _require_output(PRE_DG / "custody" / "split_integrity.json").read_text()
    )
    assert len(payload) == 18
    assert all(item["passed"] for item in payload.values())


def test_pre_dg_complete_authoritative_execution() -> None:
    manifest = pd.read_csv(
        _require_output(PRE_DG / "execution" / "execution_manifest.csv")
    )
    assert len(manifest) == 54
    assert set(manifest["seed"]) == {42, 43, 44}


def test_pre_dg_real_numerical_comparison_is_fully_classified() -> None:
    payload = json.loads(
        _require_output(
            PRE_DG / "comparison" / "comparison_result.json"
        ).read_text()
    )
    assert payload["exact_identity_checks"] == "PASS"
    assert not payload["numerical_tolerance_passed"]
    failures = [
        item
        for item in payload["run_comparisons"]
        if not item["accuracy_within_tolerance"]
    ]
    assert failures == [
        {
            "absolute_difference": 0.05104166666666665,
            "accuracy_within_tolerance": False,
            "authoritative_accuracy": 0.5494791666666666,
            "authoritative_selected_dropout": 0.3,
            "dataset_id": "paper3_tyndall",
            "reproduced_accuracy": 0.4984375,
            "reproduced_selected_dropout": 0.3,
            "seed": 44,
            "split_name": "paper3_leave_surface_BEND2_raw_condition_val",
        }
    ]
    assert all(
        item["within_tolerance"] for item in payload["split_mean_comparisons"]
    )
