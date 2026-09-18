from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows" / "14_dann_position_intervention"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(WORKFLOW))

for _workflow_module in ("data", "metrics", "model", "protocol", "target", "training"):
    sys.modules.pop(_workflow_module, None)

from data import SourceData, assert_held_position_isolated, assert_no_block_leakage, final_probe_masks
from metrics import condition_centroids, independent_probes
from protocol import classify_intervention, select_nonzero_lambda, source_retention_guardrail


def _selection_rows() -> list[dict]:
    return [
        {"lambda_max": 0.0, "run_count": 15, "held_macro_f1_mean": 0.80, "position_probe_balanced_accuracy_mean": 0.90, "p4_used": False},
        {"lambda_max": 0.03, "run_count": 15, "held_macro_f1_mean": 0.79, "position_probe_balanced_accuracy_mean": 0.70, "p4_used": False},
        {"lambda_max": 0.10, "run_count": 15, "held_macro_f1_mean": 0.795, "position_probe_balanced_accuracy_mean": 0.60, "p4_used": False},
        {"lambda_max": 0.30, "run_count": 15, "held_macro_f1_mean": 0.791, "position_probe_balanced_accuracy_mean": 0.605, "p4_used": False},
        {"lambda_max": 1.00, "run_count": 15, "held_macro_f1_mean": 0.70, "position_probe_balanced_accuracy_mean": 0.50, "p4_used": False},
    ]


def test_source_only_lambda_selection() -> None:
    result = select_nonzero_lambda(_selection_rows())
    assert result["selected_nonzero_lambda"] == 0.10
    assert result["source_only_overall_winner"] == "ERM"
    assert not result["p4_used"]


def test_source_lambda_selection_rejects_target_evidence() -> None:
    rows = _selection_rows()
    rows[1]["p4_used"] = True
    with pytest.raises(RuntimeError):
        select_nonzero_lambda(rows)


def test_source_retention_guardrail() -> None:
    assert source_retention_guardrail(0.80, 0.78)["passed"]
    assert not source_retention_guardrail(0.80, 0.779)["passed"]


def _probe_centroids() -> dict:
    values = []
    labels = []
    positions = []
    conditions = []
    for position_index, position in enumerate(("P1", "P2")):
        for tag in range(7):
            for split in range(2):
                vector = np.zeros(256, dtype=np.float64)
                vector[tag] = 4.0
                vector[20 + position_index] = 2.0
                vector[40 + split] = 0.1
                values.append(vector)
                labels.append(tag)
                positions.append(position)
                conditions.append(f"{position}-tag{tag}-split{split}")
    return {"values": np.vstack(values), "labels": np.asarray(labels), "positions": np.asarray(positions), "condition_ids": np.asarray(conditions)}


def test_independent_position_probe_construction_and_chance() -> None:
    centroids = _probe_centroids()
    train = {value for value in centroids["condition_ids"] if value.endswith("split0")}
    test = {value for value in centroids["condition_ids"] if value.endswith("split1")}
    result = independent_probes(centroids, train, test, probe_seeds=(42,))
    assert result["condition_block_disjoint"]
    assert result["position_chance"] == 0.5
    assert result["position"][0]["scaling_fit_on_probe_train_only"]


def test_tagid_probe_construction() -> None:
    centroids = _probe_centroids()
    train = {value for value in centroids["condition_ids"] if value.endswith("split0")}
    test = {value for value in centroids["condition_ids"] if value.endswith("split1")}
    result = independent_probes(centroids, train, test, probe_seeds=(43,))
    assert result["tagid"][0]["macro_f1"] > 0.9


def test_probe_rejects_cross_boundary_condition() -> None:
    centroids = _probe_centroids()
    all_conditions = set(centroids["condition_ids"].tolist())
    with pytest.raises(RuntimeError):
        independent_probes(centroids, all_conditions, all_conditions, probe_seeds=(42,))


def test_final_three_position_probe_split_is_synchronized() -> None:
    rows = []
    for position in ("P1", "P2", "P3"):
        for tag_id in range(1, 8):
            for er in range(3):
                for surface in ("A1", "A2", "A3"):
                    rows.append({"tag_id": tag_id, "er": er, "surface": surface, "position": position, "raw_condition_id": f"{position}-{tag_id}-{er}-{surface}"})
    count = len(rows)
    data = SourceData(np.zeros((count, 281)), np.zeros(count, dtype=np.int64), tuple(rows), {}, {})
    train, test = final_probe_masks(data)
    assert len(train) == 42 * 3
    assert len(test) == 21 * 3
    train_signatures = {(rows[index]["tag_id"], rows[index]["er"], rows[index]["surface"]) for index in train}
    assert len(train_signatures) == 42


def test_source_block_disjointness_detects_leakage() -> None:
    rows = (
        {"raw_condition_id": "shared", "position": "P1"},
        {"raw_condition_id": "shared", "position": "P1"},
    )
    with pytest.raises(RuntimeError, match="Condition block crosses"):
        assert_no_block_leakage(
            rows,
            {"inner_train": np.asarray([0]), "inner_validation": np.asarray([1]), "outer_held": np.asarray([], dtype=int)},
        )


def test_held_position_isolation_detects_leakage() -> None:
    rows = (
        {"raw_condition_id": "a", "position": "P1"},
        {"raw_condition_id": "b", "position": "P2"},
    )
    with pytest.raises(RuntimeError, match="Held source position leaked"):
        assert_held_position_isolated(
            rows,
            {"inner_train": np.asarray([0]), "inner_validation": np.asarray([], dtype=int), "outer_held": np.asarray([0])},
            "P1",
        )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"position_change": -0.2, "position_interval": (-0.3, -0.1), "p4_change": 0.1, "p4_interval": (0.02, 0.2), "source_change": -0.01}, "POSITION_SUPPRESSION_AND_P4_TRANSFER_BENEFIT_CONFIRMED"),
        ({"position_change": -0.2, "position_interval": (-0.3, -0.1), "p4_change": 0.0, "p4_interval": (-0.1, 0.1), "source_change": 0.0}, "POSITION_SUPPRESSION_WITHOUT_P4_TRANSFER_BENEFIT"),
        ({"position_change": -0.01, "position_interval": (-0.1, 0.1), "p4_change": 0.1, "p4_interval": (0.02, 0.2), "source_change": 0.0}, "P4_TRANSFER_BENEFIT_WITHOUT_CONFIRMED_POSITION_SUPPRESSION"),
        ({"position_change": 0.0, "position_interval": (-0.1, 0.1), "p4_change": 0.0, "p4_interval": (-0.1, 0.1), "source_change": 0.0}, "POSITION_SUPPRESSION_NOT_ACHIEVED"),
        ({"position_change": -0.2, "position_interval": (-0.3, -0.1), "p4_change": -0.1, "p4_interval": (-0.2, -0.01), "source_change": -0.1}, "POSITION_SUPPRESSION_HARMS_TAGID_TRANSFER"),
    ],
)
def test_synthetic_interpretation_cases(kwargs: dict, expected: str) -> None:
    assert classify_intervention(**kwargs) == expected


def test_protocol_defect_classification() -> None:
    assert classify_intervention(position_change=-1, position_interval=(-2, -1), p4_change=1, p4_interval=(1, 2), source_change=0, protocol_passed=False) == "FAIL_PROTOCOL_OR_LABEL_BOUNDARY_DEFECT"
