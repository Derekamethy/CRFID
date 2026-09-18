from __future__ import annotations

import json
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

from metrics import (
    aggregate_blocks,
    deterministic_majority_vote,
    paired_vector_bootstrap,
    stratified_paired_block_bootstrap,
    uniform_random_block_reference,
)
import target as target_module
from target import array_sha256, canonical_p4_uncertainty, load_target_labels_after_freeze


def test_p4_deny_before_prediction_freeze() -> None:
    with pytest.raises(RuntimeError, match="FAIL_P4_LABEL_BOUNDARY"):
        load_target_labels_after_freeze(Path("missing"), Path("missing.json"))


def test_prediction_freeze_manifest_must_be_complete(tmp_path: Path) -> None:
    path = tmp_path / "freeze.json"
    path.write_text(
        json.dumps(
            {
                "status": "P4_PREDICTIONS_FROZEN_LABELS_SEALED",
                "prediction_count": 14,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="FAIL_P4_LABEL_BOUNDARY"):
        load_target_labels_after_freeze(tmp_path, path)


def test_prediction_hash_and_block_majority_are_deterministic() -> None:
    predictions = np.asarray([0, 1, 2], dtype=np.int64)
    assert array_sha256(predictions) == array_sha256(predictions.copy())
    changed = predictions.copy()
    changed[-1] = 3
    assert array_sha256(predictions) != array_sha256(changed)
    assert deterministic_majority_vote(np.asarray([2, 2, 1])) == 2
    assert deterministic_majority_vote(np.asarray([3, 2])) == 2


def test_block_metric_has_no_row_pseudoreplication() -> None:
    true = np.repeat(np.asarray([0, 1]), 50)
    predicted = np.concatenate(
        (np.zeros(50, dtype=int), np.ones(26, dtype=int), np.zeros(24, dtype=int))
    )
    blocks = np.repeat(np.asarray(["a", "b"]), 50)
    result = aggregate_blocks(true, predicted, blocks, class_count=2)
    assert result["metrics"]["macro_f1"] == 1.0
    assert len(result["rows"]) == 2


def test_paired_bootstraps_are_reproducible() -> None:
    true = np.repeat(np.arange(7), 9)
    reference = np.tile(true, (5, 1))
    treatment = reference.copy()
    reference[:, ::9] = (reference[:, ::9] + 1) % 7
    result = stratified_paired_block_bootstrap(
        reference, treatment, true, replicates=200, seed=20_260_809
    )
    assert result["point_estimate"] > 0
    vector = paired_vector_bootstrap(
        np.asarray([-0.2, -0.1, -0.3]), replicates=100, seed=20_260_809
    )
    assert vector["interval"][1] < 0


def test_all_p4_intervals_use_the_single_preregistered_seed(monkeypatch) -> None:
    preregistration = json.loads(
        (ROOT / "configs" / "dann_v2" / "preregistered.json").read_text(
            encoding="utf-8"
        )
    )
    bootstrap = preregistration["p4_bootstrap"]
    assert bootstrap["seed"] == 20_260_809
    assert not bootstrap["seed_offsets"]
    calls = []

    def fake_bootstrap(reference, treatment, truth, **kwargs):
        calls.append(kwargs)
        return {
            "point_estimate": 0.0,
            "interval": (-0.1, 0.1),
            "replicates": kwargs["replicates"],
        }

    monkeypatch.setattr(target_module, "stratified_paired_block_bootstrap", fake_bootstrap)
    truth = np.repeat(np.arange(7), 9)
    predictions = np.tile(truth, (5, 1))
    by_arm = {
        arm: predictions.copy()
        for arm in (
            "ARM_A0_ERM_MATCHED",
            "ARM_A1_DOMAIN_POSITIVE_MATCHED",
            "ARM_A2_DANN_NEGATIVE_MATCHED",
        )
    }
    rows = canonical_p4_uncertainty(
        by_arm,
        truth,
        bootstrap_replicates=bootstrap["replicates"],
        bootstrap_seed=bootstrap["seed"],
    )
    assert len(rows) == len(calls) == 12
    assert {call["seed"] for call in calls} == {20_260_809}
    assert {row["rng_seed"] for row in rows} == {20_260_809}


def test_chance_reference_is_fixed_and_near_one_seventh() -> None:
    truth = np.repeat(np.arange(7), 9)
    first = uniform_random_block_reference(
        truth, replicates=500, seed=20_260_810
    )
    second = uniform_random_block_reference(
        truth, replicates=500, seed=20_260_810
    )
    assert first == second
    assert abs(first["mean_random_accuracy"] - 1 / 7) < 0.02
    assert abs(first["mean_random_macro_f1"] - 1 / 7) < 0.03
