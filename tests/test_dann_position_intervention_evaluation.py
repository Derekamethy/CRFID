from __future__ import annotations

import json
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

from metrics import aggregate_blocks, deterministic_majority_vote, paired_vector_bootstrap, stratified_paired_block_bootstrap
from target import array_sha256, load_target_labels_after_freeze


def test_p4_deny_before_final_stage_rule_and_label_seal(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="FAIL_P4_LABEL_BOUNDARY"):
        load_target_labels_after_freeze(tmp_path, tmp_path / "missing.json")


def test_prediction_freeze_manifest_must_be_complete(tmp_path: Path) -> None:
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps({"status": "P4_PREDICTIONS_FROZEN_LABELS_SEALED", "prediction_count": 9}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="FAIL_P4_LABEL_BOUNDARY"):
        load_target_labels_after_freeze(tmp_path, path)


def test_prediction_freezing_hash_is_deterministic_and_content_bound() -> None:
    predictions = np.asarray([0, 1, 2], dtype=np.int64)
    assert array_sha256(predictions) == array_sha256(predictions.copy())
    changed = predictions.copy(); changed[-1] = 3
    assert array_sha256(predictions) != array_sha256(changed)


def test_deterministic_block_majority_vote_and_tie_break() -> None:
    assert deterministic_majority_vote(np.asarray([2, 2, 1])) == 2
    assert deterministic_majority_vote(np.asarray([3, 2])) == 2


def test_block_macro_f1_and_no_row_pseudoreplication() -> None:
    true = np.repeat(np.asarray([0, 1]), 50)
    predicted = np.concatenate((np.zeros(50, dtype=int), np.ones(26, dtype=int), np.zeros(24, dtype=int)))
    blocks = np.repeat(np.asarray(["a", "b"]), 50)
    result = aggregate_blocks(true, predicted, blocks, class_count=2)
    assert result["metrics"]["macro_f1"] == 1.0
    assert len(result["rows"]) == 2


def test_condition_block_rejects_mixed_labels() -> None:
    with pytest.raises(RuntimeError):
        aggregate_blocks(np.asarray([0, 1]), np.asarray([0, 1]), np.asarray(["a", "a"]), class_count=2)


def test_paired_erm_dann_contrast_and_tagid_stratified_bootstrap() -> None:
    true = np.repeat(np.arange(7), 9)
    erm = np.tile(true, (5, 1))
    dann = erm.copy()
    erm[:, ::9] = (erm[:, ::9] + 1) % 7
    result = stratified_paired_block_bootstrap(erm, dann, true, replicates=200, seed=1)
    assert result["point_estimate"] > 0
    assert len(result["seedwise_contrasts"]) == 5


def test_seed_resampling_sensitivity() -> None:
    true = np.repeat(np.arange(7), 9)
    erm = np.tile(true, (5, 1))
    dann = erm.copy()
    result = stratified_paired_block_bootstrap(erm, dann, true, replicates=50, resample_training_seeds=True)
    assert result["resample_training_seeds"]
    assert result["interval"] == (0.0, 0.0)


def test_source_probe_paired_bootstrap() -> None:
    result = paired_vector_bootstrap(np.asarray([-0.2, -0.1, -0.3]), replicates=100, seed=2)
    assert result["point_estimate"] < 0
    assert result["interval"][1] < 0


def test_repeated_rows_are_not_bootstrap_units() -> None:
    true = np.repeat(np.arange(7), 9)
    predictions = np.tile(true, (5, 1))
    result = stratified_paired_block_bootstrap(predictions, predictions, true, replicates=10)
    assert len(true) == 63
    assert result["replicates"] == 10


def test_authoritative_and_earlier_patch_preservation_paths_are_additive() -> None:
    allowed = (
        "workflows/14_dann_position_intervention/",
        "configs/dann_position_intervention/",
        "tests/test_dann_position_intervention_",
        "docs/DANN_POSITION_INTERVENTION_",
        "results/canonical_metrics/dann_position_intervention/",
        "manifests/dann_position_intervention/",
    )
    assert all("patch_fixed_position" not in path and "patch_angle" not in path for path in allowed)
