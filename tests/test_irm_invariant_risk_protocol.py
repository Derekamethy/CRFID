from __future__ import annotations

import numpy as np
import pytest

from crfid.irm_invariant_risk.core import (
    FINAL_P4_EVALUATION,
    IRMProtocolError,
    P4LabelSeal,
    classify_invariance_diagnostics,
    classify_main_result,
    deterministic_block_majority_vote,
    execution_record,
    paired_source_bootstrap,
    paired_tagid_stratified_block_bootstrap,
    select_lambda_source_only,
)


def _lambda_rows():
    rows = []
    scores = {1.0: 0.50, 10.0: 0.51, 100.0: 0.505, 1000.0: 0.40}
    for value, score in scores.items():
        for fold, position in (("S1", "P1"), ("S2", "P2"), ("S3", "P3")):
            for seed in (42, 43, 44, 45, 46):
                rows.append({"lambda": value, "fold_id": fold, "held_position": position, "seed": seed, "held_macro_f1": score, "held_accuracy": score + 0.1, "validation_irm_penalty": value / 1000})
    return rows


def test_source_only_lambda_selector() -> None:
    result = select_lambda_source_only(_lambda_rows())
    assert result["selected_lambda"] == 10.0 and not result["p4_accessed"]


def test_lambda_selector_requires_all_units() -> None:
    with pytest.raises(IRMProtocolError):
        select_lambda_source_only(_lambda_rows()[:-1])


def test_paired_source_bootstrap() -> None:
    rows = [{"held_position": p, "seed": s, "erm_macro_f1": 0.5, "irm_macro_f1": 0.6} for p in ("P1", "P2", "P3") for s in (42, 43, 44, 45, 46)]
    result = paired_source_bootstrap(rows, resamples=100, seed=2)
    assert result["mean_contrast"]["interval_95"][0] > 0


def test_deterministic_block_majority_vote() -> None:
    labels = np.asarray([0] * 50 + [1] * 50)
    predictions = np.asarray([0] * 50 + [1] * 50)
    metrics, _ = deterministic_block_majority_vote(labels, predictions, ["a"] * 50 + ["b"] * 50)
    assert metrics["accuracy"] == 1


def test_deterministic_tie_breaking() -> None:
    labels = np.asarray([0] * 50)
    predictions = np.asarray([2] * 25 + [1] * 25)
    _, rows = deterministic_block_majority_vote(labels, predictions, ["a"] * 50)
    assert rows[0]["predicted_label"] == 1


def test_block_macro_f1() -> None:
    labels = np.repeat(np.arange(7), 50)
    predictions = labels.copy()
    conditions = [f"b{i}" for i in range(7) for _ in range(50)]
    metrics, _ = deterministic_block_majority_vote(labels, predictions, conditions)
    assert metrics["macro_f1"] == 1


def test_block_accuracy() -> None:
    labels = np.repeat(np.arange(7), 50)
    predictions = labels.copy()
    metrics, _ = deterministic_block_majority_vote(labels, predictions, [f"b{i}" for i in range(7) for _ in range(50)])
    assert metrics["accuracy"] == 1


def test_paired_irm_minus_erm_contrast() -> None:
    labels = np.repeat(np.arange(7), 9)
    erm = np.zeros((5, 63), dtype=int)
    irm = np.tile(labels, (5, 1))
    result = paired_tagid_stratified_block_bootstrap(labels, erm, irm, resamples=100)
    assert result["point_estimate"] > 0


def test_tagid_stratified_bootstrap() -> None:
    labels = np.repeat(np.arange(7), 9)
    values = np.tile(labels, (5, 1))
    result = paired_tagid_stratified_block_bootstrap(labels, values, values, resamples=50)
    assert result["inferential_unit"] == "TagID_stratified_condition_block"


def test_seed_resampling_sensitivity() -> None:
    labels = np.repeat(np.arange(7), 9)
    values = np.tile(labels, (5, 1))
    result = paired_tagid_stratified_block_bootstrap(labels, values, values, resamples=50, resample_seeds=True)
    assert result["resample_training_seeds"]


def test_no_row_level_pseudoreplication() -> None:
    labels = np.repeat(np.arange(7), 9)
    values = np.tile(labels, (5, 1))
    result = paired_tagid_stratified_block_bootstrap(labels, values, values, resamples=20)
    assert not result["row_level_pseudoreplication"]


def test_p4_deny_before_freeze(tmp_path) -> None:
    seal = P4LabelSeal(tmp_path)
    with pytest.raises(IRMProtocolError, match="FAIL_P4_LABEL_BOUNDARY"):
        seal.open_labels(execution_record(FINAL_P4_EVALUATION))


def test_prediction_freezing_and_hashing(tmp_path) -> None:
    seal = P4LabelSeal(tmp_path)
    record = seal.freeze_predictions({"m": np.zeros(3150, dtype=int)}, {"m": "abc"}, tmp_path / "p.npz", execution_record(FINAL_P4_EVALUATION))
    assert record["prediction_bundle_sha256"] and record["prediction_hashes"]["m"]


def test_checkpoint_prediction_binding(tmp_path) -> None:
    seal = P4LabelSeal(tmp_path)
    record = seal.freeze_predictions({"m": np.zeros(3150, dtype=int)}, {"m": "abc"}, tmp_path / "p.npz", execution_record(FINAL_P4_EVALUATION))
    assert record["checkpoint_prediction_bindings"]["m"]


def test_final_writer_atomic_persistence(tmp_path) -> None:
    seal = P4LabelSeal(tmp_path)
    seal._labels_opened = True
    seal.persist_metrics_once(tmp_path / "metrics.json", {"ok": True}, execution_record(FINAL_P4_EVALUATION))
    assert (tmp_path / "metrics.json").is_file() and not (tmp_path / "metrics.json.tmp").exists()


def test_simulated_persistence_failure(tmp_path) -> None:
    seal = P4LabelSeal(tmp_path)
    seal._labels_opened = True
    with pytest.raises(IRMProtocolError, match="FAIL_P4_FINAL_STAGE_PERSISTENCE"):
        seal.persist_metrics_once(tmp_path / "metrics.json", {"ok": True}, execution_record(FINAL_P4_EVALUATION), fail_before_replace=True)


def test_silent_post_label_retry_rejected(tmp_path) -> None:
    seal = P4LabelSeal(tmp_path)
    seal._labels_opened = True
    with pytest.raises(IRMProtocolError):
        seal.persist_metrics_once(tmp_path / "metrics.json", {}, execution_record(FINAL_P4_EVALUATION), fail_before_replace=True)
    with pytest.raises(IRMProtocolError, match="silent post-label retry rejected"):
        seal.persist_metrics_once(tmp_path / "metrics.json", {}, execution_record(FINAL_P4_EVALUATION))


def test_invariance_classification_improved() -> None:
    assert classify_invariance_diagnostics((-0.2, -0.1), (-0.1, -0.01), True) == "SOURCE_INVARIANCE_DIAGNOSTICS_IMPROVED"


def test_invariance_classification_tradeoff() -> None:
    assert classify_invariance_diagnostics((-0.2, -0.1), (-0.1, -0.01), False) == "SOURCE_INVARIANCE_DIAGNOSTICS_IMPROVED_WITH_TAGID_TRADEOFF"


def test_main_classification_source_and_p4() -> None:
    value = classify_main_result(source_mean_interval=(0.01, 0.1), source_worst_interval=(-0.1, 0.1), p4_interval=(0.01, 0.1), source_guardrail_passed=True, diagnostic_classification="SOURCE_INVARIANCE_DIAGNOSTICS_PARTIALLY_IMPROVED", unstable=False)
    assert value == "IRM_SOURCE_ROBUSTNESS_AND_P4_BENEFIT_CONFIRMED"


def test_main_classification_harms_p4() -> None:
    value = classify_main_result(source_mean_interval=(-0.1, 0.1), source_worst_interval=(-0.1, 0.1), p4_interval=(-0.2, -0.01), source_guardrail_passed=True, diagnostic_classification="SOURCE_INVARIANCE_DIAGNOSTICS_NOT_IMPROVED", unstable=False)
    assert value == "IRM_HARMS_P4_TRANSFER"
