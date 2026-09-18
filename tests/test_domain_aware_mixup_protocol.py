from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from crfid.domain_aware_mixup.core import (
    FINAL_P4_EVALUATION,
    SOURCE_LOPO_DEVELOPMENT,
    MixupProtocolError,
    P4LabelSeal,
    deterministic_block_majority_vote,
    execution_record,
    paired_source_bootstrap,
    paired_tagid_stratified_block_bootstrap,
    validate_condition_block_disjoint,
    validate_lopo_position_isolation,
)
from crfid.domain_aware_mixup.runtime import (
    CONFIG_PATH,
    PROJECT_ROOT,
    c1_lineage_binding,
    load_config,
    synthetic_p4_persistence_test,
    synthetic_validity_rows,
)


def block_case():
    labels = np.repeat(np.arange(7), 50)
    predictions = labels.copy()
    condition_ids = np.repeat([f"block_{index}" for index in range(7)], 50)
    return labels, predictions, condition_ids


def test_first_difference_dimensionality():
    assert np.diff(np.zeros((2, 281)), axis=1).shape == (2, 280)


def test_canonical_c1_lineage_binding():
    binding = c1_lineage_binding(load_config())
    assert binding["canonical_candidate"] == "C1_FIRST_DIFFERENCE_ERM_1DCNN"
    assert all(len(row["sha256"]) == 64 for row in binding["bound_files"])


def test_penultimate_embedding_binding():
    binding = c1_lineage_binding(load_config())
    assert binding["embedding_dimension"] == 256
    assert "network[:-1]" in binding["penultimate_embedding"]


def test_metadata_registry_binding_header():
    registry = PROJECT_ROOT.parent / "CRFID_research_code_v2" / "outputs" / "strict_dg" / "source_inputs" / "CANONICAL_SOURCE_REGISTRY.csv"
    if not registry.is_file():
        pytest.skip("governed source registry unavailable")
    with registry.open("r", encoding="utf-8", newline="") as handle:
        fields = next(csv.reader(handle))
    for field in ("tag_id", "er", "surface", "position", "repeat_index", "raw_condition_id"):
        assert field in fields


def test_condition_block_disjointness_passes():
    validate_condition_block_disjoint(["a", "b"], ["c", "d"])


def test_condition_block_overlap_rejected():
    with pytest.raises(MixupProtocolError):
        validate_condition_block_disjoint(["a", "b"], ["b", "c"])


@pytest.mark.parametrize("held,training", [("P1", ("P2", "P3")), ("P2", ("P1", "P3")), ("P3", ("P1", "P2"))])
def test_held_position_isolation(held, training):
    validate_lopo_position_isolation(training, [held], held)


def test_held_position_leak_rejected():
    with pytest.raises(MixupProtocolError):
        validate_lopo_position_isolation(("P1", "P2"), ["P3"], "P1")


def test_deterministic_majority_vote():
    labels, predictions, condition_ids = block_case()
    metrics, rows = deterministic_block_majority_vote(labels, predictions, condition_ids)
    assert metrics["macro_f1"] == 1.0 and len(rows) == 7


def test_deterministic_tie_breaking():
    labels = np.zeros(50, dtype=np.int64)
    predictions = np.asarray([1] * 25 + [2] * 25)
    _, rows = deterministic_block_majority_vote(labels, predictions, ["x"] * 50)
    assert rows[0]["predicted_label"] == 1


def test_block_size_custody_rejected():
    with pytest.raises(MixupProtocolError):
        deterministic_block_majority_vote(np.zeros(49), np.zeros(49), ["x"] * 49)


def test_block_accuracy_reported():
    labels, predictions, condition_ids = block_case()
    metrics, _ = deterministic_block_majority_vote(labels, predictions, condition_ids)
    assert metrics["accuracy"] == 1.0


def source_pairs():
    return [
        {"held_position": position, "seed": seed, "erm_macro_f1": .5, "irm_macro_f1": .6}
        for position in ("P1", "P2", "P3") for seed in (42, 43, 44, 45, 46)
    ]


def test_paired_source_bootstrap():
    result = paired_source_bootstrap(source_pairs(), resamples=200, seed=1)
    assert result["mean_contrast"]["point_estimate"] == pytest.approx(.1)


@pytest.mark.parametrize("resample_seeds", [False, True])
def test_tagid_stratified_bootstrap(resample_seeds):
    labels = np.repeat(np.arange(7), 9)
    left = np.zeros((5, 63), dtype=np.int64)
    right = np.tile(labels, (5, 1))
    result = paired_tagid_stratified_block_bootstrap(labels, left, right, resamples=100, seed=1, resample_seeds=resample_seeds)
    assert result["inferential_unit"] == "TagID_stratified_condition_block"
    assert result["row_level_pseudoreplication"] is False


@pytest.mark.parametrize("metric", ["macro_f1", "accuracy"])
def test_three_way_bootstrap_primitive_supports_metric(metric):
    labels = np.repeat(np.arange(7), 9)
    predictions = np.tile(labels, (5, 1))
    result = paired_tagid_stratified_block_bootstrap(labels, predictions, predictions, metric_name=metric, resamples=50, seed=3)
    assert result["point_estimate"] == 0.0


def test_p4_deny_before_final_stage(tmp_path):
    seal = P4LabelSeal(tmp_path)
    with pytest.raises(MixupProtocolError):
        seal.freeze_predictions({"x": np.zeros(3150)}, {"x": "hash"}, tmp_path / "x.npz", execution_record(SOURCE_LOPO_DEVELOPMENT, "S1"))


def test_p4_labels_sealed_before_prediction(tmp_path):
    seal = P4LabelSeal(tmp_path)
    with pytest.raises(MixupProtocolError, match="FAIL_P4_LABEL_BOUNDARY"):
        seal.open_labels(execution_record(FINAL_P4_EVALUATION))


def test_checkpoint_prediction_binding_and_hash(tmp_path):
    seal = P4LabelSeal(tmp_path)
    record = seal.freeze_predictions({"x": np.zeros(3150)}, {"x": "checkpoint"}, tmp_path / "x.npz", execution_record(FINAL_P4_EVALUATION))
    assert len(record["prediction_hashes"]["x"]) == 64
    assert len(record["checkpoint_prediction_bindings"]["x"]) == 64


def test_prediction_shape_rejected(tmp_path):
    seal = P4LabelSeal(tmp_path)
    with pytest.raises(MixupProtocolError):
        seal.freeze_predictions({"x": np.zeros(3149)}, {"x": "checkpoint"}, tmp_path / "x.npz", execution_record(FINAL_P4_EVALUATION))


def test_atomic_result_persistence_and_no_retry(tmp_path):
    result = synthetic_p4_persistence_test(tmp_path)
    assert result["exact_final_writer_succeeded"] and result["silent_retry_rejected"]


def test_synthetic_validity_suite():
    rows = synthetic_validity_rows()
    assert len(rows) >= 25 and all(row["passed"] for row in rows)


def test_config_has_no_scheduler():
    assert load_config()["training"]["scheduler"] is None


def test_config_uses_canonical_seeds():
    assert load_config()["seeds"] == [42, 43, 44, 45, 46]


def test_config_mixup_training_only():
    assert load_config()["mixup"]["training_only"] is True


def test_config_disables_label_interpolation():
    assert load_config()["mixup"]["label_interpolation"] is False


def test_config_disables_input_mixup():
    assert load_config()["mixup"]["input_space_mixup"] is False


def test_config_file_exists():
    assert CONFIG_PATH.is_file()
