from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from crfid.irm_invariant_risk.core import (
    FINAL_P4_EVALUATION,
    FINAL_SOURCE_TRAINING,
    LAMBDA_GRID,
    SOURCE_LOPO_DEVELOPMENT,
    IRMProtocolError,
    assert_penalty_connectivity,
    canonical_position_order,
    early_stop_allowed,
    execution_record,
    irmv1_objective,
    make_environment_balanced_batches,
    matched_erm_objective,
    source_retention_guardrail,
    validate_condition_block_disjoint,
    validate_execution_record,
    validate_frozen_configuration,
    validate_lopo_position_isolation,
    validate_source_fold_record,
)


ROOT = Path(__file__).resolve().parents[1]


def _terms(lambda_value: float = 10.0, step: int = 10, anneal: int = 5):
    torch.manual_seed(2)
    encoder = torch.nn.Linear(3, 4)
    classifier = torch.nn.Linear(4, 2)
    inputs = torch.tensor([[1.0, 0.0, 1.0], [-1.0, 0.0, -1.0], [1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]])
    labels = torch.tensor([1, 0, 1, 0])
    positions = np.asarray(["P1", "P1", "P2", "P2"])
    terms = irmv1_objective(classifier(encoder(inputs)), labels, positions, ("P1", "P2"), configured_lambda=lambda_value, optimizer_step=step, anneal_step=anneal)
    return encoder, classifier, terms, labels, positions


def test_canonical_c1_lineage_config_binding() -> None:
    config = json.loads((ROOT / "configs" / "irm_invariant_risk" / "canonical.json").read_text())
    assert config["canonical_candidate"] == "C1_FIRST_DIFFERENCE_ERM_1DCNN"
    assert config["model"]["parameter_count"] == 142855


def test_first_difference_dimensionality() -> None:
    assert np.diff(np.zeros((3, 281)), axis=1).shape == (3, 280)


def test_environment_mapping() -> None:
    assert canonical_position_order(["P3", "P1", "P2"]) == ("P1", "P2", "P3")


def test_environment_mapping_rejects_p4() -> None:
    with pytest.raises(IRMProtocolError):
        canonical_position_order(["P1", "P4"])


def test_source_lopo_stage_schema() -> None:
    assert execution_record(SOURCE_LOPO_DEVELOPMENT, "S1")["held_source_position"] == "P1"


def test_final_source_stage_schema() -> None:
    assert execution_record(FINAL_SOURCE_TRAINING)["target_position"] is None


def test_final_p4_stage_schema() -> None:
    assert execution_record(FINAL_P4_EVALUATION)["target_position"] == "P4"


def test_source_fold_p4_stage_separation() -> None:
    with pytest.raises(IRMProtocolError, match="FAIL_EXECUTION_STAGE_SCHEMA_DEFECT"):
        validate_source_fold_record(execution_record(FINAL_P4_EVALUATION))


def test_stage_ambiguity_rejected() -> None:
    record = execution_record(FINAL_P4_EVALUATION)
    record["development_fold"] = "S1"
    with pytest.raises(IRMProtocolError, match="FAIL_EXECUTION_STAGE_SCHEMA_DEFECT"):
        validate_execution_record(record)


def test_scalar_scale_constructed_exactly_one() -> None:
    assert float(_terms()[2].scale.detach()) == 1.0


def test_scale_gradient_create_graph_active() -> None:
    assert _terms()[2].penalty.requires_grad


def test_penalty_nonnegative_and_finite() -> None:
    penalty = _terms()[2].penalty
    assert torch.isfinite(penalty) and float(penalty.detach()) >= 0


def test_penalty_encoder_connectivity() -> None:
    encoder, classifier, terms, _, _ = _terms()
    encoder_gradient, _ = assert_penalty_connectivity(terms.penalty, encoder.weight, classifier.weight)
    assert encoder_gradient.norm() > 0


def test_penalty_classifier_connectivity() -> None:
    encoder, classifier, terms, _, _ = _terms()
    _, classifier_gradient = assert_penalty_connectivity(terms.penalty, encoder.weight, classifier.weight)
    assert classifier_gradient.norm() > 0


def test_create_graph_disabled_is_detected() -> None:
    encoder, classifier, _, labels, positions = _terms()
    logits = classifier(encoder(torch.randn(4, 3)))
    terms = irmv1_objective(logits, labels, positions, ("P1", "P2"), configured_lambda=1.0, optimizer_step=1, anneal_step=0, create_graph=False)
    with pytest.raises(IRMProtocolError):
        assert_penalty_connectivity(terms.penalty, encoder.weight, classifier.weight)


def test_detached_logits_are_detected() -> None:
    encoder, classifier, _, labels, positions = _terms()
    logits = classifier(encoder(torch.randn(4, 3)))
    terms = irmv1_objective(logits, labels, positions, ("P1", "P2"), configured_lambda=1.0, optimizer_step=1, anneal_step=0, detach_logits=True)
    with pytest.raises(IRMProtocolError):
        assert_penalty_connectivity(terms.penalty, encoder.weight, classifier.weight)


def test_environment_permutation_invariance() -> None:
    encoder, classifier, _, labels, positions = _terms()
    logits = classifier(encoder(torch.randn(4, 3)))
    left = irmv1_objective(logits, labels, positions, ("P1", "P2"), configured_lambda=10.0, optimizer_step=1, anneal_step=0)
    right = irmv1_objective(logits, labels, positions, ("P2", "P1"), configured_lambda=10.0, optimizer_step=1, anneal_step=0)
    assert torch.allclose(left.objective, right.objective)


def test_pre_anneal_matches_environment_mean_erm() -> None:
    encoder, classifier, terms, labels, positions = _terms(step=4, anneal=5)
    logits = classifier(encoder(torch.randn(4, 3)))
    irm = irmv1_objective(logits, labels, positions, ("P1", "P2"), configured_lambda=10.0, optimizer_step=4, anneal_step=5)
    erm, _ = matched_erm_objective(logits, labels, positions, ("P1", "P2"))
    assert torch.equal(irm.objective, erm)


def test_exact_anneal_boundary() -> None:
    assert _terms(step=4, anneal=5)[2].lambda_effective == 0
    assert _terms(step=5, anneal=5)[2].lambda_effective == 10


def test_post_anneal_formula() -> None:
    terms = _terms(lambda_value=100.0, step=5, anneal=5)[2]
    assert torch.allclose(terms.objective, (terms.mean_risk + 100 * terms.penalty) / 101)


def test_frozen_lambda_grid() -> None:
    assert LAMBDA_GRID == (1.0, 10.0, 100.0, 1000.0)


def test_frozen_configuration_validator() -> None:
    config = json.loads((ROOT / "configs" / "irm_invariant_risk" / "canonical.json").read_text())
    validate_frozen_configuration(config)


def test_condition_block_disjointness() -> None:
    validate_condition_block_disjoint(["a"], ["b"])
    with pytest.raises(IRMProtocolError):
        validate_condition_block_disjoint(["a"], ["a"])


def test_held_position_isolation() -> None:
    validate_lopo_position_isolation(["P2", "P3"], ["P1"], "P1")
    with pytest.raises(IRMProtocolError):
        validate_lopo_position_isolation(["P1", "P2"], ["P1"], "P1")


def test_source_retention_guardrail() -> None:
    assert source_retention_guardrail(0.48, 0.50)["passed"]
    assert not source_retention_guardrail(0.479, 0.50)["passed"]


def test_environment_balanced_batching() -> None:
    positions = np.asarray(["P1"] * 28 + ["P2"] * 28)
    labels = np.tile(np.arange(7), 8)
    plan = make_environment_balanced_batches(positions, labels, seed=42, epoch=1, stage="test", batch_size=16)
    assert all(abs(row["environment_counts"]["P1"] - row["environment_counts"]["P2"]) <= 1 for row in plan.audits)


def test_tagid_balance_where_feasible() -> None:
    positions = np.asarray(["P1"] * 28 + ["P2"] * 28)
    labels = np.tile(np.arange(7), 8)
    plan = make_environment_balanced_batches(positions, labels, seed=42, epoch=1, stage="test", batch_size=28)
    for audit in plan.audits:
        for counts in audit["tagid_counts_per_environment"].values():
            observed = [value for value in counts.values() if value]
            assert max(observed) - min(observed) <= 1


def test_irm_cannot_stop_before_anneal_intervention() -> None:
    assert not early_stop_allowed(method="irm", epochs_without_improvement=8, patience=8, completed_optimizer_steps=212, anneal_step=212)
    assert early_stop_allowed(method="irm", epochs_without_improvement=8, patience=8, completed_optimizer_steps=213, anneal_step=212)


def test_erm_keeps_canonical_patience_rule() -> None:
    assert early_stop_allowed(method="erm", epochs_without_improvement=8, patience=8, completed_optimizer_steps=10, anneal_step=212)
