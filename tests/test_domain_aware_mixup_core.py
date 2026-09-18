from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from crfid.domain_aware_mixup.core import (
    ALPHA_GRID,
    BETA,
    FINAL_P4_EVALUATION,
    FINAL_SOURCE_TRAINING,
    SOURCE_LOPO_DEVELOPMENT,
    MixupProtocolError,
    assert_mixup_connectivity,
    classify_domain_bridging_diagnostics,
    classify_main_result,
    execution_record,
    make_condition_matched_pairs,
    make_environment_balanced_batches,
    mixup_objective,
    sample_mix_coefficients,
    select_alpha_source_only,
    source_retention_guardrail,
    validate_execution_record,
    validate_frozen_configuration,
    validate_source_fold_record,
)
from crfid.domain_aware_mixup.runtime import load_config


def metadata():
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    er = np.zeros(8, dtype=np.int64)
    surfaces = np.asarray(["A1"] * 8)
    positions = np.asarray(["P1", "P1", "P2", "P2"] * 2)
    repeats = np.asarray([0, 1, 0, 1] * 2)
    roles = np.asarray(["training"] * 8)
    return labels, er, surfaces, positions, repeats, roles


@pytest.mark.parametrize(
    "stage,fold",
    [(SOURCE_LOPO_DEVELOPMENT, "S1"), (FINAL_SOURCE_TRAINING, None), (FINAL_P4_EVALUATION, None)],
)
def test_execution_stage_schema_valid(stage, fold):
    validate_execution_record(execution_record(stage, fold))


@pytest.mark.parametrize("field,value", [("target_position", "P4"), ("held_source_position", "P2"), ("training_positions", "P1"), ("development_fold", "S2")])
def test_source_stage_schema_rejects_ambiguity(field, value):
    record = execution_record(SOURCE_LOPO_DEVELOPMENT, "S1")
    record[field] = value
    with pytest.raises(MixupProtocolError, match="FAIL_EXECUTION_STAGE_SCHEMA_DEFECT"):
        validate_execution_record(record)


def test_p4_record_rejected_by_source_validator():
    with pytest.raises(MixupProtocolError):
        validate_source_fold_record(execution_record(FINAL_P4_EVALUATION))


def test_frozen_alpha_grid():
    assert ALPHA_GRID == (0.2, 0.5, 1.0)


def test_frozen_beta():
    assert BETA == 1.0


@pytest.mark.parametrize("key,value", [("alpha_grid", [0.2, 0.5]), ("within_position_alpha", 1.0), ("beta", 0.5), ("location", "input")])
def test_frozen_config_mutations_rejected(key, value):
    config = copy.deepcopy(load_config())
    config["mixup"][key] = value
    with pytest.raises(MixupProtocolError):
        validate_frozen_configuration(config)


def test_cross_pair_same_tagid():
    values = metadata()
    plan = make_condition_matched_pairs(*values, mode="cross_position", seed=1, epoch=1)
    assert np.all(values[0][plan.parent_indices] == values[0][plan.partner_indices])


def test_cross_pair_same_er():
    values = metadata()
    plan = make_condition_matched_pairs(*values, mode="cross_position", seed=1, epoch=1)
    assert np.all(values[1][plan.parent_indices] == values[1][plan.partner_indices])


def test_cross_pair_same_surface():
    values = metadata()
    plan = make_condition_matched_pairs(*values, mode="cross_position", seed=1, epoch=1)
    assert np.all(values[2][plan.parent_indices] == values[2][plan.partner_indices])


def test_cross_pair_different_position():
    values = metadata()
    plan = make_condition_matched_pairs(*values, mode="cross_position", seed=1, epoch=1)
    assert np.all(values[3][plan.parent_indices] != values[3][plan.partner_indices])


def test_within_pair_same_position():
    values = metadata()
    plan = make_condition_matched_pairs(*values, mode="within_position", seed=1, epoch=1)
    assert np.all(values[3][plan.parent_indices] == values[3][plan.partner_indices])


def test_within_pair_different_repeat():
    values = metadata()
    plan = make_condition_matched_pairs(*values, mode="within_position", seed=1, epoch=1)
    assert np.all(values[4][plan.parent_indices] != values[4][plan.partner_indices])


def test_pairing_is_deterministic():
    values = metadata()
    left = make_condition_matched_pairs(*values, mode="cross_position", seed=7, epoch=2)
    right = make_condition_matched_pairs(*values, mode="cross_position", seed=7, epoch=2)
    assert left.signature_sha256 == right.signature_sha256


def test_pairing_coverage_calculation():
    plan = make_condition_matched_pairs(*metadata(), mode="cross_position", seed=7, epoch=2)
    assert plan.coverage == 1.0 and plan.missing_count == 0


@pytest.mark.parametrize("role", ["validation", "held_source", "P4"])
def test_nontraining_parent_rejected(role):
    values = list(metadata())
    values[-1] = values[-1].copy()
    values[-1][0] = role
    with pytest.raises(MixupProtocolError, match="training partition"):
        make_condition_matched_pairs(*values, mode="cross_position", seed=1, epoch=1, parent_indices=[0])


def test_missing_pair_threshold_rejected():
    labels, er, surfaces, positions, repeats, roles = metadata()
    with pytest.raises(MixupProtocolError, match="FAIL_DOMAIN_AWARE_PAIRING_COVERAGE"):
        make_condition_matched_pairs(labels[:2], er[:2], surfaces[:2], positions[:2], repeats[:2], roles[:2], mode="cross_position", seed=1, epoch=1)


@pytest.mark.parametrize("alpha", ALPHA_GRID)
def test_beta_coefficient_generation(alpha):
    values = sample_mix_coefficients(alpha, 1000, seed=3, epoch=4, batch_index=5)
    assert values.shape == (1000,) and values.dtype == np.float32


@pytest.mark.parametrize("alpha", ALPHA_GRID)
def test_symmetric_coefficient_bounds(alpha):
    values = sample_mix_coefficients(alpha, 1000, seed=3, epoch=4, batch_index=5)
    assert np.all(values >= 0.5) and np.all(values <= 1.0)


def test_alpha_changes_strength_not_eligibility():
    plan = make_condition_matched_pairs(*metadata(), mode="cross_position", seed=2, epoch=3)
    assert all(len(sample_mix_coefficients(alpha, plan.requested_count, seed=2, epoch=3, batch_index=0)) == plan.requested_count for alpha in ALPHA_GRID)


def test_embedding_interpolation_and_gradients():
    torch.manual_seed(2)
    encoder = torch.nn.Linear(3, 4, bias=False)
    head = torch.nn.Linear(4, 2, bias=False)
    x = torch.randn(6, 3)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    z_a = encoder(x)
    z_b = encoder(x.flip(0))
    terms = mixup_objective(head(z_a), labels, z_a, z_b, labels, torch.full((6,), 0.75), head)
    gradients = assert_mixup_connectivity(terms.mixup_loss, z_a, z_b, encoder.weight, head.weight)
    assert terms.mixed_embeddings.shape == z_a.shape
    assert all(value is not None and torch.isfinite(value).all() and value.norm() > 0 for value in gradients)


def test_original_and_mixup_losses_active():
    head = torch.nn.Linear(2, 2)
    z_a = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], requires_grad=True)
    z_b = torch.tensor([[0.5, 1.0], [-0.5, -1.0]], requires_grad=True)
    labels = torch.tensor([1, 0])
    terms = mixup_objective(head(z_a), labels, z_a, z_b, labels, torch.tensor([0.5, 0.75]), head)
    assert terms.original_loss > 0 and terms.mixup_loss > 0


def test_beta_normalization_formula():
    head = torch.nn.Linear(2, 2)
    z = torch.randn(4, 2, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1])
    terms = mixup_objective(head(z), labels, z, z.flip(0), labels, torch.full((4,), 0.5), head)
    assert torch.allclose(terms.objective, (terms.original_loss + terms.mixup_loss) / 2)


def test_environment_balanced_batching():
    positions = np.asarray(["P1"] * 21 + ["P2"] * 19)
    labels = np.asarray(list(range(7)) * 5 + [0, 1, 2, 3, 4])[:40]
    plan = make_environment_balanced_batches(positions, labels, seed=1, epoch=1, stage="test", batch_size=16)
    for audit in plan.audits:
        counts = list(audit["environment_counts"].values())
        assert max(counts) - min(counts) <= 1 and min(counts) > 0


def test_environment_batch_seed_reproducibility():
    positions = np.asarray(["P1"] * 14 + ["P2"] * 14)
    labels = np.asarray(list(range(7)) * 4)
    a = make_environment_balanced_batches(positions, labels, seed=1, epoch=1, stage="test", batch_size=14)
    b = make_environment_balanced_batches(positions, labels, seed=1, epoch=1, stage="test", batch_size=14)
    assert a.signature_sha256 == b.signature_sha256


def alpha_rows():
    rows = []
    values = {0.2: [0.60, 0.59, 0.61], 0.5: [0.605, 0.605, 0.605], 1.0: [0.58, 0.62, 0.61]}
    for alpha, fold_values in values.items():
        for fold_index, position in enumerate(("P1", "P2", "P3")):
            for seed in (42, 43, 44, 45, 46):
                rows.append({"method": "cross_position_mixup", "alpha": alpha, "held_position": position, "held_macro_f1": fold_values[fold_index], "held_accuracy": fold_values[fold_index], "pair_coverage": 1.0})
    return rows


def test_source_only_alpha_selection():
    selected = select_alpha_source_only(alpha_rows())
    assert selected["selected_alpha"] == 0.5 and not selected["p4_accessed"]


@pytest.mark.parametrize("cross,erm,passed", [(0.6, 0.6, True), (0.58, 0.6, True), (0.579, 0.6, False)])
def test_source_retention_guardrail(cross, erm, passed):
    assert source_retention_guardrail(cross, erm)["passed"] is passed


@pytest.mark.parametrize(
    "distance,position,retention,control,expected",
    [
        ((-2, -1), (-2, -1), True, (-2, -1), "CROSS_POSITION_GEOMETRY_IMPROVED_WITH_TAGID_RETENTION"),
        ((-2, -1), (-2, -1), True, (-1, 1), "GENERIC_MIXUP_REGULARISATION_ONLY"),
        ((-2, -1), (-1, 1), True, (-1, 1), "GENERIC_MIXUP_REGULARISATION_ONLY"),
        ((-1, 1), (-1, 1), True, (-1, 1), "CROSS_POSITION_GEOMETRY_NOT_IMPROVED"),
        ((-2, -1), (-2, -1), False, (-2, -1), "CROSS_POSITION_MIXUP_HARMS_TAGID_REPRESENTATION"),
    ],
)
def test_diagnostic_classification(distance, position, retention, control, expected):
    assert classify_domain_bridging_diagnostics(cross_position_distance_interval=distance, position_decodability_interval=position, tagid_retention_passed=retention, cross_vs_within_interval=control) == expected


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"source_mean_interval": (0.01, 0.02), "source_worst_interval": (-.01, .01), "p4_cross_vs_erm_interval": (.01, .02), "p4_cross_vs_within_interval": (.01, .02), "p4_within_vs_erm_interval": (-.01, .01)}, "DOMAIN_AWARE_MIXUP_SOURCE_AND_P4_BENEFIT_CONFIRMED"),
        ({"source_mean_interval": (-.01, .01), "source_worst_interval": (-.01, .01), "p4_cross_vs_erm_interval": (.01, .02), "p4_cross_vs_within_interval": (.01, .02), "p4_within_vs_erm_interval": (-.01, .01)}, "DOMAIN_AWARE_MIXUP_P4_BENEFIT_CONFIRMED"),
        ({"source_mean_interval": (.01, .02), "source_worst_interval": (-.01, .01), "p4_cross_vs_erm_interval": (-.01, .01), "p4_cross_vs_within_interval": (-.01, .01), "p4_within_vs_erm_interval": (-.01, .01)}, "DOMAIN_AWARE_MIXUP_SOURCE_BENEFIT_WITHOUT_P4_BENEFIT"),
        ({"source_mean_interval": (-.01, .01), "source_worst_interval": (-.01, .01), "p4_cross_vs_erm_interval": (.01, .02), "p4_cross_vs_within_interval": (-.01, .01), "p4_within_vs_erm_interval": (.01, .02)}, "GENERIC_MIXUP_BENEFIT_NOT_DOMAIN_SPECIFIC"),
        ({"source_mean_interval": (-.01, .01), "source_worst_interval": (-.01, .01), "p4_cross_vs_erm_interval": (-.02, -.01), "p4_cross_vs_within_interval": (-.01, .01), "p4_within_vs_erm_interval": (-.01, .01)}, "DOMAIN_AWARE_MIXUP_HARMS_P4_TRANSFER"),
    ],
)
def test_main_classification(kwargs, expected):
    defaults = {"source_guardrail_passed": True, "diagnostic_classification": "CROSS_POSITION_GEOMETRY_NOT_IMPROVED", "unstable": False}
    assert classify_main_result(**kwargs, **defaults) == expected
