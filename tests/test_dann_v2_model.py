from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows" / "19_dann_v2"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(WORKFLOW))

for _workflow_module in ("data", "metrics", "model", "protocol", "target", "training"):
    sys.modules.pop(_workflow_module, None)

from data import balanced_epoch_indices, batch_composition, domain_labels, fit_first_difference
from model import C1PositionDANN, gradient_forward_matched, gradient_reverse
from protocol import LAMBDAS, registered_lambda_trajectory, scheduled_lambda, validate_lambda_grid
from training import intervention_validity, schedule_trajectory_for_entry_point


def test_canonical_c1_lineage_and_first_difference() -> None:
    model = C1PositionDANN(seed=42, domain_count=None)
    assert model.encoder[-1].__class__.__name__ == "Flatten"
    assert (model.tag_head.in_features, model.tag_head.out_features) == (256, 7)
    signals = np.arange(8 * 281, dtype=np.float64).reshape(8, 281)
    assert fit_first_difference(signals, np.arange(4)).mean.shape == (280,)


def test_positive_and_negative_gradient_operators_are_forward_identity() -> None:
    positive = torch.tensor([1.0, -2.0], requires_grad=True)
    negative = positive.detach().clone().requires_grad_(True)
    assert torch.equal(gradient_forward_matched(positive, 0.3), positive)
    assert torch.equal(gradient_reverse(negative, 0.3), negative)
    gradient_forward_matched(positive, 0.3).sum().backward()
    gradient_reverse(negative, 0.3).sum().backward()
    assert torch.allclose(positive.grad, torch.full_like(positive, 0.3))
    assert torch.allclose(negative.grad, torch.full_like(negative, -0.3))


def test_complete_dann_v2_implementation_validity_gate() -> None:
    result = intervention_validity()
    assert result["passed"]
    assert result["grl_forward_identity"]
    assert result["a1_a2_encoder_domain_gradient_sign_match"]
    assert result["a1_a2_encoder_domain_gradient_magnitude_match"]
    assert result["a1_a2_domain_head_gradients_identical"]
    assert result["a1_a2_tagid_gradient_contribution_identical"]
    assert result["lambda_zero_a1_encoder_update_matches_a0"]
    assert result["lambda_zero_a2_encoder_update_matches_a0"]
    assert result["development_final_schedule_trajectory_identical"]
    assert result["fold_local_domain_labels_correct"]
    assert not result["domain_loss_can_access_p4"]


@pytest.mark.parametrize("domain_count", [2, 3])
def test_domain_head_shape_and_parameter_count(domain_count: int) -> None:
    model = C1PositionDANN(seed=42, domain_count=domain_count)
    _, domain, embeddings = model.forward_domain_treatment(
        torch.randn(3, 1, 280), encoder_domain_coefficient=-0.1
    )
    assert embeddings.shape == (3, 256)
    assert domain.shape == (3, domain_count)
    expected = 256 * 128 + 128 + 128 * domain_count + domain_count
    assert sum(parameter.numel() for parameter in model.domain_head.parameters()) == expected


def test_domain_labels_are_fold_local_source_positions_only() -> None:
    labels, order = domain_labels(np.asarray(["P2", "P3", "P2"]))
    assert order == ("P2", "P3")
    assert labels.tolist() == [0, 1, 0]
    with pytest.raises(RuntimeError):
        domain_labels(np.asarray(["P2", "P4"]))


def test_balanced_batches_retain_every_row_and_every_domain() -> None:
    positions = np.repeat(np.asarray(["P1", "P2"]), 7 * 20)
    labels = np.tile(np.repeat(np.arange(7), 20), 2)
    order = balanced_epoch_indices(positions, labels, seed=42, epoch=1)
    assert set(order.tolist()) == set(range(len(labels)))
    for row in batch_composition(order, positions, labels):
        assert set(row["position_counts"]) == {"P1", "P2"}
        assert row["max_minus_min_cell_count"] <= 1


def test_frozen_grid_and_single_registered_schedule() -> None:
    assert validate_lambda_grid(LAMBDAS) == LAMBDAS
    assert scheduled_lambda(0.3, 0, 17) == 0.0
    trajectory = registered_lambda_trajectory(
        0.3, executed_epochs=7, registered_steps_per_epoch=17
    )
    assert len(trajectory) == 119
    development = schedule_trajectory_for_entry_point(
        "development", lambda_max=0.3, executed_epochs=7, registered_steps_per_epoch=17
    )
    final = schedule_trajectory_for_entry_point(
        "final", lambda_max=0.3, executed_epochs=7, registered_steps_per_epoch=17
    )
    assert development == final == trajectory
    assert trajectory[-1] < 0.3
    with pytest.raises(RuntimeError):
        validate_lambda_grid((0.0, 0.1))

