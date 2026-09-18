from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows" / "14_dann_position_intervention"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(WORKFLOW))

for _workflow_module in ("data", "metrics", "model", "protocol", "target", "training"):
    sys.modules.pop(_workflow_module, None)

from data import balanced_epoch_indices, batch_composition, domain_labels, fit_first_difference
from model import C1PositionDANN, gradient_reverse
from protocol import LAMBDAS, grl_strength, validate_lambda_grid
from training import _evaluate, intervention_validity


def test_canonical_c1_lineage_binding() -> None:
    model = C1PositionDANN(seed=42, domain_count=None)
    assert model.encoder[-1].__class__.__name__ == "Flatten"
    assert model.tag_head.in_features == 256
    assert model.tag_head.out_features == 7


def test_first_difference_dimensionality() -> None:
    signals = np.arange(8 * 281, dtype=np.float64).reshape(8, 281)
    state = fit_first_difference(signals, np.arange(4))
    assert state.mean.shape == (280,)


def test_gradient_reversal_forward_identity() -> None:
    values = torch.randn(4, 5, requires_grad=True)
    assert torch.equal(gradient_reverse(values, 0.3), values)


def test_gradient_reversal_backward_sign() -> None:
    values = torch.tensor([1.0, -2.0], requires_grad=True)
    gradient_reverse(values, 0.3).sum().backward()
    assert torch.allclose(values.grad, torch.full_like(values, -0.3))


def test_gradient_reversal_changes_encoder_gradient_sign_without_double_reversal() -> None:
    result = intervention_validity()
    assert result["gradient_reversal_exact_sign"]
    assert result["no_double_reversal"]


def test_lambda_zero_erm_equivalence() -> None:
    inputs = torch.randn(8, 1, 280)
    labels = torch.arange(8) % 7
    erm = C1PositionDANN(seed=43, domain_count=None)
    dann_zero = C1PositionDANN(seed=43, domain_count=2)
    assert torch.equal(erm(inputs), dann_zero(inputs))
    left = torch.optim.AdamW(erm.parameters(), lr=0.001, weight_decay=0.0001, foreach=False, fused=False)
    right = torch.optim.AdamW(dann_zero.parameters(), lr=0.001, weight_decay=0.0001, foreach=False, fused=False)
    criterion = nn.CrossEntropyLoss()
    left.zero_grad(); criterion(erm(inputs), labels).backward(); left.step()
    right.zero_grad(); criterion(dann_zero(inputs), labels).backward(); right.step()
    assert torch.equal(erm(inputs), dann_zero(inputs))


def test_lambda_zero_difference_is_detected() -> None:
    left = C1PositionDANN(seed=42, domain_count=None)
    right = C1PositionDANN(seed=42, domain_count=2)
    with torch.no_grad():
        right.tag_head.bias.add_(1.0)
    assert not torch.equal(left(torch.randn(2, 1, 280)), right(torch.randn(2, 1, 280)))


@pytest.mark.parametrize("domain_count", [2, 3])
def test_domain_head_output_dimension(domain_count: int) -> None:
    model = C1PositionDANN(seed=42, domain_count=domain_count)
    _, domain, embeddings = model.forward_dann(torch.randn(3, 1, 280), grl_strength=0.1)
    assert embeddings.shape == (3, 256)
    assert domain.shape == (3, domain_count)


def test_two_position_lopo_domain_labels() -> None:
    labels, order = domain_labels(np.asarray(["P2", "P3", "P2"]))
    assert order == ("P2", "P3")
    assert labels.tolist() == [0, 1, 0]


def test_three_position_final_domain_labels() -> None:
    labels, order = domain_labels(np.asarray(["P3", "P1", "P2"]))
    assert order == ("P1", "P2", "P3")
    assert labels.tolist() == [2, 0, 1]


def test_balanced_source_batching_and_matched_order() -> None:
    positions = np.repeat(np.asarray(["P1", "P2"]), 7 * 20)
    labels = np.tile(np.repeat(np.arange(7), 20), 2)
    first = balanced_epoch_indices(positions, labels, seed=42, epoch=1)
    second = balanced_epoch_indices(positions, labels, seed=42, epoch=1)
    assert np.array_equal(first, second)
    assert set(first.tolist()) == set(range(len(labels)))
    assert max(row["max_minus_min_cell_count"] for row in batch_composition(first, positions, labels)) <= 1


def test_dann_lambda_grid_is_frozen() -> None:
    assert validate_lambda_grid(LAMBDAS) == LAMBDAS
    with pytest.raises(RuntimeError):
        validate_lambda_grid((0.0, 0.1))


def test_grl_schedule_endpoints() -> None:
    assert grl_strength(0.3, 0.0) == 0.0
    assert 0.2999 < grl_strength(0.3, 1.0) < 0.3


def test_learning_validity_gradients_and_domain_update() -> None:
    result = intervention_validity()
    assert result["passed"]
    assert result["tagid_encoder_gradient_nonzero"]
    assert result["domain_head_parameters_update"]
    assert not result["domain_loss_can_access_p4"]


def test_held_position_is_tagid_only_not_forced_into_training_domain_head() -> None:
    model = C1PositionDANN(seed=42, domain_count=2)
    result = _evaluate(
        model,
        np.random.default_rng(1).normal(size=(4, 280)).astype(np.float32),
        np.asarray([0, 1, 2, 3]),
        np.asarray(["P1"] * 4),
    )
    assert "domain" not in result
    assert "outside the training-domain classifier" in result["domain_evaluation_skipped"]
