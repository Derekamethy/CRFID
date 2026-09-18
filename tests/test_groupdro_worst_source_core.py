from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from crfid.groupdro_worst_source.core import (
    ETA_GRID,
    GroupDROProtocolError,
    GroupDROState,
    canonical_position_order,
    erm_loss,
    groupdro_loss,
    make_group_balanced_batches,
    source_retention_guardrail,
    validate_condition_block_disjoint,
    validate_frozen_configuration,
    validate_lopo_position_isolation,
)


def test_first_difference_dimensionality_and_position_mapping() -> None:
    signals = np.zeros((6, 281), dtype=np.float64)
    assert np.diff(signals, axis=1).shape == (6, 280)
    assert canonical_position_order(["P3", "P1", "P2"]) == ("P1", "P2", "P3")
    with pytest.raises(GroupDROProtocolError):
        canonical_position_order(["P1", "P4"])


def test_uniform_q_exponentiated_gradient_normalization_and_stability() -> None:
    state = GroupDROState(0.10, ("P1", "P2"))
    assert torch.equal(state.q, torch.tensor([0.5, 0.5], dtype=torch.float64))
    state.update({"P1": torch.tensor(2.0), "P2": torch.tensor(0.5)})
    assert state.q[0] > state.q[1]
    assert torch.isclose(state.q.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert bool((state.q >= 0).all())
    state.update({"P1": torch.tensor(1e6), "P2": torch.tensor(0.0)})
    assert bool(torch.isfinite(state.q).all())


def test_equal_losses_preserve_q_and_eta_changes_update_speed() -> None:
    slow = GroupDROState(0.01, ("P1", "P2"))
    fast = GroupDROState(0.20, ("P1", "P2"))
    equal = {"P1": torch.tensor(1.0), "P2": torch.tensor(1.0)}
    slow.update(equal)
    assert torch.allclose(slow.q, torch.tensor([0.5, 0.5], dtype=torch.float64))
    losses = {"P1": torch.tensor(2.0), "P2": torch.tensor(0.0)}
    slow.update(losses)
    fast.update(losses)
    assert fast.q[0] - 0.5 > slow.q[0] - 0.5


def test_detached_q_updates_do_not_block_model_gradients_or_depend_on_mapping_order() -> None:
    logits = torch.tensor([[3.0, -1.0], [-1.0, 3.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    positions = np.asarray(["P1", "P2"])
    state = GroupDROState(0.05, ("P1", "P2"))
    objective, _, losses, _ = groupdro_loss(logits, labels, positions, state)
    objective.backward()
    assert not state.q.requires_grad
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())
    left = GroupDROState(0.05, ("P1", "P2"))
    right = GroupDROState(0.05, ("P1", "P2"))
    left.update(losses)
    right.update({"P2": losses["P2"], "P1": losses["P1"]})
    assert torch.allclose(left.q, right.q)


def test_erm_has_no_q_weighting() -> None:
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    loss, per_example = erm_loss(logits, torch.tensor([0, 1]))
    assert torch.isclose(loss, per_example.mean())
    loss.backward()
    assert logits.grad is not None


def test_group_balanced_batches_are_deterministic_and_tagid_interleaved() -> None:
    positions = np.asarray(["P2"] * 28 + ["P1"] * 28)
    labels = np.tile(np.arange(7, dtype=np.int64), 8)
    first = make_group_balanced_batches(positions, labels, seed=42, epoch=1, stage="inner", batch_size=16)
    second = make_group_balanced_batches(positions, labels, seed=42, epoch=1, stage="inner", batch_size=16)
    assert first.signature_sha256 == second.signature_sha256
    assert first.source_positions == ("P1", "P2")
    for audit in first.audits:
        counts = audit["position_counts"]
        assert abs(counts["P1"] - counts["P2"]) <= 1
        for group_counts in audit["tagid_counts"].values():
            observed = [count for count in group_counts.values() if count]
            assert max(observed) - min(observed) <= 1
    observed = np.concatenate(first.batches)
    assert np.array_equal(np.sort(observed), np.arange(len(positions)))
    assert first.replacement_count == 0


def test_condition_block_and_lopo_isolation_guards() -> None:
    validate_condition_block_disjoint({"a", "b"}, {"c"})
    with pytest.raises(GroupDROProtocolError):
        validate_condition_block_disjoint({"a", "b"}, {"b", "c"})
    validate_lopo_position_isolation({"P2", "P3"}, {"P1"}, "P1")
    with pytest.raises(GroupDROProtocolError):
        validate_lopo_position_isolation({"P1", "P2"}, {"P1"}, "P1")


def test_frozen_eta_grid_and_source_retention_guardrail() -> None:
    config = {
        "groupdro": {
            "eta_grid": list(ETA_GRID),
            "initial_q": "uniform",
            "update_frequency": "optimizer_step",
            "group_variable": "source_position",
            "q_momentum": None,
            "q_clipping": None,
        },
        "source_positions": ["P1", "P2", "P3"],
        "seeds": [42, 43, 44, 45, 46],
    }
    validate_frozen_configuration(json.loads(json.dumps(config)))
    config["groupdro"]["eta_grid"].append(0.3)
    with pytest.raises(GroupDROProtocolError):
        validate_frozen_configuration(config)
    assert source_retention_guardrail(0.79, 0.80)["retained"]
    assert not source_retention_guardrail(0.77, 0.80)["retained"]
