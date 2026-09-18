from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from crfid.irm_invariant_risk.core import LAMBDA_GRID, IRMProtocolError, irmv1_objective
from crfid.irm_invariant_risk.runtime import atomic_write_json, synthetic_validity_rows


ROOT = Path(__file__).resolve().parents[1]


def test_validation_irm_penalty_case() -> None:
    rows = synthetic_validity_rows()
    assert any(row["test"] == "penalty_finite" and row["passed"] for row in rows)


def test_environment_risk_dispersion_case() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.1, 0.0], [0.0, 0.1]], requires_grad=True)
    terms = irmv1_objective(logits, torch.tensor([0, 1, 0, 1]), ["P1", "P1", "P2", "P2"], ("P1", "P2"), configured_lambda=1.0, optimizer_step=1, anneal_step=0)
    assert terms.environment_risks["P1"] != terms.environment_risks["P2"]


def test_independent_position_probe_contract_declared() -> None:
    source = (ROOT / "src" / "crfid" / "irm_invariant_risk" / "runtime.py").read_text()
    assert "position_probe_balanced_accuracy" in source and "condition_block_disjoint" in source


def test_independent_tagid_probe_contract_declared() -> None:
    source = (ROOT / "src" / "crfid" / "irm_invariant_risk" / "runtime.py").read_text()
    assert "tagid_probe_macro_f1" in source


def test_path_sanitisation_placeholders() -> None:
    source = (ROOT / "src" / "crfid" / "irm_invariant_risk" / "runtime.py").read_text()
    assert "<GOVERNED_DATA_ROOT>" in source and "<EXTERNAL_RUNTIME_ROOT>" in source


def test_no_mixed_precision() -> None:
    config = json.loads((ROOT / "configs" / "irm_invariant_risk" / "canonical.json").read_text())
    assert not config["irmv1"]["mixed_precision"]


def test_no_optimizer_reset_at_anneal() -> None:
    config = json.loads((ROOT / "configs" / "irm_invariant_risk" / "canonical.json").read_text())
    assert not config["irmv1"]["optimizer_reset_at_anneal"]


def test_lambda_1000_finite_gradient_case() -> None:
    logits = torch.randn(8, 2, requires_grad=True)
    labels = torch.tensor([0, 1] * 4)
    positions = ["P1"] * 4 + ["P2"] * 4
    terms = irmv1_objective(logits, labels, positions, ("P1", "P2"), configured_lambda=1000.0, optimizer_step=1, anneal_step=0)
    terms.objective.backward()
    assert torch.isfinite(logits.grad).all()


def test_lambda_outside_grid_rejected() -> None:
    with pytest.raises(IRMProtocolError):
        irmv1_objective(torch.randn(4, 2, requires_grad=True), torch.tensor([0, 1, 0, 1]), ["P1", "P1", "P2", "P2"], ("P1", "P2"), configured_lambda=3.0, optimizer_step=1, anneal_step=0)


def test_atomic_json_failure_leaves_no_final(tmp_path) -> None:
    with pytest.raises(OSError):
        atomic_write_json(tmp_path / "x.json", {"x": 1}, fail_before_replace=True)
    assert not (tmp_path / "x.json").exists()


def test_frozen_grid_exact_nonzero() -> None:
    assert all(value > 0 for value in LAMBDA_GRID) and len(LAMBDA_GRID) == 4
