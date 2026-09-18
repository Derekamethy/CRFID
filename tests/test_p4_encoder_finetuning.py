from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflows/11_p4_factor_aware_few_shot"))
sys.path.insert(0, str(ROOT / "workflows/12_p4_large_calibration_curve"))
sys.path.insert(0, str(ROOT / "workflows/13_p4_trainable_linear_readout"))
sys.path.insert(0, str(ROOT / "workflows/14_p4_encoder_finetuning"))

from p4_factor_aware.models import FrozenC1CNN1D, model_state_sha256
from p4_encoder_ft.constants import (
    BUDGETS,
    FULL_PARAMETER_COUNT,
    PARTIAL_PARAMETER_COUNT,
    PARTIAL_TRAINABLE_NAMES,
    PREREGISTRATION_SHA256,
    SOURCE_EPISODE_KEYS,
)
from p4_encoder_ft.finetune import derive_seed, fine_tune, parameter_mask
from p4_encoder_ft.study import _block_correctness, _classify, _load_parent_predictions, _padded_predictions


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_model() -> FrozenC1CNN1D:
    torch.manual_seed(123)
    model = FrozenC1CNN1D()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def test_preregistration_hash_and_primary_budgets() -> None:
    path = ROOT / "configs/p4_encoder_finetuning/preregistration.json"
    assert _sha256(path) == PREREGISTRATION_SHA256
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert tuple(payload["data"]["primary_budgets"]) == BUDGETS
    assert payload["query_seal"]["training_access"] is False
    assert payload["stopping_rule"].startswith("After a valid result")


def test_source_episode_keys_are_position_local() -> None:
    assert SOURCE_EPISODE_KEYS == (("P1", 0), ("P1", 4), ("P1", 10), ("P2", 0), ("P2", 4), ("P2", 10), ("P3", 0), ("P3", 4), ("P3", 10))


def test_parameter_masks_match_preregistration() -> None:
    partial = _source_model()
    _, _, partial_mask = parameter_mask(partial, "PARTIAL_FT")
    assert {name for name, value in partial_mask.items() if value} == set(PARTIAL_TRAINABLE_NAMES)
    assert sum(parameter.numel() for parameter in partial.parameters() if parameter.requires_grad) == PARTIAL_PARAMETER_COUNT
    full = _source_model()
    _, _, full_mask = parameter_mask(full, "FULL_FT")
    assert all(full_mask.values())
    assert sum(parameter.numel() for parameter in full.parameters() if parameter.requires_grad) == FULL_PARAMETER_COUNT


def test_fitter_api_has_no_query_object() -> None:
    names = set(inspect.signature(fine_tune).parameters)
    assert not any("query" in name for name in names)
    assert names == {"source_model", "support_inputs", "support_labels", "initial_head_weight", "initial_head_bias", "arm", "learning_rate", "encoder_weight_decay", "epochs", "seed", "capture_epochs"}


def test_partial_finetuning_is_deterministic_and_does_not_mutate_source() -> None:
    source = _source_model()
    before = model_state_sha256(source)
    inputs = torch.linspace(-1.0, 1.0, 7 * 280, dtype=torch.float32).reshape(7, 1, 280)
    labels = np.arange(7, dtype=np.int64)
    kwargs = dict(
        source_model=source,
        support_inputs=inputs,
        support_labels=labels,
        initial_head_weight=source.network[-1].weight.detach().clone(),
        initial_head_bias=source.network[-1].bias.detach().clone(),
        arm="PARTIAL_FT",
        learning_rate=1e-4,
        encoder_weight_decay=1e-4,
        epochs=2,
        seed=derive_seed("test", 42),
    )
    first = fine_tune(**kwargs)[-1]
    second = fine_tune(**kwargs)[-1]
    assert model_state_sha256(source) == before
    assert first.diagnostics["adapted_model_sha256"] == second.diagnostics["adapted_model_sha256"]
    assert first.diagnostics["trainable_parameter_count"] == PARTIAL_PARAMETER_COUNT
    changed = set(str(first.diagnostics["changed_parameter_names"]).split("|"))
    assert changed <= set(PARTIAL_TRAINABLE_NAMES)
    assert first.diagnostics["all_objectives_finite"] is True


def test_seed_mapping_is_stable_and_arm_specific() -> None:
    first = derive_seed("PARTIAL_FT", 42, 104729, 0, 7)
    assert first == derive_seed("PARTIAL_FT", 42, 104729, 0, 7)
    assert first != derive_seed("FULL_FT", 42, 104729, 0, 7)
    assert 0 <= first < 2**32


def test_padded_predictions_preserve_preregistered_budget_indices() -> None:
    values = np.empty((4, 5, 20, 3150), dtype=np.int8)
    for index in range(4):
        values[index].fill(index)
    padded = _padded_predictions(values)
    assert padded.shape == (10, 5, 20, 3150)
    assert [int(padded[index, 0, 0, 0]) for index in (1, 5, 7, 9)] == [0, 1, 2, 3]


def test_parent_prediction_budget_subset_is_hash_bound() -> None:
    full, selected = _load_parent_predictions(ROOT)
    assert full.shape == (10, 5, 20, 3150)
    assert selected.shape == (4, 5, 20, 3150)


def test_classification_rules_follow_preregistered_order() -> None:
    rows = []
    for budget in BUDGETS:
        for comparison in ("PARTIAL_MINUS_FROZEN", "FULL_MINUS_FROZEN", "FULL_MINUS_PARTIAL"):
            rows.append({"budget": budget, "comparison": comparison, "practical_and_reliable_macro_f1": False, "reliable_positive_macro_f1": False})
    assert _classify(rows) == "ENCODER_ADAPTATION_DOES_NOT_RESCUE_HELD_CONDITION_TRANSFER"
    next(row for row in rows if row["budget"] == 100 and row["comparison"] == "FULL_MINUS_FROZEN")["practical_and_reliable_macro_f1"] = True
    assert _classify(rows) == "FULL_ENCODER_ADAPTATION_REQUIRED_FOR_P4_TRANSFER"
    next(row for row in rows if row["budget"] == 35 and row["comparison"] == "PARTIAL_MINUS_FROZEN")["practical_and_reliable_macro_f1"] = True
    assert _classify(rows) == "PARTIAL_ENCODER_ADAPTATION_RESCUES_P4_TRANSFER"


def test_block_correctness_preserves_source_support_and_row_axes() -> None:
    values = np.zeros((4, 5, 20, 12), dtype=np.int8)
    truth = np.zeros(12, dtype=np.int64)
    indices = np.asarray([1, 4, 8], dtype=np.int64)
    correct = _block_correctness(values, 2, indices, truth)
    assert correct.shape == (5, 20, 3)
    assert np.all(correct)
