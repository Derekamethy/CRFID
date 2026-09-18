from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from crfid.governance.recipe_freeze import create_recipe_seal, verify_recipe_seal
from crfid.models.readouts import (
    condition_block_euclidean_prototypes,
    euclidean_ncm_predict,
)
from crfid.preprocessing.representations import first_difference
from crfid.strict_runtime.hashing import canonical_json_sha256
from crfid.strict_runtime.neutral_model import initialize_model, model_state_sha256
from crfid.strict_runtime.phase3b_execution import CANDIDATE_IDS, coral_penalty
from crfid.strict_runtime.phase3b_summary import deterministic_ranking
from crfid.training.checkpoints import load_checkpoint, save_checkpoint


def test_candidate_registry_is_complete_and_ordered() -> None:
    assert CANDIDATE_IDS == (
        "C0_NEUTRAL_ERM_1DCNN",
        "C1_FIRST_DIFFERENCE_ERM_1DCNN",
        "C2_SOURCE_EUCLIDEAN_NCM_READOUT",
        "C3_SOURCE_CORAL_ERM_1DCNN",
    )


def test_first_difference_has_authoritative_length() -> None:
    values = first_difference(np.zeros((2, 281), dtype=np.float64))
    assert values.shape == (2, 280)


def test_coral_is_zero_for_equal_embeddings() -> None:
    values = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    assert coral_penalty(values, values).item() == 0.0


def test_coral_is_nonnegative_and_symmetric() -> None:
    left = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    right = left.square()
    assert coral_penalty(left, right).item() >= 0.0
    assert coral_penalty(left, right).item() == coral_penalty(right, left).item()


def test_condition_block_prototypes_are_equal_block_weighted() -> None:
    embeddings = np.asarray([[0.0], [2.0], [10.0], [14.0]])
    labels = np.asarray([0, 0, 1, 1])
    conditions = np.asarray(["a", "a", "b", "b"])
    prototypes = condition_block_euclidean_prototypes(embeddings, labels, conditions, (0, 1), expected_block_size=2)
    assert np.array_equal(prototypes, np.asarray([[1.0], [12.0]]))


def test_euclidean_ncm_uses_lowest_index_tie() -> None:
    predictions, scores = euclidean_ncm_predict(np.asarray([[1.0, 0.0]]), np.asarray([[0.0, 0.0], [2.0, 0.0]]))
    assert predictions.tolist() == [0]
    assert scores.dtype == np.float32


def test_selection_tie_is_candidate_identifier_ascending() -> None:
    comparisons = []
    for candidate in reversed(CANDIDATE_IDS):
        comparisons.append({
            "candidate_id": candidate,
            "worst_outer_held_position_macro_f1": 0.1,
            "mean_outer_held_position_macro_f1": 0.1,
            "worst_class_recall": 0.0,
            "zero_recall_class_frequency": 0.2,
            "condition_block_macro_f1": 0.1,
            "unique_signal_weighted_macro_f1": 0.1,
            "across_seed_standard_deviation": 0.01,
        })
    rule = {"selection_rule_sha256": "x", "ordered_selector": [
        {"metric": "worst_outer_held_position_macro_f1", "operation": "maximize"},
        {"metric": "mean_outer_held_position_macro_f1", "operation": "maximize"},
        {"metric": "worst_class_recall", "operation": "maximize"},
        {"metric": "zero_recall_class_frequency", "operation": "minimize"},
        {"metric": "condition_block_macro_f1", "operation": "maximize"},
        {"metric": "unique_signal_weighted_macro_f1", "operation": "maximize"},
        {"metric": "across_seed_standard_deviation", "operation": "minimize"},
        {"metric": "candidate_id", "operation": "ascending_deterministic"},
    ]}
    assert deterministic_ranking(comparisons, rule)["provisional_source_only_winner"] == CANDIDATE_IDS[0]


def test_recipe_hash_is_canonical_and_order_independent() -> None:
    assert canonical_json_sha256({"a": 1, "b": 2}) == canonical_json_sha256({"b": 2, "a": 1})
    seal = create_recipe_seal({"a": 1, "b": 2})
    assert verify_recipe_seal({"b": 2, "a": 1}, seal)


def test_checkpoint_save_load_and_hash_rejection(tmp_path) -> None:
    path = tmp_path / "model.pt"
    digest = save_checkpoint(path, {"weights": torch.arange(3)})
    assert torch.equal(load_checkpoint(path, digest)["weights"], torch.arange(3))
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(path, digest)


def test_model_initialization_is_repeatable_and_exact_size() -> None:
    first = initialize_model(42)
    second = initialize_model(42)
    assert model_state_sha256(first) == model_state_sha256(second)
    assert sum(parameter.numel() for parameter in first.parameters()) == 142855


def test_prediction_npz_serialization_round_trip(tmp_path) -> None:
    path = tmp_path / "predictions.npz"
    np.savez(path, labels=np.asarray([0, 1]), predictions=np.asarray([1, 1]), logits=np.zeros((2, 2), dtype=np.float32))
    with np.load(path, allow_pickle=False) as bundle:
        assert bundle["predictions"].dtype == np.int64
        assert bundle["logits"].dtype == np.float32

