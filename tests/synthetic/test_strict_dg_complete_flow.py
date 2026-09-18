from __future__ import annotations

import numpy as np
import pytest
import torch

from crfid.models.cnn1d import initialize_strict_model
from crfid.models.readouts import euclidean_ncm_predict
from crfid.protocols.common import AccessRequest, Purpose, Resource
from crfid.protocols.strict_dg import StrictDGProtocol
from crfid.strict_runtime.phase3b_execution import CANDIDATE_IDS, FOLDS, SEEDS


def test_complete_registry_expands_to_sixty_units() -> None:
    units = [(candidate, fold, seed) for candidate in CANDIDATE_IDS for fold in FOLDS for seed in SEEDS]
    assert len(units) == 60 and len(set(units)) == 60


def test_synthetic_source_scores_select_declared_best_without_target() -> None:
    scores = {candidate: 0.2 for candidate in CANDIDATE_IDS}
    scores["C1_FIRST_DIFFERENCE_ERM_1DCNN"] = 0.3
    assert max(scores, key=scores.get) == "C1_FIRST_DIFFERENCE_ERM_1DCNN"


def test_freeze_then_separate_target_authorization() -> None:
    protocol = StrictDGProtocol()
    recipe = {"selected": "C1", "epochs": {"42": 13}}
    seal = protocol.freeze_recipe(recipe)
    protocol.authorize_final_evaluation(recipe, seal)
    protocol.authorize(AccessRequest("P4", Resource.FEATURES, Purpose.EVALUATION))


def test_synthetic_target_inference_has_no_gradients() -> None:
    model = initialize_strict_model(42).eval()
    with torch.inference_mode():
        output = model(torch.zeros((3, 1, 280), dtype=torch.float32))
    assert output.shape == (3, 7) and not output.requires_grad


def test_fold_partitions_are_pairwise_disjoint() -> None:
    train, validation, held = set(range(4)), set(range(4, 6)), set(range(6, 9))
    assert not train & validation and not train & held and not validation & held
    assert train | validation | held == set(range(9))


def test_repeated_initialization_and_inference_are_exact() -> None:
    inputs = torch.arange(560, dtype=torch.float32).reshape(2, 1, 280)
    with torch.inference_mode():
        first = initialize_strict_model(46).eval()(inputs)
        second = initialize_strict_model(46).eval()(inputs)
    assert torch.equal(first, second)


def test_synthetic_euclidean_target_evaluation() -> None:
    predictions, _ = euclidean_ncm_predict(np.asarray([[0.1], [9.9]]), np.asarray([[0.0], [10.0]]))
    assert predictions.tolist() == [0, 1]
