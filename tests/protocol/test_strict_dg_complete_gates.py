from __future__ import annotations

import pytest

from crfid.exceptions import DomainAccessViolation, RecipeFreezeError
from crfid.protocols.common import AccessRequest, Purpose, Resource
from crfid.protocols.strict_dg import StrictDGProtocol


P4_EVAL = AccessRequest("P4", Resource.FEATURES, Purpose.EVALUATION)


def test_p4_blocked_during_source_selection() -> None:
    with pytest.raises(DomainAccessViolation):
        StrictDGProtocol().authorize(P4_EVAL)


def test_p4_blocked_during_final_source_training() -> None:
    protocol = StrictDGProtocol()
    protocol.freeze_recipe({"candidate": "C1"})
    with pytest.raises(DomainAccessViolation):
        protocol.authorize(P4_EVAL)


def test_target_authorization_blocked_before_freeze() -> None:
    protocol = StrictDGProtocol()
    with pytest.raises(RecipeFreezeError):
        protocol.authorize_final_evaluation({}, object())  # type: ignore[arg-type]


def test_altered_recipe_is_rejected() -> None:
    protocol = StrictDGProtocol()
    recipe = {"candidate": "C1"}
    seal = protocol.freeze_recipe(recipe)
    with pytest.raises(RecipeFreezeError):
        protocol.authorize_final_evaluation({"candidate": "C0"}, seal)


def test_p4_training_is_rejected_even_after_authorization() -> None:
    protocol = StrictDGProtocol()
    recipe = {"candidate": "C1"}
    seal = protocol.freeze_recipe(recipe)
    protocol.authorize_final_evaluation(recipe, seal)
    with pytest.raises(DomainAccessViolation):
        protocol.authorize(AccessRequest("P4", Resource.FEATURES, Purpose.TRAINING))


def test_p4_normalization_refit_is_rejected() -> None:
    protocol = StrictDGProtocol()
    recipe = {"candidate": "C1"}
    seal = protocol.freeze_recipe(recipe)
    protocol.authorize_final_evaluation(recipe, seal)
    with pytest.raises(DomainAccessViolation):
        protocol.authorize(AccessRequest("P4", Resource.FEATURES, Purpose.ADAPTATION))


def test_frozen_recipe_cannot_be_replaced() -> None:
    protocol = StrictDGProtocol()
    protocol.freeze_recipe({"candidate": "C1"})
    with pytest.raises(RecipeFreezeError):
        protocol.freeze_recipe({"candidate": "C0"})

