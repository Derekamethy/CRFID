"""Strict source-only development and recipe-frozen target evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..exceptions import DomainAccessViolation, RecipeFreezeError
from ..governance.recipe_freeze import RecipeSeal, create_recipe_seal, verify_recipe_seal
from .common import AccessRequest, ProtocolDeclaration, Purpose


class StrictDGProtocol:
    declaration = ProtocolDeclaration(
        "strict_source_only_dg",
        "SOURCE_ONLY_PROTOCOL_LOCKED_RETROSPECTIVE_TARGET",
        ("P1", "P2", "P3", "P4"),
        False,
    )

    def __init__(self) -> None:
        self._seal: RecipeSeal | None = None
        self._target_authorized = False

    @property
    def recipe_frozen(self) -> bool:
        return self._seal is not None

    def freeze_recipe(self, recipe: Mapping[str, Any]) -> RecipeSeal:
        if self._seal is not None:
            raise RecipeFreezeError("Strict recipe is already frozen")
        self._seal = create_recipe_seal(recipe)
        return self._seal

    def authorize_final_evaluation(self, recipe: Mapping[str, Any], seal: RecipeSeal) -> None:
        if self._seal is None or seal != self._seal or not verify_recipe_seal(recipe, seal):
            raise RecipeFreezeError("Final evaluation requires the unchanged frozen recipe")
        self._target_authorized = True

    def authorize(self, request: AccessRequest) -> None:
        if request.domain in {"P1", "P2", "P3"}:
            return
        if request.domain != "P4":
            raise DomainAccessViolation(f"Unknown strict-DG domain: {request.domain}")
        if request.purpose is not Purpose.EVALUATION:
            raise DomainAccessViolation("P4 may only be used for final evaluation")
        if not self._target_authorized:
            raise DomainAccessViolation("P4 access is blocked until recipe freeze verification")

    def assert_revision_allowed(self) -> None:
        if self._target_authorized:
            raise RecipeFreezeError("Recipe revision is forbidden after target authorization")
