"""Few-shot support/query access with a sealed adaptation boundary."""

from __future__ import annotations

from ..exceptions import DomainAccessViolation
from .common import AccessRequest, ProtocolDeclaration, Purpose, Resource


class FewShotProtocol:
    declaration = ProtocolDeclaration(
        "few_shot_p4", "CONTROLLED_TARGET_FEW_SHOT_ADAPTATION", ("P1", "P2", "P3", "P4"), True
    )

    def __init__(self) -> None:
        self._adaptation_closed = False

    def close_adaptation(self) -> None:
        self._adaptation_closed = True

    def authorize(self, request: AccessRequest) -> None:
        if request.domain in {"P1", "P2", "P3"}:
            return
        if request.domain != "P4" or request.partition not in {"support", "query"}:
            raise DomainAccessViolation("Few-shot P4 access requires support or query partition")
        if request.partition == "support":
            return
        if request.resource is Resource.FEATURES and request.purpose is Purpose.EVALUATION:
            return
        if request.resource is Resource.LABELS and request.purpose is Purpose.EVALUATION and self._adaptation_closed:
            return
        raise DomainAccessViolation("Query labels are sealed during adaptation")
