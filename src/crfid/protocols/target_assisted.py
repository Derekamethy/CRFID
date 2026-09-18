"""Protocol facade for P4 Target-Assisted Adaptation."""

from __future__ import annotations

from ..governance.target_access import TargetAccessContract
from .common import AccessRequest, ProtocolDeclaration


class TargetAssistedProtocol:
    declaration = ProtocolDeclaration(
        "p4_target_assisted_adaptation",
        "RETROSPECTIVE_P4_INFORMED_REFERENCE",
        ("P1", "P2", "P3", "P4"),
        True,
    )

    def __init__(self, target_access: TargetAccessContract) -> None:
        self.target_access = target_access

    def authorize(self, request: AccessRequest) -> None:
        self.target_access.authorize(request)
