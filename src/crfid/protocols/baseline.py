"""Pre-DG baseline access contract."""

from __future__ import annotations

from ..exceptions import DomainAccessViolation
from .common import AccessRequest, ProtocolDeclaration


class BaselineProtocol:
    declaration = ProtocolDeclaration(
        "pre_dg_baseline", "SINGLE_DOMAIN_BASELINE", ("P1", "P2", "P3"), False
    )

    def authorize(self, request: AccessRequest) -> None:
        if request.domain not in self.declaration.allowed_domains:
            raise DomainAccessViolation("Pre-DG baseline access is limited to P1-P3")
