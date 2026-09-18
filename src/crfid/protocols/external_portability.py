"""Data1 local-task portability access contract."""

from __future__ import annotations

from ..exceptions import DomainAccessViolation
from .common import AccessRequest, ProtocolDeclaration, Purpose


class ExternalPortabilityProtocol:
    declaration = ProtocolDeclaration(
        "data1_portability", "LOCAL_FOUR_CLASS_PIPELINE_PORTABILITY", ("train", "test"), False
    )
    class_order = (0, 1, 2, 3)

    def authorize(self, request: AccessRequest) -> None:
        if request.domain == "train":
            return
        if request.domain == "test" and request.purpose is Purpose.EVALUATION:
            return
        raise DomainAccessViolation("Data1 test data are evaluation-only")
