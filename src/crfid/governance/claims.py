"""Scientific claim limits for target-assisted results."""

from __future__ import annotations

from ..exceptions import ProtocolViolation


RETROSPECTIVE_CLAIM = "RETROSPECTIVE_P4_INFORMED_REFERENCE"


def validate_target_assisted_claim(claim: str) -> None:
    if claim != RETROSPECTIVE_CLAIM:
        raise ProtocolViolation(
            "The historical reuse of P4 outcomes permits only RETROSPECTIVE_P4_INFORMED_REFERENCE"
        )
