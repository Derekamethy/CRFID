"""Frozen External Data1 pipeline-validation protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..exceptions import ProtocolViolation
from ..external_data1.artifacts import canonical_json_sha256


PROTOCOL_SHA256 = "ed041a4cebc97de229211f038c9618393aaf1e45a440a6fc54582585b52169ed"


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    """Reject any change to a field that governed EV3R-R1."""

    supplied = dict(protocol)
    if canonical_json_sha256(supplied) != PROTOCOL_SHA256:
        raise ProtocolViolation("The frozen EV2R/EV2R-A1 protocol was altered")
    if supplied["class_order"] != [0, 1, 2, 3]:
        raise ProtocolViolation("The frozen Data1 class order was altered")
    if supplied["majority_tie_rule"]["selected_class"] != 0:
        raise ProtocolViolation("The frozen majority tie rule was altered")
