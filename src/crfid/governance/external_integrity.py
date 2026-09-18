"""Protocol, artifact and forbidden-mode integrity checks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..exceptions import ProtocolViolation
from ..external_data1.artifacts import public_manifest
from ..protocols.external_data1 import validate_protocol


FORBIDDEN_MODES = {
    "paper4",
    "p4",
    "target_assisted",
    "few_shot",
    "data1_ssl",
    "cross_dataset_transfer",
}


def validate_external_mode(mode: str) -> None:
    if mode.casefold() in FORBIDDEN_MODES:
        raise ProtocolViolation("Requested mode is outside External Data1 pipeline validation")
    if mode != "pipeline_validation":
        raise ProtocolViolation("Unknown External Data1 execution mode")


def validate_protocol_and_mode(protocol: Mapping[str, Any], mode: str) -> None:
    validate_protocol(protocol)
    validate_external_mode(mode)


def assert_public_paths(root: str | Path) -> None:
    for row in public_manifest(root):
        relative = row["relative_path"]
        if Path(relative).is_absolute() or ":" in relative:
            raise ProtocolViolation("Public artifact manifest contains an absolute path")

