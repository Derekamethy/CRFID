"""Data-scope allow list for the independent Data1 branch."""

from __future__ import annotations

from pathlib import Path

from ..exceptions import DomainAccessViolation
from ..external_data1.schema import DATASET_ID


_FORBIDDEN_TOKENS = (
    "paper4",
    "target_assisted",
    "few_shot",
    "data1_ssl",
    "openems",
)


def authorize_external_data(dataset_id: str, path: str | Path) -> None:
    if dataset_id != DATASET_ID:
        raise DomainAccessViolation("Only the logical external Data1 dataset is authorized")
    normalized = str(path).replace("\\", "/").casefold()
    if any(token in normalized for token in _FORBIDDEN_TOKENS):
        raise DomainAccessViolation("Input belongs to a prohibited scientific branch")
    if "/p4/" in normalized or normalized.endswith("/p4"):
        raise DomainAccessViolation("P4 input is prohibited for Data1 pipeline validation")

