"""Metadata helpers for protected condition groups."""

from __future__ import annotations

from .schemas import SampleMetadata


def condition_key(metadata: SampleMetadata) -> str:
    """Return the declared group key used for split isolation."""

    if not metadata.condition_id:
        raise ValueError("condition_id cannot be empty")
    return metadata.condition_id


def group_by_condition(metadata: list[SampleMetadata]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(metadata):
        groups.setdefault(condition_key(item), []).append(index)
    return groups
