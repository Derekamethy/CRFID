"""Candidate enumeration without solver execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .geometry import candidate_geometry


def build_candidate_geometries(
    base_geometry: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        name = str(candidate["name"])
        if name in result:
            raise ValueError(f"Duplicate redesign candidate: {name}")
        result[name] = candidate_geometry(base_geometry, candidate.get("overrides", {}))
    return result
