"""Explicit external-solver adapter boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..exceptions import SimulationDisabledError


def run_simulation(
    geometry: Mapping[str, Any],
    output_path: str | Path,
    *,
    execute: bool,
    adapter: Callable[[Mapping[str, Any], Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Call only a supplied adapter after an explicit execution decision."""

    if not execute:
        raise SimulationDisabledError("OpenEMS execution was not explicitly enabled")
    if adapter is None:
        raise SimulationDisabledError("No approved OpenEMS adapter was supplied")
    return adapter(geometry, Path(output_path))
