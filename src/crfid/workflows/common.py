"""Shared workflow planning and explicit array loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WorkflowPlan:
    workflow: str
    protocol: str
    input_path: str | None
    output_path: str
    execute_requested: bool
    status: str = "NUMERICAL_REPRODUCTION_PENDING"

    def as_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "protocol": self.protocol,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "execute_requested": self.execute_requested,
            "status": self.status,
        }


def build_plan(config: dict[str, Any], protocol: str, execute: bool) -> WorkflowPlan:
    return WorkflowPlan(
        workflow=str(config["workflow"]),
        protocol=protocol,
        input_path=config.get("input_path"),
        output_path=str(config["output_path"]),
        execute_requested=execute,
    )


def load_npz(path: str | Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as payload:
        missing = sorted(set(required).difference(payload.files))
        if missing:
            raise ValueError(f"Input archive is missing arrays: {missing}")
        return {name: np.ascontiguousarray(payload[name]) for name in required}
