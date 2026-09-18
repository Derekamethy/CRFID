"""Assumption-bounded redesign planning without implicit simulation."""

from __future__ import annotations

from typing import Any

from ..openems.redesign import build_candidate_geometries
from .common import WorkflowPlan, build_plan


def run(config: dict[str, Any], *, execute: bool = False) -> WorkflowPlan | dict[str, Any]:
    plan = build_plan(config, "ASSUMPTION_BOUNDED_PHYSICAL_REDESIGN", execute)
    geometries = build_candidate_geometries(config["geometry"], config["candidates"])
    if not execute:
        return plan
    return {
        "status": "GEOMETRY_PLAN_CREATED_OPENEMS_NOT_EXECUTED",
        "evidence_boundary": config["evidence_boundary"],
        "candidate_geometries": geometries,
    }
