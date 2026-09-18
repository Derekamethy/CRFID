"""Reusable failure-mechanism analysis workflow."""

from __future__ import annotations

from typing import Any

from ..analysis.embedding_analysis import class_domain_entanglement, separability_summary
from ..analysis.failure_analysis import collapse_summary
from .common import WorkflowPlan, build_plan, load_npz


def run(config: dict[str, Any], *, execute: bool = False) -> WorkflowPlan | dict[str, Any]:
    plan = build_plan(config, "ANALYSIS_ONLY", execute)
    if not execute:
        return plan
    payload = load_npz(str(config["input_path"]), ("embeddings", "labels", "domains", "predictions"))
    return {
        "status": "ANALYSIS_COMPLETE",
        "separability": separability_summary(payload["embeddings"], payload["labels"]),
        "entanglement": class_domain_entanglement(payload["embeddings"], payload["labels"], payload["domains"]),
        "collapse": collapse_summary(payload["labels"], payload["predictions"], class_count=int(config["class_count"])),
    }
