"""Frozen protocol assembly for the Stage1-5 Pre-DG baselines."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..governance.pre_dg_integrity import canonical_json_sha256


EXPECTED_STAGE_ORDER = (
    "validate_pre_dg_environment",
    "validate_pre_dg_inputs",
    "reconstruct_pre_dg_split",
    "fit_preprocessing",
    "build_pre_dg_model",
    "train_pre_dg_model",
    "select_checkpoint",
    "save_checkpoint",
    "reload_checkpoint",
    "generate_predictions",
    "calculate_metrics",
    "aggregate_seed_results",
    "compare_with_authoritative_reference",
    "generate_pre_dg_report",
    "package_pre_dg_review_material",
)


def build_protocol(
    canonical: Mapping[str, Any],
    data_authority: Mapping[str, Any],
    split: Mapping[str, Any],
    model: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = {
        "schema_version": 1,
        "scientific_name": canonical["scientific_name"],
        "historical_verdict": canonical["historical_verdict"],
        "datasets": list(data_authority["datasets"]),
        "split_names": [item["split_name"] for item in split["splits"]],
        "seeds": list(canonical["random_seeds"]),
        "preprocessing": model["preprocessing"],
        "model": model["model"],
        "training": model["training"],
        "evaluation": {
            "metrics": evaluation["metrics"],
            "aggregation": evaluation["aggregation"],
            "comparison_policy": evaluation["comparison_policy"],
        },
        "stage_order": list(EXPECTED_STAGE_ORDER),
        "claim_boundary": canonical["claim_boundary"],
    }
    if protocol["seeds"] != [42, 43, 44]:
        raise ValueError("Pre-DG seeds changed")
    if len(protocol["split_names"]) != 18:
        raise ValueError("Pre-DG split scope changed")
    return protocol


def protocol_sha256(protocol: Mapping[str, Any]) -> str:
    return canonical_json_sha256(dict(protocol))
