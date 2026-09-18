"""Configuration loading and validation for the branch.

A configuration is rejected -- not silently corrected -- if it would violate a
scientific boundary: any reference to the target domain in a training or selection
role, any use of Data1 labels, a corpus mixture that does not sum to one, an unknown
objective, a treatment outside the declared set, or a seed set that differs from the
frozen one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import objectives, treatments
from .paths import BRANCH_CONFIG_ROOT

FORBIDDEN_TARGET_ROLES = (
    "target_training",
    "target_selection",
    "target_adaptation",
    "target_labels_for_selection",
    "target_batch_norm_statistics",
    "target_pseudo_labels",
    "target_entropy_minimization",
    "target_calibration",
    "target_early_stopping",
    "target_prototypes",
)


class ConfigurationViolation(ValueError):
    """Raised when a configuration breaches a declared scientific boundary."""


def load(name: str) -> dict[str, Any]:
    path = BRANCH_CONFIG_ROOT / name
    if not path.is_file():
        raise ConfigurationViolation(f"Configuration is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate(payload)
    return payload


def validate(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a branch configuration and return it unchanged."""

    required = {"schema_version", "branch", "seeds", "folds", "target_policy", "data1_policy"}
    missing = sorted(required - set(config))
    if missing:
        raise ConfigurationViolation(f"Configuration fields missing: {missing}")

    if tuple(config["seeds"]) != treatments.SEEDS:
        raise ConfigurationViolation(
            f"Seed set {config['seeds']} differs from the frozen {list(treatments.SEEDS)}"
        )
    if tuple(config["folds"]) != treatments.FOLDS:
        raise ConfigurationViolation(f"Fold set {config['folds']} is not the canonical source folds")

    target_policy = config["target_policy"]
    for role in FORBIDDEN_TARGET_ROLES:
        if target_policy.get(role) is not False:
            raise ConfigurationViolation(f"Target policy must declare {role}=false")
    if target_policy.get("evaluation_only_after_preregistration") is not True:
        raise ConfigurationViolation(
            "Target policy must declare evaluation_only_after_preregistration=true"
        )

    data1_policy = config["data1_policy"]
    for role in (
        "labels_for_supervised_pretraining",
        "labels_for_objective_selection",
        "labels_for_hyperparameter_selection",
        "labels_for_checkpoint_selection",
        "labels_for_class_balancing",
        "labels_for_example_filtering",
        "labels_for_method_ranking",
        "labels_for_representation_evaluation",
    ):
        if data1_policy.get(role) is not False:
            raise ConfigurationViolation(f"Data1 policy must declare {role}=false")

    for entry in config.get("treatments", []):
        treatment_id = entry.get("treatment_id")
        if treatment_id not in treatments.TREATMENTS:
            raise ConfigurationViolation(f"Unknown treatment: {treatment_id}")
        fractions = entry.get("corpus_fractions") or {}
        if fractions and abs(sum(fractions.values()) - 1.0) > 1e-9:
            raise ConfigurationViolation(f"{treatment_id} corpus fractions do not sum to one")
        objective_id = entry.get("objective_id")
        if objective_id is not None and objective_id not in objectives.OBJECTIVE_IDS:
            raise ConfigurationViolation(f"{treatment_id} names an unknown objective: {objective_id}")

    for key in ("training_domains", "selection_domains", "ssl_domains"):
        declared = config.get(key)
        if declared is not None and "P4" in declared:
            raise ConfigurationViolation(f"{key} must not contain the sealed target domain")
    return config


def write(name: str, payload: dict[str, Any]) -> Path:
    validate(payload)
    BRANCH_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    path = BRANCH_CONFIG_ROOT / name
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path
