"""Frozen DANN v2 constants, schedule, mechanism gate, and source-only rules."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Iterable


LAMBDAS = (0.01, 0.03, 0.10, 0.30, 1.00)
FOLDS = ("S1", "S2", "S3")
SEEDS = (42, 43, 44, 45, 46)
POSITIONS = ("P1", "P2", "P3")
HELD_POSITION = {"S1": "P1", "S2": "P2", "S3": "P3"}
CLASS_ORDER = tuple(range(7))

ARM_A0 = "ARM_A0_ERM_MATCHED"
ARM_A1 = "ARM_A1_DOMAIN_POSITIVE_MATCHED"
ARM_A2 = "ARM_A2_DANN_NEGATIVE_MATCHED"
ARMS = (ARM_A0, ARM_A1, ARM_A2)

MAXIMUM_EPOCHS = 50
TASK_RETENTION_THRESHOLD = -0.02
TAGID_PROBE_RETENTION_THRESHOLD = -0.05
MACRO_F1_TIE_TOLERANCE = 0.002
DEVELOPMENT_BOOTSTRAP_SEED = 20_260_809
DEVELOPMENT_BOOTSTRAP_REPLICATES = 10_000
P4_BOOTSTRAP_SEED = 20_260_809
P4_BOOTSTRAP_REPLICATES = 10_000
CHANCE_REFERENCE_SEED = 20_260_810
CHANCE_REFERENCE_REPLICATES = 10_000


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_lambda_grid(values: Iterable[float]) -> tuple[float, ...]:
    supplied = tuple(float(value) for value in values)
    if supplied != LAMBDAS:
        raise RuntimeError(f"Frozen DANN v2 lambda grid changed: {supplied!r}")
    return supplied


def scheduled_lambda(
    lambda_max: float,
    global_update_index: int,
    registered_steps_per_epoch: int,
    *,
    maximum_epochs: int = MAXIMUM_EPOCHS,
) -> float:
    """The single treatment schedule used by every DANN v2 training entry point."""

    value = float(lambda_max)
    if value != 0.0 and value not in LAMBDAS:
        raise ValueError("lambda is outside the frozen DANN v2 grid")
    steps = int(registered_steps_per_epoch)
    epochs = int(maximum_epochs)
    index = int(global_update_index)
    if steps <= 0 or epochs != MAXIMUM_EPOCHS:
        raise ValueError("registered schedule must use the frozen 50-epoch horizon")
    maximum_registered_updates = epochs * steps
    if not 0 <= index < maximum_registered_updates:
        raise ValueError("global update index is outside the registered schedule")
    denominator = maximum_registered_updates - 1
    progress = index / denominator if denominator else 0.0
    return value * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def registered_lambda_trajectory(
    lambda_max: float,
    *,
    executed_epochs: int,
    registered_steps_per_epoch: int,
) -> list[float]:
    if not 0 <= int(executed_epochs) <= MAXIMUM_EPOCHS:
        raise ValueError("executed epochs exceed the registered horizon")
    count = int(executed_epochs) * int(registered_steps_per_epoch)
    return [
        scheduled_lambda(lambda_max, index, registered_steps_per_epoch)
        for index in range(count)
    ]


def schedule_summary(values: Iterable[float]) -> dict:
    trajectory = [float(value) for value in values]
    if not trajectory:
        return {
            "update_count": 0,
            "mean_lambda": 0.0,
            "maximum_realised_lambda": 0.0,
            "cumulative_domain_gradient_coefficient": 0.0,
        }
    return {
        "update_count": len(trajectory),
        "mean_lambda": float(sum(trajectory) / len(trajectory)),
        "maximum_realised_lambda": float(max(trajectory)),
        "cumulative_domain_gradient_coefficient": float(sum(abs(value) for value in trajectory)),
    }


def _retention(value: float, threshold: float) -> dict:
    difference = float(value)
    return {
        "difference": difference,
        "threshold": float(threshold),
        "passed": difference >= float(threshold) - 1e-15,
    }


def select_lambda(rows: list[dict]) -> dict:
    """Apply the preregistered source-only continuation rule and tie breaks."""

    by_lambda = {float(row["lambda_max"]): dict(row) for row in rows}
    if set(by_lambda) != set(LAMBDAS):
        raise RuntimeError("DANN v2 selection requires all five frozen lambdas")
    strong: list[float] = []
    weak: list[float] = []
    for value in LAMBDAS:
        row = by_lambda[value]
        if int(row.get("run_count", 0)) != 15:
            raise RuntimeError(f"Lambda {value} does not have 15 fold-by-seed A2 runs")
        if row.get("p4_used", False):
            raise RuntimeError("P4 evidence is prohibited during DANN v2 selection")
        if row.get("outer_held_lopo_used_for_selection", False):
            raise RuntimeError("Outer-held LOPO is diagnostic only")
        task = _retention(
            float(row["familiar_validation_macro_f1_difference"]),
            TASK_RETENTION_THRESHOLD,
        )
        tagid = _retention(
            float(row["tagid_probe_macro_f1_difference"]),
            TAGID_PROBE_RETENTION_THRESHOLD,
        )
        position_mean = float(row["position_probe_macro_f1_difference"])
        interval = tuple(float(item) for item in row["position_probe_difference_interval_95"])
        if len(interval) != 2:
            raise RuntimeError("Position interval must have two bounds")
        mean_reduced = position_mean < 0.0
        row["task_retention_gate"] = task
        row["tagid_representation_retention_gate"] = tagid
        row["mean_position_reduction"] = mean_reduced
        row["strong_position_reduction"] = mean_reduced and interval[1] < 0.0
        row["mechanism_eligible"] = task["passed"] and tagid["passed"] and mean_reduced
        row["position_probe_difference_interval_95"] = list(interval)
        by_lambda[value] = row
        if row["mechanism_eligible"] and row["strong_position_reduction"]:
            strong.append(value)
        elif row["mechanism_eligible"]:
            weak.append(value)

    pool = strong if strong else weak
    strength = "STRONG_POSITION_REDUCTION_CANDIDATES" if strong else (
        "WEAK_POSITION_REDUCTION_CANDIDATES" if weak else None
    )
    payload: dict = {
        "schema_version": 1,
        "rule": "source_retention_then_position_reduction_then_tagid_task_smaller_lambda",
        "tie_tolerance_absolute_macro_f1": MACRO_F1_TIE_TOLERANCE,
        "task_retention_threshold": TASK_RETENTION_THRESHOLD,
        "tagid_probe_retention_threshold": TAGID_PROBE_RETENTION_THRESHOLD,
        "weak_candidate_fallback_preregistered": True,
        "strong_candidate_lambdas": strong,
        "weak_candidate_lambdas": weak,
        "candidate_rows": [by_lambda[value] for value in LAMBDAS],
        "p4_used": False,
    }
    if not pool:
        payload.update(
            {
                "status": "NO_DANN_V2_CANDIDATE_MEETS_SOURCE_MECHANISM_GATE",
                "mechanism_candidate_class": None,
                "selected_lambda": None,
            }
        )
        payload["selection_sha256"] = canonical_json_sha256(payload)
        return payload

    best_position = min(float(by_lambda[value]["position_probe_macro_f1_difference"]) for value in pool)
    position_tied = [
        value
        for value in pool
        if float(by_lambda[value]["position_probe_macro_f1_difference"]) - best_position
        <= MACRO_F1_TIE_TOLERANCE + 1e-15
    ]
    best_tagid = max(float(by_lambda[value]["tagid_probe_macro_f1_mean"]) for value in position_tied)
    tagid_tied = [
        value
        for value in position_tied
        if best_tagid - float(by_lambda[value]["tagid_probe_macro_f1_mean"])
        <= MACRO_F1_TIE_TOLERANCE + 1e-15
    ]
    best_task = max(
        float(by_lambda[value]["familiar_validation_macro_f1_mean"])
        for value in tagid_tied
    )
    task_tied = [
        value
        for value in tagid_tied
        if best_task - float(by_lambda[value]["familiar_validation_macro_f1_mean"])
        <= MACRO_F1_TIE_TOLERANCE + 1e-15
    ]
    selected = min(task_tied)
    payload.update(
        {
            "status": "PASS_DANN_V2_SOURCE_ONLY_LAMBDA_SELECTION",
            "mechanism_candidate_class": strength,
            "selected_lambda": selected,
            "largest_position_probe_reduction": best_position,
            "position_tied_lambdas": position_tied,
            "highest_tagid_probe_within_tie": best_tagid,
            "tagid_tied_lambdas": tagid_tied,
            "highest_familiar_validation_within_tie": best_task,
            "task_tied_lambdas": task_tied,
        }
    )
    payload["selection_sha256"] = canonical_json_sha256(payload)
    return payload


def derive_final_epochs(records: list[dict], selected_lambda: float) -> dict[str, int]:
    selected = [
        row
        for row in records
        if row["arm"] == ARM_A2 and float(row["lambda_max"]) == float(selected_lambda)
    ]
    if len(selected) != 15:
        raise RuntimeError("Final epoch derivation requires 15 selected-A2 development runs")
    output: dict[str, int] = {}
    for seed in SEEDS:
        local = [row for row in selected if int(row["seed"]) == seed]
        if {row["fold"] for row in local} != set(FOLDS):
            raise RuntimeError(f"Final epoch derivation is incomplete for seed {seed}")
        epochs = [int(row["selected_epoch"]) for row in local]
        output[str(seed)] = int(statistics.median(epochs))
    return output
