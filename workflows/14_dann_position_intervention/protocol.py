"""Frozen protocol constants and source-only decision rules."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


LAMBDAS = (0.0, 0.03, 0.10, 0.30, 1.00)
NONZERO_LAMBDAS = LAMBDAS[1:]
FOLDS = ("S1", "S2", "S3")
SEEDS = (42, 43, 44, 45, 46)
POSITIONS = ("P1", "P2", "P3")
HELD_POSITION = {"S1": "P1", "S2": "P2", "S3": "P3"}
CLASS_ORDER = tuple(range(7))
SOURCE_RETENTION_GUARDRAIL = 0.02
ELIGIBILITY_MARGIN = 0.01
PROBE_TIE_MARGIN = 0.01

INTERPRETATION_CLASSES = (
    "POSITION_SUPPRESSION_AND_P4_TRANSFER_BENEFIT_CONFIRMED",
    "POSITION_SUPPRESSION_WITHOUT_P4_TRANSFER_BENEFIT",
    "P4_TRANSFER_BENEFIT_WITHOUT_CONFIRMED_POSITION_SUPPRESSION",
    "POSITION_SUPPRESSION_NOT_ACHIEVED",
    "POSITION_SUPPRESSION_HARMS_TAGID_TRANSFER",
    "DANN_MIXED_OR_UNSTABLE_RESULT",
    "BLOCKED_GOVERNED_INPUTS",
    "FAIL_PROTOCOL_OR_LABEL_BOUNDARY_DEFECT",
)


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
        raise RuntimeError(f"Frozen DANN lambda grid changed: {supplied!r}")
    return supplied


def grl_strength(lambda_max: float, progress: float) -> float:
    """Preregistered logistic gradient-reversal schedule."""

    import math

    value = float(lambda_max)
    p = float(progress)
    if value not in LAMBDAS or not 0.0 <= p <= 1.0:
        raise ValueError("lambda or normalized progress is outside the frozen protocol")
    return value * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def select_nonzero_lambda(rows: list[dict]) -> dict:
    """Apply the frozen lexicographic rule to source-only aggregate rows."""

    by_lambda = {float(row["lambda_max"]): row for row in rows}
    if set(by_lambda) != set(LAMBDAS):
        raise RuntimeError("Source-only selection requires all five frozen lambdas")
    for value in LAMBDAS:
        row = by_lambda[value]
        if int(row.get("run_count", 0)) != 15:
            raise RuntimeError(f"Lambda {value} does not have 15 fold/seed runs")
        if row.get("p4_used", False):
            raise RuntimeError("P4 evidence is prohibited during lambda selection")
    best_nonzero = max(
        float(by_lambda[value]["held_macro_f1_mean"]) for value in NONZERO_LAMBDAS
    )
    eligible = [
        value
        for value in NONZERO_LAMBDAS
        if best_nonzero - float(by_lambda[value]["held_macro_f1_mean"])
        <= ELIGIBILITY_MARGIN + 1e-15
    ]
    lowest_probe = min(
        float(by_lambda[value]["position_probe_balanced_accuracy_mean"])
        for value in eligible
    )
    near_lowest = [
        value
        for value in eligible
        if float(by_lambda[value]["position_probe_balanced_accuracy_mean"])
        - lowest_probe
        <= PROBE_TIE_MARGIN + 1e-15
    ]
    selected = min(near_lowest)
    overall = max(
        LAMBDAS,
        key=lambda value: (
            float(by_lambda[value]["held_macro_f1_mean"]),
            -float(value),
        ),
    )
    payload = {
        "schema_version": 1,
        "rule": "source_only_frozen_lexicographic_v1",
        "best_nonzero_held_macro_f1": best_nonzero,
        "eligible_nonzero_lambdas": eligible,
        "selected_nonzero_lambda": selected,
        "source_only_overall_winner": "ERM" if overall == 0.0 else "DANN",
        "source_only_overall_winner_lambda": overall,
        "p4_used": False,
    }
    payload["selection_sha256"] = canonical_json_sha256(payload)
    return payload


def source_retention_guardrail(erm_macro_f1: float, dann_macro_f1: float) -> dict:
    change = float(dann_macro_f1) - float(erm_macro_f1)
    return {
        "threshold": -SOURCE_RETENTION_GUARDRAIL,
        "change": change,
        "passed": change >= -SOURCE_RETENTION_GUARDRAIL - 1e-15,
        "sacrifices_source_tagid_performance": change < -SOURCE_RETENTION_GUARDRAIL,
    }


def classify_intervention(
    *,
    position_change: float,
    position_interval: tuple[float, float],
    p4_change: float,
    p4_interval: tuple[float, float],
    source_change: float,
    source_interval: tuple[float, float] | None = None,
    protocol_passed: bool = True,
    stable: bool = True,
) -> str:
    """Return exactly one preregistered interpretation class."""

    if not protocol_passed:
        return "FAIL_PROTOCOL_OR_LABEL_BOUNDARY_DEFECT"
    position_confirmed = position_change < 0.0 and position_interval[1] < 0.0
    p4_improved = p4_change > 0.0 and p4_interval[0] > 0.0
    p4_worsened = p4_change < 0.0 and p4_interval[1] < 0.0
    source_worsened = (
        source_interval is not None
        and source_change < 0.0
        and source_interval[1] < 0.0
    )
    if position_confirmed and (p4_worsened or source_worsened):
        return "POSITION_SUPPRESSION_HARMS_TAGID_TRANSFER"
    if position_confirmed and p4_improved and source_change >= -SOURCE_RETENTION_GUARDRAIL:
        return "POSITION_SUPPRESSION_AND_P4_TRANSFER_BENEFIT_CONFIRMED"
    if position_confirmed and not p4_improved:
        return "POSITION_SUPPRESSION_WITHOUT_P4_TRANSFER_BENEFIT"
    if p4_improved and not position_confirmed:
        return "P4_TRANSFER_BENEFIT_WITHOUT_CONFIRMED_POSITION_SUPPRESSION"
    if not position_confirmed and stable:
        return "POSITION_SUPPRESSION_NOT_ACHIEVED"
    return "DANN_MIXED_OR_UNSTABLE_RESULT"
