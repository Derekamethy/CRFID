"""Immutable constants sealed by the preregistration."""

from __future__ import annotations

from p4_factor_aware.constants import CHECKPOINT_SEEDS, QUERY_FOLDS


EXPERIMENT_ID = "p4_large_calibration_curve_v1"
CLAIM_STRICT_DG = "STRICT_SOURCE_ONLY_DG"
CLAIM_TARGET_ASSISTED = "TARGET_ASSISTED_LABELLED_TARGET_CALIBRATION"
METHOD_ZERO = "FS0_FROZEN_SOURCE_HEAD"
METHOD_POSITIVE = "TARGET_ONLY_COSINE_PROTOTYPE"

TOTAL_BUDGETS = (0, 7, 10, 20, 21, 35, 50, 100, 200, 500)
POSITIVE_BUDGETS = TOTAL_BUDGETS[1:]
SHOT_LABELS = {7: "1-shot", 21: "3-shot", 35: "5-shot"}
MAXIMUM_BUDGET = max(TOTAL_BUDGETS)
MAXIMUM_SUPPORT_BLOCKS = 42

SUPPORT_SEEDS = (
    104729,
    130363,
    155921,
    181081,
    205439,
    230003,
    254959,
    280001,
    305093,
    330017,
    355031,
    380041,
    405019,
    430007,
    455003,
    480017,
    505027,
    530017,
    555029,
    580031,
)

BOOTSTRAP_SEED = 20260809
BOOTSTRAP_REPLICATES = 10_000
PRACTICAL_DELTA = 0.05

PREREGISTRATION_SHA256 = (
    "30e65854c09773f3ff64054cf04fabc215a9e9fa27919ffc9de448d1f48bb99d"
)

STRICT_PER_SEED = {
    42: {"accuracy": 0.1726984126984127, "macro_f1": 0.12305412236755582},
    43: {"accuracy": 0.11936507936507937, "macro_f1": 0.09655707116594889},
    44: {"accuracy": 0.1292063492063492, "macro_f1": 0.10079515335097167},
    45: {"accuracy": 0.20412698412698413, "macro_f1": 0.1445645243313212},
    46: {"accuracy": 0.15777777777777777, "macro_f1": 0.096183546957079},
}
STRICT_AGGREGATE = {
    "accuracy": 0.15663492063492063,
    "macro_f1": 0.11223088363457531,
}

HISTORICAL_REFERENCES = {
    "historical_few_shot_prototype": {
        7: 0.112159,
        21: 0.145141,
        35: 0.144360,
        "protocol_comparable": False,
    },
    "retrospective_target_informed": {
        "macro_f1": 0.5835693346352661,
        "protocol_comparable": False,
    },
    "matched_within_condition_linear_probe": {
        "macro_f1": 0.983494,
        "protocol_comparable": False,
    },
}


def budget_label(budget: int) -> str:
    if budget == 0:
        return "0 labels (strict source-only DG)"
    if budget in SHOT_LABELS:
        return f"{SHOT_LABELS[budget]} ({budget} labels)"
    return f"{budget} labels"
