"""Treatment definitions.

All treatments share one backbone family, one supervised fine-tuning implementation,
one label set, one seed set, one source-side selection protocol, one final evaluator
and one reporting format. They differ only in what the backbone trunk is initialised
from, and in nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import objectives
from .pretrain import DATA1_CORPUS, PAPER4_CORPUS

T0 = "T0_MATCHED_SUPERVISED_ONLY"
T1 = "T1_PAPER4_SOURCE_ONLY_SSL"
T2 = "T2_DATA1_ONLY_SSL"
T3 = "T3_JOINT_BALANCED_SSL"
T4 = "T4_JOINT_PHYSICS_AWARE_SSL"
T5 = "T5_JOINT_DOMAIN_INVARIANT_SSL"

TREATMENT_ORDER = (T0, T1, T2, T3, T4, T5)

SEEDS = (42, 43, 44, 45, 46)
FOLDS = ("S1", "S2", "S3")


@dataclass(frozen=True)
class Treatment:
    treatment_id: str
    description: str
    uses_ssl: bool
    corpus_fractions: dict[str, float] = field(default_factory=dict)
    objective_role: str | None = None  # "GENERIC" | "PHYSICS_AWARE"
    domain_confusion_weight: float = 0.0
    gated: bool = False

    def as_record(self) -> dict[str, Any]:
        return {
            "treatment_id": self.treatment_id,
            "description": self.description,
            "uses_ssl": self.uses_ssl,
            "corpus_fractions": dict(sorted(self.corpus_fractions.items())),
            "objective_role": self.objective_role,
            "domain_confusion_weight": self.domain_confusion_weight,
            "gated": self.gated,
            "uses_data1_labels": False,
            "uses_target_domain": False,
        }


TREATMENTS: dict[str, Treatment] = {
    T0: Treatment(
        treatment_id=T0,
        description=(
            "Matched supervised-only control: no self-supervision, no Data1, "
            "Paper4 P1-P3 supervised training in the same backbone and budget."
        ),
        uses_ssl=False,
    ),
    T1: Treatment(
        treatment_id=T1,
        description=(
            "Unlabelled Paper4 P1-P3 self-supervised pretraining, then Paper4 P1-P3 "
            "supervised fine-tuning. Isolates self-supervision without external data."
        ),
        uses_ssl=True,
        corpus_fractions={PAPER4_CORPUS: 1.0},
        objective_role="GENERIC",
    ),
    T2: Treatment(
        treatment_id=T2,
        description=(
            "Unlabelled Data1 self-supervised pretraining only, then Paper4 P1-P3 "
            "supervised fine-tuning. Tests whether Data1 alone yields a transferable "
            "initialisation."
        ),
        uses_ssl=True,
        corpus_fractions={DATA1_CORPUS: 1.0},
        objective_role="GENERIC",
    ),
    T3: Treatment(
        treatment_id=T3,
        description=(
            "Joint balanced self-supervised pretraining with a preregistered "
            "Paper4/Data1 batch ratio, then Paper4 P1-P3 supervised fine-tuning."
        ),
        uses_ssl=True,
        corpus_fractions={PAPER4_CORPUS: 0.5, DATA1_CORPUS: 0.5},
        objective_role="GENERIC",
    ),
    T4: Treatment(
        treatment_id=T4,
        description=(
            "Joint balanced self-supervised pretraining using the source-selected "
            "physics-aware objective, then Paper4 P1-P3 supervised fine-tuning."
        ),
        uses_ssl=True,
        corpus_fractions={PAPER4_CORPUS: 0.5, DATA1_CORPUS: 0.5},
        objective_role="PHYSICS_AWARE",
    ),
    T5: Treatment(
        treatment_id=T5,
        description=(
            "Joint self-supervised pretraining with an explicit gradient-reversal "
            "dataset-identity confusion term. Gated on positive source-side evidence "
            "from T3 or T4."
        ),
        uses_ssl=True,
        corpus_fractions={PAPER4_CORPUS: 0.5, DATA1_CORPUS: 0.5},
        objective_role="GENERIC",
        domain_confusion_weight=0.1,
        gated=True,
    ),
}


def objective_for(
    treatment: Treatment, *, generic: str, physics_aware: str
) -> str:
    if not treatment.uses_ssl:
        raise ValueError(f"{treatment.treatment_id} does not use an SSL objective")
    if treatment.objective_role == "PHYSICS_AWARE":
        return physics_aware
    return generic


def validate_selected_objectives(generic: str, physics_aware: str) -> None:
    if generic not in objectives.GENERIC_FAMILY:
        raise ValueError(f"{generic} is not in the declared generic family")
    if physics_aware not in objectives.PHYSICS_AWARE_FAMILY:
        raise ValueError(f"{physics_aware} is not in the declared physics-aware family")
