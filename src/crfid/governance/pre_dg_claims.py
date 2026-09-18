"""Permitted and prohibited claim boundaries for Pre-DG evidence."""

from __future__ import annotations


EVIDENCE_LEVELS = {
    "MODEL_RETRAINING_REPRODUCTION",
    "EXACT_ARTIFACT_MATCH",
    "TOLERANCE_MATCH",
}

PERMITTED_CLAIM = (
    "The final Stage1-5 single-dataset baseline is learnable under its frozen "
    "non-strict grouped-random and leave-domain protocols."
)

PROHIBITED_LABELS = {
    "strict DG",
    "unseen-position evaluation",
    "target adaptation",
    "external validation",
    "fair zero-shot DG",
}


def validate_pre_dg_claim(text: str) -> None:
    lowered = text.lower()
    for label in PROHIBITED_LABELS:
        if label.lower() in lowered:
            raise ValueError(f"Prohibited Pre-DG claim label: {label}")
