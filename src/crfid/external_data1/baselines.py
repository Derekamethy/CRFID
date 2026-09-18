"""Frozen Data1 majority-class context baseline."""

from __future__ import annotations

from collections import Counter

import numpy as np

from ..exceptions import ProtocolViolation
from .schema import CLASS_ORDER


FORMAL_NAME = "D1_TRAIN_MAJORITY_BASELINE_TIE_BROKEN_BY_FROZEN_CLASS_ORDER"
SELECTION_REASON = "FIRST_TIED_CLASS_IN_FROZEN_GLOBAL_CLASS_ORDER"


def majority_tie_decision(train_labels: np.ndarray) -> dict[str, object]:
    values = np.asarray(train_labels, dtype=np.int64)
    counts = Counter(int(value) for value in values)
    if set(counts).difference(CLASS_ORDER) or not len(values):
        raise ValueError("Majority baseline requires non-empty four-class training labels")
    maximum = max(counts.values())
    tied = [label for label in CLASS_ORDER if counts[label] == maximum]
    selected = tied[0]
    if [counts[label] for label in CLASS_ORDER] != [1400, 1400, 1400, 1400]:
        raise ProtocolViolation("Frozen Data1 training class counts changed")
    if tied != list(CLASS_ORDER) or selected != 0:
        raise ProtocolViolation("Frozen majority tie decision changed")
    return {
        "formal_name": FORMAL_NAME,
        "training_class_counts": [counts[label] for label in CLASS_ORDER],
        "maximum_training_count": maximum,
        "tied_labels": tied,
        "frozen_global_class_order": list(CLASS_ORDER),
        "selected_class": selected,
        "selection_reason": SELECTION_REASON,
        "test_information_used": False,
    }


def predict_majority(sample_count: int, selected_class: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if sample_count <= 0 or selected_class not in CLASS_ORDER:
        raise ValueError("Invalid majority prediction request")
    predictions = np.full(sample_count, selected_class, dtype=np.int64)
    scores = np.zeros((sample_count, len(CLASS_ORDER)), dtype=np.float64)
    scores[:, selected_class] = 1.0
    return predictions, scores

