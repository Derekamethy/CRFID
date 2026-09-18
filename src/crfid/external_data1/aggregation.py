"""Four-class metric and equal-seed aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..evaluation.metrics import classification_metrics
from .schema import TEST_SETS


def metric_bundle(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    shared = classification_metrics(truth, prediction, class_count=4)
    confusion = np.asarray(shared["confusion_matrix"], dtype=np.int64)
    support = confusion.sum(axis=1)
    return {
        "labels": [0, 1, 2, 3],
        "sample_count": int(confusion.sum()),
        "accuracy": shared["accuracy"],
        "macro_f1": shared["macro_f1"],
        "balanced_accuracy": float(np.mean(shared["per_class_recall"])),
        "per_class_precision": {
            str(index): float(value)
            for index, value in enumerate(shared["per_class_precision"])
        },
        "per_class_recall": {
            str(index): float(value)
            for index, value in enumerate(shared["per_class_recall"])
        },
        "per_class_f1": {
            str(index): float(value)
            for index, value in enumerate(shared["per_class_f1"])
        },
        "support": {str(index): int(value) for index, value in enumerate(support)},
        "confusion_matrix": confusion.tolist(),
        "zero_division": 0,
    }


def grouped_metrics(
    truth: np.ndarray, prediction: np.ndarray, groups: Sequence[str]
) -> dict[str, dict[str, Any]]:
    group_array = np.asarray(groups)
    return {
        group: metric_bundle(truth[group_array == group], prediction[group_array == group])
        for group in TEST_SETS
    }


def summarize_values(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("Cannot summarize empty values")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "sample_standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "population_standard_deviation": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def summarize_cnn_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        metric: summarize_values([float(run["metrics"][metric]) for run in runs])
        for metric in ("accuracy", "macro_f1", "balanced_accuracy")
    } | {
        "per_class_recall": {
            str(label): summarize_values(
                [
                    float(run["metrics"]["per_class_recall"][str(label)])
                    for run in runs
                ]
            )
            for label in range(4)
        },
        "seeds": [int(run["seed"]) for run in runs],
    }


def prediction_mismatch_count(
    reproduced: np.ndarray, authoritative: np.ndarray
) -> int:
    left = np.asarray(reproduced, dtype=np.int64)
    right = np.asarray(authoritative, dtype=np.int64)
    if left.shape != right.shape:
        raise ValueError("Prediction comparison arrays are not aligned")
    return int(np.count_nonzero(left != right))
