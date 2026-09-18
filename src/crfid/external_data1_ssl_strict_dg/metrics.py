"""Explicit classification metrics with documented zero-division handling.

Every metric is computed over the fixed label order ``0..6``, including classes absent
from a particular evaluation population, so denominators never change silently between
treatments. ``zero_division=0`` is applied to precision, recall and F1 exactly as
``sklearn.metrics.f1_score(average="macro", zero_division=0)`` does; a unit test pins
this equivalence.

Standard deviations reported by this branch are **population** (``ddof=0``) unless a
field name says ``sample``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CLASS_ORDER = tuple(range(7))
ZERO_DIVISION_POLICY = "precision_recall_f1_set_to_zero_when_denominator_is_zero"
STANDARD_DEVIATION_POLICY = "population_ddof_0"


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    truth = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("Labels and predictions must be aligned one-dimensional arrays")
    size = len(CLASS_ORDER)
    if truth.size and (truth.min() < 0 or truth.max() >= size):
        raise ValueError("Label outside the fixed class order")
    if predicted.size and (predicted.min() < 0 or predicted.max() >= size):
        raise ValueError("Prediction outside the fixed class order")
    matrix = np.zeros((size, size), dtype=np.int64)
    np.add.at(matrix, (truth, predicted), 1)
    return matrix


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numerator, dtype=np.float64)
    valid = denominator > 0
    out[valid] = numerator[valid] / denominator[valid]
    return out


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    """Accuracy, macro-F1, per-class recall/precision/F1 and diagnostics."""

    matrix = confusion_matrix(labels, predictions)
    total = int(matrix.sum())
    if total == 0:
        raise ValueError("Cannot score an empty evaluation population")
    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted_count = matrix.sum(axis=0).astype(np.float64)
    recall = _safe_divide(true_positive, support)
    precision = _safe_divide(true_positive, predicted_count)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    counts = predicted_count / float(total)
    entropy_terms = np.where(counts > 0, -counts * np.log(np.maximum(counts, 1e-300)), 0.0)
    return {
        "sample_count": total,
        "accuracy": float(true_positive.sum() / total),
        "macro_f1": float(f1.mean()),
        "per_class_recall": [float(value) for value in recall],
        "per_class_precision": [float(value) for value in precision],
        "per_class_f1": [float(value) for value in f1],
        "per_class_support": [int(value) for value in support],
        "worst_class_recall": float(recall.min()),
        "zero_recall_class_count": int((recall == 0.0).sum()),
        "predicted_class_counts": [int(value) for value in predicted_count],
        "dominant_predicted_class_fraction": float(counts.max()),
        "predicted_distribution_entropy": float(entropy_terms.sum()),
        "class_collapse_count": int((predicted_count == 0).sum()),
        "label_order": list(CLASS_ORDER),
        "zero_division_policy": ZERO_DIVISION_POLICY,
        "confusion_matrix": matrix.tolist(),
    }


def prediction_entropy(probabilities: np.ndarray) -> float:
    """Mean predictive entropy in nats over the evaluated population."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CLASS_ORDER):
        raise ValueError("Probabilities must be [N,7]")
    clipped = np.clip(values, 1e-300, 1.0)
    return float((-(clipped * np.log(clipped)).sum(axis=1)).mean())


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def argmax_predictions(logits: np.ndarray) -> np.ndarray:
    """Lowest-index tie-breaking argmax, matching the canonical prediction rule."""

    return np.asarray(logits, dtype=np.float32).argmax(axis=1).astype(np.int64)


def aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "population_standard_deviation": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "count": int(array.size),
    }


def paired_difference(treatment: list[float], control: list[float]) -> dict[str, Any]:
    """Seed-paired difference summary used by every hypothesis in this branch."""

    first = np.asarray(treatment, dtype=np.float64)
    second = np.asarray(control, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1 or first.size == 0:
        raise ValueError("Paired comparison requires aligned non-empty vectors")
    differences = first - second
    return {
        "per_seed_difference": [float(value) for value in differences],
        "mean_difference": float(differences.mean()),
        "population_standard_deviation": float(differences.std(ddof=0)),
        "improved_seed_count": int((differences > 0).sum()),
        "unchanged_seed_count": int((differences == 0).sum()),
        "degraded_seed_count": int((differences < 0).sum()),
        "seed_count": int(differences.size),
        "minimum_difference": float(differences.min()),
        "maximum_difference": float(differences.max()),
    }
