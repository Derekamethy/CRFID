"""Sample, condition-block and duplicate-aware classification metrics."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np


def confusion_matrix(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    class_count: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    actual = np.asarray(truth, dtype=np.int64)
    predicted = np.asarray(prediction, dtype=np.int64)
    if actual.ndim != 1 or actual.shape != predicted.shape:
        raise ValueError("Truth and prediction must be aligned vectors")
    if np.any((actual < 0) | (actual >= class_count)) or np.any(
        (predicted < 0) | (predicted >= class_count)
    ):
        raise ValueError("Label outside declared class range")
    if weights is None:
        matrix = np.zeros((class_count, class_count), dtype=np.int64)
        np.add.at(matrix, (actual, predicted), 1)
    else:
        supplied = np.asarray(weights, dtype=np.float64)
        if supplied.shape != actual.shape or np.any(supplied <= 0):
            raise ValueError("Weights must be positive and aligned")
        matrix = np.zeros((class_count, class_count), dtype=np.float64)
        np.add.at(matrix, (actual, predicted), supplied)
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or np.any(values < 0):
        raise ValueError("Confusion matrix must be square and non-negative")
    true_positive = np.diag(values).astype(np.float64)
    true_support = values.sum(axis=1).astype(np.float64)
    predicted_support = values.sum(axis=0).astype(np.float64)
    precision = np.divide(true_positive, predicted_support, out=np.zeros_like(true_positive), where=predicted_support > 0)
    recall = np.divide(true_positive, true_support, out=np.zeros_like(true_positive), where=true_support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    total = float(values.sum())
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_f1": float(f1.mean()) if len(f1) else 0.0,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "worst_class_recall": float(recall.min()) if len(recall) else 0.0,
        "zero_recall_class_count": int(np.count_nonzero(recall == 0)),
        "confusion_matrix": values.tolist(),
        "total_weight": total,
    }


def classification_metrics(
    truth: np.ndarray, prediction: np.ndarray, *, class_count: int, weights: np.ndarray | None = None
) -> dict[str, Any]:
    return metrics_from_confusion(confusion_matrix(truth, prediction, class_count=class_count, weights=weights))


def condition_block_metrics(
    truth: np.ndarray, logits: np.ndarray, condition_ids: Sequence[str], *, class_count: int
) -> dict[str, Any]:
    actual = np.asarray(truth, dtype=np.int64)
    scores = np.asarray(logits, dtype=np.float32)
    if scores.shape != (len(actual), class_count) or len(condition_ids) != len(actual):
        raise ValueError("Condition metric arrays are not aligned")
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for index, condition in enumerate(condition_ids):
        groups.setdefault(str(condition), []).append(index)
    block_truth: list[int] = []
    block_prediction: list[int] = []
    for condition, indices in groups.items():
        labels = np.unique(actual[indices])
        if len(labels) != 1:
            raise ValueError(f"Condition has multiple labels: {condition}")
        block_truth.append(int(labels[0]))
        block_prediction.append(int(np.argmax(scores[indices].astype(np.float64).mean(axis=0))))
    return classification_metrics(
        np.asarray(block_truth), np.asarray(block_prediction), class_count=class_count
    )
