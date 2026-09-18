"""Deterministic sample, condition-block, and duplicate-aware metrics."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np


CLASS_COUNT = 7


def confusion_matrix(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray | None = None,
    class_count: int = CLASS_COUNT,
) -> np.ndarray:
    true_labels = np.asarray(true_labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if true_labels.shape != predictions.shape:
        raise ValueError("True-label and prediction shapes differ")
    if np.any((true_labels < 0) | (true_labels >= class_count)):
        raise ValueError("True labels outside declared class range")
    if np.any((predictions < 0) | (predictions >= class_count)):
        raise ValueError("Predictions outside declared class range")
    if weights is None:
        matrix = np.zeros((class_count, class_count), dtype=np.int64)
        np.add.at(matrix, (true_labels, predictions), 1)
    else:
        supplied = np.asarray(weights, dtype=np.float64)
        if supplied.shape != true_labels.shape or np.any(supplied <= 0):
            raise ValueError("Metric weights must be positive and shape-aligned")
        matrix = np.zeros((class_count, class_count), dtype=np.float64)
        np.add.at(matrix, (true_labels, predictions), supplied)
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> dict:
    matrix = np.asarray(matrix)
    total = float(matrix.sum())
    true_support = matrix.sum(axis=1).astype(np.float64)
    predicted_support = matrix.sum(axis=0).astype(np.float64)
    true_positive = np.diag(matrix).astype(np.float64)
    recall = np.divide(
        true_positive,
        true_support,
        out=np.zeros_like(true_positive),
        where=true_support > 0,
    )
    f1_denominator = 2.0 * true_positive + (predicted_support - true_positive) + (
        true_support - true_positive
    )
    per_class_f1 = np.divide(
        2.0 * true_positive,
        f1_denominator,
        out=np.zeros_like(true_positive),
        where=f1_denominator > 0,
    )
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_f1": float(per_class_f1.mean()),
        "per_class_recall": [float(value) for value in recall],
        "per_class_f1": [float(value) for value in per_class_f1],
        "zero_recall_class_count": int(np.count_nonzero(recall == 0.0)),
        "worst_class_recall": float(recall.min()),
        "confusion_matrix": matrix.tolist(),
        "total_weight": total,
    }


def classification_metrics(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict:
    return metrics_from_confusion(confusion_matrix(true_labels, predictions, weights))


def aggregate_condition_blocks(
    true_labels: np.ndarray,
    logits: np.ndarray,
    condition_ids: list[str] | np.ndarray,
) -> dict:
    true_labels = np.asarray(true_labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)
    if logits.shape != (len(true_labels), CLASS_COUNT):
        raise ValueError("Condition aggregation requires [N, 7] logits")
    if len(condition_ids) != len(true_labels):
        raise ValueError("Condition IDs are not sample-aligned")
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for index, condition_id in enumerate(condition_ids):
        groups.setdefault(str(condition_id), []).append(index)

    rows = []
    block_true = []
    block_predicted = []
    for condition_id, indices in groups.items():
        labels = np.unique(true_labels[indices])
        if len(labels) != 1:
            raise ValueError(f"Condition block has multiple labels: {condition_id}")
        mean_logits = logits[indices].astype(np.float64).mean(axis=0)
        prediction = int(np.argmax(mean_logits))
        label = int(labels[0])
        rows.append(
            {
                "raw_condition_id": condition_id,
                "sample_count": len(indices),
                "true_label": label,
                "predicted_label": prediction,
                "mean_logits": [float(value) for value in mean_logits],
            }
        )
        block_true.append(label)
        block_predicted.append(prediction)
    metrics = classification_metrics(
        np.asarray(block_true, dtype=np.int64),
        np.asarray(block_predicted, dtype=np.int64),
    )
    return {"rows": rows, "metrics": metrics}


def evaluate_all_levels(
    true_labels: np.ndarray,
    logits: np.ndarray,
    condition_ids: list[str] | np.ndarray,
    unique_signal_weights: np.ndarray,
) -> dict:
    logits = np.asarray(logits, dtype=np.float32)
    predictions = np.argmax(logits, axis=1).astype(np.int64)
    sample = classification_metrics(true_labels, predictions)
    condition = aggregate_condition_blocks(true_labels, logits, condition_ids)
    unique_weighted = classification_metrics(
        true_labels, predictions, np.asarray(unique_signal_weights, dtype=np.float64)
    )
    return {
        "predictions": predictions,
        "sample": sample,
        "condition": condition,
        "unique_signal_weighted": unique_weighted,
    }
