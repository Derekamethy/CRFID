"""Class-collapse and confusion diagnostics."""

from __future__ import annotations

import numpy as np

from ..evaluation.metrics import confusion_matrix, metrics_from_confusion


def collapse_summary(truth: np.ndarray, prediction: np.ndarray, *, class_count: int) -> dict[str, object]:
    matrix = confusion_matrix(truth, prediction, class_count=class_count)
    metrics = metrics_from_confusion(matrix)
    histogram = np.bincount(np.asarray(prediction, dtype=np.int64), minlength=class_count)
    total = int(histogram.sum())
    return {
        "zero_recall_class_count": metrics["zero_recall_class_count"],
        "worst_class_recall": metrics["worst_class_recall"],
        "predicted_class_histogram": histogram.tolist(),
        "dominant_predicted_class_fraction": float(histogram.max() / total) if total else 0.0,
    }


def strongest_confusions(matrix: np.ndarray, limit: int = 5) -> list[dict[str, int]]:
    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Confusion matrix must be square")
    rows = []
    for actual in range(values.shape[0]):
        for predicted in range(values.shape[1]):
            if actual != predicted and values[actual, predicted] > 0:
                rows.append({"actual": actual, "predicted": predicted, "count": int(values[actual, predicted])})
    return sorted(rows, key=lambda row: (-row["count"], row["actual"], row["predicted"]))[:limit]
