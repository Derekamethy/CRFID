"""Row and majority-vote physical condition-block metrics."""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass

import numpy as np

from .constants import CLASS_ORDER, ROWS_PER_BLOCK
from .protocol import ProtocolViolation


def confusion_matrix(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    actual = np.asarray(truth, dtype=np.int64)
    predicted = np.asarray(prediction, dtype=np.int64)
    if actual.shape != predicted.shape or actual.ndim != 1:
        raise ValueError("truth and predictions must be aligned vectors")
    if np.any((actual < 0) | (actual >= 7)) or np.any((predicted < 0) | (predicted >= 7)):
        raise ValueError("class index outside the frozen class order")
    matrix = np.zeros((7, 7), dtype=np.int64)
    np.add.at(matrix, (actual, predicted), 1)
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, object]:
    values = np.asarray(matrix)
    actual = values.sum(axis=1).astype(np.float64)
    predicted = values.sum(axis=0).astype(np.float64)
    tp = np.diag(values).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros(7), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros(7), where=actual > 0)
    f1 = np.divide(2 * tp, actual + predicted, out=np.zeros(7), where=(actual + predicted) > 0)
    total = int(values.sum())
    return {
        "accuracy": float(tp.sum() / total) if total else 0.0,
        "macro_f1": float(f1.mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": values,
    }


def metrics_from_predictions(truth: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    return metrics_from_confusion(confusion_matrix(truth, prediction))


def majority_vote(predictions: np.ndarray) -> int:
    values = np.asarray(predictions, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("majority vote requires a nonempty prediction vector")
    counts = np.bincount(values, minlength=7)
    return int(np.argmax(counts))


@dataclass(frozen=True)
class QueryEvaluation:
    row_metrics: dict[str, object]
    block_metrics: dict[str, object]
    block_truth: np.ndarray
    block_prediction: np.ndarray
    block_ids: tuple[str, ...]
    block_agreement: np.ndarray
    predicted_histogram: np.ndarray


def evaluate_query(
    *, truth: np.ndarray, predictions: np.ndarray, block_ids: tuple[str, ...]
) -> QueryEvaluation:
    actual = np.asarray(truth, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    if actual.shape != predicted.shape or len(block_ids) != len(actual):
        raise ValueError("query truth, predictions and block IDs are not aligned")
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for index, block_id in enumerate(block_ids):
        groups.setdefault(block_id, []).append(index)
    block_truth: list[int] = []
    block_prediction: list[int] = []
    agreement: list[float] = []
    for block_id, indices in groups.items():
        if len(indices) != ROWS_PER_BLOCK:
            raise ProtocolViolation(f"query block {block_id} does not retain all 50 repetitions")
        labels = np.unique(actual[indices])
        if len(labels) != 1:
            raise ProtocolViolation("query condition block contains multiple TagID labels")
        vote = majority_vote(predicted[indices])
        block_truth.append(int(labels[0]))
        block_prediction.append(vote)
        agreement.append(float(np.mean(predicted[indices] == vote)))
    if len(groups) != 21:
        raise ProtocolViolation("outer-fold query must contain 21 physical blocks")
    block_actual = np.asarray(block_truth, dtype=np.int64)
    block_predicted = np.asarray(block_prediction, dtype=np.int64)
    return QueryEvaluation(
        row_metrics=metrics_from_predictions(actual, predicted),
        block_metrics=metrics_from_predictions(block_actual, block_predicted),
        block_truth=block_actual,
        block_prediction=block_predicted,
        block_ids=tuple(groups),
        block_agreement=np.asarray(agreement, dtype=np.float64),
        predicted_histogram=np.bincount(predicted, minlength=7),
    )


def assert_no_row_level_pseudoreplication(
    *, inference_unit: str, block_count: int, row_count: int
) -> None:
    if inference_unit != "condition_block" or block_count * ROWS_PER_BLOCK != row_count:
        raise ProtocolViolation("ROW_LEVEL_PSEUDOREPLICATION_PROHIBITED")
