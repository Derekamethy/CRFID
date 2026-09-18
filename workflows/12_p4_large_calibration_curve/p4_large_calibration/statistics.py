"""Metrics and crossed hierarchical physical-block bootstrap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from p4_factor_aware.metrics import confusion_matrix, metrics_from_confusion

from .constants import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, TOTAL_BUDGETS


def extended_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    metrics = metrics_from_confusion(confusion_matrix(truth, prediction))
    recall = np.asarray(metrics["recall"], dtype=np.float64)
    f1 = np.asarray(metrics["f1"], dtype=np.float64)
    return {
        **metrics,
        "balanced_accuracy": float(recall.mean()),
        "worst_class_f1": float(f1.min()),
    }


def block_layout(data: object) -> tuple[list[tuple[int, int, int]], list[np.ndarray], np.ndarray]:
    keys = sorted(data._groups)
    indices = [np.asarray(data._groups[key], dtype=np.int64) for key in keys]
    truth = np.asarray([key[0] for key in keys], dtype=np.int64)
    if len(keys) != 63 or any(len(value) != 50 for value in indices):
        raise ValueError("expected 63 complete 50-row condition blocks")
    return keys, indices, truth


def prediction_histograms_by_block(
    predictions: np.ndarray, block_indices: list[np.ndarray]
) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.int64)
    if values.ndim != 4:
        raise ValueError("predictions must be [budget, source, support, row]")
    output = np.zeros((*values.shape[:3], len(block_indices), 7), dtype=np.int16)
    for block_index, indices in enumerate(block_indices):
        selected = values[..., indices]
        for class_index in range(7):
            output[..., block_index, class_index] = np.sum(
                selected == class_index, axis=-1
            )
    if np.any(output.sum(axis=-1) != 50):
        raise ValueError("block prediction histograms do not retain all 50 rows")
    return output


def majority_predictions(histograms: np.ndarray) -> np.ndarray:
    return np.argmax(np.asarray(histograms), axis=-1).astype(np.int64)


def _bootstrap_draws(
    truth_by_block: np.ndarray,
    *,
    replicates: int,
    source_count: int,
    support_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = np.asarray(truth_by_block, dtype=np.int64)
    if truth.shape != (63,):
        raise ValueError("bootstrap requires 63 physical blocks")
    groups = [np.flatnonzero(truth == class_index) for class_index in range(7)]
    if any(len(group) != 9 for group in groups):
        raise ValueError("bootstrap requires nine blocks per TagID")
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, 63), dtype=np.uint8)
    replicate_index = np.arange(replicates)
    for group in groups:
        draws = rng.choice(group, size=(replicates, 9), replace=True)
        for column in range(9):
            np.add.at(weights, (replicate_index, draws[:, column]), 1)

    source_draws = rng.integers(0, source_count, size=(replicates, source_count))
    source_counts = np.zeros((replicates, source_count), dtype=np.uint8)
    for column in range(source_count):
        np.add.at(source_counts, (replicate_index, source_draws[:, column]), 1)

    support_draws = rng.integers(0, support_count, size=(replicates, support_count))
    support_counts = np.zeros((replicates, support_count), dtype=np.uint8)
    for column in range(support_count):
        np.add.at(support_counts, (replicate_index, support_draws[:, column]), 1)
    return weights, source_counts, support_counts


@dataclass(frozen=True)
class BootstrapResult:
    macro_f1: np.ndarray
    accuracy: np.ndarray
    block_weights_sha256_payload: np.ndarray
    source_counts: np.ndarray
    support_counts: np.ndarray


def hierarchical_bootstrap(
    block_histograms: np.ndarray,
    truth_by_block: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    chunk_size: int = 250,
) -> BootstrapResult:
    histograms = np.asarray(block_histograms)
    expected = (len(TOTAL_BUDGETS), 5, 20, 63, 7)
    if histograms.shape != expected:
        raise ValueError(f"block histograms must have shape {expected}")
    if replicates <= 0:
        raise ValueError("bootstrap replicate count must be positive")
    weights, source_counts, support_counts = _bootstrap_draws(
        truth_by_block,
        replicates=replicates,
        source_count=5,
        support_count=20,
        seed=seed,
    )

    # Flatten as block x (budget, source, support, predicted class) so each
    # bootstrap chunk becomes two dense, deterministic matrix products.
    flat_hist = (
        histograms.transpose(3, 0, 1, 2, 4)
        .reshape(63, -1)
        .astype(np.float64, copy=False)
    )
    one_hot_truth = np.eye(7, dtype=np.int16)[np.asarray(truth_by_block, dtype=np.int64)]
    true_positive_hist = histograms * one_hot_truth[None, None, None, :, :]
    flat_true_positive = (
        true_positive_hist.transpose(3, 0, 1, 2, 4)
        .reshape(63, -1)
        .astype(np.float64, copy=False)
    )

    macro_samples = np.empty((replicates, len(TOTAL_BUDGETS)), dtype=np.float64)
    accuracy_samples = np.empty_like(macro_samples)
    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        block_weight = weights[start:stop].astype(np.float64, copy=False)
        shape = (stop - start, len(TOTAL_BUDGETS), 5, 20, 7)
        predicted_count = (block_weight @ flat_hist).reshape(shape)
        true_positive = (block_weight @ flat_true_positive).reshape(shape)
        f1 = np.divide(
            2.0 * true_positive,
            450.0 + predicted_count,
            out=np.zeros_like(true_positive),
            where=(450.0 + predicted_count) > 0,
        )
        unit_macro = f1.mean(axis=-1)
        unit_accuracy = true_positive.sum(axis=-1) / 3150.0
        source_weight = source_counts[start:stop].astype(np.float64, copy=False)
        support_weight = support_counts[start:stop].astype(np.float64, copy=False)
        macro_samples[start:stop] = np.einsum(
            "rs,rt,rkst->rk",
            source_weight,
            support_weight,
            unit_macro,
            optimize=True,
        ) / 100.0
        accuracy_samples[start:stop] = np.einsum(
            "rs,rt,rkst->rk",
            source_weight,
            support_weight,
            unit_accuracy,
            optimize=True,
        ) / 100.0
    return BootstrapResult(
        macro_f1=macro_samples,
        accuracy=accuracy_samples,
        block_weights_sha256_payload=weights,
        source_counts=source_counts,
        support_counts=support_counts,
    )


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    vector = np.asarray(values, dtype=np.float64)
    return float(np.percentile(vector, 2.5)), float(np.percentile(vector, 97.5))


def aggregate_metric_from_histograms(histograms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(histograms, dtype=np.float64)
    if values.shape != (len(TOTAL_BUDGETS), 5, 20, 63, 7):
        raise ValueError("unexpected histogram shape")
    predicted = values.sum(axis=3)
    truth = np.repeat(np.arange(7), 9)
    one_hot = np.eye(7)[truth]
    true_positive = (values * one_hot[None, None, None, :, :]).sum(axis=3)
    f1 = np.divide(
        2.0 * true_positive,
        450.0 + predicted,
        out=np.zeros_like(true_positive),
        where=(450.0 + predicted) > 0,
    )
    macro = f1.mean(axis=-1)
    accuracy = true_positive.sum(axis=-1) / 3150.0
    return macro, accuracy
