"""Paired TagID-stratified physical-block uncertainty and seed sensitivities."""

from __future__ import annotations

import numpy as np


def stratified_block_bootstrap_indices(
    truth: np.ndarray, *, replicates: int, seed: int
) -> np.ndarray:
    labels = np.asarray(truth, dtype=np.int64)
    if labels.ndim != 1 or len(labels) != 63:
        raise ValueError("bootstrap requires the 63 physical P4 blocks")
    groups = [np.flatnonzero(labels == class_index) for class_index in range(7)]
    if any(len(group) != 9 for group in groups):
        raise ValueError("bootstrap requires nine blocks within each TagID")
    rng = np.random.default_rng(seed)
    sampled = [rng.choice(group, size=(replicates, 9), replace=True) for group in groups]
    return np.concatenate(sampled, axis=1).astype(np.int64)


def _bootstrap_metric_by_unit(
    truth: np.ndarray,
    predictions: np.ndarray,
    sampled_indices: np.ndarray,
    *,
    metric: str,
    chunk_size: int = 200,
) -> np.ndarray:
    actual = np.asarray(truth, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    if predicted.ndim != 2 or predicted.shape[1] != len(actual):
        raise ValueError("prediction matrix must be [paired_seed_unit, block]")
    output = np.empty((len(sampled_indices), len(predicted)), dtype=np.float64)
    for start in range(0, len(sampled_indices), chunk_size):
        sample = sampled_indices[start : start + chunk_size]
        sampled_truth = actual[sample]
        sampled_prediction = predicted[:, sample]
        if metric == "accuracy":
            values = np.mean(
                sampled_prediction == sampled_truth[None, :, :], axis=2
            ).T
        elif metric == "macro_f1":
            macro = np.zeros((len(predicted), len(sample)), dtype=np.float64)
            for class_index in range(7):
                actual_count = np.sum(sampled_truth == class_index, axis=1)[None, :]
                predicted_count = np.sum(
                    sampled_prediction == class_index, axis=2
                )
                true_positive = np.sum(
                    (sampled_prediction == class_index)
                    & (sampled_truth[None, :, :] == class_index),
                    axis=2,
                )
                denominator = actual_count + predicted_count
                macro += np.divide(
                    2.0 * true_positive,
                    denominator,
                    out=np.zeros_like(true_positive, dtype=np.float64),
                    where=denominator > 0,
                ) / 7.0
            values = macro.T
        else:
            raise ValueError("metric must be macro_f1 or accuracy")
        output[start : start + len(sample)] = values
    return output


def bootstrap_metric_by_unit(
    truth: np.ndarray,
    predictions: np.ndarray,
    sampled_indices: np.ndarray,
    *,
    metric: str,
) -> np.ndarray:
    """Return one bootstrapped metric per replicate and paired seed unit."""

    return _bootstrap_metric_by_unit(
        truth, predictions, sampled_indices, metric=metric
    )


def paired_bootstrap_effects(
    *,
    truth: np.ndarray,
    baseline_predictions: np.ndarray,
    comparison_predictions: np.ndarray,
    sampled_indices: np.ndarray,
    metric: str,
) -> np.ndarray:
    baseline = _bootstrap_metric_by_unit(
        truth, baseline_predictions, sampled_indices, metric=metric
    )
    comparison = _bootstrap_metric_by_unit(
        truth, comparison_predictions, sampled_indices, metric=metric
    )
    return comparison - baseline


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    vector = np.asarray(values, dtype=np.float64)
    return float(np.percentile(vector, 2.5)), float(np.percentile(vector, 97.5))


def seed_resampling_sensitivity(
    effects_by_replicate_and_unit: np.ndarray, *, seed: int
) -> dict[str, np.ndarray]:
    effects = np.asarray(effects_by_replicate_and_unit, dtype=np.float64)
    if effects.ndim != 2 or effects.shape[1] != 25:
        raise ValueError("seed sensitivity requires 25 source-seed x support-seed units")
    cube = effects.reshape(len(effects), 5, 5)
    rng = np.random.default_rng(seed)
    source_draws = rng.integers(0, 5, size=(len(effects), 5))
    support_draws = rng.integers(0, 5, size=(len(effects), 5))
    block_only = cube.mean(axis=(1, 2))
    block_source = np.empty(len(effects), dtype=np.float64)
    block_support = np.empty(len(effects), dtype=np.float64)
    block_both = np.empty(len(effects), dtype=np.float64)
    for index in range(len(effects)):
        source_selected = cube[index, source_draws[index], :]
        support_selected = cube[index, :, support_draws[index]]
        both_selected = cube[index][np.ix_(source_draws[index], support_draws[index])]
        block_source[index] = source_selected.mean()
        block_support[index] = support_selected.mean()
        block_both[index] = both_selected.mean()
    return {
        "TAGID_STRATIFIED_BLOCK": block_only,
        "BLOCK_PLUS_SOURCE_SEED": block_source,
        "BLOCK_PLUS_SUPPORT_SEED": block_support,
        "BLOCK_PLUS_BOTH_SEEDS": block_both,
    }


def fold_aggregate_sensitivity(
    fold_effect_by_unit: np.ndarray, *, replicates: int, seed: int
) -> np.ndarray:
    values = np.asarray(fold_effect_by_unit, dtype=np.float64)
    if values.shape != (25, 3):
        raise ValueError("fold sensitivity requires [25 paired units, 3 folds]")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, 3, size=(replicates, 3))
    output = np.empty(replicates, dtype=np.float64)
    for index, folds in enumerate(sampled):
        output[index] = values[:, folds].mean()
    return output
