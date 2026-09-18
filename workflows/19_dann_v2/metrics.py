"""Block-aware metrics, independent probes, diagnostics, and uncertainty."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, silhouette_score
from torch import nn


def classification_metrics(
    true: np.ndarray, predicted: np.ndarray, *, class_count: int
) -> dict:
    true = np.asarray(true, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    matrix = confusion_matrix(true, predicted, labels=np.arange(class_count)).astype(np.int64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted_support = matrix.sum(axis=0).astype(np.float64)
    tp = np.diag(matrix).astype(np.float64)
    precision = np.divide(tp, predicted_support, out=np.zeros_like(tp), where=predicted_support > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1_denominator = precision + recall
    per_class_f1 = np.divide(
        2.0 * precision * recall,
        f1_denominator,
        out=np.zeros_like(tp),
        where=f1_denominator > 0,
    )
    return {
        "accuracy": float(tp.sum() / matrix.sum()) if matrix.sum() else 0.0,
        "macro_f1": float(per_class_f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def deterministic_majority_vote(predictions: np.ndarray, class_count: int = 7) -> int:
    counts = np.bincount(np.asarray(predictions, dtype=np.int64), minlength=class_count)
    return int(np.flatnonzero(counts == counts.max())[0])


def aggregate_blocks(
    true: np.ndarray,
    predicted: np.ndarray,
    block_ids: np.ndarray,
    *,
    class_count: int = 7,
) -> dict:
    true = np.asarray(true, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    block_ids = np.asarray(block_ids, dtype=str)
    if not (true.shape == predicted.shape == block_ids.shape):
        raise ValueError("Block inputs are not aligned")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, block in enumerate(block_ids):
        groups[str(block)].append(index)
    rows = []
    block_true = []
    block_predicted = []
    for block, indices in sorted(groups.items()):
        labels = np.unique(true[indices])
        if len(labels) != 1:
            raise RuntimeError(f"Block {block} has multiple TagID labels")
        vote = deterministic_majority_vote(predicted[indices], class_count)
        label = int(labels[0])
        agreement = float(np.mean(predicted[indices] == vote))
        rows.append(
            {
                "block_id": block,
                "true_label": label,
                "predicted_label": vote,
                "sample_count": len(indices),
                "within_block_agreement": agreement,
            }
        )
        block_true.append(label)
        block_predicted.append(vote)
    return {
        "rows": rows,
        "metrics": classification_metrics(
            np.asarray(block_true), np.asarray(block_predicted), class_count=class_count
        ),
        "block_true": np.asarray(block_true, dtype=np.int64),
        "block_predicted": np.asarray(block_predicted, dtype=np.int64),
    }


def condition_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
    condition_ids: np.ndarray,
) -> dict:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, condition in enumerate(np.asarray(condition_ids, dtype=str)):
        groups[str(condition)].append(index)
    values = []
    output_labels = []
    output_positions = []
    output_conditions = []
    for condition, indices in sorted(groups.items()):
        local_labels = np.unique(labels[indices])
        local_positions = np.unique(positions[indices])
        if len(local_labels) != 1 or len(local_positions) != 1:
            raise RuntimeError("Condition-block centroid custody failed")
        values.append(embeddings[indices].mean(axis=0))
        output_labels.append(int(local_labels[0]))
        output_positions.append(str(local_positions[0]))
        output_conditions.append(condition)
    return {
        "values": np.ascontiguousarray(np.vstack(values), dtype=np.float64),
        "labels": np.asarray(output_labels, dtype=np.int64),
        "positions": np.asarray(output_positions, dtype=str),
        "condition_ids": np.asarray(output_conditions, dtype=str),
    }


def fit_linear_probe(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    test_values: np.ndarray,
    test_labels: np.ndarray,
    *,
    class_count: int,
    seed: int,
    learning_rate: float = 0.01,
    weight_decay: float = 0.0001,
    maximum_epochs: int = 200,
) -> dict:
    """Fixed multinomial probe with train-only scaling and no model-family tuning."""

    train_values = np.asarray(train_values, dtype=np.float64)
    test_values = np.asarray(test_values, dtype=np.float64)
    mean = train_values.mean(axis=0)
    scale = train_values.std(axis=0, ddof=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    train = np.ascontiguousarray(((train_values - mean) / scale).astype(np.float32))
    test = np.ascontiguousarray(((test_values - mean) / scale).astype(np.float32))
    torch.manual_seed(int(seed) + 4_000_001)
    model = nn.Linear(train.shape[1], int(class_count))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    criterion = nn.CrossEntropyLoss()
    tensor = torch.from_numpy(train)
    labels = torch.from_numpy(np.asarray(train_labels, dtype=np.int64))
    generator = torch.Generator(device="cpu")
    for epoch in range(1, int(maximum_epochs) + 1):
        generator.manual_seed(int(seed) * 1_000_000 + epoch)
        order = torch.randperm(len(tensor), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(tensor[order]), labels[order])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        predictions = model(torch.from_numpy(test)).argmax(dim=1).numpy().astype(np.int64)
    true = np.asarray(test_labels, dtype=np.int64)
    return {
        "seed": int(seed),
        "balanced_accuracy": float(balanced_accuracy_score(true, predictions)),
        "macro_f1": float(f1_score(true, predictions, labels=np.arange(class_count), average="macro", zero_division=0)),
        "accuracy": float(np.mean(true == predictions)),
        "chance_balanced_accuracy": 1.0 / float(class_count),
        "predictions": predictions,
        "true_labels": true,
        "scaling_fit_on_probe_train_only": True,
        "selection_rule": "fixed_final_epoch",
    }


def independent_probes(
    centroids: dict,
    train_conditions: set[str],
    test_conditions: set[str],
    *,
    probe_seeds: tuple[int, ...],
) -> dict:
    conditions = centroids["condition_ids"]
    train_mask = np.asarray([value in train_conditions for value in conditions])
    test_mask = np.asarray([value in test_conditions for value in conditions])
    if np.any(train_mask & test_mask) or not np.all(train_mask | test_mask):
        raise RuntimeError("Probe condition blocks are not a disjoint complete split")
    position_order = tuple(sorted(np.unique(centroids["positions"]).tolist()))
    position_map = {value: index for index, value in enumerate(position_order)}
    position_labels = np.asarray([position_map[value] for value in centroids["positions"]])
    if set(position_labels[train_mask]) != set(range(len(position_order))) or set(position_labels[test_mask]) != set(range(len(position_order))):
        raise RuntimeError("Every source position must occur in probe train and test")
    position_results = []
    tag_results = []
    for seed in probe_seeds:
        position_results.append(
            fit_linear_probe(
                centroids["values"][train_mask],
                position_labels[train_mask],
                centroids["values"][test_mask],
                position_labels[test_mask],
                class_count=len(position_order),
                seed=seed,
            )
        )
        tag_results.append(
            fit_linear_probe(
                centroids["values"][train_mask],
                centroids["labels"][train_mask],
                centroids["values"][test_mask],
                centroids["labels"][test_mask],
                class_count=7,
                seed=seed,
            )
        )
    return {
        "position_order": position_order,
        "position": position_results,
        "tagid": tag_results,
        "position_balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in position_results])),
        "position_macro_f1_mean": float(np.mean([row["macro_f1"] for row in position_results])),
        "tagid_macro_f1_mean": float(np.mean([row["macro_f1"] for row in tag_results])),
        "position_chance": 1.0 / len(position_order),
        "condition_block_disjoint": True,
    }


def representation_diagnostics(centroids: dict) -> dict:
    values = np.asarray(centroids["values"], dtype=np.float64)
    labels = np.asarray(centroids["labels"], dtype=np.int64)
    positions = np.asarray(centroids["positions"], dtype=str)
    norms = np.linalg.norm(values, axis=1)
    position_silhouette = float(silhouette_score(values, positions))
    tag_silhouette = float(silhouette_score(values, labels))
    same_tag_cross_position = []
    different_tag = []
    same_position_cross_tag = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            distance = float(np.linalg.norm(values[left] - values[right]))
            if labels[left] == labels[right] and positions[left] != positions[right]:
                same_tag_cross_position.append(distance)
            if labels[left] != labels[right]:
                different_tag.append(distance)
                if positions[left] == positions[right]:
                    same_position_cross_tag.append(distance)
    within_class_cross_position = float(np.mean(same_tag_cross_position))
    between_class = float(np.mean(different_tag))
    same_position_between_class = float(np.mean(same_position_cross_tag))
    return {
        "position_silhouette": position_silhouette,
        "tagid_silhouette": tag_silhouette,
        "embedding_norm_mean": float(norms.mean()),
        "embedding_norm_std": float(norms.std(ddof=0)),
        "embedding_norm_q05": float(np.quantile(norms, 0.05)),
        "embedding_norm_q50": float(np.quantile(norms, 0.50)),
        "embedding_norm_q95": float(np.quantile(norms, 0.95)),
        "within_class_cross_position_distance": within_class_cross_position,
        "between_class_distance": between_class,
        "position_vs_tagid_separability_ratio": within_class_cross_position / same_position_between_class,
    }


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    result = np.quantile(np.asarray(values, dtype=np.float64), [0.025, 0.975])
    return float(result[0]), float(result[1])


def stratified_paired_block_bootstrap(
    erm_predictions: np.ndarray,
    dann_predictions: np.ndarray,
    true_labels: np.ndarray,
    *,
    replicates: int = 10_000,
    seed: int = 20_260_809,
    resample_training_seeds: bool = False,
    metric: str = "macro_f1",
) -> dict:
    """TagID-stratified block bootstrap preserving method and seed pairing."""

    erm = np.asarray(erm_predictions, dtype=np.int64)
    dann = np.asarray(dann_predictions, dtype=np.int64)
    true = np.asarray(true_labels, dtype=np.int64)
    if erm.shape != dann.shape or erm.ndim != 2 or erm.shape[1] != len(true):
        raise ValueError("Expected paired [training_seed, block] predictions")
    strata = [np.flatnonzero(true == tag) for tag in range(7)]
    if any(len(indices) == 0 for indices in strata):
        raise ValueError("Every TagID stratum is required")
    if metric not in {"macro_f1", "accuracy"}:
        raise ValueError("Unsupported paired block-bootstrap metric")
    generator = np.random.default_rng(int(seed))

    def score(labels: np.ndarray, predictions: np.ndarray) -> float:
        if metric == "accuracy":
            return float(np.mean(labels == predictions))
        return classification_metrics(labels, predictions, class_count=7)["macro_f1"]

    contrasts = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        sampled_blocks = np.concatenate(
            [generator.choice(indices, size=len(indices), replace=True) for indices in strata]
        )
        sampled_seeds = (
            generator.integers(0, erm.shape[0], size=erm.shape[0])
            if resample_training_seeds
            else np.arange(erm.shape[0])
        )
        per_seed = []
        sampled_true = true[sampled_blocks]
        for training_seed in sampled_seeds:
            erm_metric = score(sampled_true, erm[training_seed, sampled_blocks])
            dann_metric = score(sampled_true, dann[training_seed, sampled_blocks])
            per_seed.append(dann_metric - erm_metric)
        contrasts[replicate] = float(np.mean(per_seed))
    seedwise = []
    for training_seed in range(erm.shape[0]):
        erm_metric = score(true, erm[training_seed])
        dann_metric = score(true, dann[training_seed])
        seedwise.append(dann_metric - erm_metric)
    return {
        "point_estimate": float(np.mean(seedwise)),
        "interval": percentile_interval(contrasts),
        "seedwise_contrasts": seedwise,
        "replicates": int(replicates),
        "resample_training_seeds": bool(resample_training_seeds),
        "metric": metric,
    }


def paired_vector_bootstrap(
    differences: np.ndarray,
    *,
    replicates: int = 10_000,
    seed: int = 20_260_809,
) -> dict:
    """Paired unit bootstrap for source fold/seed or probe-seed contrasts."""

    values = np.asarray(differences, dtype=np.float64)
    generator = np.random.default_rng(int(seed))
    draws = generator.choice(values, size=(int(replicates), len(values)), replace=True).mean(axis=1)
    return {
        "point_estimate": float(values.mean()),
        "interval": percentile_interval(draws),
        "unit_contrasts": values.tolist(),
        "replicates": int(replicates),
    }


def uniform_random_block_reference(
    true_labels: np.ndarray,
    *,
    replicates: int = 10_000,
    seed: int = 20_260_810,
    class_count: int = 7,
) -> dict:
    """Descriptive uniform-random reference under the exact block-label structure."""

    true = np.asarray(true_labels, dtype=np.int64)
    if true.ndim != 1 or set(np.unique(true).tolist()) != set(range(class_count)):
        raise ValueError("Chance reference requires the fixed seven-class block structure")
    generator = np.random.default_rng(int(seed))
    accuracy = np.empty(int(replicates), dtype=np.float64)
    macro_f1 = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        predicted = generator.integers(0, class_count, size=len(true), dtype=np.int64)
        metrics = classification_metrics(true, predicted, class_count=class_count)
        accuracy[index] = metrics["accuracy"]
        macro_f1[index] = metrics["macro_f1"]
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "prediction_distribution": "uniform_over_fixed_seven_classes",
        "block_count": int(len(true)),
        "analytic_accuracy_chance": 1.0 / float(class_count),
        "mean_random_accuracy": float(accuracy.mean()),
        "accuracy_interval_95": list(percentile_interval(accuracy)),
        "mean_random_macro_f1": float(macro_f1.mean()),
        "macro_f1_interval_95": list(percentile_interval(macro_f1)),
    }


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])
