"""Prototype definitions used by P4 Target-Assisted Adaptation."""

from __future__ import annotations

import numpy as np


def l2_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Expected a finite matrix")
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def normalized_source_prototypes(
    embeddings: np.ndarray, labels: np.ndarray, class_order: tuple[int, ...]
) -> np.ndarray:
    values = l2_rows(embeddings)
    target = np.asarray(labels, dtype=np.int64)
    if target.shape != (len(values),):
        raise ValueError("Source embeddings and labels are not aligned")
    means: list[np.ndarray] = []
    for class_index in class_order:
        selected = values[target == class_index]
        if not len(selected):
            raise ValueError(f"Class {class_index} has no source rows")
        means.append(selected.mean(axis=0, dtype=np.float64))
    return np.ascontiguousarray(l2_rows(np.vstack(means)), dtype=np.float64)


def blend_prototypes(
    source: np.ndarray, target: np.ndarray, *, target_weight: float
) -> np.ndarray:
    """Testable utility; not used by the retained historical method."""

    if not 0.0 <= target_weight <= 1.0:
        raise ValueError("target_weight must be in [0, 1]")
    left, right = l2_rows(source), l2_rows(target)
    if left.shape != right.shape:
        raise ValueError("Prototype shapes differ")
    return l2_rows((1.0 - target_weight) * left + target_weight * right)


def build_target_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sample_ids: list[str],
    *,
    shot_count: int,
) -> np.ndarray:
    """Preserved condition-controlled target prototype utility."""

    values = np.asarray(embeddings, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or target.shape != (len(values),) or len(sample_ids) != len(values):
        raise ValueError("Target prototype inputs are not aligned")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Target prototype sample identities are duplicated")
    classes = sorted(np.unique(target).tolist())
    centers = []
    for class_index in classes:
        selected = values[target == class_index]
        if len(selected) != shot_count:
            raise ValueError(f"Class {class_index} does not have exactly {shot_count} shots")
        centers.append(selected.mean(axis=0, dtype=np.float64))
    return np.ascontiguousarray(np.vstack(centers), dtype=np.float64)


def squared_euclidean_predict(
    embeddings: np.ndarray, prototypes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(embeddings, dtype=np.float64)
    centers = np.asarray(prototypes, dtype=np.float64)
    if values.ndim != 2 or centers.ndim != 2 or values.shape[1] != centers.shape[1]:
        raise ValueError("Embedding and prototype dimensions differ")
    distances = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1).astype(np.int64), distances
