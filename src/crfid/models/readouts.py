"""Source nearest-class-mean readouts."""

from __future__ import annotations

import numpy as np

from .embeddings import l2_normalize


def class_prototypes(
    embeddings: np.ndarray, labels: np.ndarray, class_order: tuple[int, ...]
) -> np.ndarray:
    values = l2_normalize(embeddings)
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (values.shape[0],):
        raise ValueError("Prototype labels and embeddings are not aligned")
    centers = []
    for class_index in class_order:
        selected = values[y == class_index]
        if selected.size == 0:
            raise ValueError(f"Class {class_index} has no prototype samples")
        centers.append(selected.mean(axis=0))
    return l2_normalize(np.stack(centers))


def cosine_ncm_predict(embeddings: np.ndarray, prototypes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = l2_normalize(embeddings) @ l2_normalize(prototypes).T
    return np.argmax(scores, axis=1).astype(np.int64), scores


def condition_block_euclidean_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    condition_ids: np.ndarray,
    class_order: tuple[int, ...],
    *,
    expected_block_size: int = 50,
) -> np.ndarray:
    """Fit equal-weight class prototypes from complete condition-block centroids."""

    values = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    conditions = np.asarray(condition_ids, dtype=str)
    if values.ndim != 2 or y.shape != (len(values),) or conditions.shape != (len(values),):
        raise ValueError("Embeddings, labels, and condition IDs are not aligned")
    block_values: list[np.ndarray] = []
    block_labels: list[int] = []
    for condition in sorted(set(conditions.tolist()), key=str.casefold):
        selected = np.flatnonzero(conditions == condition)
        unique_labels = np.unique(y[selected])
        if len(selected) != expected_block_size or len(unique_labels) != 1:
            raise ValueError(f"Condition block is incomplete or label-mixed: {condition}")
        block_values.append(values[selected].mean(axis=0, dtype=np.float64))
        block_labels.append(int(unique_labels[0]))
    blocks = np.ascontiguousarray(np.vstack(block_values), dtype=np.float64)
    block_y = np.asarray(block_labels, dtype=np.int64)
    prototypes = []
    for class_index in class_order:
        selected = blocks[block_y == class_index]
        if not len(selected):
            raise ValueError(f"Class {class_index} has no complete condition block")
        prototypes.append(selected.mean(axis=0, dtype=np.float64))
    return np.ascontiguousarray(np.vstack(prototypes), dtype=np.float64)


def euclidean_ncm_predict(
    embeddings: np.ndarray, prototypes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Predict with negative squared Euclidean distance and lowest-index ties."""

    values = np.asarray(embeddings, dtype=np.float64)
    centers = np.asarray(prototypes, dtype=np.float64)
    if values.ndim != 2 or centers.ndim != 2 or values.shape[1] != centers.shape[1]:
        raise ValueError("Embedding and prototype dimensions differ")
    scores = -np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmax(scores, axis=1).astype(np.int64), np.ascontiguousarray(scores, dtype=np.float32)
