"""Embedding separability and position-encoding diagnostics."""

from __future__ import annotations

import numpy as np

from ..models.embeddings import l2_normalize


def separability_summary(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    values = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or y.shape != (len(values),) or len(np.unique(y)) < 2:
        raise ValueError("Separability requires aligned embeddings with at least two classes")
    global_mean = values.mean(axis=0)
    within = 0.0
    between = 0.0
    for label in sorted(np.unique(y).tolist()):
        group = values[y == label]
        center = group.mean(axis=0)
        within += float(np.sum((group - center) ** 2))
        between += float(len(group) * np.sum((center - global_mean) ** 2))
    return {
        "within_class_sum_squares": within,
        "between_class_sum_squares": between,
        "between_within_ratio": between / max(within, 1e-12),
    }


def nearest_centroid_encoding_accuracy(embeddings: np.ndarray, domain_labels: np.ndarray) -> float:
    """Measure how strongly domains are encoded by leave-one-out centroids."""

    values = l2_normalize(embeddings)
    domains = np.asarray(domain_labels)
    if domains.shape != (len(values),) or len(values) < 3:
        raise ValueError("Domain labels and embeddings are not aligned")
    unique = sorted(np.unique(domains).tolist(), key=str)
    predictions = []
    for index in range(len(values)):
        centers = []
        for domain in unique:
            mask = domains == domain
            mask[index] = False
            if not mask.any():
                raise ValueError("Each domain needs at least two samples")
            centers.append(l2_normalize(values[mask].mean(axis=0, keepdims=True))[0])
        predictions.append(unique[int(np.argmax(values[index] @ np.stack(centers).T))])
    return float(np.mean(np.asarray(predictions, dtype=object) == domains))


def class_domain_entanglement(
    embeddings: np.ndarray, class_labels: np.ndarray, domain_labels: np.ndarray
) -> dict[str, float]:
    class_ratio = separability_summary(embeddings, class_labels)["between_within_ratio"]
    _, encoded_domains = np.unique(domain_labels, return_inverse=True)
    domain_ratio = separability_summary(embeddings, encoded_domains)["between_within_ratio"]
    return {
        "class_separability_ratio": class_ratio,
        "domain_separability_ratio": domain_ratio,
        "domain_to_class_ratio": domain_ratio / max(class_ratio, 1e-12),
    }
