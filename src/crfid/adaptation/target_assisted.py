"""Authoritative retrospective P4-informed source NCM readout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .prototypes import normalized_source_prototypes


@dataclass(frozen=True)
class TargetInformedNCMResult:
    prototypes: np.ndarray
    distances: np.ndarray
    predictions: np.ndarray


def execute_target_informed_ncm(
    source_embeddings: np.ndarray,
    source_labels: np.ndarray,
    target_embeddings: np.ndarray,
    *,
    class_order: tuple[int, ...] = tuple(range(7)),
) -> TargetInformedNCMResult:
    """Apply the historical normalized-mean-of-normalized cosine NCM rule.

    Target labels are deliberately absent here. They are authorized only in
    the separate retrospective selection and metric stages.
    """

    prototypes = normalized_source_prototypes(source_embeddings, source_labels, class_order)
    target = np.asarray(target_embeddings, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != prototypes.shape[1] or not np.isfinite(target).all():
        raise ValueError("Target embeddings must be finite and dimension-aligned")
    norms = np.linalg.norm(target, axis=1, keepdims=True)
    normalized = target / np.maximum(norms, 1e-12)
    distances = np.ascontiguousarray(normalized @ prototypes.T, dtype=np.float64)
    predictions = np.argmax(distances, axis=1).astype(np.int64)
    return TargetInformedNCMResult(prototypes, distances, predictions)
