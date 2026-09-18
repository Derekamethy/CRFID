"""Deterministic classifier helpers."""

from __future__ import annotations

import numpy as np


def linear_scores(embeddings: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    weights = np.asarray(weight, dtype=np.float32)
    offsets = np.asarray(bias, dtype=np.float32)
    if values.ndim != 2 or weights.ndim != 2 or weights.shape[1] != values.shape[1]:
        raise ValueError("Linear-classifier feature dimensions differ")
    if offsets.shape != (weights.shape[0],):
        raise ValueError("Linear-classifier bias shape is invalid")
    return values @ weights.T + offsets


def predict_lowest_tie(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Scores must be a finite matrix")
    return np.argmax(values, axis=1).astype(np.int64)
