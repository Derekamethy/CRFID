"""Embedding validation and normalization."""

from __future__ import annotations

import numpy as np


def l2_normalize(values: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Embeddings must be a finite matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, epsilon)


def validate_embeddings(values: np.ndarray, dimension: int = 256) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != dimension or not np.isfinite(array).all():
        raise ValueError(f"Expected finite [N,{dimension}] embeddings")
    return array
