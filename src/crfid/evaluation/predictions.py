"""Prediction creation helpers."""

from __future__ import annotations

import numpy as np


def predict_from_logits(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Logits must be a finite [N,C] matrix")
    return np.argmax(values, axis=1).astype(np.int64)


def mean_prediction_entropy(logits: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=1)
    return float(entropy.mean())
