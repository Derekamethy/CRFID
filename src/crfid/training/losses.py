"""Validated loss helpers."""

from __future__ import annotations

import numpy as np


def cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or y.shape != (values.shape[0],) or values.shape[0] == 0:
        raise ValueError("Logits and labels are not aligned")
    if np.any((y < 0) | (y >= values.shape[1])):
        raise ValueError("Label outside logit class range")
    shifted = values - values.max(axis=1, keepdims=True)
    log_sum = np.log(np.exp(shifted).sum(axis=1))
    return float(np.mean(log_sum - shifted[np.arange(len(y)), y]))
