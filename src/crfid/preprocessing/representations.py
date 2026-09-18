"""Canonical raw and derivative signal representations."""

from __future__ import annotations

import numpy as np


def first_difference(signals: np.ndarray) -> np.ndarray:
    """Compute x[i+1]-x[i] without padding."""

    values = np.asarray(signals)
    if values.ndim not in (1, 2) or values.shape[-1] < 2 or not np.isfinite(values).all():
        raise ValueError("Signals must be finite rank-one or rank-two arrays")
    return np.ascontiguousarray(np.diff(values, axis=-1))


def two_channel_raw_gradient(signals: np.ndarray) -> np.ndarray:
    """Create a two-channel raw/gradient representation with aligned length."""

    values = np.asarray(signals, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite [N,L] signals")
    gradient = np.gradient(values.astype(np.float64), axis=1).astype(np.float32)
    return np.stack([values, gradient], axis=1)


def add_channel_axis(signals: np.ndarray) -> np.ndarray:
    values = np.asarray(signals, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Expected [N,L] signals")
    return values[:, None, :]
