"""Compact, interpretable spectral feature extraction."""

from __future__ import annotations

import numpy as np

from .reference_peaks import local_extrema_indices


def extract_physical_features(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or len(values) < 5 or not np.isfinite(values).all():
        raise ValueError("Physical features require a finite one-dimensional signal")
    minima = local_extrema_indices(values, mode="min")
    gradient = np.diff(values)
    return np.asarray(
        [
            values.mean(),
            values.std(ddof=0),
            values.max() - values.min(),
            values.min(),
            values.max(),
            float(len(minima)),
            np.mean(np.abs(gradient)),
            np.max(np.abs(gradient)),
        ],
        dtype=np.float64,
    )
