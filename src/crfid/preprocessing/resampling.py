"""Validated one-dimensional physical-axis interpolation."""

from __future__ import annotations

import numpy as np

from ..data.validation import validate_frequency_axis


def resample_signal(
    signal: np.ndarray, source_axis: np.ndarray, target_axis: np.ndarray
) -> np.ndarray:
    source = validate_frequency_axis(source_axis, len(source_axis))
    target = validate_frequency_axis(target_axis, len(target_axis))
    values = np.asarray(signal, dtype=np.float64)
    if values.shape != source.shape or not np.isfinite(values).all():
        raise ValueError("Signal and source frequency axis must be finite and aligned")
    if target[0] < source[0] or target[-1] > source[-1]:
        raise ValueError("Target frequency grid cannot extrapolate beyond the source")
    return np.interp(target, source, values).astype(np.float32)


def uniform_grid(start: float, stop: float, points: int) -> np.ndarray:
    if not start < stop or points < 2:
        raise ValueError("Uniform-grid bounds or point count are invalid")
    return np.linspace(start, stop, points, dtype=np.float64)
