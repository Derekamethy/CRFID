"""Array, label and alignment validation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..exceptions import DataValidationError


def validate_supervised_arrays(
    inputs: np.ndarray, labels: np.ndarray, *, signal_length: int, class_order: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(inputs)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != signal_length or y.shape != (x.shape[0],):
        raise DataValidationError("Supervised array shapes are not aligned")
    if x.shape[0] == 0 or not np.isfinite(x).all():
        raise DataValidationError("Inputs must be non-empty and finite")
    if set(np.unique(y)).difference(int(item) for item in class_order):
        raise DataValidationError("Labels are outside the declared class order")
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def validate_frequency_axis(axis: np.ndarray, expected_length: int) -> np.ndarray:
    values = np.asarray(axis, dtype=np.float64)
    if values.shape != (expected_length,) or not np.isfinite(values).all():
        raise DataValidationError("Frequency-axis shape or finite-value check failed")
    if not np.all(np.diff(values) > 0):
        raise DataValidationError("Frequency axis must be strictly increasing")
    return values
