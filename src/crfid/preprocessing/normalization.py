"""Featurewise population standardization fitted on declared data only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray
    minimum_scale: float = 1e-12

    @classmethod
    def fit(cls, inputs: np.ndarray, *, minimum_scale: float = 1e-12) -> "Standardizer":
        values = np.asarray(inputs, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
            raise ValueError("Standardizer fit requires finite non-empty [N,F] data")
        if minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive")
        mean = values.mean(axis=0, dtype=np.float64)
        raw_scale = values.std(axis=0, ddof=0, dtype=np.float64)
        scale = np.where(raw_scale < minimum_scale, 1.0, raw_scale)
        return cls(np.ascontiguousarray(mean), np.ascontiguousarray(scale), minimum_scale)

    def transform(self, inputs: np.ndarray, *, dtype: np.dtype = np.float32) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float64)
        if values.ndim != 2 or values.shape[1:] != self.mean.shape:
            raise ValueError("Transform features do not match the fitted state")
        transformed = (values - self.mean) / self.scale
        if not np.isfinite(transformed).all():
            raise ValueError("Non-finite standardized value")
        return np.ascontiguousarray(transformed, dtype=dtype)
