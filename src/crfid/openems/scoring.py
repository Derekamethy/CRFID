"""Global pairwise spectral-separation scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def centered(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array - array.mean(axis=-1, keepdims=True)


def pairwise_distances(
    spectra: Mapping[str, np.ndarray], codes: Sequence[str], weights: np.ndarray
) -> list[dict[str, float | str]]:
    supplied = np.asarray(weights, dtype=np.float64)
    if supplied.ndim != 1 or np.any(supplied < 0) or not np.isclose(supplied.sum(), 1.0):
        raise ValueError("Weights must be non-negative and sum to one")
    rows: list[dict[str, float | str]] = []
    for left_index, left in enumerate(codes):
        for right in codes[left_index + 1 :]:
            first = np.asarray(spectra[left], dtype=np.float64)
            second = np.asarray(spectra[right], dtype=np.float64)
            if first.shape != supplied.shape or second.shape != supplied.shape:
                raise ValueError("Spectrum and weight shapes differ")
            delta = centered(first[None, :])[0] - centered(second[None, :])[0]
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "weighted_rms": float(np.sqrt(np.sum(supplied * delta**2))),
                    "uniform_rms": float(np.sqrt(np.mean(delta**2))),
                }
            )
    return sorted(rows, key=lambda row: (float(row["weighted_rms"]), str(row["left"]), str(row["right"])))


def minimum_pair_score(rows: list[dict[str, float | str]]) -> tuple[float, str]:
    if not rows:
        raise ValueError("At least one pair is required")
    first = min(rows, key=lambda row: float(row["weighted_rms"]))
    return float(first["weighted_rms"]), f"{first['left']}-{first['right']}"
