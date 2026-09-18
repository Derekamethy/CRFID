"""Reference-peak extraction on declared frequency windows."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def local_extrema_indices(signal: np.ndarray, *, mode: str = "min") -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("Peak extraction requires a finite vector with at least three points")
    if mode == "min":
        mask = (values[1:-1] < values[:-2]) & (values[1:-1] <= values[2:])
    elif mode == "max":
        mask = (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
    else:
        raise ValueError("mode must be 'min' or 'max'")
    return np.flatnonzero(mask) + 1


def strongest_window_peaks(
    signal: np.ndarray,
    frequency_axis: np.ndarray,
    windows: Mapping[str, tuple[float, float]],
    *,
    mode: str = "min",
) -> dict[str, dict[str, float | int | bool]]:
    values = np.asarray(signal, dtype=np.float64)
    axis = np.asarray(frequency_axis, dtype=np.float64)
    if values.shape != axis.shape or not np.all(np.diff(axis) > 0):
        raise ValueError("Signal and increasing frequency axis must align")
    extrema = set(local_extrema_indices(values, mode=mode).tolist())
    result: dict[str, dict[str, float | int | bool]] = {}
    for name, (low, high) in windows.items():
        candidates = np.flatnonzero((axis >= low) & (axis <= high))
        if candidates.size == 0:
            raise ValueError(f"Empty frequency window: {name}")
        key = min if mode == "min" else max
        index = key(candidates.tolist(), key=lambda item: values[item])
        result[name] = {
            "index": int(index),
            "frequency": float(axis[index]),
            "value": float(values[index]),
            "interior_local_extremum": index in extrema,
        }
    return result


def tag_bits(class_index: int) -> tuple[int, int, int]:
    """Map canonical class 0..6 to three-bit tag identifier 1..7."""

    if class_index not in range(7):
        raise ValueError("class_index must be in 0..6")
    return tuple(int(bit) for bit in f"{class_index + 1:03b}")  # type: ignore[return-value]
