"""Equal-weight seed summaries."""

from __future__ import annotations

import numpy as np


def mean_population_std(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty list")
    return {"count": int(array.size), "mean": float(array.mean()), "standard_deviation": float(array.std(ddof=0))}
