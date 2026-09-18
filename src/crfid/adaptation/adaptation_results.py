"""Deterministic target-assisted metric aggregation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


def aggregate_metrics(rows: Iterable[Mapping[str, float]]) -> dict[str, float | int]:
    records = list(rows)
    if not records:
        raise ValueError("At least one run is required")
    accuracy = np.asarray([float(row["accuracy"]) for row in records], dtype=np.float64)
    macro_f1 = np.asarray([float(row["macro_f1"]) for row in records], dtype=np.float64)
    return {
        "run_count": len(records),
        "accuracy_mean": float(accuracy.mean()),
        "accuracy_std_population": float(accuracy.std(ddof=0)),
        "macro_f1_mean": float(macro_f1.mean()),
        "macro_f1_std_population": float(macro_f1.std(ddof=0)),
    }
