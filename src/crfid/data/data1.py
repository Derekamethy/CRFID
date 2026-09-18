"""Data1 local four-class schema and explicit CSV loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .schemas import DatasetSchema
from ..exceptions import DataValidationError


CLASS_ORDER = (0, 1, 2, 3)
TRAIN_GROUPS = ("set_3",)
TEST_GROUPS = ("set_4", "set_5", "set_6", "set_7", "set_8", "set_9")
SCHEMA = DatasetSchema("data1", CLASS_ORDER, 1601, TRAIN_GROUPS + TEST_GROUPS)


def load_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one Data1 CSV after explicit caller authorization."""

    table = np.atleast_2d(
        np.loadtxt(Path(path), delimiter=",", skiprows=1, dtype=np.float64)
    )
    if table.ndim != 2 or table.shape[1] != 1602 or not np.isfinite(table).all():
        raise DataValidationError("Data1 table must be finite with 1,602 columns")
    labels_float = table[:, -1]
    labels = labels_float.astype(np.int64)
    if not np.array_equal(labels_float, labels.astype(np.float64)):
        raise DataValidationError("Data1 labels must be integral")
    if set(np.unique(labels)).difference(CLASS_ORDER):
        raise DataValidationError("Data1 label is outside local class order")
    return np.ascontiguousarray(table[:, :-1]), np.ascontiguousarray(labels)


def deterministic_majority_label(labels: np.ndarray) -> int:
    """Choose the most frequent training label, with lowest-index ties."""

    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or values.size == 0 or set(np.unique(values)).difference(CLASS_ORDER):
        raise DataValidationError("Majority labels must be a non-empty local-label vector")
    counts = np.bincount(values, minlength=len(CLASS_ORDER))
    return int(np.flatnonzero(counts == counts.max())[0])
