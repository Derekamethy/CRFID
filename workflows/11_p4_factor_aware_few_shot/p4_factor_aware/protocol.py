"""Outer-fold construction and the fail-closed query-label boundary."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .constants import ALL_CELLS, CLASS_ORDER, QUERY_FOLDS, cell_id


class ProtocolViolation(RuntimeError):
    """A material preregistered protocol violation."""


class QueryLabelBoundaryError(ProtocolViolation):
    """Raised on any query-label access before prediction freeze."""


def validate_outer_folds() -> None:
    seen: list[tuple[int, int]] = []
    for fold, cells in QUERY_FOLDS.items():
        if len(cells) != 3 or len(set(cells)) != 3:
            raise ProtocolViolation(f"fold {fold} does not contain three unique cells")
        if {cell[0] for cell in cells} != {0, 1, 2}:
            raise ProtocolViolation(f"fold {fold} does not cover every ER")
        if {cell[1] for cell in cells} != {0, 1, 2}:
            raise ProtocolViolation(f"fold {fold} does not cover every surface")
        seen.extend(cells)
    if sorted(seen) != sorted(ALL_CELLS):
        raise ProtocolViolation("outer folds do not cover every factor cell exactly once")


def outer_fold_manifest_rows() -> list[dict[str, object]]:
    validate_outer_folds()
    rows: list[dict[str, object]] = []
    for fold, query_cells in QUERY_FOLDS.items():
        query_set = set(query_cells)
        for class_index in CLASS_ORDER:
            for cell in ALL_CELLS:
                rows.append(
                    {
                        "fold": fold,
                        "class_index": class_index,
                        "factor_cell": cell_id(cell),
                        "partition": "QUERY" if cell in query_set else "SUPPORT_CANDIDATE",
                        "physical_rows": 50,
                    }
                )
    return rows


def prediction_sha256(predictions: np.ndarray) -> str:
    values = np.ascontiguousarray(predictions, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


@dataclass
class QueryLabelSeal:
    """Own query labels while exposing only a freeze-then-open interface."""

    _labels: np.ndarray
    _state: str = "SEALED"
    _prediction_hash: str = ""
    _metrics_computed: bool = False
    access_log: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        labels = np.ascontiguousarray(self._labels, dtype=np.int64)
        if labels.ndim != 1 or np.any((labels < 0) | (labels >= len(CLASS_ORDER))):
            raise ValueError("invalid sealed query labels")
        self._labels = labels
        self._log("QUERY_LABELS_SEALED", labels_opened=False)

    def _log(self, event: str, **fields: object) -> None:
        self.access_log.append(
            {"sequence": len(self.access_log) + 1, "event": event, **fields}
        )

    @property
    def state(self) -> str:
        return self._state

    @property
    def prediction_hash(self) -> str:
        return self._prediction_hash

    def freeze_predictions(self, predictions: np.ndarray) -> str:
        if self._state != "SEALED":
            raise QueryLabelBoundaryError("FAIL_QUERY_LABEL_BOUNDARY")
        values = np.asarray(predictions, dtype=np.int64)
        if values.shape != self._labels.shape:
            raise ValueError("query predictions are not aligned with the sealed labels")
        self._prediction_hash = prediction_sha256(values)
        self._state = "PREDICTIONS_FROZEN"
        self._log(
            "QUERY_PREDICTIONS_FROZEN",
            labels_opened=False,
            prediction_sha256=self._prediction_hash,
            prediction_count=len(values),
        )
        return self._prediction_hash

    def open_labels(self) -> np.ndarray:
        if self._state != "PREDICTIONS_FROZEN":
            self._log("PREMATURE_QUERY_LABEL_ACCESS_BLOCKED", labels_opened=False)
            raise QueryLabelBoundaryError("FAIL_QUERY_LABEL_BOUNDARY")
        self._state = "LABELS_OPENED"
        self._log(
            "QUERY_LABELS_OPENED_AFTER_PREDICTION_FREEZE",
            labels_opened=True,
            prediction_sha256=self._prediction_hash,
        )
        return self._labels.copy()

    def mark_metrics_computed(self) -> None:
        if self._state != "LABELS_OPENED" or self._metrics_computed:
            raise QueryLabelBoundaryError("FAIL_QUERY_LABEL_BOUNDARY")
        self._metrics_computed = True
        self._state = "METRICS_COMPUTED"
        self._log("QUERY_METRICS_COMPUTED_ONCE", labels_opened=True)


def assert_no_query_informed_selection(
    *,
    query_labels: object | None = None,
    query_predictions: object | None = None,
    query_embeddings: object | None = None,
    query_metrics: object | None = None,
) -> None:
    if any(
        item is not None
        for item in (query_labels, query_predictions, query_embeddings, query_metrics)
    ):
        raise ProtocolViolation("PROHIBITED_QUERY_INFORMED_SUPPORT_SELECTION")


def validate_complete_oof_blocks(rows: Iterable[dict[str, object]]) -> None:
    query = [row for row in rows if row["partition"] == "QUERY"]
    keys = [
        (int(row["class_index"]), str(row["factor_cell"]))
        for row in query
    ]
    if len(keys) != 63 or len(set(keys)) != 63:
        raise ProtocolViolation("outer query manifest does not contain 63 unique blocks")
