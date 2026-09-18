"""Evaluation and prediction serialization contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.metrics import classification_metrics
from ..governance.pre_dg_integrity import sha256_file


def evaluate_predictions(
    truth: np.ndarray, prediction: np.ndarray, *, class_count: int
) -> dict[str, Any]:
    return classification_metrics(
        np.asarray(truth, dtype=np.int64),
        np.asarray(prediction, dtype=np.int64),
        class_count=class_count,
    )


def save_predictions(
    path: str | Path,
    sample_ids: tuple[str, ...],
    truth: np.ndarray,
    prediction: np.ndarray,
    logits: np.ndarray,
) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        sample_id=np.asarray(sample_ids, dtype=str),
        truth=np.asarray(truth, dtype=np.int64),
        prediction=np.asarray(prediction, dtype=np.int64),
        logits=np.asarray(logits, dtype=np.float32),
    )
    return sha256_file(destination)


def load_predictions(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as payload:
        result = {name: np.asarray(payload[name]) for name in payload.files}
    expected = {"sample_id", "truth", "prediction", "logits"}
    if set(result) != expected:
        raise ValueError("Prediction artifact schema changed")
    count = len(result["truth"])
    if any(len(result[name]) != count for name in expected):
        raise ValueError("Prediction artifact arrays are not aligned")
    return result
