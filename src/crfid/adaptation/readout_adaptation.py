"""Serialization for the source-fitted target-informed readout state."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_readout_state(path: str | Path, prototypes: np.ndarray, class_order: tuple[int, ...]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        prototypes=np.asarray(prototypes, dtype=np.float64),
        class_order=np.asarray(class_order, dtype=np.int64),
        trainable_parameter_names=np.asarray([], dtype=str),
    )


def load_readout_state(path: str | Path) -> tuple[np.ndarray, tuple[int, ...]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        prototypes = np.asarray(payload["prototypes"], dtype=np.float64)
        class_order = tuple(int(value) for value in payload["class_order"])
        trainable = payload["trainable_parameter_names"]
    if trainable.size:
        raise ValueError("The retained readout cannot contain trainable parameters")
    return prototypes, class_order
