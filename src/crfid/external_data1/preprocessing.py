"""Training-only raw and first-difference standardization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..exceptions import ProtocolViolation


@dataclass(frozen=True)
class StandardizationState:
    representation: str
    mean: np.ndarray
    scale: np.ndarray
    original_population_std: np.ndarray
    replacement_mask: np.ndarray
    fit_partition: str = "set_3"

    @property
    def feature_count(self) -> int:
        return int(self.mean.size)


def represent(inputs: np.ndarray, representation: str) -> np.ndarray:
    values = np.asarray(inputs, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 1601:
        raise ValueError("Data1 representation inputs must have shape [N,1601]")
    if representation == "raw":
        return values
    if representation == "first_difference":
        return np.diff(values, axis=1)
    raise ValueError(f"Unsupported Data1 representation: {representation}")


def fit_standardization(
    train_inputs: np.ndarray, representation: str, *, partition: str = "set_3"
) -> tuple[np.ndarray, StandardizationState]:
    if partition != "set_3":
        raise ProtocolViolation("Data1 preprocessing may be fit only on set_3")
    represented = represent(train_inputs, representation)
    mean = represented.mean(axis=0, dtype=np.float64)
    original_scale = represented.std(axis=0, ddof=0, dtype=np.float64)
    scale = original_scale.copy()
    replacement = scale < 1e-12
    scale[replacement] = 1.0
    state = StandardizationState(
        representation,
        mean,
        scale,
        original_scale,
        replacement,
    )
    return transform_with_state(train_inputs, state), state


def transform_with_state(
    inputs: np.ndarray, state: StandardizationState, *, allow_refit: bool = False
) -> np.ndarray:
    if allow_refit:
        raise ProtocolViolation("Test preprocessing refit is forbidden")
    represented = represent(inputs, state.representation)
    if represented.shape[1] != state.feature_count:
        raise ValueError("Preprocessing state dimension mismatch")
    return np.ascontiguousarray(
        ((represented - state.mean) / state.scale).astype(np.float32)
    )


def save_standardization(path: str | Path, state: StandardizationState) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        representation=np.asarray(state.representation),
        mean=state.mean.astype("<f8", copy=False),
        scale=state.scale.astype("<f8", copy=False),
        original_population_std=state.original_population_std.astype("<f8", copy=False),
        replacement_mask=state.replacement_mask.astype(np.uint8),
        fit_partition=np.asarray(state.fit_partition),
    )


def load_standardization(path: str | Path) -> StandardizationState:
    with np.load(Path(path), allow_pickle=False) as payload:
        return StandardizationState(
            representation=str(payload["representation"].item()),
            mean=np.ascontiguousarray(payload["mean"], dtype=np.float64),
            scale=np.ascontiguousarray(payload["scale"], dtype=np.float64),
            original_population_std=np.ascontiguousarray(
                payload["original_population_std"], dtype=np.float64
            ),
            replacement_mask=np.ascontiguousarray(
                payload["replacement_mask"], dtype=np.uint8
            ).astype(bool),
            fit_partition=str(payload["fit_partition"].item()),
        )

