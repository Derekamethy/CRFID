"""Configuration-driven preprocessing composition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .normalization import Standardizer
from .representations import add_channel_axis, first_difference


@dataclass(frozen=True)
class FittedTransform:
    representation: str
    standardizer: Standardizer

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        values = first_difference(inputs) if self.representation == "first_difference" else np.asarray(inputs)
        return add_channel_axis(self.standardizer.transform(values))


def fit_transform(inputs: np.ndarray, representation: str) -> tuple[FittedTransform, np.ndarray]:
    if representation not in {"raw", "first_difference"}:
        raise ValueError("Unknown representation")
    values = first_difference(inputs) if representation == "first_difference" else np.asarray(inputs)
    state = FittedTransform(representation, Standardizer.fit(values))
    return state, state.transform(inputs)
