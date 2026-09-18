"""Frozen train-only channel scaler and Stage4 representation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..governance.pre_dg_integrity import sha256_file


@dataclass(frozen=True)
class TrainOnlyChannelScaler:
    mean: np.ndarray
    scale: np.ndarray
    fit_partition: str = "train"
    ddof: int = 0

    @classmethod
    def fit(cls, inputs: np.ndarray, *, partition: str) -> "TrainOnlyChannelScaler":
        if partition != "train":
            raise PermissionError("Pre-DG preprocessing may be fit on train only")
        values = np.asarray(inputs)
        if values.ndim != 3 or tuple(values.shape[1:]) != (2, 512):
            raise ValueError(f"Expected training inputs [N,2,512], got {values.shape}")
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError("Pre-DG inputs must be finite float32")
        mean = values.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        scale = values.std(axis=(0, 2), ddof=0, keepdims=True).astype(np.float32)
        scale[scale == 0] = np.float32(1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float32)
        if values.ndim != 3 or tuple(values.shape[1:]) != (2, 512):
            raise ValueError(f"Expected inputs [N,2,512], got {values.shape}")
        transformed = ((values - self.mean) / self.scale).astype(np.float32)
        if not np.isfinite(transformed).all():
            raise ValueError("Pre-DG preprocessing produced non-finite values")
        return transformed

    def save(self, path: str | Path) -> str:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            destination,
            mean=np.asarray(self.mean, dtype=np.float32),
            scale=np.asarray(self.scale, dtype=np.float32),
            fit_partition=np.asarray(self.fit_partition),
            ddof=np.asarray(self.ddof, dtype=np.int64),
        )
        return sha256_file(destination)

    @classmethod
    def load(cls, path: str | Path) -> "TrainOnlyChannelScaler":
        with np.load(Path(path), allow_pickle=False) as payload:
            state = cls(
                mean=np.asarray(payload["mean"], dtype=np.float32),
                scale=np.asarray(payload["scale"], dtype=np.float32),
                fit_partition=str(payload["fit_partition"].item()),
                ddof=int(payload["ddof"].item()),
            )
        if state.fit_partition != "train" or state.ddof != 0:
            raise ValueError("Serialized Pre-DG preprocessing state is not authoritative")
        return state


def native_physical_two_channel(
    signal: np.ndarray, source_axis: np.ndarray, target_length: int = 512
) -> np.ndarray:
    """Reconstruct the frozen Stage4 single-signal plus zero-channel transform."""
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    axis = np.asarray(source_axis, dtype=np.float64).reshape(-1)
    if len(values) != len(axis) or len(axis) < 2 or not np.all(np.diff(axis) > 0):
        raise ValueError("Signal and physical axis are not aligned")
    target = np.linspace(float(axis[0]), float(axis[-1]), target_length, dtype=np.float64)
    channel = np.interp(target, axis, values.astype(np.float64)).astype(np.float32)
    return np.stack([channel, np.zeros(target_length, dtype=np.float32)])
