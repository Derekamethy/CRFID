"""Deterministic, class-identity-preserving augmentations for the SSL stage.

Every augmentation is driven by an explicitly supplied ``numpy`` generator so that a
seed reproduces the exact view stream. All operate on standardized first-difference
windows of shape ``[B, 1, L]`` and are shape- and finiteness-preserving.

Augmentations that could change tag identity are deliberately absent: no class-mixing,
no large frequency shift, no time reversal, no per-position permutation. The
amplitude-scaling and noise magnitudes are small and are *source-selected*, never
target-selected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AugmentationConfig:
    amplitude_scale_range: tuple[float, float] = (0.9, 1.1)
    noise_standard_deviation: float = 0.05
    baseline_offset_range: tuple[float, float] = (-0.05, 0.05)
    local_mask_span_count: int = 2
    local_mask_span_length: int = 8
    circular_shift_limit: int = 4
    apply_probability: float = 0.5

    def as_record(self) -> dict[str, object]:
        return {
            "amplitude_scale_range": list(self.amplitude_scale_range),
            "noise_standard_deviation": self.noise_standard_deviation,
            "baseline_offset_range": list(self.baseline_offset_range),
            "local_mask_span_count": self.local_mask_span_count,
            "local_mask_span_length": self.local_mask_span_length,
            "circular_shift_limit": self.circular_shift_limit,
            "apply_probability": self.apply_probability,
            "class_identity_preserving": True,
        }


def random_window(
    signals: np.ndarray, window: int, generator: np.random.Generator
) -> np.ndarray:
    """Crop each row to a random contiguous window of exactly ``window`` positions."""

    data = np.asarray(signals, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("Windowing expects [N,L] arrays")
    length = data.shape[1]
    if window > length:
        raise ValueError(f"Window {window} exceeds signal length {length}")
    starts = generator.integers(0, length - window + 1, size=data.shape[0])
    offsets = starts[:, None] + np.arange(window)[None, :]
    return np.ascontiguousarray(np.take_along_axis(data, offsets, axis=1))


def _maybe(generator: np.random.Generator, probability: float, count: int) -> np.ndarray:
    return generator.random(count) < probability


def augment(
    views: np.ndarray, config: AugmentationConfig, generator: np.random.Generator
) -> np.ndarray:
    """Apply the configured augmentation stack to ``[B,1,L]`` standardized windows."""

    data = np.array(views, dtype=np.float32, copy=True)
    if data.ndim != 3 or data.shape[1] != 1:
        raise ValueError("Augmentation expects [B,1,L] arrays")
    batch, _, length = data.shape

    selected = _maybe(generator, config.apply_probability, batch)
    low, high = config.amplitude_scale_range
    scales = np.where(selected, generator.uniform(low, high, batch), 1.0).astype(np.float32)
    data *= scales[:, None, None]

    selected = _maybe(generator, config.apply_probability, batch)
    low, high = config.baseline_offset_range
    offsets = np.where(selected, generator.uniform(low, high, batch), 0.0).astype(np.float32)
    data += offsets[:, None, None]

    if config.noise_standard_deviation > 0:
        selected = _maybe(generator, config.apply_probability, batch)
        noise = generator.normal(0.0, config.noise_standard_deviation, data.shape).astype(np.float32)
        data += noise * selected[:, None, None].astype(np.float32)

    if config.circular_shift_limit > 0:
        shifts = generator.integers(
            -config.circular_shift_limit, config.circular_shift_limit + 1, size=batch
        )
        index = (np.arange(length)[None, :] - shifts[:, None]) % length
        data = np.take_along_axis(data, index[:, None, :], axis=2)

    if config.local_mask_span_count > 0 and config.local_mask_span_length > 0:
        for _ in range(config.local_mask_span_count):
            selected = _maybe(generator, config.apply_probability, batch)
            starts = generator.integers(0, max(1, length - config.local_mask_span_length), batch)
            span = np.arange(config.local_mask_span_length)[None, :]
            positions = starts[:, None] + span
            mask = np.zeros((batch, length), dtype=bool)
            np.put_along_axis(mask, positions, True, axis=1)
            mask &= selected[:, None]
            data[:, 0, :][mask] = 0.0

    data = np.ascontiguousarray(data, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("Augmentation produced a non-finite value")
    return data


def contiguous_span_mask(
    batch: int,
    length: int,
    *,
    span_count: int,
    span_length: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Boolean ``[B,L]`` mask marking the contiguous spans to be reconstructed."""

    if span_count <= 0 or span_length <= 0 or span_length >= length:
        raise ValueError("Invalid contiguous span-mask configuration")
    mask = np.zeros((batch, length), dtype=bool)
    span = np.arange(span_length)[None, :]
    for _ in range(span_count):
        starts = generator.integers(0, length - span_length + 1, batch)
        np.put_along_axis(mask, starts[:, None] + span, True, axis=1)
    return mask
