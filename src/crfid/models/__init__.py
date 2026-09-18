"""Canonical model architectures and deterministic readouts."""

from .cnn1d import ModelSpec, NumpyNeutralCNN1D, build_torch_model

__all__ = ["ModelSpec", "NumpyNeutralCNN1D", "build_torch_model"]
