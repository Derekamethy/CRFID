"""Frozen one-dimensional CNN definitions with an import-safe NumPy verifier."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..exceptions import DependencyUnavailableError

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # Optional dependency; imports remain safe.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_channels: int
    input_length: int
    class_count: int
    embedding_dimension: int


STRICT_SPEC = ModelSpec("strict_groupnorm_cnn", 1, 280, 7, 256)
BASELINE_SPEC = ModelSpec("pre_dg_deep_cnn", 2, 512, 7, 512)


def torch_available() -> bool:
    return torch is not None


if nn is not None:

    class NeutralCNN1D(nn.Module):  # type: ignore[misc]
        """Canonical class-symmetric GroupNorm network."""

        def __init__(self, class_count: int = 7) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv1d(1, 64, 7, padding=3),
                nn.GroupNorm(8, 64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, 5, padding=2),
                nn.GroupNorm(8, 128),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(128, 256, 3, padding=1),
                nn.GroupNorm(8, 256),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(256, class_count),
            )

        def forward(self, inputs: Any) -> Any:
            return self.network(inputs)

        def encode(self, inputs: Any) -> Any:
            return self.network[:13](inputs)


    class _ConvBlock(nn.Module):  # type: ignore[misc]
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 7, padding=3),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )

        def forward(self, inputs: Any) -> Any:
            return self.block(inputs)


    class BaselineDeepCNN1D(nn.Module):  # type: ignore[misc]
        """Frozen pre-DG two-channel baseline."""

        def __init__(self, dropout: float = 0.0) -> None:
            super().__init__()
            self.features = nn.Sequential(
                _ConvBlock(2, 128),
                _ConvBlock(128, 256),
                _ConvBlock(256, 256),
                _ConvBlock(256, 128),
                nn.Conv1d(128, 64, 7, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(8),
                nn.Flatten(),
            )
            self.projection = nn.Sequential(nn.Linear(64 * 8, 512), nn.ReLU(), nn.Dropout(dropout))
            self.output = nn.Linear(512, 7)

        def forward(self, inputs: Any) -> Any:
            return self.output(self.projection(self.features(inputs)))


else:

    class NeutralCNN1D:  # type: ignore[no-redef]
        def __init__(self, class_count: int = 7) -> None:
            raise DependencyUnavailableError("PyTorch is required to construct the strict CNN")


    class BaselineDeepCNN1D:  # type: ignore[no-redef]
        def __init__(self, dropout: float = 0.0) -> None:
            raise DependencyUnavailableError("PyTorch is required to construct the baseline CNN")


def build_torch_model(name: str, *, dropout: float = 0.0) -> Any:
    """Construct an explicitly selected PyTorch architecture."""

    if name == STRICT_SPEC.name:
        return NeutralCNN1D()
    if name == BASELINE_SPEC.name:
        return BaselineDeepCNN1D(dropout=dropout)
    raise ValueError(f"Unknown model architecture: {name}")


def initialize_strict_model(seed: int) -> Any:
    """Construct the strict CNN with the frozen explicit initialization."""

    return initialize_neutral_model(seed, class_count=7)


def initialize_neutral_model(seed: int, *, class_count: int) -> Any:
    """Construct the canonical CNN with a task-local output head."""

    if torch is None or nn is None:
        raise DependencyUnavailableError("PyTorch is required to initialize the strict CNN")
    torch.manual_seed(int(seed))
    model = NeutralCNN1D(class_count=class_count)
    torch.manual_seed(int(seed))
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, a=0.0, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    nn.init.uniform_(module.bias, -fan_in**-0.5, fan_in**-0.5)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    return model


def model_state_sha256(model_or_state: Any) -> str:
    """Hash a model state independently of checkpoint container bytes."""

    if torch is None:
        raise DependencyUnavailableError("PyTorch is required for model-state hashing")
    state = model_or_state.state_dict() if hasattr(model_or_state, "state_dict") else model_or_state
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def parameter_count(model: Any) -> int:
    if torch is None:
        raise DependencyUnavailableError("PyTorch is required for parameter counting")
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _conv_same(inputs: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    kernel = weight.shape[2]
    padded = np.pad(inputs, ((0, 0), (0, 0), (kernel // 2, kernel // 2)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel, axis=2)
    return np.einsum("nclk,ock->nol", windows, weight, optimize=True) + bias[None, :, None]


def _group_norm(inputs: np.ndarray, groups: int = 8, epsilon: float = 1e-5) -> np.ndarray:
    batch, channels, length = inputs.shape
    shaped = inputs.reshape(batch, groups, channels // groups, length)
    mean = shaped.mean(axis=(2, 3), keepdims=True)
    variance = shaped.var(axis=(2, 3), keepdims=True)
    return ((shaped - mean) / np.sqrt(variance + epsilon)).reshape(inputs.shape)


def _max_pool2(inputs: np.ndarray) -> np.ndarray:
    length = inputs.shape[2] - inputs.shape[2] % 2
    return inputs[:, :, :length].reshape(inputs.shape[0], inputs.shape[1], -1, 2).max(axis=3)


class NumpyNeutralCNN1D:
    """Numerical mirror used only for dependency-free synthetic forward checks."""

    def __init__(self, seed: int = 0, class_count: int = 7) -> None:
        generator = np.random.default_rng(seed)

        def weights(out_channels: int, in_channels: int, kernel: int) -> np.ndarray:
            bound = np.sqrt(6.0 / (in_channels * kernel))
            return generator.uniform(-bound, bound, (out_channels, in_channels, kernel)).astype(np.float32)

        self.w1, self.b1 = weights(64, 1, 7), np.zeros(64, dtype=np.float32)
        self.w2, self.b2 = weights(128, 64, 5), np.zeros(128, dtype=np.float32)
        self.w3, self.b3 = weights(256, 128, 3), np.zeros(256, dtype=np.float32)
        bound = np.sqrt(6.0 / 256.0)
        self.linear = generator.uniform(-bound, bound, (class_count, 256)).astype(np.float32)
        self.linear_bias = np.zeros(class_count, dtype=np.float32)

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float32)
        if values.ndim != 3 or values.shape[1] != 1 or values.shape[2] < 4:
            raise ValueError("Strict CNN inputs must have shape [N,1,L] with L>=4")
        if not np.isfinite(values).all():
            raise ValueError("Model inputs must be finite")
        values = _max_pool2(np.maximum(_group_norm(_conv_same(values, self.w1, self.b1)), 0.0))
        values = _max_pool2(np.maximum(_group_norm(_conv_same(values, self.w2, self.b2)), 0.0))
        values = np.maximum(_group_norm(_conv_same(values, self.w3, self.b3)), 0.0)
        return values.mean(axis=2, dtype=np.float64).astype(np.float32)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        embeddings = self.encode(inputs)
        return embeddings @ self.linear.T + self.linear_bias
