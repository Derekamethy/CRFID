"""Final Paper3/Paper4 Pre-DG DeepCNN architecture."""

from __future__ import annotations

from typing import Any

from ..exceptions import DependencyUnavailableError

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - optional import contract
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


if nn is not None:

    class _ConvBlock(nn.Module):  # type: ignore[misc]
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=1, padding=3),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )

        def forward(self, inputs: Any) -> Any:
            return self.block(inputs)


    class PreDGDeepCNN(nn.Module):  # type: ignore[misc]
        """Shared final backbone with a dataset-local output dimension."""

        def __init__(self, class_count: int, dropout: float) -> None:
            super().__init__()
            if class_count not in {7, 8}:
                raise ValueError("Final Pre-DG class count must be 7 or 8")
            if dropout not in {0.0, 0.3}:
                raise ValueError("Final Pre-DG dropout must be 0.0 or 0.3")
            self.features = nn.Sequential(
                _ConvBlock(2, 128),
                _ConvBlock(128, 256),
                _ConvBlock(256, 256),
                _ConvBlock(256, 128),
                nn.Conv1d(128, 64, kernel_size=7, stride=1, padding=3),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(8),
                nn.Flatten(),
            )
            self.projection = nn.Sequential(
                nn.Linear(64 * 8, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.output = nn.Linear(512, class_count)
            self.class_count = class_count
            self.dropout_value = float(dropout)

        def forward(self, inputs: Any) -> Any:
            if inputs.ndim != 3 or tuple(inputs.shape[1:]) != (2, 512):
                raise ValueError(f"Expected [N,2,512] input, got {tuple(inputs.shape)}")
            return self.output(self.projection(self.features(inputs)))

else:

    class PreDGDeepCNN:  # type: ignore[no-redef]
        def __init__(self, class_count: int, dropout: float) -> None:
            raise DependencyUnavailableError("PyTorch is required for Pre-DG")


def build_pre_dg_model(class_count: int, dropout: float) -> Any:
    return PreDGDeepCNN(class_count=class_count, dropout=dropout)


def expected_parameter_count(class_count: int) -> int:
    return {8: 1_245_896, 7: 1_245_383}[class_count]


def verify_model_contract(model: Any, class_count: int) -> dict[str, object]:
    observed = int(sum(parameter.numel() for parameter in model.parameters()))
    expected = expected_parameter_count(class_count)
    if observed != expected:
        raise ValueError(f"Pre-DG parameter count changed: {observed} != {expected}")
    return {
        "architecture": "PreDGDeepCNN",
        "input_shape": [None, 2, 512],
        "output_dimension": class_count,
        "parameter_count": observed,
        "verified": True,
    }
