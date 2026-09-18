"""Backbone, projection head, span decoder and dataset-domain head.

The backbone is structurally and numerically identical to the canonical Strict-DG
``NeutralSourceOnlyCNN1D`` (142,855 parameters for seven classes) and is initialised by
the same explicit procedure, so ``T0`` is a genuinely matched control rather than a
re-architected baseline. It is already length-agnostic because it ends in
``AdaptiveAvgPool1d(1)``; no modification was needed to accept Data1.

Auxiliary heads are separate modules initialised from their own seed stream, so
attaching them cannot perturb the backbone's initial weights.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - torch is required to run, not to import
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


EMBEDDING_DIMENSION = 256
CANONICAL_PARAMETER_COUNT_SEVEN_CLASS = 142855
FEATURE_MAP_STRIDE = 4  # two MaxPool1d(2) stages


def _require_torch() -> None:
    if torch is None or nn is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for this branch")


if nn is not None:

    class Backbone(nn.Module):  # type: ignore[misc]
        """Canonical class-symmetric GroupNorm CNN, unchanged."""

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
            """Length-invariant 256-dimensional embedding."""

            return self.network[:13](inputs)

        def feature_map(self, inputs: Any) -> Any:
            """Pre-pooling feature map ``[B, 256, L/4]`` used by the span decoder."""

            return self.network[:11](inputs)

        def trunk_state(self) -> dict[str, Any]:
            """Every parameter except the task-specific output layer."""

            return {
                key: value
                for key, value in self.state_dict().items()
                if not key.startswith("network.13.")
            }

        def load_trunk_state(self, state: dict[str, Any]) -> None:
            missing = [key for key in self.trunk_state() if key not in state]
            if missing:
                raise RuntimeError(f"Trunk state is incomplete: {missing[:4]}")
            own = self.state_dict()
            for key, value in state.items():
                if key.startswith("network.13."):
                    raise RuntimeError("Trunk state must not carry an output head")
                if key not in own:
                    raise RuntimeError(f"Unexpected trunk key: {key}")
                if tuple(own[key].shape) != tuple(value.shape):
                    raise RuntimeError(f"Trunk shape mismatch at {key}")
            self.load_state_dict(state, strict=False)

    class ProjectionHead(nn.Module):  # type: ignore[misc]
        """Two-layer contrastive projection head."""

        def __init__(self, output_dimension: int = 128) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(EMBEDDING_DIMENSION, EMBEDDING_DIMENSION),
                nn.ReLU(),
                nn.Linear(EMBEDDING_DIMENSION, output_dimension),
            )

        def forward(self, embeddings: Any) -> Any:
            return self.network(embeddings)

    class SpanDecoder(nn.Module):  # type: ignore[misc]
        """Upsample the trunk feature map back to input resolution for masked spans."""

        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.ConvTranspose1d(EMBEDDING_DIMENSION, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 1, 3, padding=1),
            )

        def forward(self, feature_map: Any) -> Any:
            return self.network(feature_map).squeeze(1)

    class _GradientReversal(torch.autograd.Function):  # type: ignore[misc]
        @staticmethod
        def forward(ctx: Any, inputs: Any, strength: float) -> Any:
            ctx.strength = float(strength)
            return inputs.view_as(inputs)

        @staticmethod
        def backward(ctx: Any, gradient: Any) -> Any:
            return gradient.neg() * ctx.strength, None

    class DatasetDomainHead(nn.Module):  # type: ignore[misc]
        """Gradient-reversed binary dataset-identity classifier (T5 only).

        The label it predicts is *dataset identity* (Paper4 vs Data1), never a Data1
        class label.
        """

        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(EMBEDDING_DIMENSION, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )

        def forward(self, embeddings: Any, strength: float) -> Any:
            return self.network(_GradientReversal.apply(embeddings, strength))

else:  # pragma: no cover

    class Backbone:  # type: ignore[no-redef]
        def __init__(self, class_count: int = 7) -> None:
            _require_torch()

    class ProjectionHead:  # type: ignore[no-redef]
        def __init__(self, output_dimension: int = 128) -> None:
            _require_torch()

    class SpanDecoder:  # type: ignore[no-redef]
        def __init__(self) -> None:
            _require_torch()

    class DatasetDomainHead:  # type: ignore[no-redef]
        def __init__(self) -> None:
            _require_torch()


def initialize_backbone(seed: int, *, class_count: int = 7) -> "Backbone":
    """Construct the backbone with the canonical explicit initialisation."""

    _require_torch()
    torch.manual_seed(int(seed))
    model = Backbone(class_count=class_count)
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


def initialize_auxiliary(module: Any, seed: int) -> Any:
    """Initialise an auxiliary head from its own seed stream."""

    _require_torch()
    torch.manual_seed(int(seed))
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
                nn.init.kaiming_uniform_(child.weight, a=0.0, mode="fan_in", nonlinearity="relu")
                if child.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(child.weight)
                    nn.init.uniform_(child.bias, -fan_in**-0.5, fan_in**-0.5)
    return module


def parameter_count(module: Any) -> int:
    _require_torch()
    return int(sum(parameter.numel() for parameter in module.parameters()))


def trainable_parameter_count(module: Any) -> int:
    _require_torch()
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def state_sha256(state: Any) -> str:
    """Hash a model state independently of checkpoint container bytes."""

    _require_torch()
    values = state.state_dict() if hasattr(state, "state_dict") else state
    digest = hashlib.sha256()
    for key in sorted(values):
        tensor = values[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def embed_numpy(model: Any, inputs: np.ndarray, *, batch_size: int = 512) -> np.ndarray:
    """Batched length-invariant embedding extraction in evaluation mode."""

    _require_torch()
    model.eval()
    chunks = []
    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.float32))
        for start in range(0, tensor.shape[0], batch_size):
            chunks.append(model.encode(tensor[start : start + batch_size]).cpu().numpy())
    return np.ascontiguousarray(np.concatenate(chunks, axis=0))


def logits_numpy(model: Any, inputs: np.ndarray, *, batch_size: int = 512) -> np.ndarray:
    _require_torch()
    model.eval()
    chunks = []
    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.float32))
        for start in range(0, tensor.shape[0], batch_size):
            chunks.append(model(tensor[start : start + batch_size]).cpu().numpy())
    return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)
