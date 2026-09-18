"""Frozen, class-symmetric Phase 2 neutral baseline model."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any

import numpy as np
import torch
from torch import nn


EXPECTED_PARAMETER_COUNT = 142_855
_DETERMINISM_CONFIGURED = False


class NeutralSourceOnlyCNN1D(nn.Module):
    """The single predeclared neutral architecture; no optional components."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 7),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def configure_determinism() -> None:
    global _DETERMINISM_CONFIGURED
    if _DETERMINISM_CONFIGURED:
        return
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    _DETERMINISM_CONFIGURED = True


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def apply_explicit_initialization(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_uniform_(
                module.weight, a=0.0, mode="fan_in", nonlinearity="relu"
            )
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1.0 / math.sqrt(fan_in)
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, nn.GroupNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def initialize_model(seed: int) -> NeutralSourceOnlyCNN1D:
    configure_determinism()
    set_all_seeds(seed)
    model = NeutralSourceOnlyCNN1D()
    # Reset immediately before the declared initializer so constructor defaults do
    # not define the final parameter draw sequence.
    torch.manual_seed(seed)
    apply_explicit_initialization(model)
    count = parameter_count(model)
    if count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Neutral model parameter count changed: {count} != {EXPECTED_PARAMETER_COUNT}"
        )
    return model


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def model_state_sha256(model_or_state: nn.Module | dict[str, torch.Tensor]) -> str:
    state = model_or_state.state_dict() if isinstance(model_or_state, nn.Module) else model_or_state
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def gradient_hashes(model: nn.Module) -> dict:
    per_parameter: dict[str, str | None] = {}
    aggregate = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            per_parameter[name] = None
            aggregate.update(f"{name}|NONE".encode("utf-8"))
            continue
        value_hash = tensor_sha256(parameter.grad)
        per_parameter[name] = value_hash
        aggregate.update(f"{name}|{value_hash}".encode("utf-8"))
    return {"aggregate_sha256": aggregate.hexdigest(), "per_parameter": per_parameter}


def _hash_nested(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        digest.update(b"TENSOR")
        digest.update(tensor_sha256(value).encode("ascii"))
    elif isinstance(value, dict):
        digest.update(b"DICT")
        for key in sorted(value, key=lambda item: str(item)):
            digest.update(str(key).encode("utf-8"))
            _hash_nested(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"SEQUENCE")
        for item in value:
            _hash_nested(digest, item)
    else:
        digest.update(repr(value).encode("utf-8"))


def optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()
    _hash_nested(digest, optimizer.state_dict())
    return digest.hexdigest()
