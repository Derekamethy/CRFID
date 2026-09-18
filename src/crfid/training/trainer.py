"""Narrow source-supervised PyTorch training loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..exceptions import DependencyUnavailableError
from .reproducibility import set_deterministic_seed


@dataclass(frozen=True)
class TrainingResult:
    model: Any
    epoch_losses: tuple[float, ...]


def train_source_model(
    model: Any,
    inputs: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> TrainingResult:
    """Train only caller-supplied source arrays; no target flags exist."""

    try:
        import torch
        import torch.nn.functional as functional
    except ModuleNotFoundError as exc:
        raise DependencyUnavailableError("PyTorch is required for training") from exc
    x = np.asarray(inputs, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 3 or y.shape != (x.shape[0],) or x.shape[0] == 0:
        raise ValueError("Training arrays are not aligned")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    set_deterministic_seed(seed)
    features = torch.from_numpy(x)
    targets = torch.from_numpy(y)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(targets), generator=generator)
        total = 0.0
        for start in range(0, len(targets), batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(model(features[indices]), targets[indices])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        losses.append(total / len(targets))
    return TrainingResult(model, tuple(losses))
