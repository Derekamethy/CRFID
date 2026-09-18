"""Supervised fine-tuning on Paper4 P1-P3, matched to the canonical Strict-DG recipe.

The canonical source-side protocol has two stages per fold, both reproduced here:

1. *inner selection* - train on ``inner_train``, early stop on ``inner_validation``
   Macro-F1 (maximum 50 epochs, patience 8, minimum improvement 1e-12), yielding a
   selected epoch;
2. *outer refit* - re-initialise from the same seed and train on ``outer_development``
   (``inner_train`` + ``inner_validation``) for exactly that many epochs, then score the
   held-out source domain.

Each stage fits its own featurewise standardizer on its own training partition, and each
epoch draws its shuffle from ``seed * 1_000_000 + stage_offset + epoch``, matching the
canonical implementation exactly.

The final model is trained on all 9,450 source rows for the per-seed median of the three
fold-selected inner epochs -- a source-only rule that reproduces the canonical
``final_epochs_by_seed`` for all five seeds.

Identical code runs for every treatment; treatments differ only in the trunk state the
backbone starts from.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]

from . import model as model_module
from . import metrics

INNER_STAGE_OFFSET = 0
OUTER_STAGE_OFFSET = 100000
FINAL_STAGE_OFFSET = 100000
PERMUTATION_SEED_STRIDE = 1_000_000


def permutation_seed(seed: int, stage_offset: int, epoch: int) -> int:
    """Canonical per-epoch shuffle seed: ``seed * 1_000_000 + offset + epoch``."""

    return int(seed) * PERMUTATION_SEED_STRIDE + int(stage_offset) + int(epoch)


@dataclass(frozen=True)
class SupervisedConfig:
    maximum_epochs: int = 50
    early_stopping_patience: int = 8
    minimum_improvement: float = 1e-12
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    flooding_level: float | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "maximum_epochs": self.maximum_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "minimum_improvement": self.minimum_improvement,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "optimizer": "AdamW",
            "loss": "unweighted_cross_entropy",
            "flooding_level": self.flooding_level,
            "early_stopping_signal": "inner_validation_macro_f1",
            "target_used_for_stopping": False,
        }


@dataclass
class SupervisedResult:
    state: dict[str, Any]
    state_sha256: str
    selected_epoch: int
    epochs_executed: int
    epoch_losses: tuple[float, ...]
    validation_macro_f1: tuple[float, ...]
    best_validation_macro_f1: float
    optimizer_steps: int
    wall_clock_seconds: float


def _epoch(
    backbone: Any,
    optimizer: Any,
    features: Any,
    targets: Any,
    *,
    seed: int,
    stage_offset: int,
    epoch: int,
    config: SupervisedConfig,
) -> tuple[float, int]:
    backbone.train()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(permutation_seed(seed, stage_offset, epoch))
    permutation = torch.randperm(targets.shape[0], generator=generator)
    total = 0.0
    steps = 0
    for start in range(0, targets.shape[0], config.batch_size):
        indices = permutation[start : start + config.batch_size]
        optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(backbone(features[indices]), targets[indices])
        if config.flooding_level is not None:
            level = float(config.flooding_level)
            loss = (loss - level).abs() + level
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(indices)
        steps += 1
    return total / float(targets.shape[0]), steps


def train_with_early_stopping(
    *,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    validation_inputs: np.ndarray,
    validation_labels: np.ndarray,
    seed: int,
    config: SupervisedConfig,
    trunk_state: dict[str, Any] | None = None,
    stage_offset: int = INNER_STAGE_OFFSET,
) -> SupervisedResult:
    """Canonical inner loop: early stopping on source pseudo-target validation only."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for supervised training")
    backbone = model_module.initialize_backbone(seed)
    if trunk_state is not None:
        backbone.load_trunk_state(trunk_state)

    features = torch.from_numpy(np.ascontiguousarray(train_inputs, dtype=np.float32))
    targets = torch.from_numpy(np.ascontiguousarray(train_labels, dtype=np.int64))
    optimizer = torch.optim.AdamW(
        backbone.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    started = time.perf_counter()
    losses: list[float] = []
    validation_scores: list[float] = []
    best_score = -np.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    stalled = 0
    steps = 0
    for epoch in range(1, config.maximum_epochs + 1):
        loss, epoch_steps = _epoch(
            backbone,
            optimizer,
            features,
            targets,
            seed=seed,
            stage_offset=stage_offset,
            epoch=epoch,
            config=config,
        )
        losses.append(loss)
        steps += epoch_steps
        logits = model_module.logits_numpy(backbone, validation_inputs)
        score = metrics.classification_metrics(
            validation_labels, metrics.argmax_predictions(logits)
        )["macro_f1"]
        validation_scores.append(score)
        if score > best_score + config.minimum_improvement:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(backbone.state_dict())
            stalled = 0
        else:
            stalled += 1
            if stalled >= config.early_stopping_patience:
                break
    if best_state is None:  # pragma: no cover - maximum_epochs >= 1 guarantees a state
        raise RuntimeError("Early stopping produced no state")
    return SupervisedResult(
        state=best_state,
        state_sha256=model_module.state_sha256(best_state),
        selected_epoch=best_epoch,
        epochs_executed=len(losses),
        epoch_losses=tuple(losses),
        validation_macro_f1=tuple(validation_scores),
        best_validation_macro_f1=float(best_score),
        optimizer_steps=steps,
        wall_clock_seconds=time.perf_counter() - started,
    )


def train_fixed_epochs(
    *,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    epochs: int,
    seed: int,
    config: SupervisedConfig,
    trunk_state: dict[str, Any] | None = None,
    stage_offset: int = FINAL_STAGE_OFFSET,
) -> SupervisedResult:
    """Final-model training for a source-decided epoch count, with no validation."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for supervised training")
    if epochs <= 0:
        raise ValueError("Final training requires a positive epoch count")
    backbone = model_module.initialize_backbone(seed)
    if trunk_state is not None:
        backbone.load_trunk_state(trunk_state)
    features = torch.from_numpy(np.ascontiguousarray(train_inputs, dtype=np.float32))
    targets = torch.from_numpy(np.ascontiguousarray(train_labels, dtype=np.int64))
    optimizer = torch.optim.AdamW(
        backbone.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    started = time.perf_counter()
    losses: list[float] = []
    steps = 0
    for epoch in range(1, epochs + 1):
        loss, epoch_steps = _epoch(
            backbone,
            optimizer,
            features,
            targets,
            seed=seed,
            stage_offset=stage_offset,
            epoch=epoch,
            config=config,
        )
        losses.append(loss)
        steps += epoch_steps
    state = copy.deepcopy(backbone.state_dict())
    return SupervisedResult(
        state=state,
        state_sha256=model_module.state_sha256(state),
        selected_epoch=epochs,
        epochs_executed=epochs,
        epoch_losses=tuple(losses),
        validation_macro_f1=(),
        best_validation_macro_f1=float("nan"),
        optimizer_steps=steps,
        wall_clock_seconds=time.perf_counter() - started,
    )


def median_epoch(selected_epochs: list[int]) -> int:
    """Per-seed final epoch rule: median of the fold-selected inner epochs.

    Verified against the canonical ``final_epochs_by_seed`` for all five seeds.
    """

    values = sorted(int(value) for value in selected_epochs)
    if not values:
        raise ValueError("Median epoch requires at least one fold result")
    return int(values[len(values) // 2]) if len(values) % 2 else int(
        round((values[len(values) // 2 - 1] + values[len(values) // 2]) / 2)
    )
