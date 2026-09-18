"""Self-supervised pretraining with an explicit balanced joint sampler.

An SSL corpus is a pair of standardized arrays for the *same* samples: the
first-difference view and the amplitude view. No labels of any kind are carried.

The joint sampler draws a declared fraction of every batch from each corpus, so the
Paper4/Data1 mixture is a preregistered constant rather than an emergent property of
corpus sizes. Every treatment runs the same number of optimizer steps at the same
batch size, which is what makes the SSL comparisons compute-matched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]

from . import model as model_module
from . import objectives
from .augmentations import random_window
from .carrier import SSL_WINDOW_LENGTH, FeatureStandardizer, first_difference

PAPER4_CORPUS = "paper4_source"
DATA1_CORPUS = "data1_external"
DATASET_INDEX = {PAPER4_CORPUS: 0, DATA1_CORPUS: 1}


class SslCorpusViolation(RuntimeError):
    pass


@dataclass(frozen=True)
class SslCorpus:
    """Label-free standardized SSL corpus for one dataset."""

    name: str
    difference: np.ndarray  # [N, L-1] float32, standardized
    amplitude: np.ndarray  # [N, L] float32, standardized
    zero_scale_replacements: int

    def __post_init__(self) -> None:
        if self.difference.shape[0] != self.amplitude.shape[0]:
            raise SslCorpusViolation(f"{self.name}: view row counts differ")
        if self.amplitude.shape[1] != self.difference.shape[1] + 1:
            raise SslCorpusViolation(f"{self.name}: amplitude/difference lengths are inconsistent")
        if self.difference.shape[1] < SSL_WINDOW_LENGTH:
            raise SslCorpusViolation(f"{self.name}: shorter than the SSL window")

    @property
    def sample_count(self) -> int:
        return int(self.difference.shape[0])


def build_ssl_corpus(name: str, raw_signals: np.ndarray) -> SslCorpus:
    """Standardize both views of one dataset using only that dataset's own statistics."""

    raw = np.asarray(raw_signals, dtype=np.float64)
    difference = first_difference(raw)
    difference_standardizer = FeatureStandardizer.fit(difference)
    amplitude_standardizer = FeatureStandardizer.fit(raw)
    return SslCorpus(
        name=name,
        difference=difference_standardizer.transform(difference),
        amplitude=amplitude_standardizer.transform(raw),
        zero_scale_replacements=(
            difference_standardizer.zero_scale_replacement_count
            + amplitude_standardizer.zero_scale_replacement_count
        ),
    )


@dataclass(frozen=True)
class PretrainConfig:
    objective: objectives.ObjectiveConfig
    steps_per_epoch: int = 32
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    window_length: int = SSL_WINDOW_LENGTH
    corpus_fractions: dict[str, float] = field(default_factory=lambda: {PAPER4_CORPUS: 1.0})
    domain_confusion_weight: float = 0.0
    seed_offset: int = 200000

    @property
    def total_steps(self) -> int:
        return int(self.steps_per_epoch * self.epochs)

    def as_record(self) -> dict[str, Any]:
        return {
            "objective": self.objective.as_record(),
            "steps_per_epoch": self.steps_per_epoch,
            "epochs": self.epochs,
            "total_optimizer_steps": self.total_steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "optimizer": "AdamW",
            "window_length": self.window_length,
            "corpus_fractions": dict(sorted(self.corpus_fractions.items())),
            "domain_confusion_weight": self.domain_confusion_weight,
            "seed_offset": self.seed_offset,
            "uses_any_label": False,
        }


@dataclass
class PretrainResult:
    trunk_state: dict[str, Any]
    trunk_sha256: str
    epoch_losses: tuple[float, ...]
    total_steps: int
    wall_clock_seconds: float
    trainable_parameters: int
    final_domain_accuracy: float | None


def _batch_allocation(fractions: dict[str, float], batch_size: int) -> dict[str, int]:
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise SslCorpusViolation(f"Corpus fractions must sum to one, got {total}")
    allocation = {name: int(round(batch_size * value)) for name, value in sorted(fractions.items())}
    drift = batch_size - sum(allocation.values())
    if drift:
        first = sorted(allocation)[0]
        allocation[first] += drift
    if any(count <= 0 for count in allocation.values()):
        raise SslCorpusViolation("Every declared corpus must contribute at least one row")
    return allocation


def pretrain(
    corpora: dict[str, SslCorpus],
    config: PretrainConfig,
    *,
    seed: int,
) -> PretrainResult:
    """Run self-supervised pretraining and return the trunk state only."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch is required for pretraining")
    declared = set(config.corpus_fractions)
    if declared != set(corpora):
        raise SslCorpusViolation(
            f"Declared corpora {sorted(declared)} do not match supplied {sorted(corpora)}"
        )
    allocation = _batch_allocation(config.corpus_fractions, config.batch_size)

    stream_seed = int(seed) + int(config.seed_offset)
    torch.manual_seed(stream_seed)
    generator = np.random.default_rng(stream_seed)

    backbone = model_module.initialize_backbone(seed)
    projection = model_module.initialize_auxiliary(
        model_module.ProjectionHead(), stream_seed + 1
    )
    decoder = model_module.initialize_auxiliary(model_module.SpanDecoder(), stream_seed + 2)
    domain_head = (
        model_module.initialize_auxiliary(model_module.DatasetDomainHead(), stream_seed + 3)
        if config.domain_confusion_weight > 0
        else None
    )

    parameters = list(backbone.parameters())
    if config.objective.objective_id in objectives.REQUIRES_PROJECTION:
        parameters += list(projection.parameters())
    if config.objective.objective_id in objectives.REQUIRES_DECODER:
        parameters += list(decoder.parameters())
    if domain_head is not None:
        parameters += list(domain_head.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )

    backbone.train()
    projection.train()
    decoder.train()
    if domain_head is not None:
        domain_head.train()

    started = time.perf_counter()
    epoch_losses: list[float] = []
    domain_accuracy: float | None = None
    for _ in range(config.epochs):
        running = 0.0
        for _ in range(config.steps_per_epoch):
            difference_parts: list[np.ndarray] = []
            amplitude_parts: list[np.ndarray] = []
            dataset_parts: list[np.ndarray] = []
            for name in sorted(allocation):
                corpus = corpora[name]
                count = allocation[name]
                rows = generator.integers(0, corpus.sample_count, count)
                starts = generator.integers(
                    0, corpus.difference.shape[1] - config.window_length + 1, count
                )
                offsets = starts[:, None] + np.arange(config.window_length)[None, :]
                difference_parts.append(
                    np.take_along_axis(corpus.difference[rows], offsets, axis=1)
                )
                amplitude_parts.append(
                    np.take_along_axis(corpus.amplitude[rows], offsets, axis=1)
                )
                dataset_parts.append(np.full(count, DATASET_INDEX[name], dtype=np.int64))

            difference_window = np.ascontiguousarray(
                np.concatenate(difference_parts, axis=0)[:, None, :], dtype=np.float32
            )
            amplitude_window = np.ascontiguousarray(
                np.concatenate(amplitude_parts, axis=0)[:, None, :], dtype=np.float32
            )
            dataset_identity = np.concatenate(dataset_parts, axis=0)

            optimizer.zero_grad(set_to_none=True)
            loss = objectives.compute_loss(
                config=config.objective,
                backbone=backbone,
                projection=projection,
                decoder=decoder,
                difference_window=difference_window,
                raw_window=amplitude_window,
                generator=generator,
            )
            if domain_head is not None:
                embeddings = backbone.encode(torch.from_numpy(difference_window))
                logits = domain_head(embeddings, 1.0)
                targets = torch.from_numpy(dataset_identity)
                domain_loss = functional.cross_entropy(logits, targets)
                loss = loss + config.domain_confusion_weight * domain_loss
                domain_accuracy = float(
                    (logits.detach().argmax(dim=1) == targets).float().mean()
                )
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
        epoch_losses.append(running / config.steps_per_epoch)

    trunk_state = {
        key: value.detach().cpu().clone() for key, value in backbone.trunk_state().items()
    }
    return PretrainResult(
        trunk_state=trunk_state,
        trunk_sha256=model_module.state_sha256(trunk_state),
        epoch_losses=tuple(epoch_losses),
        total_steps=config.total_steps,
        wall_clock_seconds=time.perf_counter() - started,
        trainable_parameters=int(sum(p.numel() for p in parameters)),
        final_domain_accuracy=domain_accuracy,
    )


def window_probe(corpus: SslCorpus, count: int, seed: int) -> np.ndarray:
    """Deterministic window sample, used by tests and diagnostics."""

    generator = np.random.default_rng(seed)
    rows = generator.integers(0, corpus.sample_count, count)
    return random_window(corpus.difference[rows], SSL_WINDOW_LENGTH, generator)
