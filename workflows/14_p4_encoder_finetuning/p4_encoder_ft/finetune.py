"""Deterministic support-only partial/full C1 fine-tuning."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from p4_factor_aware.metrics import metrics_from_predictions
from p4_factor_aware.models import FrozenC1CNN1D, model_state_sha256
from p4_factor_aware.protocol import ProtocolViolation

from .constants import (
    ENCODER_PARAMETER_COUNT,
    FULL_PARAMETER_COUNT,
    FULL_TRAINABLE_PREFIXES,
    HEAD_PARAMETER_COUNT,
    HEAD_PROXIMAL_LAMBDA,
    PARTIAL_PARAMETER_COUNT,
    PARTIAL_TRAINABLE_NAMES,
)


def array_sha256(value: np.ndarray) -> str:
    item = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(item.dtype).encode("ascii"))
    digest.update(str(tuple(item.shape)).encode("ascii"))
    digest.update(item.tobytes(order="C"))
    return digest.hexdigest()


def derive_seed(*parts: object, prefix: str = "P4_ENCODER_FT_V1") -> int:
    payload = "|".join([prefix, *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def clone_source_model(source: FrozenC1CNN1D) -> FrozenC1CNN1D:
    model = FrozenC1CNN1D()
    model.load_state_dict({name: value.detach().clone() for name, value in source.state_dict().items()}, strict=True)
    return model


def parameter_mask(model: FrozenC1CNN1D, arm: str) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], dict[str, bool]]:
    if arm not in {"PARTIAL_FT", "FULL_FT"}:
        raise ValueError("unknown fine-tuning arm")
    encoder: list[torch.nn.Parameter] = []
    head: list[torch.nn.Parameter] = []
    mask: dict[str, bool] = {}
    for name, parameter in model.named_parameters():
        trainable = name in PARTIAL_TRAINABLE_NAMES if arm == "PARTIAL_FT" else name.startswith(FULL_TRAINABLE_PREFIXES)
        parameter.requires_grad_(trainable)
        parameter.grad = None
        mask[name] = trainable
        if trainable:
            (head if name.startswith("network.13.") else encoder).append(parameter)
    observed = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    expected = PARTIAL_PARAMETER_COUNT if arm == "PARTIAL_FT" else FULL_PARAMETER_COUNT
    if observed != expected or sum(parameter.numel() for parameter in encoder) != expected - HEAD_PARAMETER_COUNT:
        raise ProtocolViolation("FINETUNING_PARAMETER_MASK_MISMATCH")
    return encoder, head, mask


def encoder_state_sha256(model: FrozenC1CNN1D) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith("network.13."):
            continue
        item = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(item.dtype).encode("ascii"))
        digest.update(str(tuple(item.shape)).encode("ascii"))
        digest.update(item.numpy().tobytes(order="C"))
    return digest.hexdigest()


def embedding_forward(model: FrozenC1CNN1D, inputs: torch.Tensor, *, batch_size: int = 512) -> torch.Tensor:
    batches = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            batches.append(model.network[:-1](inputs[start : start + batch_size]))
    return torch.cat(batches, dim=0).detach().cpu().contiguous()


def predict(model: FrozenC1CNN1D, inputs: torch.Tensor, *, batch_size: int = 512) -> tuple[np.ndarray, torch.Tensor]:
    model.eval()
    embeddings = embedding_forward(model, inputs, batch_size=batch_size)
    with torch.inference_mode():
        logits = model.network[-1](embeddings)
        predictions = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)
    return np.ascontiguousarray(predictions), embeddings


@dataclass
class FineTuneCheckpoint:
    epoch: int
    model: FrozenC1CNN1D
    diagnostics: dict[str, object]


def fine_tune(
    *,
    source_model: FrozenC1CNN1D,
    support_inputs: torch.Tensor,
    support_labels: np.ndarray,
    initial_head_weight: torch.Tensor,
    initial_head_bias: torch.Tensor,
    arm: str,
    learning_rate: float,
    encoder_weight_decay: float,
    epochs: int,
    seed: int,
    capture_epochs: Iterable[int] = (),
) -> list[FineTuneCheckpoint]:
    """Fine-tune without accepting any query object."""

    if epochs <= 0 or learning_rate <= 0 or encoder_weight_decay < 0:
        raise ValueError("invalid fine-tuning hyperparameters")
    labels_array = np.ascontiguousarray(support_labels, dtype=np.int64)
    counts = np.bincount(labels_array, minlength=7)
    if len(labels_array) < 7 or np.any(counts == 0) or int(counts.max() - counts.min()) > 1:
        raise ProtocolViolation("INVALID_FINETUNING_SUPPORT_CLASS_COUNTS")
    set_determinism(seed)
    model = clone_source_model(source_model)
    source_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    source_encoder_hash = encoder_state_sha256(model)
    anchor_weight = source_state["network.13.weight"].detach().clone()
    anchor_bias = source_state["network.13.bias"].detach().clone()
    with torch.no_grad():
        model.network[-1].weight.copy_(initial_head_weight)
        model.network[-1].bias.copy_(initial_head_bias)
    encoder_parameters, head_parameters, mask = parameter_mask(model, arm)
    trainable_count = sum(parameter.numel() for parameter in (*encoder_parameters, *head_parameters))
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "weight_decay": encoder_weight_decay},
            {"params": head_parameters, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        foreach=False,
        fused=False,
    )
    inputs = support_inputs.detach().cpu().to(dtype=torch.float32).contiguous()
    labels = torch.from_numpy(labels_array)
    capture = set(int(value) for value in capture_epochs) | {int(epochs)}
    checkpoints: list[FineTuneCheckpoint] = []
    trajectory: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        cross_entropy = F.cross_entropy(logits, labels, reduction="mean")
        head_displacement = torch.sum((model.network[-1].weight - anchor_weight) ** 2) + torch.sum((model.network[-1].bias - anchor_bias) ** 2)
        proximal = HEAD_PROXIMAL_LAMBDA * head_displacement / HEAD_PARAMETER_COUNT
        objective = cross_entropy + proximal
        if not torch.isfinite(objective):
            raise ProtocolViolation("NONFINITE_FINETUNING_OBJECTIVE")
        objective.backward()
        optimizer.step()
        trajectory.append({"epoch": epoch, "cross_entropy": float(cross_entropy.detach()), "proximal_penalty": float(proximal.detach()), "objective": float(objective.detach())})
        if epoch in capture:
            snapshot = clone_source_model(model)
            snapshot.eval()
            for parameter in snapshot.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            support_prediction, _ = predict(snapshot, inputs)
            support_metrics = metrics_from_predictions(labels_array, support_prediction)
            changed = []
            for name, value in snapshot.state_dict().items():
                if not torch.equal(value, source_state[name]):
                    changed.append(name)
            expected_changed = {name for name, is_trainable in mask.items() if is_trainable}
            if not set(changed).issubset(expected_changed) or not expected_changed.intersection(changed):
                raise ProtocolViolation("FINETUNING_CHANGED_PARAMETER_MASK_MISMATCH")
            encoder_displacement_sq = 0.0
            for name, value in snapshot.state_dict().items():
                if not name.startswith("network.13."):
                    encoder_displacement_sq += float(torch.sum((value - source_state[name]) ** 2))
            checkpoints.append(
                FineTuneCheckpoint(
                    epoch=epoch,
                    model=snapshot,
                    diagnostics={
                        "support_rows": len(labels_array),
                        "class_count_min": int(counts.min()),
                        "class_count_max": int(counts.max()),
                        "support_accuracy": float(support_metrics["accuracy"]),
                        "support_macro_f1": float(support_metrics["macro_f1"]),
                        "support_worst_class_f1": float(np.min(support_metrics["f1"])),
                        "final_cross_entropy": trajectory[-1]["cross_entropy"],
                        "final_proximal_penalty": trajectory[-1]["proximal_penalty"],
                        "final_objective": trajectory[-1]["objective"],
                        "all_objectives_finite": all(np.isfinite(row["objective"]) for row in trajectory),
                        "source_encoder_sha256": source_encoder_hash,
                        "adapted_encoder_sha256": encoder_state_sha256(snapshot),
                        "adapted_model_sha256": model_state_sha256(snapshot),
                        "encoder_displacement_frobenius": float(np.sqrt(encoder_displacement_sq)),
                        "changed_parameter_names": "|".join(sorted(changed)),
                        "trainable_parameter_names": "|".join(sorted(expected_changed)),
                        "trainable_parameter_count": trainable_count,
                        "optimizer_state_is_unit_local": True,
                    },
                )
            )
    if [item.epoch for item in checkpoints] != sorted(capture):
        raise ProtocolViolation("FINETUNING_CHECKPOINT_CAPTURE_MISMATCH")
    return checkpoints


def representation_displacement(source_embedding: torch.Tensor, adapted_embedding: torch.Tensor) -> dict[str, float]:
    source = source_embedding.detach().cpu().to(dtype=torch.float64)
    adapted = adapted_embedding.detach().cpu().to(dtype=torch.float64)
    if source.shape != adapted.shape or source.ndim != 2 or source.shape[1] != 256:
        raise ValueError("embedding shapes differ")
    cosine = F.cosine_similarity(source, adapted, dim=1, eps=1e-12)
    rms = torch.sqrt(torch.mean((adapted - source) ** 2))
    return {"embedding_mean_cosine_to_source": float(cosine.mean()), "embedding_rms_displacement": float(rms)}


def centroid_ratio(embeddings: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(embeddings, dtype=np.float64)
    groups = np.asarray(labels)
    unique = np.unique(groups)
    global_mean = values.mean(axis=0)
    between = 0.0
    within = 0.0
    for group in unique:
        selected = values[groups == group]
        center = selected.mean(axis=0)
        between += len(selected) * float(np.sum((center - global_mean) ** 2))
        within += float(np.sum((selected - center) ** 2))
    return between / max(within, 1e-15)
