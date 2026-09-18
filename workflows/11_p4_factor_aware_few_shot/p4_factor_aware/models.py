"""Hash-bound frozen C1 encoder, cosine prototypes, and historical head adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .constants import (
    CHECKPOINT_FILE_HASHES,
    CHECKPOINT_STATE_HASHES,
    DIFFERENCED_SIGNAL_LENGTH,
    EMBEDDING_DIMENSION,
    HEAD_LAMBDAS,
    HEAD_SUPPORT_FIT_ACCURACY_THRESHOLD,
    HEAD_SUPPORT_FIT_MACRO_F1_THRESHOLD,
    PREPROCESSING_FILES,
    RAW_SIGNAL_LENGTH,
)
from .metrics import metrics_from_predictions
from .protocol import ProtocolViolation


EXPECTED_PARAMETER_COUNT = 142_855
HEAD_PARAMETER_COUNT = 7 * EMBEDDING_DIMENSION + 7


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenC1CNN1D(nn.Module):
    """Exact canonical C1 architecture from the released Few-Shot lineage."""

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


def load_frozen_model(checkpoint_path: str | Path, seed: int) -> tuple[FrozenC1CNN1D, dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.is_file() or sha256_file(path) != CHECKPOINT_FILE_HASHES[seed]:
        raise ProtocolViolation("FROZEN_CHECKPOINT_HASH_MISMATCH")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("method_id") != "C1_FIRST_DIFFERENCE_ERM_1DCNN"
        or int(payload.get("seed")) != seed
        or bool(payload.get("p4_used"))
    ):
        raise ProtocolViolation("FROZEN_CHECKPOINT_IDENTITY_MISMATCH")
    model = FrozenC1CNN1D()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise ProtocolViolation("FROZEN_CHECKPOINT_ARCHITECTURE_MISMATCH")
    observed_state = model_state_sha256(model)
    if observed_state != CHECKPOINT_STATE_HASHES[seed] or observed_state != payload["model_state_sha256"]:
        raise ProtocolViolation("FROZEN_CHECKPOINT_STATE_MISMATCH")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return model, payload


def load_preprocessing(preprocessing_directory: str | Path) -> tuple[np.ndarray, np.ndarray]:
    root = Path(preprocessing_directory)
    arrays = []
    for filename, expected_hash in PREPROCESSING_FILES.items():
        path = root / filename
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ProtocolViolation("FROZEN_PREPROCESSING_HASH_MISMATCH")
        value = np.load(path, allow_pickle=False)
        if value.shape != (DIFFERENCED_SIGNAL_LENGTH,) or value.dtype != np.float64:
            raise ProtocolViolation("FROZEN_PREPROCESSING_SHAPE_MISMATCH")
        arrays.append(np.ascontiguousarray(value, dtype=np.float64))
    mean, scale = arrays
    if np.any(scale < 1e-12) or not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ProtocolViolation("FROZEN_PREPROCESSING_VALUE_MISMATCH")
    return mean, scale


def preprocess(raw_signals: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> torch.Tensor:
    values = np.asarray(raw_signals, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != RAW_SIGNAL_LENGTH:
        raise ValueError("raw signals must be [N, 281]")
    transformed = (np.diff(values, axis=1) - mean) / scale
    result = torch.from_numpy(np.ascontiguousarray(transformed, dtype=np.float32))[:, None, :]
    if result.shape[1:] != (1, DIFFERENCED_SIGNAL_LENGTH):
        raise ProtocolViolation("FIRST_DIFFERENCE_DIMENSION_MISMATCH")
    return result


def extract_embeddings(
    model: FrozenC1CNN1D, inputs: torch.Tensor, *, batch_size: int = 512
) -> torch.Tensor:
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ProtocolViolation("SOURCE_ENCODER_NOT_FROZEN")
    batches = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            batches.append(model.network[:-1](inputs[start : start + batch_size]))
    embeddings = torch.cat(batches, dim=0).cpu().contiguous()
    if embeddings.dtype != torch.float32 or embeddings.shape != (len(inputs), EMBEDDING_DIMENSION):
        raise ProtocolViolation("FROZEN_EMBEDDING_DIMENSION_MISMATCH")
    return embeddings


def frozen_source_predictions(model: FrozenC1CNN1D, query_embeddings: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        logits = model.network[-1](query_embeddings.to(dtype=torch.float32))
        predictions = torch.argmax(logits, dim=1)
    return np.ascontiguousarray(predictions.cpu().numpy(), dtype=np.int64)


def _l2_normalize(values: torch.Tensor, *, axis: int) -> torch.Tensor:
    norms = torch.linalg.vector_norm(values, dim=axis, keepdim=True)
    if torch.any(~torch.isfinite(norms)) or torch.any(norms <= 0):
        raise ProtocolViolation("ZERO_OR_NONFINITE_COSINE_NORM")
    return values / norms


def build_cosine_prototypes(
    support_embeddings: torch.Tensor, support_labels: np.ndarray, shot_count: int
) -> torch.Tensor:
    embeddings = support_embeddings.detach().cpu().to(dtype=torch.float64)
    labels = np.asarray(support_labels, dtype=np.int64)
    if embeddings.shape != (7 * shot_count, EMBEDDING_DIMENSION):
        raise ValueError("support embeddings violate the matched shot budget")
    normalized = _l2_normalize(embeddings, axis=1)
    prototypes = []
    for class_index in range(7):
        selected = normalized[torch.from_numpy(np.flatnonzero(labels == class_index))]
        if selected.shape != (shot_count, EMBEDDING_DIMENSION):
            raise ValueError("support labels are not class-balanced")
        prototypes.append(_l2_normalize(selected.mean(dim=0, keepdim=True), axis=1)[0])
    result = torch.stack(prototypes, dim=0)
    if result.shape != (7, EMBEDDING_DIMENSION):
        raise ProtocolViolation("COSINE_PROTOTYPE_SHAPE_MISMATCH")
    return result


def cosine_similarity_predict(
    query_embeddings: torch.Tensor, prototypes: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    queries = _l2_normalize(query_embeddings.detach().cpu().to(dtype=torch.float64), axis=1)
    centers = _l2_normalize(prototypes.detach().cpu().to(dtype=torch.float64), axis=1)
    similarities = queries @ centers.T
    predictions = torch.argmax(similarities, dim=1)
    return (
        np.ascontiguousarray(predictions.numpy(), dtype=np.int64),
        np.ascontiguousarray(similarities.numpy(), dtype=np.float64),
    )


def _named_head_sha256(weight: torch.Tensor, bias: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(
        (("network.13.weight", weight), ("network.13.bias", bias))
    ):
        item = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(item.dtype).encode("ascii"))
        digest.update(str(tuple(item.shape)).encode("ascii"))
        digest.update(item.numpy().tobytes(order="C"))
    return digest.hexdigest()


def adapt_linear_head(
    *,
    support_embeddings: torch.Tensor,
    support_labels: np.ndarray,
    original_weight: torch.Tensor,
    original_bias: torch.Tensor,
    shot_count: int,
) -> dict[str, Any]:
    """Reuse the released full-batch float64 LBFGS proximal-head recipe exactly."""

    regularization_lambda = HEAD_LAMBDAS[shot_count]
    embeddings = support_embeddings.detach().cpu().to(dtype=torch.float64).contiguous()
    labels = torch.from_numpy(np.ascontiguousarray(support_labels, dtype=np.int64))
    if embeddings.shape != (7 * shot_count, EMBEDDING_DIMENSION):
        raise ValueError("support embeddings violate the matched shot budget")
    if not torch.all(torch.bincount(labels, minlength=7) == shot_count):
        raise ValueError("head support must be class-balanced")
    anchor_weight = original_weight.detach().cpu().to(dtype=torch.float64).contiguous().clone()
    anchor_bias = original_bias.detach().cpu().to(dtype=torch.float64).contiguous().clone()
    anchor_hash = _named_head_sha256(anchor_weight, anchor_bias)
    weight = anchor_weight.clone().requires_grad_(True)
    bias = anchor_bias.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [weight, bias],
        max_iter=100,
        history_size=20,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )
    trajectory: list[float] = []

    def components() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = F.linear(embeddings, weight, bias)
        support_ce = F.cross_entropy(logits, labels, reduction="mean")
        displacement = torch.sum((weight - anchor_weight) ** 2) + torch.sum(
            (bias - anchor_bias) ** 2
        )
        proximal = regularization_lambda * displacement / HEAD_PARAMETER_COUNT
        return support_ce, proximal, support_ce + proximal

    with torch.no_grad():
        initial_ce, _, initial_objective = components()

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        _, _, objective = components()
        if not torch.isfinite(objective):
            raise ProtocolViolation("NONFINITE_HISTORICAL_HEAD_OBJECTIVE")
        trajectory.append(float(objective.detach()))
        objective.backward()
        return objective

    optimizer.step(closure)
    final_ce, final_proximal, final_objective = components()
    state = optimizer.state[weight]
    weight32 = weight.detach().to(dtype=torch.float32).contiguous()
    bias32 = bias.detach().to(dtype=torch.float32).contiguous()
    with torch.inference_mode():
        support_prediction = torch.argmax(
            F.linear(embeddings.to(torch.float32), weight32, bias32), dim=1
        ).numpy()
    support_metrics = metrics_from_predictions(np.asarray(support_labels), support_prediction)
    fit_reached = bool(
        support_metrics["accuracy"] >= HEAD_SUPPORT_FIT_ACCURACY_THRESHOLD
        and support_metrics["macro_f1"] >= HEAD_SUPPORT_FIT_MACRO_F1_THRESHOLD
    )
    if _named_head_sha256(anchor_weight, anchor_bias) != anchor_hash:
        raise ProtocolViolation("SOURCE_HEAD_ANCHOR_CHANGED")
    return {
        "weight": weight32,
        "bias": bias32,
        "regularization_lambda": regularization_lambda,
        "initial_support_cross_entropy": float(initial_ce),
        "support_cross_entropy": float(final_ce.detach()),
        "proximal_penalty": float(final_proximal.detach()),
        "total_objective": float(final_objective.detach()),
        "optimizer_iterations": int(state.get("n_iter", 0)),
        "optimizer_function_evaluations": int(state.get("func_evals", 0)),
        "loss_trajectory": trajectory,
        "support_accuracy": float(support_metrics["accuracy"]),
        "support_macro_f1": float(support_metrics["macro_f1"]),
        "support_fit_threshold_reached": fit_reached,
        "adapted_head_sha256": _named_head_sha256(weight32, bias32),
        "anchor_head_sha256": anchor_hash,
        "optimizer_completed_finite": bool(
            torch.isfinite(final_objective)
            and float(final_objective.detach()) <= float(initial_objective) + 1e-12
            and int(state.get("n_iter", 0)) <= 100
        ),
    }


def linear_head_predict(
    query_embeddings: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> np.ndarray:
    with torch.inference_mode():
        predictions = torch.argmax(
            F.linear(query_embeddings.to(torch.float32), weight, bias), dim=1
        )
    return np.ascontiguousarray(predictions.numpy(), dtype=np.int64)
