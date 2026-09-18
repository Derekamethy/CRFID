"""Deterministic source-anchored linear softmax fitting on P4 support only."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from p4_factor_aware.metrics import metrics_from_predictions
from p4_factor_aware.protocol import ProtocolViolation

from .constants import HEAD_PARAMETER_COUNT, SELECTED_LAMBDA


def _head_sha256(weight: torch.Tensor, bias: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for name, value in (("bias", bias), ("weight", weight)):
        item = value.detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(item.dtype).encode("ascii"))
        digest.update(str(tuple(item.shape)).encode("ascii"))
        digest.update(item.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class LinearFit:
    weight: torch.Tensor
    bias: torch.Tensor
    diagnostics: dict[str, object]


def fit_linear_readout(
    *,
    support_embeddings: torch.Tensor,
    support_labels: np.ndarray,
    anchor_weight: torch.Tensor,
    anchor_bias: torch.Tensor,
    regularization_lambda: float = SELECTED_LAMBDA,
) -> LinearFit:
    """Fit only a cloned linear head; query objects are intentionally absent."""

    embeddings = support_embeddings.detach().cpu().to(dtype=torch.float64).contiguous()
    labels_array = np.ascontiguousarray(support_labels, dtype=np.int64)
    labels = torch.from_numpy(labels_array)
    if embeddings.ndim != 2 or embeddings.shape[1] != 256 or len(embeddings) != len(labels):
        raise ValueError("support embeddings must be [budget,256]")
    counts = np.bincount(labels_array, minlength=7)
    if len(embeddings) < 7 or np.any(counts == 0) or int(counts.max() - counts.min()) > 1:
        raise ProtocolViolation("INVALID_LINEAR_SUPPORT_CLASS_COUNTS")
    if not np.isfinite(regularization_lambda) or regularization_lambda <= 0:
        raise ValueError("regularization lambda must be positive")

    anchor_w = anchor_weight.detach().cpu().to(dtype=torch.float64).contiguous().clone()
    anchor_b = anchor_bias.detach().cpu().to(dtype=torch.float64).contiguous().clone()
    if anchor_w.shape != (7, 256) or anchor_b.shape != (7,):
        raise ValueError("source head must be Linear(256,7)")
    anchor_hash = _head_sha256(anchor_w, anchor_b)
    weight = anchor_w.clone().requires_grad_(True)
    bias = anchor_b.clone().requires_grad_(True)
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
        cross_entropy = F.cross_entropy(logits, labels, reduction="mean")
        displacement = torch.sum((weight - anchor_w) ** 2) + torch.sum((bias - anchor_b) ** 2)
        proximal = regularization_lambda * displacement / HEAD_PARAMETER_COUNT
        return cross_entropy, proximal, cross_entropy + proximal

    with torch.no_grad():
        initial_ce, _, initial_objective = components()

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        _, _, objective = components()
        if not torch.isfinite(objective):
            raise ProtocolViolation("NONFINITE_LINEAR_OBJECTIVE")
        trajectory.append(float(objective.detach()))
        objective.backward()
        return objective

    optimizer.step(closure)
    final_ce, final_proximal, final_objective = components()
    gradients = torch.autograd.grad(final_objective, (weight, bias))
    gradient_inf = max(float(torch.max(torch.abs(value))) for value in gradients)
    state = optimizer.state[weight]
    weight32 = weight.detach().to(dtype=torch.float32).contiguous()
    bias32 = bias.detach().to(dtype=torch.float32).contiguous()
    with torch.inference_mode():
        support_prediction = torch.argmax(
            F.linear(embeddings.to(dtype=torch.float32), weight32, bias32), dim=1
        ).numpy()
    support_metrics = metrics_from_predictions(labels_array, support_prediction)
    anchor_unchanged = _head_sha256(anchor_w, anchor_b) == anchor_hash
    completed = bool(
        anchor_unchanged
        and torch.isfinite(final_objective)
        and float(final_objective.detach()) <= float(initial_objective) + 1e-12
        and int(state.get("n_iter", 0)) <= 100
    )
    if not anchor_unchanged:
        raise ProtocolViolation("SOURCE_HEAD_ANCHOR_CHANGED")
    return LinearFit(
        weight=weight32,
        bias=bias32,
        diagnostics={
            "support_rows": len(labels_array),
            "class_count_min": int(counts.min()),
            "class_count_max": int(counts.max()),
            "regularization_lambda": float(regularization_lambda),
            "initial_support_cross_entropy": float(initial_ce),
            "support_cross_entropy": float(final_ce.detach()),
            "proximal_penalty": float(final_proximal.detach()),
            "initial_objective": float(initial_objective),
            "final_objective": float(final_objective.detach()),
            "objective_reduction": float(initial_objective - final_objective.detach()),
            "gradient_inf_norm": gradient_inf,
            "optimizer_iterations": int(state.get("n_iter", 0)),
            "optimizer_function_evaluations": int(state.get("func_evals", 0)),
            "closure_evaluations": len(trajectory),
            "support_accuracy": float(support_metrics["accuracy"]),
            "support_macro_f1": float(support_metrics["macro_f1"]),
            "support_worst_class_f1": float(np.min(support_metrics["f1"])),
            "weight_displacement_frobenius": float(torch.linalg.vector_norm(weight.detach() - anchor_w)),
            "bias_displacement_l2": float(torch.linalg.vector_norm(bias.detach() - anchor_b)),
            "anchor_head_sha256": anchor_hash,
            "adapted_head_sha256": _head_sha256(weight32, bias32),
            "optimizer_completed_finite": completed,
        },
    )


def linear_predict(embeddings: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> np.ndarray:
    values = embeddings.detach().cpu().to(dtype=torch.float32)
    with torch.inference_mode():
        prediction = torch.argmax(F.linear(values, weight, bias), dim=1)
    return np.ascontiguousarray(prediction.numpy(), dtype=np.int64)
