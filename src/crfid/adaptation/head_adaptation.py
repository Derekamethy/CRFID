"""Support-only proximal linear-head adaptation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..exceptions import DependencyUnavailableError


LAMBDA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def adapt_linear_head_lbfgs(
    support_embeddings: np.ndarray,
    support_labels: np.ndarray,
    original_weight: np.ndarray,
    original_bias: np.ndarray,
    *,
    regularization_lambda: float,
) -> dict[str, Any]:
    """Adapt from labeled support only; the interface accepts no query labels."""

    if regularization_lambda not in LAMBDA_GRID:
        raise ValueError("regularization_lambda is outside the frozen grid")
    try:
        import torch
        import torch.nn.functional as functional
    except ModuleNotFoundError as exc:
        raise DependencyUnavailableError("PyTorch is required for LBFGS head adaptation") from exc
    embeddings = torch.as_tensor(np.asarray(support_embeddings), dtype=torch.float64)
    labels = torch.as_tensor(np.asarray(support_labels), dtype=torch.int64)
    anchor_weight = torch.as_tensor(np.asarray(original_weight), dtype=torch.float64).clone()
    anchor_bias = torch.as_tensor(np.asarray(original_bias), dtype=torch.float64).clone()
    if embeddings.ndim != 2 or labels.shape != (len(embeddings),):
        raise ValueError("Support embeddings and labels are not aligned")
    class_count, dimension = anchor_weight.shape
    if anchor_bias.shape != (class_count,) or embeddings.shape[1] != dimension:
        raise ValueError("Source-head shapes are invalid")
    counts = torch.bincount(labels, minlength=class_count)
    if torch.any(counts == 0) or not torch.all(counts == counts[0]):
        raise ValueError("Support must be balanced and cover every class")
    weight = anchor_weight.clone().requires_grad_(True)
    bias = anchor_bias.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [weight, bias], max_iter=100, history_size=20, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12
    )
    parameter_count = float(weight.numel() + bias.numel())

    def objective() -> Any:
        cross_entropy = functional.cross_entropy(functional.linear(embeddings, weight, bias), labels)
        proximal = regularization_lambda * (
            torch.sum((weight - anchor_weight) ** 2) + torch.sum((bias - anchor_bias) ** 2)
        ) / parameter_count
        return cross_entropy + proximal

    def closure() -> Any:
        optimizer.zero_grad(set_to_none=True)
        value = objective()
        if not torch.isfinite(value):
            raise RuntimeError("Non-finite head-adaptation objective")
        value.backward()
        return value

    initial = float(objective().detach())
    optimizer.step(closure)
    final = float(objective().detach())
    if final > initial + 1e-12:
        raise RuntimeError("Head adaptation did not reduce its support objective")
    return {
        "weight": weight.detach().to(dtype=torch.float32).numpy(),
        "bias": bias.detach().to(dtype=torch.float32).numpy(),
        "initial_objective": initial,
        "final_objective": final,
        "regularization_lambda": regularization_lambda,
    }
