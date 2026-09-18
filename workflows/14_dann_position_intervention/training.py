"""Matched ERM/DANN development and final source-only training."""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import (
    PreprocessingState,
    SourceData,
    balanced_epoch_indices,
    batch_composition,
    domain_labels,
    final_probe_masks,
    fit_first_difference,
    transform,
)
from metrics import (
    aggregate_blocks,
    classification_metrics,
    condition_centroids,
    independent_probes,
    representation_diagnostics,
)
from model import C1PositionDANN
from protocol import HELD_POSITION, SEEDS, grl_strength


OPTIMIZER = {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "betas": (0.9, 0.999),
    "epsilon": 1e-8,
    "amsgrad": False,
    "foreach": False,
    "fused": False,
}
BATCH_SIZE = 256
MAXIMUM_EPOCHS = 50
PATIENCE = 8
MINIMUM_IMPROVEMENT = 1e-12


def _optimizer(model: nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=OPTIMIZER["learning_rate"],
        weight_decay=OPTIMIZER["weight_decay"],
        betas=OPTIMIZER["betas"],
        eps=OPTIMIZER["epsilon"],
        amsgrad=OPTIMIZER["amsgrad"],
        foreach=OPTIMIZER["foreach"],
        fused=OPTIMIZER["fused"],
    )


def model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _gradient_norm(parameters: list[torch.Tensor]) -> float:
    values = [parameter.grad.detach().norm().item() for parameter in parameters if parameter.grad is not None]
    return float(sum(values))


def _train_epoch(
    *,
    model: C1PositionDANN,
    optimizer: torch.optim.Optimizer,
    inputs: np.ndarray,
    tag_labels: np.ndarray,
    positions: np.ndarray,
    lambda_max: float,
    seed: int,
    epoch: int,
    progress_epoch_budget: int,
) -> tuple[dict, dict | None]:
    model.train()
    order = balanced_epoch_indices(positions, tag_labels, seed=seed, epoch=epoch)
    batches = math.ceil(len(order) / BATCH_SIZE)
    domain, domain_order = domain_labels(positions)
    tag_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    total_tag = total_domain = total = 0.0
    count = 0
    first_audit = None
    for batch_index, start in enumerate(range(0, len(order), BATCH_SIZE)):
        selected = order[start : start + BATCH_SIZE]
        tensor = torch.from_numpy(inputs[selected]).unsqueeze(1)
        tag_tensor = torch.from_numpy(tag_labels[selected].astype(np.int64, copy=False))
        optimizer.zero_grad(set_to_none=True)
        if lambda_max == 0.0:
            tag_logits = model(tensor)
            tag_loss = tag_criterion(tag_logits, tag_tensor)
            domain_loss = torch.zeros((), dtype=tag_loss.dtype)
            strength = 0.0
            loss = tag_loss
        else:
            denominator = max(1, int(progress_epoch_budget) * batches - 1)
            progress = ((epoch - 1) * batches + batch_index) / denominator
            strength = grl_strength(lambda_max, min(1.0, progress))
            tag_logits, domain_logits, _ = model.forward_dann(tensor, grl_strength=strength)
            tag_loss = tag_criterion(tag_logits, tag_tensor)
            domain_tensor = torch.from_numpy(domain[selected].astype(np.int64, copy=False))
            domain_loss = domain_criterion(domain_logits, domain_tensor)
            loss = tag_loss + domain_loss
        loss.backward()
        if epoch == 1 and batch_index == 0:
            first_audit = {
                "tagid_encoder_gradient_norm": _gradient_norm(list(model.encoder.parameters())),
                "tagid_head_gradient_norm": _gradient_norm(list(model.tag_head.parameters())),
                "domain_head_gradient_norm": (
                    _gradient_norm(list(model.domain_head.parameters()))
                    if model.domain_head is not None
                    else 0.0
                ),
                "grl_strength": float(strength),
                "position_order": list(domain_order),
                "batch_composition": batch_composition(order, positions, tag_labels)[0],
            }
        optimizer.step()
        batch_count = len(selected)
        total += float(loss.detach()) * batch_count
        total_tag += float(tag_loss.detach()) * batch_count
        total_domain += float(domain_loss.detach()) * batch_count
        count += batch_count
    return {
        "total_loss": total / count,
        "tagid_loss": total_tag / count,
        "domain_loss": total_domain / count,
        "sample_count": count,
    }, first_audit


def infer(
    model: C1PositionDANN,
    inputs: np.ndarray,
    *,
    positions: np.ndarray | None = None,
) -> dict:
    model.eval()
    tag_logits = []
    embeddings = []
    domain_logits = []
    with torch.inference_mode():
        for start in range(0, len(inputs), BATCH_SIZE):
            tensor = torch.from_numpy(inputs[start : start + BATCH_SIZE]).unsqueeze(1)
            embedding = model.encode(tensor)
            embeddings.append(embedding.cpu())
            tag_logits.append(model.tag_head(embedding).cpu())
            if model.domain_head is not None:
                domain_logits.append(model.domain_head(embedding).cpu())
    result = {
        "tag_logits": torch.cat(tag_logits).numpy().astype(np.float32),
        "embeddings": torch.cat(embeddings).numpy().astype(np.float32),
    }
    if domain_logits:
        result["domain_logits"] = torch.cat(domain_logits).numpy().astype(np.float32)
    return result


def _evaluate(
    model: C1PositionDANN,
    inputs: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
) -> dict:
    output = infer(model, inputs, positions=positions)
    predicted = np.argmax(output["tag_logits"], axis=1).astype(np.int64)
    result = {
        "tagid": classification_metrics(labels, predicted, class_count=7),
        "tag_predictions": predicted,
        "embeddings": output["embeddings"],
    }
    unique_positions = tuple(sorted(np.unique(np.asarray(positions, dtype=str)).tolist()))
    expected_domain_count = (
        model.domain_head.network[-1].out_features
        if model.domain_head is not None
        else 0
    )
    if "domain_logits" in output and len(unique_positions) == expected_domain_count:
        domain, order = domain_labels(positions)
        domain_predictions = np.argmax(output["domain_logits"], axis=1).astype(np.int64)
        result["domain"] = classification_metrics(domain, domain_predictions, class_count=len(order))
        result["domain_order"] = order
    elif "domain_logits" in output:
        result["domain_evaluation_skipped"] = (
            "partition positions are outside the training-domain classifier label space"
        )
    return result


def intervention_validity(seed: int = 42) -> dict:
    """Direct learning-validity checks, independent of scientific results."""

    torch.manual_seed(seed)
    inputs = torch.randn(16, 1, 280)
    tag_labels = torch.arange(16) % 7
    domain_labels_tensor = torch.arange(16) % 2
    model = C1PositionDANN(seed=seed, domain_count=2)
    tag_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    model.zero_grad(set_to_none=True)
    embeddings = model.encode(inputs)
    tag_loss = tag_criterion(model.tag_head(embeddings), tag_labels)
    tag_loss.backward()
    tag_encoder_norm = _gradient_norm(list(model.encoder.parameters()))
    tag_head_norm = _gradient_norm(list(model.tag_head.parameters()))

    model.zero_grad(set_to_none=True)
    embeddings = model.encode(inputs)
    domain_loss_plain = domain_criterion(model.domain_head(embeddings), domain_labels_tensor)
    domain_loss_plain.backward()
    plain = [parameter.grad.detach().clone() for parameter in model.encoder.parameters()]

    model.zero_grad(set_to_none=True)
    _, domain_logits, _ = model.forward_dann(inputs, grl_strength=1.0)
    domain_loss_reversed = domain_criterion(domain_logits, domain_labels_tensor)
    domain_loss_reversed.backward()
    reversed_gradients = [parameter.grad.detach().clone() for parameter in model.encoder.parameters()]
    sign_matches = [torch.allclose(right, -left, rtol=1e-5, atol=1e-7) for left, right in zip(plain, reversed_gradients)]
    domain_head_norm = _gradient_norm(list(model.domain_head.parameters()))

    before = [parameter.detach().clone() for parameter in model.domain_head.parameters()]
    optimizer = _optimizer(model)
    optimizer.step()
    updated = any(not torch.equal(left, right) for left, right in zip(before, model.domain_head.parameters()))

    erm = C1PositionDANN(seed=seed, domain_count=None)
    dann_zero = C1PositionDANN(seed=seed, domain_count=2)
    initial_equal = torch.equal(erm(inputs), dann_zero(inputs))
    return {
        "tagid_encoder_gradient_nonzero": tag_encoder_norm > 0.0,
        "tagid_head_gradient_nonzero": tag_head_norm > 0.0,
        "domain_head_gradient_nonzero": domain_head_norm > 0.0,
        "domain_head_parameters_update": updated,
        "encoder_domain_gradient_nonzero": any(torch.count_nonzero(value).item() > 0 for value in plain),
        "gradient_reversal_exact_sign": all(sign_matches),
        "no_double_reversal": all(sign_matches),
        "lambda_zero_initial_tag_logits_equal": initial_equal,
        "domain_loss_can_access_p4": False,
        "passed": all(
            (
                tag_encoder_norm > 0.0,
                tag_head_norm > 0.0,
                domain_head_norm > 0.0,
                updated,
                all(sign_matches),
                initial_equal,
            )
        ),
    }


def development_run(
    data: SourceData,
    *,
    fold: str,
    seed: int,
    lambda_max: float,
    probe_seeds: tuple[int, ...] = SEEDS,
) -> dict:
    parts = data.splits[fold]
    positions_all = data.positions
    inner_state = fit_first_difference(data.signals, parts["inner_train"])
    inner_inputs = transform(data.signals, parts["inner_train"], inner_state)
    validation_inputs = transform(data.signals, parts["inner_validation"], inner_state)
    inner_positions = positions_all[parts["inner_train"]]
    validation_positions = positions_all[parts["inner_validation"]]
    domain_count = None if lambda_max == 0.0 else 2
    model = C1PositionDANN(seed=seed, domain_count=domain_count)
    optimizer = _optimizer(model)
    best_metric = -float("inf")
    best_epoch = 0
    best_state = None
    without_improvement = 0
    history = []
    first_step = None
    for epoch in range(1, MAXIMUM_EPOCHS + 1):
        losses, audit = _train_epoch(
            model=model,
            optimizer=optimizer,
            inputs=inner_inputs,
            tag_labels=data.labels[parts["inner_train"]],
            positions=inner_positions,
            lambda_max=lambda_max,
            seed=seed,
            epoch=epoch,
            progress_epoch_budget=MAXIMUM_EPOCHS,
        )
        if audit is not None:
            first_step = audit
        validation = _evaluate(
            model,
            validation_inputs,
            data.labels[parts["inner_validation"]],
            validation_positions,
        )
        metric = validation["tagid"]["macro_f1"]
        improved = metric > best_metric + MINIMUM_IMPROVEMENT
        if improved:
            best_metric = metric
            best_epoch = epoch
            best_state = _clone_state(model)
            without_improvement = 0
        else:
            without_improvement += 1
        history.append(
            {
                "epoch": epoch,
                **losses,
                "validation_accuracy": validation["tagid"]["accuracy"],
                "validation_macro_f1": metric,
                "domain_validation_accuracy": validation.get("domain", {}).get("accuracy"),
                "improved": improved,
            }
        )
        if without_improvement >= PATIENCE:
            break
    if best_state is None or first_step is None:
        raise RuntimeError("Source-only checkpoint selection failed")

    outer_indices = np.sort(np.concatenate((parts["inner_train"], parts["inner_validation"])))
    outer_state = fit_first_difference(data.signals, outer_indices)
    outer_inputs = transform(data.signals, outer_indices, outer_state)
    outer_positions = positions_all[outer_indices]
    outer_model = C1PositionDANN(seed=seed, domain_count=domain_count)
    outer_optimizer = _optimizer(outer_model)
    outer_first_step = None
    outer_history = []
    for epoch in range(1, best_epoch + 1):
        losses, audit = _train_epoch(
            model=outer_model,
            optimizer=outer_optimizer,
            inputs=outer_inputs,
            tag_labels=data.labels[outer_indices],
            positions=outer_positions,
            lambda_max=lambda_max,
            seed=seed,
            epoch=epoch,
            progress_epoch_budget=MAXIMUM_EPOCHS,
        )
        if audit is not None:
            outer_first_step = audit
        outer_history.append({"epoch": epoch, **losses})
    held_inputs = transform(data.signals, parts["outer_held"], outer_state)
    held = _evaluate(
        outer_model,
        held_inputs,
        data.labels[parts["outer_held"]],
        positions_all[parts["outer_held"]],
    )
    development = _evaluate(
        outer_model,
        outer_inputs,
        data.labels[outer_indices],
        outer_positions,
    )
    held_blocks = aggregate_blocks(
        data.labels[parts["outer_held"]],
        held["tag_predictions"],
        data.condition_ids[parts["outer_held"]],
    )
    centroids = condition_centroids(
        development["embeddings"],
        data.labels[outer_indices],
        outer_positions,
        data.condition_ids[outer_indices],
    )
    train_conditions = set(data.condition_ids[parts["inner_train"]].tolist())
    test_conditions = set(data.condition_ids[parts["inner_validation"]].tolist())
    probes = independent_probes(
        centroids, train_conditions, test_conditions, probe_seeds=probe_seeds
    )
    diagnostics = representation_diagnostics(centroids)
    return {
        "fold": fold,
        "held_position": HELD_POSITION[fold],
        "seed": int(seed),
        "lambda_max": float(lambda_max),
        "method": "ERM" if lambda_max == 0.0 else "DANN",
        "selected_epoch": best_epoch,
        "epochs_executed": len(history),
        "checkpoint_metric": "inner_validation_row_macro_f1",
        "checkpoint_used_domain_accuracy": False,
        "held_row_metrics": held["tagid"],
        "held_block_metrics": held_blocks["metrics"],
        "source_development_metrics": development["tagid"],
        "domain_train_accuracy": development.get("domain", {}).get("accuracy"),
        "domain_validation_accuracy": history[best_epoch - 1]["domain_validation_accuracy"],
        "position_probe_balanced_accuracy": probes["position_balanced_accuracy_mean"],
        "position_probe_macro_f1": probes["position_macro_f1_mean"],
        "tagid_probe_macro_f1": probes["tagid_macro_f1_mean"],
        "probe_position_chance": probes["position_chance"],
        "probe_results": {
            "position": [
                {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in row.items()}
                for row in probes["position"]
            ],
            "tagid": [
                {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in row.items()}
                for row in probes["tagid"]
            ],
        },
        "representation": diagnostics,
        "model_state_sha256": model_state_sha256(outer_model),
        "inner_preprocessing_sha256": inner_state.semantic_sha256,
        "outer_preprocessing_sha256": outer_state.semantic_sha256,
        "first_step": first_step,
        "outer_first_step": outer_first_step,
        "source_samples_matched_key": f"{fold}:{seed}",
        "p4_used": False,
        "held_evaluated_once_after_checkpoint_freeze": True,
    }


def train_final_pair(
    data: SourceData,
    *,
    seed: int,
    selected_lambda: float,
    epochs: int,
    output_root: Path,
    probe_seeds: tuple[int, ...] = SEEDS,
) -> list[dict]:
    indices = np.arange(len(data.labels), dtype=np.int64)
    state = fit_first_difference(data.signals, indices)
    inputs = transform(data.signals, indices, state)
    positions = data.positions
    train_probe_rows, test_probe_rows = final_probe_masks(data)
    train_conditions = set(data.condition_ids[train_probe_rows].tolist())
    test_conditions = set(data.condition_ids[test_probe_rows].tolist())
    records = []
    output_root.mkdir(parents=True, exist_ok=True)
    for method, lambda_max in (("ERM", 0.0), ("DANN", float(selected_lambda))):
        model = C1PositionDANN(seed=seed, domain_count=None if method == "ERM" else 3)
        optimizer = _optimizer(model)
        first_step = None
        for epoch in range(1, int(epochs) + 1):
            _, audit = _train_epoch(
                model=model,
                optimizer=optimizer,
                inputs=inputs,
                tag_labels=data.labels,
                positions=positions,
                lambda_max=lambda_max,
                seed=seed,
                epoch=epoch,
                progress_epoch_budget=int(epochs),
            )
            if audit is not None:
                first_step = audit
        source = _evaluate(model, inputs, data.labels, positions)
        centroids = condition_centroids(
            source["embeddings"], data.labels, positions, data.condition_ids
        )
        probes = independent_probes(
            centroids, train_conditions, test_conditions, probe_seeds=probe_seeds
        )
        diagnostics = representation_diagnostics(centroids)
        checkpoint_path = output_root / method.lower() / f"seed_{seed}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "method": method,
            "lambda_max": lambda_max,
            "seed": int(seed),
            "epochs": int(epochs),
            "state_dict": _clone_state(model),
            "model_state_sha256": model_state_sha256(model),
            "source_positions": ["P1", "P2", "P3"],
            "p4_used": False,
        }
        torch.save(payload, checkpoint_path)
        checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        records.append(
            {
                "method": method,
                "lambda_max": lambda_max,
                "seed": int(seed),
                "epochs": int(epochs),
                "model_state_sha256": payload["model_state_sha256"],
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_path": str(checkpoint_path),
                "preprocessing_sha256": state.semantic_sha256,
                "source_row_metrics": source["tagid"],
                "domain_train_accuracy": source.get("domain", {}).get("accuracy"),
                "position_probe_balanced_accuracy": probes["position_balanced_accuracy_mean"],
                "position_probe_macro_f1": probes["position_macro_f1_mean"],
                "tagid_probe_macro_f1": probes["tagid_macro_f1_mean"],
                "probe_results": {
                    "position": [
                        {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in row.items()}
                        for row in probes["position"]
                    ],
                    "tagid": [
                        {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in row.items()}
                        for row in probes["tagid"]
                    ],
                },
                "representation": diagnostics,
                "first_step": first_step,
                "p4_used": False,
            }
        )
    return records


def load_final_model(checkpoint: Path) -> tuple[C1PositionDANN, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = C1PositionDANN(
        seed=int(payload["seed"]),
        domain_count=None if payload["method"] == "ERM" else 3,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    if model_state_sha256(model) != payload["model_state_sha256"]:
        raise RuntimeError("Final checkpoint model-state hash mismatch")
    return model, payload
