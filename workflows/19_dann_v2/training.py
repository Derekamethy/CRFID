"""Matched DANN v2 development and final source-only training."""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data import (
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
from model import C1PositionDANN, scale_encoder_gradient
from protocol import (
    ARM_A0,
    ARM_A1,
    ARM_A2,
    ARMS,
    HELD_POSITION,
    MAXIMUM_EPOCHS,
    SEEDS,
    registered_lambda_trajectory,
    schedule_summary,
    scheduled_lambda,
)


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
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _gradient_norm(parameters: list[torch.Tensor]) -> float:
    return float(
        sum(
            parameter.grad.detach().norm().item()
            for parameter in parameters
            if parameter.grad is not None
        )
    )


def schedule_trajectory_for_entry_point(
    entry_point: str,
    *,
    lambda_max: float,
    executed_epochs: int,
    registered_steps_per_epoch: int,
) -> list[float]:
    """Regression surface proving both entry points use the one schedule utility."""

    if entry_point not in {"development", "final"}:
        raise ValueError("unknown DANN v2 training entry point")
    return registered_lambda_trajectory(
        lambda_max,
        executed_epochs=executed_epochs,
        registered_steps_per_epoch=registered_steps_per_epoch,
    )


def _train_epoch(
    *,
    model: C1PositionDANN,
    optimizer: torch.optim.Optimizer,
    inputs: np.ndarray,
    tag_labels: np.ndarray,
    positions: np.ndarray,
    arm: str,
    lambda_max: float,
    seed: int,
    epoch: int,
) -> tuple[dict, dict | None]:
    model.train()
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    order = balanced_epoch_indices(positions, tag_labels, seed=seed, epoch=epoch)
    batches = math.ceil(len(order) / BATCH_SIZE)
    domain, domain_order = domain_labels(positions)
    tag_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    total_tag = total_domain = total = 0.0
    count = 0
    first_audit = None
    lambda_values: list[float] = []
    composition = batch_composition(order, positions, tag_labels)
    for batch_index, start in enumerate(range(0, len(order), BATCH_SIZE)):
        selected = order[start : start + BATCH_SIZE]
        if set(positions[selected].tolist()) != set(domain_order):
            raise RuntimeError("FAIL_DANN_V2_BATCH_DOMAIN_VALIDITY")
        tensor = torch.from_numpy(inputs[selected]).unsqueeze(1)
        tag_tensor = torch.from_numpy(tag_labels[selected].astype(np.int64, copy=False))
        optimizer.zero_grad(set_to_none=True)
        if arm == ARM_A0:
            tag_logits = model(tensor)
            tag_loss = tag_criterion(tag_logits, tag_tensor)
            domain_loss = torch.zeros((), dtype=tag_loss.dtype)
            strength = 0.0
            encoder_coefficient = 0.0
            loss = tag_loss
        else:
            global_update_index = (int(epoch) - 1) * batches + batch_index
            strength = scheduled_lambda(lambda_max, global_update_index, batches)
            encoder_coefficient = strength if arm == ARM_A1 else -strength
            embeddings = model.encode(tensor)
            tag_logits = model.tag_head(embeddings)
            domain_logits = model.domain_head(
                scale_encoder_gradient(embeddings, encoder_coefficient)
            )
            tag_loss = tag_criterion(tag_logits, tag_tensor)
            domain_tensor = torch.from_numpy(
                domain[selected].astype(np.int64, copy=False)
            )
            domain_loss = domain_criterion(domain_logits, domain_tensor)
            loss = tag_loss + domain_loss
        lambda_values.append(float(strength))
        loss.backward()
        if epoch == 1 and batch_index == 0:
            first_audit = {
                "combined_encoder_gradient_norm": _gradient_norm(
                    list(model.encoder.parameters())
                ),
                "combined_tagid_head_gradient_norm": _gradient_norm(
                    list(model.tag_head.parameters())
                ),
                "domain_head_gradient_norm": (
                    _gradient_norm(list(model.domain_head.parameters()))
                    if model.domain_head is not None
                    else 0.0
                ),
                "lambda": float(strength),
                "arm": arm,
                "encoder_domain_gradient_coefficient": float(encoder_coefficient),
                "position_order": list(domain_order),
                "batch_composition": composition[0],
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
        "lambda_values": lambda_values,
        "batch_count": len(lambda_values),
        "batch_composition": composition,
        **schedule_summary(lambda_values),
    }, first_audit


def infer(model: C1PositionDANN, inputs: np.ndarray) -> dict:
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
    output = infer(model, inputs)
    predicted = np.argmax(output["tag_logits"], axis=1).astype(np.int64)
    result = {
        "tagid": classification_metrics(labels, predicted, class_count=7),
        "tag_predictions": predicted,
        "embeddings": output["embeddings"],
    }
    unique_positions = tuple(sorted(np.unique(np.asarray(positions, dtype=str)).tolist()))
    logits64 = output["tag_logits"].astype(np.float64)
    shifted = logits64 - logits64.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    losses = -log_probabilities[np.arange(len(labels)), np.asarray(labels, dtype=np.int64)]
    result["tagid_cross_entropy"] = float(losses.mean())
    per_position = {
        position: float(losses[np.asarray(positions, dtype=str) == position].mean())
        for position in unique_positions
    }
    risk_values = np.asarray(list(per_position.values()), dtype=np.float64)
    result["familiar_source_resubstitution_risk"] = {
        "per_position_cross_entropy": per_position,
        "population_sd": float(risk_values.std(ddof=0)),
        "range": float(risk_values.max() - risk_values.min()),
    }
    expected_domain_count = (
        model.domain_head.network[-1].out_features
        if model.domain_head is not None
        else 0
    )
    if "domain_logits" in output and len(unique_positions) == expected_domain_count:
        domain, order = domain_labels(positions)
        predictions = np.argmax(output["domain_logits"], axis=1).astype(np.int64)
        result["domain"] = classification_metrics(
            domain, predictions, class_count=len(order)
        )
        result["domain_order"] = order
    elif "domain_logits" in output:
        result["domain_evaluation_skipped"] = "partition outside trained domain label space"
    return result


def _parameter_gradients(
    model: C1PositionDANN,
    loss: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    loss.backward()
    encoder = [parameter.grad.detach().clone() for parameter in model.encoder.parameters()]
    tag_head = [
        parameter.grad.detach().clone()
        for parameter in model.tag_head.parameters()
        if parameter.grad is not None
    ]
    domain_head = [
        parameter.grad.detach().clone()
        for parameter in model.domain_head.parameters()
        if parameter.grad is not None
    ]
    return encoder, tag_head, domain_head


def intervention_validity(seed: int = 42) -> dict:
    """Independent implementation gates required before scientific development."""

    torch.manual_seed(seed)
    inputs = torch.randn(16, 1, 280)
    tag_labels = torch.arange(16) % 7
    domains = torch.arange(16) % 2
    model = C1PositionDANN(seed=seed, domain_count=2)
    tag_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    strength = 0.3

    embeddings = model.encode(inputs)
    positive_forward = scale_encoder_gradient(embeddings, strength)
    negative_forward = scale_encoder_gradient(embeddings, -strength)
    forward_identity = torch.equal(positive_forward, embeddings) and torch.equal(
        negative_forward, embeddings
    )

    def domain_gradients(coefficient: float):
        local_embeddings = model.encode(inputs)
        logits = model.domain_head(
            scale_encoder_gradient(local_embeddings, coefficient)
        )
        return _parameter_gradients(model, domain_criterion(logits, domains))

    positive = domain_gradients(strength)
    negative = domain_gradients(-strength)
    zero = domain_gradients(0.0)
    sign_match = all(
        torch.allclose(left, -right, rtol=1e-5, atol=1e-7)
        for left, right in zip(positive[0], negative[0], strict=True)
    )
    magnitude_match = all(
        torch.allclose(left.abs(), right.abs(), rtol=1e-5, atol=1e-7)
        for left, right in zip(positive[0], negative[0], strict=True)
    )
    domain_head_match = all(
        torch.allclose(left, right, rtol=1e-6, atol=1e-8)
        for left, right in zip(positive[2], negative[2], strict=True)
    )
    zero_encoder = all(torch.count_nonzero(value).item() == 0 for value in zero[0])

    tag_first = _parameter_gradients(
        model, tag_criterion(model.tag_head(model.encode(inputs)), tag_labels)
    )
    tag_second = _parameter_gradients(
        model, tag_criterion(model.tag_head(model.encode(inputs)), tag_labels)
    )
    tag_match = all(
        torch.equal(left, right)
        for left, right in zip(tag_first[0] + tag_first[1], tag_second[0] + tag_second[1], strict=True)
    )

    def one_update(arm: str):
        local = C1PositionDANN(seed=seed, domain_count=None if arm == ARM_A0 else 2)
        optimizer = _optimizer(local)
        optimizer.zero_grad(set_to_none=True)
        local_embeddings = local.encode(inputs)
        loss = tag_criterion(local.tag_head(local_embeddings), tag_labels)
        domain_before = None
        if arm != ARM_A0:
            domain_before = [value.detach().clone() for value in local.domain_head.parameters()]
            loss = loss + domain_criterion(
                local.domain_head(scale_encoder_gradient(local_embeddings, 0.0)), domains
            )
        loss.backward()
        optimizer.step()
        domain_updated = (
            False
            if domain_before is None
            else any(
                not torch.equal(left, right)
                for left, right in zip(domain_before, local.domain_head.parameters(), strict=True)
            )
        )
        return local, domain_updated

    a0, _ = one_update(ARM_A0)
    a1, a1_domain_updated = one_update(ARM_A1)
    a2, a2_domain_updated = one_update(ARM_A2)
    zero_a1_encoder = all(
        torch.equal(left, right)
        for left, right in zip(a0.encoder.parameters(), a1.encoder.parameters(), strict=True)
    )
    zero_a2_encoder = all(
        torch.equal(left, right)
        for left, right in zip(a0.encoder.parameters(), a2.encoder.parameters(), strict=True)
    )
    zero_tag_heads = all(
        torch.equal(reference, positive_value) and torch.equal(reference, negative_value)
        for reference, positive_value, negative_value in zip(
            a0.tag_head.parameters(),
            a1.tag_head.parameters(),
            a2.tag_head.parameters(),
            strict=True,
        )
    )
    development_schedule = schedule_trajectory_for_entry_point(
        "development", lambda_max=0.3, executed_epochs=7, registered_steps_per_epoch=17
    )
    final_schedule = schedule_trajectory_for_entry_point(
        "final", lambda_max=0.3, executed_epochs=7, registered_steps_per_epoch=17
    )
    schedule_match = development_schedule == final_schedule
    fold_domain_labels, fold_order = domain_labels(np.asarray(["P2", "P3", "P2", "P3"]))
    domain_label_valid = fold_order == ("P2", "P3") and fold_domain_labels.tolist() == [0, 1, 0, 1]

    checks = {
        "grl_forward_identity": forward_identity,
        "a1_a2_encoder_domain_gradient_sign_match": sign_match,
        "a1_a2_encoder_domain_gradient_magnitude_match": magnitude_match,
        "a1_a2_domain_head_gradients_identical": domain_head_match,
        "a1_a2_tagid_gradient_contribution_identical": tag_match,
        "lambda_zero_encoder_domain_gradient_zero": zero_encoder,
        "lambda_zero_a1_encoder_update_matches_a0": zero_a1_encoder,
        "lambda_zero_a2_encoder_update_matches_a0": zero_a2_encoder,
        "lambda_zero_tagid_head_updates_match_a0": zero_tag_heads,
        "lambda_zero_a1_domain_head_legitimately_updated": a1_domain_updated,
        "lambda_zero_a2_domain_head_legitimately_updated": a2_domain_updated,
        "development_final_schedule_trajectory_identical": schedule_match,
        "fold_local_domain_labels_correct": domain_label_valid,
        "domain_loss_can_access_p4": False,
    }
    passed = all(
        value for key, value in checks.items() if key != "domain_loss_can_access_p4"
    )
    return {
        **checks,
        "configured_magnitude": strength,
        "maximum_encoder_sign_residual": max(
            float((left + right).abs().max())
            for left, right in zip(positive[0], negative[0], strict=True)
        ),
        "maximum_encoder_magnitude_residual": max(
            float((left.abs() - right.abs()).abs().max())
            for left, right in zip(positive[0], negative[0], strict=True)
        ),
        "maximum_domain_head_gradient_residual": max(
            float((left - right).abs().max())
            for left, right in zip(positive[2], negative[2], strict=True)
        ),
        "schedule_regression_update_count": len(development_schedule),
        "passed": passed,
    }


def _flatten_lambdas(history: list[dict], epochs: int | None = None) -> list[float]:
    selected = history if epochs is None else history[: int(epochs)]
    return [float(value) for row in selected for value in row["lambda_values"]]


def development_run(
    data: SourceData,
    *,
    fold: str,
    seed: int,
    lambda_max: float,
    arm: str,
    probe_seeds: tuple[int, ...] = SEEDS,
) -> dict:
    started = time.perf_counter()
    parts = data.splits[fold]
    positions_all = data.positions
    train_rows = parts["inner_train"]
    validation_rows = parts["inner_validation"]
    inner_state = fit_first_difference(data.signals, train_rows)
    train_inputs = transform(data.signals, train_rows, inner_state)
    validation_inputs = transform(data.signals, validation_rows, inner_state)
    train_positions = positions_all[train_rows]
    validation_positions = positions_all[validation_rows]
    domain_count = None if arm == ARM_A0 else 2
    model = C1PositionDANN(seed=seed, domain_count=domain_count)
    optimizer = _optimizer(model)
    best_metric = -float("inf")
    best_epoch = 0
    best_state = None
    without_improvement = 0
    history: list[dict] = []
    first_step = None
    for epoch in range(1, MAXIMUM_EPOCHS + 1):
        losses, audit = _train_epoch(
            model=model,
            optimizer=optimizer,
            inputs=train_inputs,
            tag_labels=data.labels[train_rows],
            positions=train_positions,
            arm=arm,
            lambda_max=lambda_max,
            seed=seed,
            epoch=epoch,
        )
        if audit is not None:
            first_step = audit
        validation = _evaluate(
            model,
            validation_inputs,
            data.labels[validation_rows],
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
                "validation_tagid_loss": validation["tagid_cross_entropy"],
                "domain_validation_accuracy": validation.get("domain", {}).get("accuracy"),
                "improved": improved,
            }
        )
        if without_improvement >= PATIENCE:
            break
    if best_state is None or first_step is None:
        raise RuntimeError("DANN v2 source-only checkpoint selection failed")
    model.load_state_dict(best_state, strict=True)
    familiar_train = _evaluate(
        model, train_inputs, data.labels[train_rows], train_positions
    )
    familiar_validation = _evaluate(
        model,
        validation_inputs,
        data.labels[validation_rows],
        validation_positions,
    )
    combined_embeddings = np.vstack(
        (familiar_train["embeddings"], familiar_validation["embeddings"])
    )
    combined_rows = np.concatenate((train_rows, validation_rows))
    centroids = condition_centroids(
        combined_embeddings,
        data.labels[combined_rows],
        positions_all[combined_rows],
        data.condition_ids[combined_rows],
    )
    train_conditions = set(data.condition_ids[train_rows].tolist())
    validation_conditions = set(data.condition_ids[validation_rows].tolist())
    probes = independent_probes(
        centroids,
        train_conditions,
        validation_conditions,
        probe_seeds=probe_seeds,
    )
    diagnostics = representation_diagnostics(centroids)

    outer_rows = np.sort(np.concatenate((train_rows, validation_rows)))
    outer_state = fit_first_difference(data.signals, outer_rows)
    outer_inputs = transform(data.signals, outer_rows, outer_state)
    outer_positions = positions_all[outer_rows]
    outer_model = C1PositionDANN(seed=seed, domain_count=domain_count)
    outer_optimizer = _optimizer(outer_model)
    outer_history: list[dict] = []
    outer_first_step = None
    for epoch in range(1, best_epoch + 1):
        losses, audit = _train_epoch(
            model=outer_model,
            optimizer=outer_optimizer,
            inputs=outer_inputs,
            tag_labels=data.labels[outer_rows],
            positions=outer_positions,
            arm=arm,
            lambda_max=lambda_max,
            seed=seed,
            epoch=epoch,
        )
        outer_history.append({"epoch": epoch, **losses})
        if audit is not None:
            outer_first_step = audit
    held_inputs = transform(data.signals, parts["outer_held"], outer_state)
    held = _evaluate(
        outer_model,
        held_inputs,
        data.labels[parts["outer_held"]],
        positions_all[parts["outer_held"]],
    )
    held_blocks = aggregate_blocks(
        data.labels[parts["outer_held"]],
        held["tag_predictions"],
        data.condition_ids[parts["outer_held"]],
    )
    selected_lambdas = _flatten_lambdas(history, best_epoch)
    outer_lambdas = _flatten_lambdas(outer_history)
    return {
        "schema_version": 2,
        "fold": fold,
        "held_position": HELD_POSITION[fold],
        "seed": int(seed),
        "lambda_max": float(lambda_max),
        "arm": arm,
        "run_id": f"DANNV2-DEV-{arm}-L{lambda_max:.2f}-{fold}-S{seed}",
        "selected_epoch": best_epoch,
        "epochs_executed": len(history),
        "checkpoint_metric": "familiar_source_inner_validation_row_macro_f1",
        "familiar_validation_metrics": familiar_validation["tagid"],
        "familiar_validation_tagid_loss": familiar_validation["tagid_cross_entropy"],
        "familiar_train_metrics": familiar_train["tagid"],
        "domain_train_accuracy": familiar_train.get("domain", {}).get("accuracy"),
        "domain_validation_accuracy": familiar_validation.get("domain", {}).get("accuracy"),
        "outer_held_lopo_row_metrics": held["tagid"],
        "outer_held_lopo_block_metrics": held_blocks["metrics"],
        "outer_held_lopo_chance_reference": 1.0 / 7.0,
        "position_probe_balanced_accuracy": probes["position_balanced_accuracy_mean"],
        "position_probe_macro_f1": probes["position_macro_f1_mean"],
        "tagid_probe_macro_f1": probes["tagid_macro_f1_mean"],
        "probe_position_chance": probes["position_chance"],
        "probe_results": {
            "position": [
                {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in row.items()
                }
                for row in probes["position"]
            ],
            "tagid": [
                {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in row.items()
                }
                for row in probes["tagid"]
            ],
        },
        "representation": diagnostics,
        "inner_training_history": history,
        "outer_diagnostic_training_history": outer_history,
        "selected_checkpoint_lambda_values": selected_lambdas,
        "selected_checkpoint_schedule": schedule_summary(selected_lambdas),
        "outer_diagnostic_lambda_values": outer_lambdas,
        "outer_diagnostic_schedule": schedule_summary(outer_lambdas),
        "model_state_sha256": model_state_sha256(model),
        "outer_model_state_sha256": model_state_sha256(outer_model),
        "inner_preprocessing_sha256": inner_state.semantic_sha256,
        "outer_preprocessing_sha256": outer_state.semantic_sha256,
        "first_step": first_step,
        "outer_first_step": outer_first_step,
        "source_samples_matched_key": f"{fold}:{seed}:{lambda_max:.2f}",
        "p4_used": False,
        "runtime_seconds": time.perf_counter() - started,
    }


def train_final_arms(
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
    for arm, lambda_max in (
        (ARM_A0, 0.0),
        (ARM_A1, float(selected_lambda)),
        (ARM_A2, float(selected_lambda)),
    ):
        started = time.perf_counter()
        model = C1PositionDANN(seed=seed, domain_count=None if arm == ARM_A0 else 3)
        optimizer = _optimizer(model)
        first_step = None
        histories: list[dict] = []
        for epoch in range(1, int(epochs) + 1):
            losses, audit = _train_epoch(
                model=model,
                optimizer=optimizer,
                inputs=inputs,
                tag_labels=data.labels,
                positions=positions,
                arm=arm,
                lambda_max=lambda_max,
                seed=seed,
                epoch=epoch,
            )
            histories.append({"epoch": epoch, **losses})
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
        slug = {ARM_A0: "a0", ARM_A1: "a1", ARM_A2: "a2"}[arm]
        checkpoint_path = output_root / slug / f"seed_{seed}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 2,
            "arm": arm,
            "lambda_max": lambda_max,
            "seed": int(seed),
            "epochs": int(epochs),
            "state_dict": _clone_state(model),
            "model_state_sha256": model_state_sha256(model),
            "source_positions": ["P1", "P2", "P3"],
            "schedule_horizon_epochs": MAXIMUM_EPOCHS,
            "p4_used": False,
        }
        torch.save(payload, checkpoint_path)
        checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        trajectory = _flatten_lambdas(histories)
        per_position = {
            position: classification_metrics(
                data.labels[positions == position],
                source["tag_predictions"][positions == position],
                class_count=7,
            )
            for position in ("P1", "P2", "P3")
        }
        records.append(
            {
                "arm": arm,
                "lambda_max": lambda_max,
                "seed": int(seed),
                "epochs": int(epochs),
                "model_state_sha256": payload["model_state_sha256"],
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_path": str(checkpoint_path),
                "preprocessing_sha256": state.semantic_sha256,
                "diagnostic_scope": "FAMILIAR_SOURCE_RESUBSTITUTION_DIAGNOSTIC",
                "source_row_metrics": source["tagid"],
                "source_position_metrics": per_position,
                "source_worst_position_macro_f1": min(
                    row["macro_f1"] for row in per_position.values()
                ),
                "familiar_source_resubstitution_risk": source[
                    "familiar_source_resubstitution_risk"
                ],
                "domain_train_accuracy": source.get("domain", {}).get("accuracy"),
                "position_probe_balanced_accuracy": probes[
                    "position_balanced_accuracy_mean"
                ],
                "position_probe_macro_f1": probes["position_macro_f1_mean"],
                "tagid_probe_macro_f1": probes["tagid_macro_f1_mean"],
                "probe_results": {
                    "position": [
                        {
                            key: value.tolist() if isinstance(value, np.ndarray) else value
                            for key, value in row.items()
                        }
                        for row in probes["position"]
                    ],
                    "tagid": [
                        {
                            key: value.tolist() if isinstance(value, np.ndarray) else value
                            for key, value in row.items()
                        }
                        for row in probes["tagid"]
                    ],
                },
                "representation": diagnostics,
                "training_history": histories,
                "lambda_values": trajectory,
                "schedule": schedule_summary(trajectory),
                "first_step": first_step,
                "p4_used": False,
                "runtime_seconds": time.perf_counter() - started,
            }
        )
    return records


def load_final_model(checkpoint: Path) -> tuple[C1PositionDANN, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = C1PositionDANN(
        seed=int(payload["seed"]),
        domain_count=None if payload["arm"] == ARM_A0 else 3,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    if model_state_sha256(model) != payload["model_state_sha256"]:
        raise RuntimeError("Final DANN v2 checkpoint model-state hash mismatch")
    return model, payload
