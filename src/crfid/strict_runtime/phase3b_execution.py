"""Frozen Phase 3B candidate execution primitives.

This module has no target-data loader.  It consumes only canonical Phase 1
artifacts and verified Phase 2 checkpoints inside the active implementation
root.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .governance import write_json
from .hashing import array_sha256, canonical_json_sha256, hash_lines, sha256_file
from .neutral_data import CanonicalPhase2Data, PartitionData
from .neutral_metrics import evaluate_all_levels
from .neutral_model import (
    gradient_hashes,
    initialize_model,
    model_state_sha256,
    optimizer_state_sha256,
    parameter_count,
    tensor_sha256,
)
from .paths import PROJECT_ROOT
from .scale_policy import MODE_CLIP_AT_MINIMUM, apply_scale_policy


CLASS_COUNT = 7
FOLDS = ("S1", "S2", "S3")
SEEDS = (42, 43, 44, 45, 46)
C0 = "C0_NEUTRAL_ERM_1DCNN"
C1 = "C1_FIRST_DIFFERENCE_ERM_1DCNN"
C2 = "C2_SOURCE_EUCLIDEAN_NCM_READOUT"
C3 = "C3_SOURCE_CORAL_ERM_1DCNN"
CANDIDATE_IDS = (C0, C1, C2, C3)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def artifact_record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "relative_path": path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def partition_indices(data: CanonicalPhase2Data, fold_id: str, stage: str, name: str) -> np.ndarray:
    if stage == "inner_selection":
        if name not in {"inner_train", "inner_validation"}:
            raise ValueError(f"Invalid inner partition: {name}")
        return data.partitions[fold_id][name].copy()
    if stage == "outer_refit":
        if name == "outer_development":
            return np.sort(
                np.concatenate(
                    (
                        data.partitions[fold_id]["inner_train"],
                        data.partitions[fold_id]["inner_validation"],
                    )
                )
            )
        if name == "outer_held":
            return data.partitions[fold_id]["outer_held"].copy()
    raise ValueError(f"Invalid partition request: {fold_id}/{stage}/{name}")


def _partition_from_values(
    data: CanonicalPhase2Data, indices: np.ndarray, values: np.ndarray
) -> PartitionData:
    rows = [data.registry_rows[int(index)] for index in indices]
    return PartitionData(
        registry_rows=np.asarray(indices, dtype=np.int64).copy(),
        inputs=np.ascontiguousarray(values, dtype=np.float32),
        labels=np.ascontiguousarray(data.labels[indices], dtype=np.int64),
        sample_ids=[row["sample_id"] for row in rows],
        condition_ids=[row["raw_condition_id"] for row in rows],
        exact_signal_hashes=[row["exact_signal_sha256"] for row in rows],
        unique_signal_weights=np.asarray(
            [row["unique_signal_weight"] for row in rows], dtype=np.float64
        ),
    )


def fit_first_difference_preprocessing(
    data: CanonicalPhase2Data, output_root: Path
) -> dict[tuple[str, str], dict]:
    """Fit diff-then-feature-standardization on each permitted fit partition."""

    transformed = np.ascontiguousarray(np.diff(data.signals, n=1, axis=1), dtype=np.float64)
    if transformed.shape != (len(data.signals), data.signals.shape[1] - 1):
        raise RuntimeError(f"Frozen first-difference shape changed: {transformed.shape}")
    states: dict[tuple[str, str], dict] = {}
    for fold_id in FOLDS:
        for stage, fit_name in (
            ("inner_selection", "inner_train"),
            ("outer_refit", "outer_development"),
        ):
            indices = partition_indices(data, fold_id, stage, fit_name)
            fit = transformed[indices]
            mean = np.ascontiguousarray(fit.mean(axis=0), dtype=np.float64)
            raw_scale = np.ascontiguousarray(fit.std(axis=0, ddof=0), dtype=np.float64)
            # Historical selection-stage behaviour, preserved bit-for-bit via an
            # explicit mode. See crfid.strict_runtime.scale_policy.
            scale = apply_scale_policy(raw_scale, mode=MODE_CLIP_AT_MINIMUM)
            directory = output_root / C1 / fold_id
            directory.mkdir(parents=True, exist_ok=True)
            mean_path = directory / f"{stage}_mean_float64.npy"
            scale_path = directory / f"{stage}_scale_float64.npy"
            state_path = directory / f"{stage}_state.json"
            np.save(mean_path, mean, allow_pickle=False)
            np.save(scale_path, scale, allow_pickle=False)
            semantic = {
                "schema_version": 1,
                "candidate_id": C1,
                "fold_id": fold_id,
                "stage": stage,
                "fit_partition": fit_name,
                "fit_registry_rows_sha256": hash_lines(str(int(value)) for value in indices),
                "fit_sample_count": int(len(indices)),
                "operation_order": [
                    "first_difference_x[i+1]-x[i]_without_padding",
                    "featurewise_population_mean_and_std_fit_on_declared_training_partition",
                    "standardize_with_scale_clipped_at_1e-12",
                ],
                "input_feature_count": 281,
                "output_feature_count": 280,
                "boundary_handling": "no_padding_drop_original_feature_index_0_boundary",
                "minimum_scale": 1e-12,
                "mean_array_sha256": array_sha256(mean),
                "scale_array_sha256": array_sha256(scale),
                "mean_file": artifact_record(mean_path),
                "scale_file": artifact_record(scale_path),
                "p4_used": False,
            }
            state = {**semantic, "state_sha256": canonical_json_sha256(semantic)}
            write_json(state_path, state)
            state["state_file"] = artifact_record(state_path)
            states[(fold_id, stage)] = state
    return states


def first_difference_partition(
    data: CanonicalPhase2Data,
    states: dict[tuple[str, str], dict],
    fold_id: str,
    stage: str,
    name: str,
) -> PartitionData:
    indices = partition_indices(data, fold_id, stage, name)
    state = states[(fold_id, stage)]
    mean = np.load(state["mean_file"]["path"], allow_pickle=False)
    scale = np.load(state["scale_file"]["path"], allow_pickle=False)
    differenced = np.ascontiguousarray(
        np.diff(data.signals[indices], n=1, axis=1), dtype=np.float64
    )
    values = np.ascontiguousarray((differenced - mean) / scale, dtype=np.float32)
    return _partition_from_values(data, indices, values)


def candidate_partitions(
    data: CanonicalPhase2Data,
    candidate_id: str,
    fold_id: str,
    first_difference_states: dict[tuple[str, str], dict],
) -> dict[str, PartitionData]:
    if candidate_id == C1:
        return {
            "inner_train": first_difference_partition(
                data, first_difference_states, fold_id, "inner_selection", "inner_train"
            ),
            "inner_validation": first_difference_partition(
                data,
                first_difference_states,
                fold_id,
                "inner_selection",
                "inner_validation",
            ),
            "outer_development": first_difference_partition(
                data,
                first_difference_states,
                fold_id,
                "outer_refit",
                "outer_development",
            ),
            "outer_held": first_difference_partition(
                data, first_difference_states, fold_id, "outer_refit", "outer_held"
            ),
        }
    if candidate_id not in {C0, C2, C3}:
        raise ValueError(candidate_id)
    return {
        "inner_train": data.transformed_partition(
            fold_id, "inner_selection", "inner_train"
        ),
        "inner_validation": data.transformed_partition(
            fold_id, "inner_selection", "inner_validation"
        ),
        "outer_development": data.transformed_partition(
            fold_id, "outer_refit", "outer_development"
        ),
        "outer_held": data.transformed_partition(fold_id, "outer_refit", "outer_held"),
    }


def extract_logits_embeddings(
    model: nn.Module, partition: PartitionData, batch_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    inputs = torch.from_numpy(partition.inputs).unsqueeze(1)
    feature_extractor = model.network[:-1]
    classifier = model.network[-1]
    logits: list[torch.Tensor] = []
    embeddings: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            values = feature_extractor(inputs[start : start + batch_size])
            embeddings.append(values.detach().cpu())
            logits.append(classifier(values).detach().cpu())
    return (
        torch.cat(logits).numpy().astype(np.float32, copy=False),
        torch.cat(embeddings).numpy().astype(np.float32, copy=False),
    )


def _per_class_precision(matrix: np.ndarray) -> list[float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    true_positive = np.diag(matrix)
    predicted = matrix.sum(axis=0)
    return [
        float(value)
        for value in np.divide(
            true_positive,
            predicted,
            out=np.zeros_like(true_positive),
            where=predicted > 0,
        )
    ]


def _mean_prediction_entropy(logits: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return float(
        np.mean(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-300, None)), axis=1))
    )


def evaluate_logits(partition: PartitionData, logits: np.ndarray) -> dict:
    evaluation = evaluate_all_levels(
        partition.labels,
        np.asarray(logits, dtype=np.float32),
        partition.condition_ids,
        partition.unique_signal_weights,
    )
    for level in ("sample", "unique_signal_weighted"):
        evaluation[level]["per_class_precision"] = _per_class_precision(
            np.asarray(evaluation[level]["confusion_matrix"])
        )
    evaluation["condition"]["metrics"]["per_class_precision"] = _per_class_precision(
        np.asarray(evaluation["condition"]["metrics"]["confusion_matrix"])
    )
    predictions = np.asarray(evaluation["predictions"], dtype=np.int64)
    counts = np.bincount(predictions, minlength=CLASS_COUNT)
    evaluation["dominant_predicted_class_fraction"] = float(counts.max() / len(predictions))
    evaluation["mean_prediction_entropy"] = _mean_prediction_entropy(logits)
    evaluation["logits"] = np.asarray(logits, dtype=np.float32)
    return evaluation


def save_prediction_bundle(
    run_directory: Path,
    stem: str,
    partition: PartitionData,
    evaluation: dict,
) -> dict:
    run_directory.mkdir(parents=True, exist_ok=True)
    path = run_directory / f"{stem}_predictions.npz"
    np.savez(
        path,
        registry_rows=partition.registry_rows.astype(np.int64, copy=False),
        true_labels=partition.labels.astype(np.int64, copy=False),
        predictions=np.asarray(evaluation["predictions"], dtype=np.int64),
        logits=np.asarray(evaluation["logits"], dtype=np.float32),
        unique_signal_weights=partition.unique_signal_weights.astype(np.float64, copy=False),
        sample_ids=np.asarray(partition.sample_ids, dtype=str),
        condition_ids=np.asarray(partition.condition_ids, dtype=str),
        exact_signal_hashes=np.asarray(partition.exact_signal_hashes, dtype=str),
    )
    return artifact_record(path)


def metrics_only(evaluation: dict) -> dict:
    return {
        "sample": evaluation["sample"],
        "condition": evaluation["condition"]["metrics"],
        "unique_signal_weighted": evaluation["unique_signal_weighted"],
        "dominant_predicted_class_fraction": evaluation[
            "dominant_predicted_class_fraction"
        ],
        "mean_prediction_entropy": evaluation["mean_prediction_entropy"],
    }


def _write_history(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return artifact_record(path)


def _optimizer(model: nn.Module, base_config: dict) -> torch.optim.AdamW:
    settings = base_config["optimizer"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        betas=tuple(float(value) for value in settings["betas"]),
        eps=float(settings["epsilon"]),
        weight_decay=float(settings["weight_decay"]),
        amsgrad=bool(settings["amsgrad"]),
        foreach=bool(settings["foreach"]),
        fused=bool(settings["fused"]),
    )


def _criterion(base_config: dict) -> nn.CrossEntropyLoss:
    settings = base_config["loss"]
    if settings["weight"] is not None or float(settings["label_smoothing"]) != 0.0:
        raise RuntimeError("Frozen candidate loss must remain unweighted unsmoothed CE")
    return nn.CrossEntropyLoss(reduction="mean")


def _initialize_with_equality_audit(seed: int) -> tuple[nn.Module, dict]:
    first = initialize_model(seed)
    first_hash = model_state_sha256(first)
    repeated = initialize_model(seed)
    repeated_hash = model_state_sha256(repeated)
    if first_hash != repeated_hash:
        raise RuntimeError(f"Same-seed initialization differs: {seed}")
    return first, {
        "seed": seed,
        "first_initial_model_state_sha256": first_hash,
        "repeated_initial_model_state_sha256": repeated_hash,
        "repeated_same_seed_initialization_equal": True,
        "parameter_count": parameter_count(first),
    }


def _permutation(
    sample_count: int, seed: int, epoch: int, stage: str, base_config: dict
) -> tuple[torch.Tensor, int]:
    training = base_config["training"]
    offset = (
        int(training["inner_stage_offset"])
        if stage == "inner_selection"
        else int(training["outer_stage_offset"])
    )
    permutation_seed = seed * 1_000_000 + offset + epoch
    generator = torch.Generator(device="cpu")
    generator.manual_seed(permutation_seed)
    return torch.randperm(sample_count, generator=generator), permutation_seed


def _permutation_sha256(permutation: torch.Tensor) -> str:
    values = permutation.detach().cpu().numpy().astype("<i8", copy=False)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _save_checkpoint(
    path: Path,
    *,
    candidate_id: str,
    model: nn.Module,
    fold_id: str,
    seed: int,
    stage: str,
    epochs: int,
    candidate_config_sha256: str,
    split_sha256: str,
    preprocessing_sha256: str,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = _clone_state(model)
    payload = {
        "schema_version": 1,
        "phase": "3B",
        "method_id": candidate_id,
        "fold_id": fold_id,
        "seed": seed,
        "stage": stage,
        "epochs": epochs,
        "candidate_config_sha256": candidate_config_sha256,
        "split_sha256": split_sha256,
        "preprocessing_sha256": preprocessing_sha256,
        "model_state_sha256": model_state_sha256(state),
        "model_state_dict": state,
        "p4_used": False,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        **artifact_record(path),
        "model_state_sha256": payload["model_state_sha256"],
        "saved_before_outer_held_evaluation": stage == "outer_refit",
    }


def load_checkpoint_model(path: Path, seed: int) -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = initialize_model(seed)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if model_state_sha256(model) != payload["model_state_sha256"]:
        raise RuntimeError(f"Checkpoint model-state hash mismatch: {path}")
    return model, payload


def _erm_train_epoch(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    partition: PartitionData,
    fold_id: str,
    seed: int,
    epoch: int,
    stage: str,
    base_config: dict,
    initial_hash: str,
    candidate_id: str,
) -> tuple[dict, dict | None]:
    model.train()
    batch_size = int(base_config["training"]["batch_size"])
    permutation, permutation_seed = _permutation(
        len(partition.labels), seed, epoch, stage, base_config
    )
    total_loss = 0.0
    total_count = 0
    first_step = None
    for start in range(0, len(permutation), batch_size):
        selected = permutation[start : start + batch_size].numpy()
        inputs = torch.from_numpy(partition.inputs[selected]).unsqueeze(1)
        labels = torch.from_numpy(partition.labels[selected])
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        if epoch == 1 and start == 0:
            gradients = gradient_hashes(model)
            first_step = {
                "candidate_id": candidate_id,
                "fold_id": fold_id,
                "seed": seed,
                "stage": stage,
                "initial_model_state_sha256": initial_hash,
                "epoch_1_permutation_seed": permutation_seed,
                "epoch_1_permutation_sha256": _permutation_sha256(permutation),
                "epoch_1_sample_permutation_sha256": hash_lines(
                    partition.sample_ids[int(index)] for index in permutation.numpy()
                ),
                "first_batch_sample_ids": [partition.sample_ids[int(index)] for index in selected],
                "first_batch_registry_rows": [
                    int(partition.registry_rows[int(index)]) for index in selected
                ],
                "first_batch_tensor_sha256": tensor_sha256(inputs),
                "first_logits_sha256": tensor_sha256(logits),
                "first_loss": float(loss.detach().cpu().item()),
                "loss_components": {
                    "cross_entropy": float(loss.detach().cpu().item()),
                    "coral": 0.0,
                    "alignment_weight": 0.0,
                },
                "gradient_aggregate_sha256": gradients["aggregate_sha256"],
                "per_parameter_gradient_sha256": gradients["per_parameter"],
            }
        optimizer.step()
        if first_step is not None and epoch == 1 and start == 0:
            first_step["post_first_step_model_state_sha256"] = model_state_sha256(model)
            first_step["post_first_step_optimizer_state_sha256"] = optimizer_state_sha256(
                optimizer
            )
        count = len(selected)
        total_loss += float(loss.detach().cpu().item()) * count
        total_count += count
    return {
        "total_loss": total_loss / total_count,
        "cross_entropy": total_loss / total_count,
        "coral": 0.0,
    }, first_step


def coral_penalty(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] < 2:
        raise ValueError("CORAL requires equal [batch, embedding] tensors with batch >= 2")
    left_centered = left - left.mean(dim=0, keepdim=True)
    right_centered = right - right.mean(dim=0, keepdim=True)
    left_covariance = left_centered.T @ left_centered / float(left.shape[0] - 1)
    right_covariance = right_centered.T @ right_centered / float(right.shape[0] - 1)
    dimension = float(left.shape[1])
    return torch.sum((left_covariance - right_covariance) ** 2) / (
        4.0 * dimension * dimension
    )


def _coral_train_epoch(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    partition: PartitionData,
    positions: np.ndarray,
    fold_id: str,
    seed: int,
    epoch: int,
    stage: str,
    base_config: dict,
    initial_hash: str,
) -> tuple[dict, dict | None]:
    model.train()
    per_position_batch_size = 128
    unique_positions = sorted(np.unique(positions).tolist())
    if len(unique_positions) != 2:
        raise RuntimeError(
            f"CORAL development partition must contain exactly two source positions: {unique_positions}"
        )
    indices_by_position = [np.flatnonzero(positions == value) for value in unique_positions]
    if len(indices_by_position[0]) != len(indices_by_position[1]):
        raise RuntimeError("Frozen paired-domain batching requires equal source-position counts")
    training = base_config["training"]
    offset = (
        int(training["inner_stage_offset"])
        if stage == "inner_selection"
        else int(training["outer_stage_offset"])
    )
    permutation_seed = seed * 1_000_000 + offset + epoch
    generator = torch.Generator(device="cpu")
    generator.manual_seed(permutation_seed)
    permutations = [
        torch.randperm(len(indices), generator=generator) for indices in indices_by_position
    ]
    total_loss = 0.0
    total_ce = 0.0
    total_coral = 0.0
    total_count = 0
    first_step = None
    feature_extractor = model.network[:-1]
    classifier = model.network[-1]
    for start in range(0, len(permutations[0]), per_position_batch_size):
        local_left = permutations[0][start : start + per_position_batch_size].numpy()
        local_right = permutations[1][start : start + per_position_batch_size].numpy()
        selected_left = indices_by_position[0][local_left]
        selected_right = indices_by_position[1][local_right]
        if len(selected_left) != len(selected_right):
            raise RuntimeError("CORAL paired chunks differ in size")
        left_inputs = torch.from_numpy(partition.inputs[selected_left]).unsqueeze(1)
        right_inputs = torch.from_numpy(partition.inputs[selected_right]).unsqueeze(1)
        left_labels = torch.from_numpy(partition.labels[selected_left])
        right_labels = torch.from_numpy(partition.labels[selected_right])
        optimizer.zero_grad(set_to_none=True)
        left_embeddings = feature_extractor(left_inputs)
        right_embeddings = feature_extractor(right_inputs)
        left_logits = classifier(left_embeddings)
        right_logits = classifier(right_embeddings)
        logits = torch.cat((left_logits, right_logits), dim=0)
        labels = torch.cat((left_labels, right_labels), dim=0)
        ce = criterion(logits, labels)
        coral = coral_penalty(left_embeddings, right_embeddings)
        loss = ce + 0.1 * coral
        loss.backward()
        if epoch == 1 and start == 0:
            combined_inputs = torch.cat((left_inputs, right_inputs), dim=0)
            combined_indices = np.concatenate((selected_left, selected_right))
            gradients = gradient_hashes(model)
            first_step = {
                "candidate_id": C3,
                "fold_id": fold_id,
                "seed": seed,
                "stage": stage,
                "initial_model_state_sha256": initial_hash,
                "domain_order": unique_positions,
                "epoch_1_permutation_seed": permutation_seed,
                "epoch_1_domain_permutation_sha256": {
                    unique_positions[index]: _permutation_sha256(permutations[index])
                    for index in range(2)
                },
                "first_batch_sample_ids": [
                    partition.sample_ids[int(index)] for index in combined_indices
                ],
                "first_batch_registry_rows": [
                    int(partition.registry_rows[int(index)]) for index in combined_indices
                ],
                "first_batch_position_counts": {
                    unique_positions[0]: len(selected_left),
                    unique_positions[1]: len(selected_right),
                },
                "first_batch_tensor_sha256": tensor_sha256(combined_inputs),
                "first_logits_sha256": tensor_sha256(logits),
                "first_loss": float(loss.detach().cpu().item()),
                "loss_components": {
                    "cross_entropy": float(ce.detach().cpu().item()),
                    "coral": float(coral.detach().cpu().item()),
                    "alignment_weight": 0.1,
                },
                "gradient_aggregate_sha256": gradients["aggregate_sha256"],
                "per_parameter_gradient_sha256": gradients["per_parameter"],
            }
        optimizer.step()
        if first_step is not None and epoch == 1 and start == 0:
            first_step["post_first_step_model_state_sha256"] = model_state_sha256(model)
            first_step["post_first_step_optimizer_state_sha256"] = optimizer_state_sha256(
                optimizer
            )
        count = len(labels)
        total_loss += float(loss.detach().cpu().item()) * count
        total_ce += float(ce.detach().cpu().item()) * count
        total_coral += float(coral.detach().cpu().item()) * count
        total_count += count
    return {
        "total_loss": total_loss / total_count,
        "cross_entropy": total_ce / total_count,
        "coral": total_coral / total_count,
    }, first_step


def _positions_for_partition(data: CanonicalPhase2Data, partition: PartitionData) -> np.ndarray:
    return np.asarray(
        [data.registry_rows[int(index)]["position"] for index in partition.registry_rows],
        dtype=str,
    )


def run_trainable_candidate(
    *,
    candidate_id: str,
    candidate_config_sha256: str,
    fold_id: str,
    seed: int,
    data: CanonicalPhase2Data,
    partitions: dict[str, PartitionData],
    base_config: dict,
    split_sha256: str,
    inner_preprocessing_sha256: str,
    outer_preprocessing_sha256: str,
    run_directory: Path,
) -> dict:
    if candidate_id not in {C0, C1, C3}:
        raise ValueError(candidate_id)
    maximum_epochs = 50
    patience = 8
    minimum_improvement = 1e-12
    criterion = _criterion(base_config)

    inner_model, inner_initialization = _initialize_with_equality_audit(seed)
    inner_initial_hash = inner_initialization["first_initial_model_state_sha256"]
    inner_optimizer = _optimizer(inner_model, base_config)
    inner_history: list[dict] = []
    best_metric = -float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    inner_first_step = None
    inner_positions = _positions_for_partition(data, partitions["inner_train"])
    for epoch in range(1, maximum_epochs + 1):
        if candidate_id in {C0, C1}:
            losses, audit = _erm_train_epoch(
                model=inner_model,
                optimizer=inner_optimizer,
                criterion=criterion,
                partition=partitions["inner_train"],
                fold_id=fold_id,
                seed=seed,
                epoch=epoch,
                stage="inner_selection",
                base_config=base_config,
                initial_hash=inner_initial_hash,
                candidate_id=candidate_id,
            )
        else:
            losses, audit = _coral_train_epoch(
                model=inner_model,
                optimizer=inner_optimizer,
                criterion=criterion,
                partition=partitions["inner_train"],
                positions=inner_positions,
                fold_id=fold_id,
                seed=seed,
                epoch=epoch,
                stage="inner_selection",
                base_config=base_config,
                initial_hash=inner_initial_hash,
            )
        if audit is not None:
            inner_first_step = audit
        validation_logits, _ = extract_logits_embeddings(
            inner_model, partitions["inner_validation"]
        )
        validation = evaluate_logits(partitions["inner_validation"], validation_logits)
        metric = float(validation["sample"]["macro_f1"])
        improved = metric > best_metric + minimum_improvement
        if improved:
            best_metric = metric
            best_epoch = epoch
            best_state = _clone_state(inner_model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        inner_history.append(
            {
                "epoch": epoch,
                "train_total_loss": losses["total_loss"],
                "train_cross_entropy": losses["cross_entropy"],
                "train_coral": losses["coral"],
                "inner_validation_accuracy": validation["sample"]["accuracy"],
                "inner_validation_macro_f1": metric,
                "improved": improved,
                "epochs_without_improvement": epochs_without_improvement,
            }
        )
        if epochs_without_improvement >= patience:
            break
    if best_state is None or inner_first_step is None:
        raise RuntimeError("Frozen inner selection failed")
    inner_model.load_state_dict(best_state, strict=True)
    selected_logits, _ = extract_logits_embeddings(
        inner_model, partitions["inner_validation"]
    )
    inner_evaluation = evaluate_logits(partitions["inner_validation"], selected_logits)
    inner_checkpoint = _save_checkpoint(
        run_directory / "inner_selected_checkpoint.pt",
        candidate_id=candidate_id,
        model=inner_model,
        fold_id=fold_id,
        seed=seed,
        stage="inner_selection_selected",
        epochs=best_epoch,
        candidate_config_sha256=candidate_config_sha256,
        split_sha256=split_sha256,
        preprocessing_sha256=inner_preprocessing_sha256,
    )

    outer_model, outer_initialization = _initialize_with_equality_audit(seed)
    outer_initial_hash = outer_initialization["first_initial_model_state_sha256"]
    if outer_initial_hash != inner_initial_hash:
        raise RuntimeError("Inner and outer initial states differ")
    outer_optimizer = _optimizer(outer_model, base_config)
    outer_history: list[dict] = []
    outer_first_step = None
    outer_positions = _positions_for_partition(data, partitions["outer_development"])
    for epoch in range(1, best_epoch + 1):
        if candidate_id in {C0, C1}:
            losses, audit = _erm_train_epoch(
                model=outer_model,
                optimizer=outer_optimizer,
                criterion=criterion,
                partition=partitions["outer_development"],
                fold_id=fold_id,
                seed=seed,
                epoch=epoch,
                stage="outer_refit",
                base_config=base_config,
                initial_hash=outer_initial_hash,
                candidate_id=candidate_id,
            )
        else:
            losses, audit = _coral_train_epoch(
                model=outer_model,
                optimizer=outer_optimizer,
                criterion=criterion,
                partition=partitions["outer_development"],
                positions=outer_positions,
                fold_id=fold_id,
                seed=seed,
                epoch=epoch,
                stage="outer_refit",
                base_config=base_config,
                initial_hash=outer_initial_hash,
            )
        if audit is not None:
            outer_first_step = audit
        outer_history.append(
            {
                "epoch": epoch,
                "train_total_loss": losses["total_loss"],
                "train_cross_entropy": losses["cross_entropy"],
                "train_coral": losses["coral"],
            }
        )
    if outer_first_step is None:
        raise RuntimeError("Outer-refit first-step audit missing")
    outer_checkpoint = _save_checkpoint(
        run_directory / "outer_refit_checkpoint.pt",
        candidate_id=candidate_id,
        model=outer_model,
        fold_id=fold_id,
        seed=seed,
        stage="outer_refit",
        epochs=best_epoch,
        candidate_config_sha256=candidate_config_sha256,
        split_sha256=split_sha256,
        preprocessing_sha256=outer_preprocessing_sha256,
    )
    loaded_model, payload = load_checkpoint_model(
        Path(outer_checkpoint["path"]), seed
    )
    development_logits, development_embeddings = extract_logits_embeddings(
        loaded_model, partitions["outer_development"]
    )
    development_evaluation = evaluate_logits(
        partitions["outer_development"], development_logits
    )
    held_evaluation_started = utc_now()
    held_logits, held_embeddings = extract_logits_embeddings(
        loaded_model, partitions["outer_held"]
    )
    held_evaluation = evaluate_logits(partitions["outer_held"], held_logits)
    return {
        "candidate_id": candidate_id,
        "fold_id": fold_id,
        "seed": seed,
        "selected_epoch": best_epoch,
        "inner_epochs_executed": len(inner_history),
        "stopped_early": len(inner_history) < maximum_epochs,
        "inner_selected_metric": best_metric,
        "inner_history": inner_history,
        "outer_history": outer_history,
        "inner_initialization": inner_initialization,
        "outer_initialization": outer_initialization,
        "inner_first_step": inner_first_step,
        "outer_first_step": outer_first_step,
        "inner_checkpoint": inner_checkpoint,
        "outer_checkpoint": outer_checkpoint,
        "checkpoint_payload_model_state_sha256": payload["model_state_sha256"],
        "checkpoint_verified_before_outer_held": True,
        "outer_held_evaluation_started_at_utc": held_evaluation_started,
        "inner_evaluation": inner_evaluation,
        "development_evaluation": development_evaluation,
        "held_evaluation": held_evaluation,
        "development_embeddings": development_embeddings,
        "held_embeddings": held_embeddings,
        "training_partitions": {
            "inner": ["inner_train"],
            "outer": ["inner_train", "inner_validation"],
        },
        "epoch_selection_partition": "inner_validation",
        "outer_held_inspected_during_selection": False,
        "outer_held_evaluation_count": 1,
        "model_updates_after_outer_held_evaluation": 0,
    }


def _evaluation_from_npz(partition: PartitionData, path: Path) -> dict:
    with np.load(path, allow_pickle=False) as bundle:
        if not np.array_equal(bundle["registry_rows"], partition.registry_rows):
            raise RuntimeError(f"Prediction registry rows differ from partition: {path}")
        if not np.array_equal(bundle["true_labels"], partition.labels):
            raise RuntimeError(f"Prediction labels differ from partition: {path}")
        logits = np.asarray(bundle["logits"], dtype=np.float32)
    return evaluate_logits(partition, logits)


def phase2_run_manifest(fold_id: str, seed: int) -> tuple[Path, dict]:
    path = (
        PROJECT_ROOT
        / "06_neutral_baseline"
        / "runs"
        / fold_id
        / f"seed_{seed}"
        / "RUN_MANIFEST.json"
    )
    return path, read_json(path)


def construct_reused_c0(
    *,
    fold_id: str,
    seed: int,
    partitions: dict[str, PartitionData],
) -> dict:
    manifest_path, source = phase2_run_manifest(fold_id, seed)
    if source["method_id"] != C0 or not source["run_complete"]:
        raise RuntimeError("Canonical Phase 2 control manifest is invalid")
    checkpoint_path = Path(source["outer_refit"]["checkpoint"]["path"])
    if sha256_file(checkpoint_path) != source["outer_refit"]["checkpoint"]["file_sha256"]:
        raise RuntimeError("Canonical C0 checkpoint hash mismatch")
    model, payload = load_checkpoint_model(checkpoint_path, seed)
    development_logits, development_embeddings = extract_logits_embeddings(
        model, partitions["outer_development"]
    )
    development = evaluate_logits(partitions["outer_development"], development_logits)
    held_source_path = Path(source["outer_refit"]["predictions"]["npz"]["path"])
    if sha256_file(held_source_path) != source["outer_refit"]["predictions"]["npz"]["sha256"]:
        raise RuntimeError("Canonical C0 outer-held prediction hash mismatch")
    held = _evaluation_from_npz(partitions["outer_held"], held_source_path)
    held_logits, held_embeddings = extract_logits_embeddings(model, partitions["outer_held"])
    if not np.array_equal(held_logits, held["logits"]):
        raise RuntimeError("Re-evaluated C0 logits differ from canonical Phase 2 predictions")
    return {
        "candidate_id": C0,
        "fold_id": fold_id,
        "seed": seed,
        "execution_status": "REUSED_CANONICAL_CONTROL",
        "selected_epoch": int(source["inner_selection"]["selected_epoch"]),
        "inner_epochs_executed": int(source["inner_selection"]["epochs_executed"]),
        "stopped_early": bool(source["inner_selection"]["stopped_early"]),
        "inner_selected_metric": float(
            source["inner_selection"]["selected_inner_validation_sample_macro_f1"]
        ),
        "inner_evaluation": None,
        "development_evaluation": development,
        "held_evaluation": held,
        "development_embeddings": development_embeddings,
        "held_embeddings": held_embeddings,
        "source_phase2_manifest": {
            **artifact_record(manifest_path),
            "source_checkpoint": artifact_record(checkpoint_path),
            "source_outer_held_predictions": artifact_record(held_source_path),
            "source_model_state_sha256": payload["model_state_sha256"],
        },
        "initialization": source["initialization"],
        "first_steps": source["first_steps"],
        "checkpoint_verified_before_outer_held": True,
        "training_performed_in_phase3b": False,
        "training_partitions": source["outer_refit"]["training_partitions"],
        "epoch_selection_partition": source["inner_selection"][
            "epoch_selection_partition"
        ],
        "outer_held_inspected_during_selection": source["inner_selection"][
            "outer_held_inspected_during_epoch_selection"
        ],
        "outer_held_evaluation_count_in_phase3b": 1,
        "model_updates_after_outer_held_evaluation": 0,
    }


def _condition_embedding_centroids(
    partition: PartitionData, embeddings: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for index, condition_id in enumerate(partition.condition_ids):
        groups.setdefault(condition_id, []).append(index)
    values: list[np.ndarray] = []
    labels: list[int] = []
    condition_ids: list[str] = []
    for condition_id in sorted(groups, key=str.casefold):
        selected = groups[condition_id]
        if len(selected) != 50:
            raise RuntimeError(f"Condition block is not complete: {condition_id}")
        unique_labels = np.unique(partition.labels[selected])
        if len(unique_labels) != 1:
            raise RuntimeError(f"Condition block labels differ: {condition_id}")
        values.append(np.asarray(embeddings[selected], dtype=np.float64).mean(axis=0))
        labels.append(int(unique_labels[0]))
        condition_ids.append(condition_id)
    return (
        np.ascontiguousarray(np.vstack(values), dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        condition_ids,
    )


def _ncm_logits(embeddings: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    centers = np.asarray(prototypes, dtype=np.float64)
    squared_distances = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.ascontiguousarray(-squared_distances, dtype=np.float32)


def construct_c2(
    *,
    fold_id: str,
    seed: int,
    partitions: dict[str, PartitionData],
    run_directory: Path,
    source_run_directory: Path | None = None,
) -> dict:
    if source_run_directory is None:
        raise RuntimeError("C2 requires the newly executed C0 source run")
    source_manifest_path = source_run_directory / "RUN_MANIFEST.json"
    source = read_json(source_manifest_path)
    if source["candidate_id"] != C0 or source["fold_id"] != fold_id or int(source["seed"]) != seed:
        raise RuntimeError("C2 source run identity mismatch")
    checkpoint_record = source["checkpoints"]["outer"]
    checkpoint_path = Path(checkpoint_record["path"])
    if sha256_file(checkpoint_path) != checkpoint_record["sha256"]:
        raise RuntimeError("C2 source encoder checkpoint hash mismatch")
    model, payload = load_checkpoint_model(checkpoint_path, seed)
    _, development_embeddings = extract_logits_embeddings(
        model, partitions["outer_development"]
    )
    _, held_embeddings = extract_logits_embeddings(model, partitions["outer_held"])
    condition_values, condition_labels, condition_ids = _condition_embedding_centroids(
        partitions["outer_development"], development_embeddings
    )
    prototypes = np.ascontiguousarray(
        np.vstack(
            [condition_values[condition_labels == label].mean(axis=0) for label in range(7)]
        ),
        dtype=np.float64,
    )
    if prototypes.shape != (7, 256):
        raise RuntimeError(f"C2 prototype shape changed: {prototypes.shape}")
    run_directory.mkdir(parents=True, exist_ok=True)
    prototype_path = run_directory / "source_development_prototypes_float64.npy"
    np.save(prototype_path, prototypes, allow_pickle=False)
    prototype_fit = {
            "partition": "outer_development",
            "prototype_grain": "equal_weight_complete_condition_block_embedding_centroids",
            "condition_block_count": len(condition_ids),
            "condition_blocks_per_class": [
                int(np.count_nonzero(condition_labels == label)) for label in range(7)
            ],
            "fit_registry_rows_sha256": hash_lines(
                str(int(value)) for value in partitions["outer_development"].registry_rows
            ),
            "fit_condition_ids_sha256": hash_lines(condition_ids),
            "outer_held_registry_rows_used": False,
            "outer_held_labels_used": False,
            "inner_validation_used_as_separate_prototype_fit_partition": False,
            "metric": "euclidean_squared_distance_equivalent_for_argmin",
            "prototype_array_sha256": array_sha256(prototypes),
            "prototype_file": artifact_record(prototype_path),
    }
    development_logits = _ncm_logits(development_embeddings, prototypes)
    held_logits = _ncm_logits(held_embeddings, prototypes)
    development = evaluate_logits(partitions["outer_development"], development_logits)
    held = evaluate_logits(partitions["outer_held"], held_logits)
    return {
        "candidate_id": C2,
        "fold_id": fold_id,
        "seed": seed,
        "execution_status": "REUSED_CANONICAL_ENCODER_NEW_SOURCE_ONLY_READOUT",
        "selected_epoch": int(source["selected_inner_epoch"]),
        "inner_epochs_executed": int(source["inner_epochs_executed"]),
        "stopped_early": bool(source["stopped_early"]),
        "inner_selected_metric": float(source["inner_selected_sample_macro_f1"]),
        "inner_evaluation": None,
        "development_evaluation": development,
        "held_evaluation": held,
        "development_embeddings": development_embeddings,
        "held_embeddings": held_embeddings,
        "prototype_fit": prototype_fit,
        "source_phase2_manifest": {
            **artifact_record(source_manifest_path),
            "source_checkpoint": artifact_record(checkpoint_path),
            "source_model_state_sha256": payload["model_state_sha256"],
        },
        "training_performed_in_phase3b": False,
        "training_partitions": ["inner_train", "inner_validation"],
        "epoch_selection_partition": "inner_validation_via_canonical_C0",
        "outer_held_inspected_during_selection": False,
        "outer_held_evaluation_count_in_phase3b": 1,
        "model_updates_after_outer_held_evaluation": 0,
    }


def save_completed_run(
    *,
    result: dict,
    partitions: dict[str, PartitionData],
    run_directory: Path,
    candidate_config_sha256: str,
    registry_file_sha256: str,
    split_file_sha256: str,
    inner_preprocessing_sha256: str,
    outer_preprocessing_sha256: str,
    is_repeat: bool,
) -> dict:
    run_directory.mkdir(parents=True, exist_ok=True)
    predictions = {
        "outer_development": save_prediction_bundle(
            run_directory,
            "outer_development",
            partitions["outer_development"],
            result["development_evaluation"],
        ),
        "outer_held": save_prediction_bundle(
            run_directory,
            "outer_held",
            partitions["outer_held"],
            result["held_evaluation"],
        ),
    }
    if result.get("inner_evaluation") is not None:
        predictions["inner_selected"] = save_prediction_bundle(
            run_directory,
            "inner_selected",
            partitions["inner_validation"],
            result["inner_evaluation"],
        )
    metrics = {
        "inner_selected": (
            metrics_only(result["inner_evaluation"])
            if result.get("inner_evaluation") is not None
            else None
        ),
        "outer_development": metrics_only(result["development_evaluation"]),
        "outer_held": metrics_only(result["held_evaluation"]),
        "development_to_held_sample_accuracy_gap": float(
            result["development_evaluation"]["sample"]["accuracy"]
            - result["held_evaluation"]["sample"]["accuracy"]
        ),
        "development_to_held_sample_macro_f1_gap": float(
            result["development_evaluation"]["sample"]["macro_f1"]
            - result["held_evaluation"]["sample"]["macro_f1"]
        ),
    }
    metrics_path = run_directory / "metrics.json"
    write_json(metrics_path, metrics)
    write_json(
        run_directory / "confusion_matrices.json",
        {
            level: metrics["outer_held"][level]["confusion_matrix"]
            for level in ("sample", "condition", "unique_signal_weighted")
        },
    )
    histories = {}
    if result.get("inner_history"):
        histories["inner"] = _write_history(
            run_directory / "inner_epoch_history.csv", result["inner_history"]
        )
    if result.get("outer_history"):
        histories["outer"] = _write_history(
            run_directory / "outer_refit_history.csv", result["outer_history"]
        )
    first_steps = {}
    if result.get("inner_first_step") is not None:
        first_steps["inner_selection"] = result["inner_first_step"]
    if result.get("outer_first_step") is not None:
        first_steps["outer_refit"] = result["outer_first_step"]
    if first_steps:
        write_json(run_directory / "first_step_audit.json", first_steps)
    execution_status = result.get(
        "execution_status", "NEWLY_TRAINED_FROZEN_CANDIDATE"
    )
    manifest = {
        "schema_version": 1,
        "phase": "3B",
        "candidate_id": result["candidate_id"],
        "fold_id": result["fold_id"],
        "seed": result["seed"],
        "execution_status": execution_status,
        "is_predeclared_reproducibility_repeat": is_repeat,
        "candidate_config_sha256": candidate_config_sha256,
        "frozen_registry_file_sha256": registry_file_sha256,
        "split_file_sha256": split_file_sha256,
        "preprocessing": {
            "inner_state_sha256": inner_preprocessing_sha256,
            "outer_state_sha256": outer_preprocessing_sha256,
            "fit_on_training_partitions_only": True,
        },
        "selected_inner_epoch": int(result["selected_epoch"]),
        "inner_epochs_executed": int(result["inner_epochs_executed"]),
        "stopped_early": bool(result["stopped_early"]),
        "inner_selected_sample_macro_f1": float(result["inner_selected_metric"]),
        "training_partitions": result["training_partitions"],
        "epoch_selection_partition": result["epoch_selection_partition"],
        "outer_held_inspected_during_selection": result[
            "outer_held_inspected_during_selection"
        ],
        "outer_held_evaluation_count_in_phase3b": result.get(
            "outer_held_evaluation_count_in_phase3b",
            result.get("outer_held_evaluation_count", 1),
        ),
        "model_updates_after_outer_held_evaluation": result[
            "model_updates_after_outer_held_evaluation"
        ],
        "training_performed_in_phase3b": result.get(
            "training_performed_in_phase3b", True
        ),
        "checkpoints": {
            "inner": result.get("inner_checkpoint"),
            "outer": result.get("outer_checkpoint"),
            "source_phase2": result.get("source_phase2_manifest"),
        },
        "initialization": {
            "inner": result.get("inner_initialization"),
            "outer": result.get("outer_initialization"),
            "reused_phase2": result.get("initialization"),
        },
        "first_steps": first_steps or result.get("first_steps"),
        "prototype_fit": result.get("prototype_fit"),
        "mechanism": result.get("mechanism"),
        "predictions": predictions,
        "metrics": {**artifact_record(metrics_path), "values": metrics},
        "histories": histories,
        "all_classes_treated_symmetrically": True,
        "seed_selected_or_ranked": False,
        "p4_numerical_content_accessed": False,
        "run_complete": True,
    }
    manifest_path = run_directory / "RUN_MANIFEST.json"
    write_json(manifest_path, manifest)
    return {**manifest, "run_manifest": artifact_record(manifest_path)}


def _condition_geometry(
    data: CanonicalPhase2Data,
    development: PartitionData,
    held: PartitionData,
    development_values: np.ndarray,
    held_values: np.ndarray,
) -> dict:
    registry_rows = np.concatenate((development.registry_rows, held.registry_rows))
    values = np.ascontiguousarray(
        np.concatenate((development_values, held_values), axis=0), dtype=np.float64
    )
    groups: dict[str, list[int]] = {}
    for local_index, registry_row in enumerate(registry_rows):
        condition_id = data.registry_rows[int(registry_row)]["raw_condition_id"]
        groups.setdefault(condition_id, []).append(local_index)
    condition_ids = sorted(groups, key=str.casefold)
    centroids = []
    labels = []
    positions = []
    tags = []
    for condition_id in condition_ids:
        selected = groups[condition_id]
        if len(selected) != 50:
            raise RuntimeError(f"Mechanism condition block incomplete: {condition_id}")
        rows = [data.registry_rows[int(registry_rows[index])] for index in selected]
        labels_here = {row["label_index"] for row in rows}
        positions_here = {row["position"] for row in rows}
        if len(labels_here) != 1 or len(positions_here) != 1:
            raise RuntimeError(f"Mechanism condition custody failed: {condition_id}")
        centroids.append(values[selected].mean(axis=0))
        labels.append(next(iter(labels_here)))
        positions.append(next(iter(positions_here)))
        tags.append(rows[0]["tag_id"])
    return {
        "condition_ids": condition_ids,
        "values": np.ascontiguousarray(np.vstack(centroids), dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int64),
        "positions": np.asarray(positions, dtype=str),
        "tags": np.asarray(tags, dtype=np.int64),
    }


def _geometry_metrics(condition_data: dict) -> dict:
    # Imported lazily to keep Phase 3B tied to the already-audited diagnostic
    # definitions without creating a second metric implementation.
    from .phase3a_diagnostics import (
        diagnostic_position_probe,
        fisher_separation,
        safe_silhouette,
    )

    values = np.asarray(condition_data["values"], dtype=np.float64)
    labels = np.asarray(condition_data["labels"], dtype=np.int64)
    positions = np.asarray(condition_data["positions"])
    fisher = fisher_separation(values, labels)
    probe = diagnostic_position_probe(condition_data)
    return {
        "tag_silhouette_condition_blocks": safe_silhouette(values, labels),
        "position_silhouette_condition_blocks": safe_silhouette(values, positions),
        "tag_fisher_ratio": fisher["fisher_ratio"],
        "tag_between_to_within_distance_ratio": fisher[
            "between_to_within_distance_ratio"
        ],
        "position_probe_validation_accuracy": probe["validation_accuracy"],
        "position_probe_validation_macro_f1": probe["validation_macro_f1"],
        "condition_block_count": len(values),
    }


def save_mechanism_artifact(
    *,
    result: dict,
    data: CanonicalPhase2Data,
    partitions: dict[str, PartitionData],
    run_directory: Path,
) -> dict:
    run_directory.mkdir(parents=True, exist_ok=True)
    candidate_id = result["candidate_id"]
    payload = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "fold_id": result["fold_id"],
        "seed": result["seed"],
        "p4_used": False,
    }
    arrays: dict[str, np.ndarray] = {}
    if candidate_id in {C0, C1}:
        input_geometry = _condition_geometry(
            data,
            partitions["outer_development"],
            partitions["outer_held"],
            partitions["outer_development"].inputs,
            partitions["outer_held"].inputs,
        )
        payload["input_geometry"] = _geometry_metrics(input_geometry)
        arrays.update(
            {
                "input_condition_values": input_geometry["values"],
                "input_condition_labels": input_geometry["labels"],
                "input_condition_positions": input_geometry["positions"],
                "input_condition_tags": input_geometry["tags"],
                "input_condition_ids": np.asarray(input_geometry["condition_ids"], dtype=str),
            }
        )
    if candidate_id in {C0, C1, C3}:
        embedding_geometry = _condition_geometry(
            data,
            partitions["outer_development"],
            partitions["outer_held"],
            result["development_embeddings"],
            result["held_embeddings"],
        )
        payload["embedding_geometry"] = _geometry_metrics(embedding_geometry)
        arrays.update(
            {
                "embedding_condition_values": embedding_geometry["values"],
                "embedding_condition_labels": embedding_geometry["labels"],
                "embedding_condition_positions": embedding_geometry["positions"],
                "embedding_condition_tags": embedding_geometry["tags"],
                "embedding_condition_ids": np.asarray(
                    embedding_geometry["condition_ids"], dtype=str
                ),
            }
        )
    if candidate_id == C2:
        prototypes = np.load(
            result["prototype_fit"]["prototype_file"]["path"], allow_pickle=False
        )
        held = np.asarray(result["held_embeddings"], dtype=np.float64)
        distances = np.sum((held[:, None, :] - prototypes[None, :, :]) ** 2, axis=2)
        ordered = np.sort(distances, axis=1)
        pairwise = []
        for left in range(CLASS_COUNT):
            for right in range(left + 1, CLASS_COUNT):
                pairwise.append(float(np.linalg.norm(prototypes[left] - prototypes[right])))
        payload["ncm_readout"] = {
            "mean_nearest_to_second_nearest_squared_distance_margin": float(
                np.mean(ordered[:, 1] - ordered[:, 0])
            ),
            "minimum_class_prototype_distance": min(pairwise),
            "mean_class_prototype_distance": float(np.mean(pairwise)),
            "prototype_fit_outer_development_only": True,
        }
        arrays.update(
            {
                "source_development_prototypes": prototypes,
                "held_nearest_two_squared_distances": ordered[:, :2],
            }
        )
    payload["held_sample_macro_f1"] = result["held_evaluation"]["sample"]["macro_f1"]
    payload["held_worst_class_recall"] = result["held_evaluation"]["sample"][
        "worst_class_recall"
    ]
    payload["held_zero_recall_class_count"] = result["held_evaluation"]["sample"][
        "zero_recall_class_count"
    ]
    payload["development_to_held_macro_f1_gap"] = float(
        result["development_evaluation"]["sample"]["macro_f1"]
        - result["held_evaluation"]["sample"]["macro_f1"]
    )
    arrays_path = run_directory / "mechanism_inputs.npz"
    np.savez(arrays_path, **arrays)
    payload["mechanism_inputs"] = artifact_record(arrays_path)
    payload["mechanism_sha256"] = canonical_json_sha256(payload)
    path = run_directory / "MECHANISM.json"
    write_json(path, payload)
    return {**payload, "artifact": artifact_record(path)}


def compare_repeat(canonical_directory: Path, repeat_directory: Path) -> dict:
    comparisons = {}
    for stem in ("outer_development", "outer_held"):
        canonical_path = canonical_directory / f"{stem}_predictions.npz"
        repeat_path = repeat_directory / f"{stem}_predictions.npz"
        with np.load(canonical_path, allow_pickle=False) as canonical, np.load(
            repeat_path, allow_pickle=False
        ) as repeated:
            keys_equal = set(canonical.files) == set(repeated.files)
            arrays_equal = keys_equal and all(
                np.array_equal(canonical[key], repeated[key]) for key in canonical.files
            )
        comparisons[stem] = {
            "keys_equal": keys_equal,
            "arrays_exactly_equal": arrays_equal,
            "canonical_sha256": sha256_file(canonical_path),
            "repeat_sha256": sha256_file(repeat_path),
        }
    canonical_manifest = read_json(canonical_directory / "RUN_MANIFEST.json")
    repeat_manifest = read_json(repeat_directory / "RUN_MANIFEST.json")
    metrics_equal = canonical_manifest["metrics"]["values"] == repeat_manifest["metrics"][
        "values"
    ]
    selected_epoch_equal = (
        canonical_manifest["selected_inner_epoch"]
        == repeat_manifest["selected_inner_epoch"]
    )
    first_steps_equal = canonical_manifest["first_steps"] == repeat_manifest["first_steps"]
    exact = (
        all(value["arrays_exactly_equal"] for value in comparisons.values())
        and metrics_equal
        and selected_epoch_equal
        and first_steps_equal
    )
    return {
        "prediction_comparisons": comparisons,
        "metrics_exactly_equal": metrics_equal,
        "selected_epoch_equal": selected_epoch_equal,
        "first_steps_exactly_equal": first_steps_equal,
        "complete_repeat_exact": exact,
    }
