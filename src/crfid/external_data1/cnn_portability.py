"""Exact four-class raw and first-difference CNN portability execution."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

from ..exceptions import DependencyUnavailableError
from ..models.cnn1d import (
    initialize_neutral_model,
    model_state_sha256,
    parameter_count,
)
from ..training.checkpoints import load_checkpoint, save_checkpoint
from .preprocessing import (
    fit_standardization,
    load_standardization,
    save_standardization,
    transform_with_state,
)


SEEDS = (42, 43, 44, 45, 46)
EPOCHS = {42: 13, 43: 13, 44: 9, 45: 10, 46: 9}
PARAMETER_COUNT = 142_084
_DETERMINISM_CONFIGURED = False


def configure_determinism() -> None:
    global _DETERMINISM_CONFIGURED
    if _DETERMINISM_CONFIGURED:
        return
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED=0 must be set before scientific execution")
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise DependencyUnavailableError("PyTorch is required for Data1 CNN execution") from exc
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _DETERMINISM_CONFIGURED = True


def _permutation_sha256(permutation: object) -> str:
    values = permutation.detach().cpu().numpy().astype("<i8", copy=False)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def train_cnn(
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    *,
    seed: int,
    epochs: int,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> tuple[object, dict[str, object]]:
    configure_determinism()
    import torch
    from torch import nn

    inputs = np.ascontiguousarray(train_inputs, dtype=np.float32)
    labels_array = np.ascontiguousarray(train_labels, dtype=np.int64)
    if inputs.ndim != 2 or labels_array.shape != (len(inputs),):
        raise ValueError("CNN training arrays are not aligned")
    model = initialize_neutral_model(seed, class_count=4)
    initial_hash = model_state_sha256(model)
    if parameter_count(model) != PARAMETER_COUNT:
        raise RuntimeError("Four-class CNN parameter count changed")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0001,
        amsgrad=False,
        foreach=False,
        fused=False,
    )
    criterion = nn.CrossEntropyLoss(
        weight=None, reduction="mean", label_smoothing=0.0
    )
    records: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        permutation_seed = seed * 1_000_000 + 100_000 + epoch
        generator = torch.Generator(device="cpu")
        generator.manual_seed(permutation_seed)
        permutation = torch.randperm(len(labels_array), generator=generator)
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for start in range(0, len(permutation), 256):
            selected = permutation[start : start + 256].numpy()
            batch_inputs = torch.from_numpy(inputs[selected]).unsqueeze(1)
            batch_labels = torch.from_numpy(labels_array[selected])
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_inputs), batch_labels)
            loss.backward()
            optimizer.step()
            count = len(selected)
            loss_sum += float(loss.detach().cpu().item()) * count
            sample_count += count
        record: dict[str, object] = {
            "epoch": epoch,
            "train_cross_entropy": loss_sum / sample_count,
            "sample_count": sample_count,
            "batch_size": 256,
            "permutation_seed": permutation_seed,
            "permutation_sha256": _permutation_sha256(permutation),
            "epoch_runtime_seconds": time.perf_counter() - epoch_started,
            "test_access": False,
        }
        records.append(record)
        if progress is not None:
            progress(record)
    return model, {
        "seed": seed,
        "fixed_epochs": epochs,
        "initial_model_state_sha256": initial_hash,
        "final_model_state_sha256": model_state_sha256(model),
        "parameter_count": PARAMETER_COUNT,
        "training_runtime_seconds": time.perf_counter() - started,
        "epoch_records": records,
    }


def save_and_reload_checkpoint(
    checkpoint_path: str | Path,
    model: object,
    *,
    run_id: str,
    representation: str,
    seed: int,
    fixed_epoch: int,
    preprocessing_sha256: str,
) -> tuple[object, dict[str, object]]:
    state_hash = model_state_sha256(model)
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "representation": representation,
        "seed": seed,
        "fixed_final_epoch": fixed_epoch,
        "model_state_sha256": state_hash,
        "preprocessing_sha256": preprocessing_sha256,
        "source_weights_used": False,
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
    }
    checkpoint_sha256 = save_checkpoint(checkpoint_path, payload)
    loaded = load_checkpoint(checkpoint_path, checkpoint_sha256)
    reloaded_model = initialize_neutral_model(seed, class_count=4)
    reloaded_model.load_state_dict(loaded["model_state_dict"], strict=True)
    if model_state_sha256(reloaded_model) != state_hash:
        raise RuntimeError("Reloaded Data1 checkpoint state changed")
    return reloaded_model, {
        "file_sha256": checkpoint_sha256,
        "model_state_sha256": state_hash,
        "fixed_final_epoch": fixed_epoch,
        "reload_verified": True,
    }


def infer_cnn(model: object, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    values = np.ascontiguousarray(inputs, dtype=np.float32)
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(values), 256):
            tensor = torch.from_numpy(values[start : start + 256]).unsqueeze(1)
            parts.append(torch.softmax(model(tensor), dim=1).cpu().numpy())
    scores = np.concatenate(parts, axis=0).astype(np.float64, copy=False)
    return (
        np.ascontiguousarray(scores.argmax(axis=1), dtype=np.int64),
        np.ascontiguousarray(scores, dtype=np.float64),
    )


def prepare_representation(
    train_inputs: np.ndarray,
    representation: str,
    state_path: str | Path,
) -> tuple[np.ndarray, object]:
    transformed, state = fit_standardization(train_inputs, representation)
    save_standardization(state_path, state)
    loaded = load_standardization(state_path)
    if not (
        np.array_equal(state.mean, loaded.mean)
        and np.array_equal(state.scale, loaded.scale)
        and state.representation == loaded.representation
    ):
        raise RuntimeError("Reloaded preprocessing state changed")
    return transformed, loaded


def transform_test(inputs: np.ndarray, state: object) -> np.ndarray:
    return transform_with_state(inputs, state)

