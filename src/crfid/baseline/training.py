"""Historical Pre-DG training, selection, checkpoint and reload flow."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..governance.pre_dg_integrity import require_file_sha256
from ..models.cnn1d import model_state_sha256
from ..training.checkpoints import load_checkpoint, save_checkpoint
from .artifacts import write_json
from .model import build_pre_dg_model, verify_model_contract


@dataclass(frozen=True)
class CandidateTraining:
    dataset_id: str
    split_name: str
    seed: int
    dropout: float
    validation_accuracy: float
    validation_loss: float
    best_epoch: int
    epochs_ran: int
    checkpoint_path: str
    checkpoint_sha256: str
    model_state_sha256: str
    history_path: str
    record_path: str


def set_reproduction_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def encode_labels(values: np.ndarray, class_order: tuple[int, ...]) -> np.ndarray:
    mapping = {int(label): index for index, label in enumerate(class_order)}
    encoded = np.asarray([mapping[int(value)] for value in values], dtype=np.int64)
    if tuple(sorted(np.unique(encoded).tolist())) != tuple(range(len(class_order))):
        raise ValueError("A Pre-DG partition lost a class")
    return encoded


def _loader(inputs: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.float32)),
            torch.from_numpy(np.array(labels, dtype=np.int64, copy=True, order="C")),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def _evaluate(model: Any, loader: Any, device: Any) -> dict[str, Any]:
    import torch

    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss_total = 0.0
    row_count = 0
    truth_parts = []
    prediction_parts = []
    logits_parts = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss_total += float(loss.detach().cpu()) * len(labels)
            row_count += len(labels)
            truth_parts.append(labels.detach().cpu().numpy().astype(np.int64))
            prediction_parts.append(logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64))
            logits_parts.append(logits.detach().cpu().numpy().astype(np.float32))
    return {
        "loss": loss_total / max(row_count, 1),
        "accuracy": float(
            np.mean(np.concatenate(truth_parts) == np.concatenate(prediction_parts))
        ),
        "truth": np.concatenate(truth_parts),
        "prediction": np.concatenate(prediction_parts),
        "logits": np.concatenate(logits_parts),
    }


def _is_better(
    current: dict[str, Any], best: dict[str, Any] | None, minimum_improvement: float
) -> bool:
    if best is None:
        return True
    if float(current["accuracy"]) > float(best["accuracy"]) + minimum_improvement:
        return True
    if math.isclose(
        float(current["accuracy"]),
        float(best["accuracy"]),
        rel_tol=1e-9,
        abs_tol=minimum_improvement,
    ):
        return float(current["loss"]) < float(best["loss"]) - minimum_improvement
    return False


def _load_cached_candidate(record_path: Path) -> CandidateTraining | None:
    if not record_path.is_file():
        return None
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    candidate = CandidateTraining(**payload)
    require_file_sha256(candidate.checkpoint_path, candidate.checkpoint_sha256)
    if not Path(candidate.history_path).is_file():
        raise FileNotFoundError(candidate.history_path)
    return candidate


def train_candidate(
    *,
    dataset_id: str,
    split_name: str,
    seed: int,
    dropout: float,
    class_count: int,
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    validation_inputs: np.ndarray,
    validation_labels: np.ndarray,
    training_config: dict[str, Any],
    checkpoint_path: str | Path,
    history_path: str | Path,
    record_path: str | Path,
    resume: bool = True,
) -> CandidateTraining:
    import torch

    record = Path(record_path)
    if resume:
        cached = _load_cached_candidate(record)
        if cached is not None:
            return cached
    set_reproduction_seed(seed)
    device = torch.device("cpu")
    batch_size = int(training_config["batch_size"])
    train_loader = _loader(train_inputs, train_labels, batch_size, True)
    validation_loader = _loader(validation_inputs, validation_labels, batch_size, False)
    model = build_pre_dg_model(class_count, dropout).to(device)
    verify_model_contract(model, class_count)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    criterion = torch.nn.CrossEntropyLoss()
    best_state: dict[str, Any] | None = None
    best_evaluation: dict[str, Any] | None = None
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(training_config["max_epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Pre-DG training loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(labels)
            total_rows += len(labels)
        validation = _evaluate(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_rows, 1),
                "validation_loss": float(validation["loss"]),
                "validation_accuracy": float(validation["accuracy"]),
            }
        )
        if _is_better(
            validation,
            best_evaluation,
            float(training_config["minimum_improvement"]),
        ):
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_evaluation = validation
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(training_config["patience"]):
                break
    if best_state is None or best_evaluation is None:
        raise RuntimeError("Pre-DG training did not produce a checkpoint")
    state_sha256 = model_state_sha256(best_state)
    checkpoint = Path(checkpoint_path)
    checkpoint_sha256 = save_checkpoint(
        checkpoint,
        {
            "model_state": best_state,
            "model_state_sha256": state_sha256,
            "dataset_id": dataset_id,
            "split_name": split_name,
            "seed": int(seed),
            "dropout": float(dropout),
            "class_count": int(class_count),
            "best_epoch": int(best_epoch),
            "validation_accuracy": float(best_evaluation["accuracy"]),
            "validation_loss": float(best_evaluation["loss"]),
        },
    )
    write_json(history_path, history)
    candidate = CandidateTraining(
        dataset_id=dataset_id,
        split_name=split_name,
        seed=int(seed),
        dropout=float(dropout),
        validation_accuracy=float(best_evaluation["accuracy"]),
        validation_loss=float(best_evaluation["loss"]),
        best_epoch=int(best_epoch),
        epochs_ran=len(history),
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=checkpoint_sha256,
        model_state_sha256=state_sha256,
        history_path=str(Path(history_path)),
        record_path=str(record),
    )
    write_json(record, asdict(candidate))
    return candidate


def select_candidate(candidates: list[CandidateTraining]) -> CandidateTraining:
    if {candidate.dropout for candidate in candidates} != {0.0, 0.3}:
        raise ValueError("Both authoritative dropout candidates are required")
    return sorted(
        candidates,
        key=lambda item: (
            -item.validation_accuracy,
            item.validation_loss,
            item.dropout,
        ),
    )[0]


def reload_checkpoint_model(candidate: CandidateTraining, class_count: int) -> Any:
    payload = load_checkpoint(candidate.checkpoint_path, candidate.checkpoint_sha256)
    model = build_pre_dg_model(class_count, candidate.dropout)
    model.load_state_dict(payload["model_state"])
    if model_state_sha256(model) != candidate.model_state_sha256:
        raise ValueError("Reloaded Pre-DG model state hash mismatch")
    model.eval()
    return model


def predict_reloaded_model(
    model: Any, inputs: np.ndarray, labels: np.ndarray, batch_size: int
) -> dict[str, Any]:
    import torch

    return _evaluate(
        model,
        _loader(inputs, labels, int(batch_size), False),
        torch.device("cpu"),
    )
