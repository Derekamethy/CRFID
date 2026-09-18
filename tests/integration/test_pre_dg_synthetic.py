from __future__ import annotations

from pathlib import Path

import numpy as np

from crfid.baseline.evaluation import evaluate_predictions
from crfid.baseline.preprocessing import TrainOnlyChannelScaler
from crfid.baseline.training import (
    predict_reloaded_model,
    reload_checkpoint_model,
    select_candidate,
    train_candidate,
)


def _synthetic(seed: int = 2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(7), 4)
    values = rng.normal(scale=0.05, size=(len(labels), 2, 512)).astype(np.float32)
    values[:, 0] += labels[:, None] / 3.0
    values[:, 1] = 0.0
    return values, labels.astype(np.int64)


def test_complete_synthetic_train_select_reload_predict(tmp_path: Path) -> None:
    inputs, labels = _synthetic()
    train_x, val_x, test_x = inputs[:14], inputs[14:21], inputs[21:]
    train_y, val_y, test_y = labels[:14], labels[14:21], labels[21:]
    scaler = TrainOnlyChannelScaler.fit(train_x, partition="train")
    state = tmp_path / "state.npz"
    scaler.save(state)
    loaded = TrainOnlyChannelScaler.load(state)
    config = {
        "batch_size": 7,
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
        "max_epochs": 1,
        "patience": 1,
        "minimum_improvement": 0.000001,
    }
    candidates = []
    for dropout in (0.0, 0.3):
        tag = str(dropout).replace(".", "p")
        candidates.append(
            train_candidate(
                dataset_id="paper4_depolarizing",
                split_name="synthetic",
                seed=42,
                dropout=dropout,
                class_count=7,
                train_inputs=loaded.transform(train_x),
                train_labels=train_y,
                validation_inputs=loaded.transform(val_x),
                validation_labels=val_y,
                training_config=config,
                checkpoint_path=tmp_path / f"{tag}.pt",
                history_path=tmp_path / f"{tag}_history.json",
                record_path=tmp_path / f"{tag}_record.json",
                resume=False,
            )
        )
    selected = select_candidate(candidates)
    model = reload_checkpoint_model(selected, 7)
    prediction = predict_reloaded_model(
        model, loaded.transform(test_x), test_y, batch_size=7
    )
    metrics = evaluate_predictions(
        prediction["truth"], prediction["prediction"], class_count=7
    )
    assert len(prediction["prediction"]) == 7
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_candidate_resume_is_deterministic(tmp_path: Path) -> None:
    inputs, labels = _synthetic()
    config = {
        "batch_size": 14,
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
        "max_epochs": 1,
        "patience": 1,
        "minimum_improvement": 0.000001,
    }
    kwargs = dict(
        dataset_id="paper4_depolarizing",
        split_name="synthetic",
        seed=42,
        dropout=0.0,
        class_count=7,
        train_inputs=inputs[:14],
        train_labels=labels[:14],
        validation_inputs=inputs[14:21],
        validation_labels=labels[14:21],
        training_config=config,
        checkpoint_path=tmp_path / "model.pt",
        history_path=tmp_path / "history.json",
        record_path=tmp_path / "record.json",
    )
    first = train_candidate(**kwargs, resume=False)
    second = train_candidate(**kwargs, resume=True)
    assert first == second
