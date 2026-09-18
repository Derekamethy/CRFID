"""Frozen Phase 6 training, one-time P4 parsing, inference and metrics.

Target access in this module is machine-enforced. ``load_p4_once``,
``save_prediction_bundle`` and ``evaluate_logits`` all require a
:class:`~crfid.governance.strict_target_authorization.TargetAccessToken` that
can only be minted by ``authorize_target_access`` after the frozen release has
been verified on disk. There is deliberately no default authorization and no
boolean override.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..governance.strict_target_authorization import TargetAccessToken, require_token
from .governance import write_json
from .hashing import (
    array_sha256,
    canonical_json_sha256,
    exact_signal_sha256,
    sha256_file,
    stable_id,
)
from .neutral_data import CanonicalPhase2Data, PartitionData
from .neutral_model import (
    initialize_model,
    model_state_sha256,
    parameter_count,
)
from .phase3b_execution import (
    C1,
    _criterion,
    _erm_train_epoch,
    _initialize_with_equality_audit,
    _optimizer,
)


CLASS_COUNT = 7
SEEDS = (42, 43, 44, 45, 46)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class P4Dataset:
    rows: list[dict]
    signals: np.ndarray
    labels: np.ndarray
    condition_ids: list[str]
    sample_ids: list[str]
    exact_signal_hashes: list[str]
    unique_signal_weights: np.ndarray
    access_records: list[dict]


def build_all_source_partition(
    *, project_root: Path, frozen_mean: np.ndarray, frozen_scale: np.ndarray
) -> tuple[CanonicalPhase2Data, PartitionData, dict]:
    """UNREACHABLE IN v2 -- retained only as legacy reference.

    This function requires ``CanonicalPhase2Data.from_config_path`` and
    ``configs/phase2_neutral_baseline.json``, neither of which exists in this
    repository, so calling it raises. The all-source partition actually used by
    the canonical release is built inline in
    ``scripts/train_and_freeze_strict_dg.py``. Note also that the clip-at-1e-12
    rule below differs from the canonical replace-with-1.0 rule; see
    ``docs/strict_dg_runtime_configuration.md``.
    """

    data = CanonicalPhase2Data.from_config_path(
        project_root / "configs" / "phase2_neutral_baseline.json"
    )
    differenced = np.ascontiguousarray(np.diff(data.signals, n=1, axis=1), dtype=np.float64)
    recomputed_mean = np.ascontiguousarray(differenced.mean(axis=0), dtype=np.float64)
    recomputed_scale = np.ascontiguousarray(
        np.clip(differenced.std(axis=0, ddof=0), 1e-12, None), dtype=np.float64
    )
    if not np.array_equal(recomputed_mean, frozen_mean):
        raise RuntimeError("Frozen all-source preprocessing mean differs")
    if not np.array_equal(recomputed_scale, frozen_scale):
        raise RuntimeError("Frozen all-source preprocessing scale differs")
    inputs = np.ascontiguousarray(
        ((differenced - frozen_mean) / frozen_scale).astype(np.float32)
    )
    indices = np.arange(len(data.registry_rows), dtype=np.int64)
    partition = PartitionData(
        registry_rows=indices,
        inputs=inputs,
        labels=np.ascontiguousarray(data.labels, dtype=np.int64),
        sample_ids=[row["sample_id"] for row in data.registry_rows],
        condition_ids=[row["raw_condition_id"] for row in data.registry_rows],
        exact_signal_hashes=[row["exact_signal_sha256"] for row in data.registry_rows],
        unique_signal_weights=np.asarray(
            [row["unique_signal_weight"] for row in data.registry_rows], dtype=np.float64
        ),
    )
    audit = {
        "sample_count": len(indices),
        "condition_block_count": len(set(partition.condition_ids)),
        "positions": sorted({row["position"] for row in data.registry_rows}),
        "input_shape": list(inputs.shape),
        "input_dtype": inputs.dtype.str,
        "recomputed_mean_array_sha256": array_sha256(recomputed_mean),
        "recomputed_scale_array_sha256": array_sha256(recomputed_scale),
        "transformed_float32_array_sha256": array_sha256(inputs),
        "p4_used": False,
    }
    if (
        audit["sample_count"] != 9450
        or audit["condition_block_count"] != 189
        or audit["positions"] != ["P1", "P2", "P3"]
        or audit["input_shape"] != [9450, 280]
    ):
        raise RuntimeError("Frozen all-source training population differs")
    return data, partition, audit


def train_final_model(
    *,
    project_root: Path,
    output_directory: Path,
    partition: PartitionData,
    base_config: dict,
    seed: int,
    epochs: int,
    recipe_seal_sha256: str,
    candidate_config_sha256: str,
    preprocessing_state_sha256: str,
    source_registry_sha256: str,
    source_signals_sha256: str,
    audit_role: str,
) -> dict:
    """Train the exact C1 ERM path; this function has no P4 input.

    NOT USED BY THE CANONICAL RELEASE. The five final C1 models were produced by
    the inline loop in ``scripts/train_and_freeze_strict_dg.py``; this variant is
    retained as legacy reference and differs in its ``fold_id`` label and
    artifact layout. It is excluded from the canonical code manifest.
    """

    if seed not in SEEDS or epochs <= 0:
        raise ValueError("Seed or epoch is outside the frozen recipe")
    output_directory.mkdir(parents=True, exist_ok=True)
    run_config = {
        "schema_version": 1,
        "method_id": C1,
        "seed": seed,
        "epochs": epochs,
        "recipe_seal_sha256": recipe_seal_sha256,
        "candidate_config_sha256": candidate_config_sha256,
        "preprocessing_state_sha256": preprocessing_state_sha256,
        "source_registry_sha256": source_registry_sha256,
        "source_signals_sha256": source_signals_sha256,
        "sample_count": 9450,
        "batch_size": 256,
        "shuffle_seed_formula": "seed * 1000000 + 100000 + epoch",
        "optimizer": base_config["optimizer"],
        "loss": base_config["loss"],
        "scheduler": None,
        "early_stopping": False,
        "p4_used": False,
    }
    run_config_sha256 = canonical_json_sha256(run_config)
    write_json(output_directory / "RUN_CONFIG.json", {**run_config, "run_config_sha256": run_config_sha256})

    started_at = utc_now()
    model, initialization = _initialize_with_equality_audit(seed)
    initial_hash = initialization["first_initial_model_state_sha256"]
    if parameter_count(model) != 142855:
        raise RuntimeError("Frozen C1 parameter count differs")
    optimizer = _optimizer(model, base_config)
    criterion = _criterion(base_config)
    history: list[dict] = []
    first_step = None
    for epoch in range(1, epochs + 1):
        losses, audit = _erm_train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            partition=partition,
            fold_id="ALL_SOURCE_P1_P2_P3",
            seed=seed,
            epoch=epoch,
            stage="outer_refit",
            base_config=base_config,
            initial_hash=initial_hash,
        )
        if audit is not None:
            first_step = audit
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": losses["total_loss"],
                "train_cross_entropy": losses["cross_entropy"],
                "train_coral": losses["coral"],
            }
        )
        print(
            json.dumps(
                {
                    "event": "FROZEN_TRAIN_EPOCH_COMPLETE",
                    "audit_role": audit_role,
                    "seed": seed,
                    "epoch": epoch,
                    "epochs": epochs,
                    "loss": losses["total_loss"],
                }
            ),
            flush=True,
        )
    if first_step is None or len(history) != epochs:
        raise RuntimeError("Frozen training audit is incomplete")

    history_path = output_directory / "TRAINING_HISTORY.csv"
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    first_step_payload = {
        "schema_version": 1,
        "audit_role": audit_role,
        "seed": seed,
        "epochs": epochs,
        "initialization": initialization,
        "first_step": first_step,
        "p4_used": False,
    }
    first_step_payload["first_step_audit_sha256"] = canonical_json_sha256(
        first_step_payload
    )
    write_json(output_directory / "FIRST_STEP_AUDIT.json", first_step_payload)

    final_model_state_sha256 = model_state_sha256(model)
    checkpoint_payload = {
        "schema_version": 1,
        "phase": 6,
        "method_id": C1,
        "seed": seed,
        "epochs": epochs,
        "stage": "frozen_all_source_final",
        "recipe_seal_sha256": recipe_seal_sha256,
        "run_config_sha256": run_config_sha256,
        "candidate_config_sha256": candidate_config_sha256,
        "preprocessing_state_sha256": preprocessing_state_sha256,
        "source_registry_sha256": source_registry_sha256,
        "source_signals_sha256": source_signals_sha256,
        "model_state_sha256": final_model_state_sha256,
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "p4_used": False,
    }
    checkpoint_path = output_directory / "FINAL_CHECKPOINT.pt"
    temporary = output_directory / "FINAL_CHECKPOINT.pt.tmp"
    torch.save(checkpoint_payload, temporary)
    temporary.replace(checkpoint_path)
    saved_at = utc_now()
    result = {
        "schema_version": 1,
        "audit_role": audit_role,
        "method_id": C1,
        "seed": seed,
        "epochs": epochs,
        "training_started_at_utc": started_at,
        "training_completed_at_utc": saved_at,
        "checkpoint_saved_at_utc": saved_at,
        "run_config_sha256": run_config_sha256,
        "candidate_config_sha256": candidate_config_sha256,
        "initial_model_state_sha256": initial_hash,
        "final_model_state_sha256": final_model_state_sha256,
        "first_step_audit_sha256": first_step_payload["first_step_audit_sha256"],
        "first_batch_sample_ids": first_step["first_batch_sample_ids"],
        "first_batch_registry_rows": first_step["first_batch_registry_rows"],
        "first_batch_tensor_sha256": first_step["first_batch_tensor_sha256"],
        "first_logits_sha256": first_step["first_logits_sha256"],
        "first_loss": first_step["first_loss"],
        "gradient_aggregate_sha256": first_step["gradient_aggregate_sha256"],
        "per_parameter_gradient_sha256": first_step[
            "per_parameter_gradient_sha256"
        ],
        "post_first_step_model_state_sha256": first_step[
            "post_first_step_model_state_sha256"
        ],
        "post_first_step_optimizer_state_sha256": first_step[
            "post_first_step_optimizer_state_sha256"
        ],
        "training_history_sha256": sha256_file(history_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_file_sha256": sha256_file(checkpoint_path),
        "p4_used_for_training": False,
    }
    result["run_manifest_semantic_sha256"] = canonical_json_sha256(result)
    write_json(output_directory / "RUN_MANIFEST.json", result)
    return result


def _integral_value(value: str, role: str) -> int:
    number = float(value)
    if not np.isfinite(number) or number != int(number):
        raise ValueError(f"{role} must be an integer")
    return int(number)


def _active_one_hot(row: dict[str, str], columns: tuple[str, ...], role: str) -> str:
    values = {column: float(row[column]) for column in columns}
    if any(value not in (0.0, 1.0) for value in values.values()):
        raise ValueError(f"Invalid {role} one-hot values")
    active = [column for column, value in values.items() if value == 1.0]
    if len(active) != 1:
        raise ValueError(f"Expected one active {role}")
    return active[0]


def load_p4_once(
    *,
    custody_manifest: dict,
    source_config: dict,
    token: TargetAccessToken,
    phase: int = 6,
) -> P4Dataset:
    """Read each P4 file once; hash and parse the same in-memory byte stream.

    Requires a validated :class:`TargetAccessToken`. The token is re-validated
    against the frozen release on disk before a single target byte is read, so
    this loader cannot run before the recipe, preprocessing state and all five
    checkpoints have been frozen and verified.
    """

    token = require_token(token)
    token.release_features(purpose=f"strict_dg_phase_{int(phase)}_final_evaluation")

    metadata_columns = tuple(source_config["metadata_columns"])
    point_count = int(source_config["signal"]["point_count"])
    signal_columns = tuple(str(index) for index in range(point_count))
    expected_header = metadata_columns + signal_columns
    p4_records = sorted(
        (row for row in custody_manifest["inputs"] if row["domain"] == "P4"),
        key=lambda row: Path(row["absolute_path"]).name,
    )
    if len(p4_records) != 3:
        raise RuntimeError("Phase 0 custody does not identify exactly three P4 files")

    rows: list[dict] = []
    signals: list[np.ndarray] = []
    condition_counts: Counter[str] = Counter()
    access_records: list[dict] = []
    for source_order, custody in enumerate(p4_records):
        path = Path(custody["absolute_path"]).resolve()
        expected_surface = path.name.split("_")[0].upper()
        if path.name.upper() != f"{expected_surface}_P4.CSV" or expected_surface not in {
            "A1",
            "A2",
            "A3",
        }:
            raise RuntimeError(f"Noncanonical P4 file name: {path.name}")
        started = utc_now()
        with path.open("rb") as handle:
            raw_bytes = handle.read()
        completed = utc_now()
        observed_hash = hashlib.sha256(raw_bytes).hexdigest()
        if len(raw_bytes) != int(custody["size_bytes"]) or observed_hash != custody["sha256"]:
            raise RuntimeError(f"P4 file differs from Phase 0 custody: {path}")
        decoded = raw_bytes.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(decoded, newline=""))
        if reader.fieldnames is None or tuple(reader.fieldnames) != expected_header:
            raise RuntimeError(f"Unexpected P4 schema: {path.name}")
        file_rows = 0
        file_hash_id = stable_id("src", f"{path.name}|{observed_hash}")
        for source_row_index, raw_row in enumerate(reader):
            surface = _active_one_hot(
                raw_row, tuple(reversed(metadata_columns[:3])), "surface"
            )
            position = _active_one_hot(raw_row, ("P1", "P2", "P3", "P4"), "position")
            if surface != expected_surface or position != "P4":
                raise RuntimeError(f"P4 file/domain mismatch: {path.name}/{source_row_index}")
            tag_id = _integral_value(raw_row["TagID"], "TagID")
            er = _integral_value(raw_row["ER"], "ER")
            if tag_id not in range(1, 8) or er not in {0, 1, 2}:
                raise RuntimeError("Unexpected P4 Tag/ER metadata")
            signal = np.fromiter(
                (float(raw_row[column]) for column in signal_columns),
                dtype="<f8",
                count=point_count,
            )
            if signal.shape != (281,) or not np.isfinite(signal).all():
                raise RuntimeError("Invalid P4 signal")
            signal_hash = exact_signal_sha256(signal)
            condition_key = f"tag={tag_id}|er={er}|surface={surface}|position=P4"
            condition_id = stable_id("cond", condition_key)
            repeat_index = condition_counts[condition_id]
            condition_counts[condition_id] += 1
            sample_key = (
                f"{path.name}|row={source_row_index}|{condition_key}|signal={signal_hash}"
            )
            rows.append(
                {
                    "registry_row": len(rows),
                    "sample_id": stable_id("sample", sample_key, length=32),
                    "raw_condition_id": condition_id,
                    "repeat_group_id": condition_id,
                    "repeat_id": f"{condition_id}_r{repeat_index:02d}",
                    "repeat_index": repeat_index,
                    "tag_id": tag_id,
                    "label_index": tag_id - 1,
                    "er": er,
                    "surface": surface,
                    "position": "P4",
                    "source_file_id": file_hash_id,
                    "source_file_name": path.name,
                    "source_row_index": source_row_index,
                    "source_csv_line_number": source_row_index + 2,
                    "exact_signal_sha256": signal_hash,
                    "exact_signal_multiplicity": 0,
                    "unique_signal_weight": 0.0,
                    "signal_point_count": 281,
                    "signal_dtype": "<f8",
                }
            )
            signals.append(signal)
            file_rows += 1
        if file_rows != 1050:
            raise RuntimeError(f"Unexpected P4 row count in {path.name}: {file_rows}")
        access_records.append(
            {
                "absolute_path": str(path),
                "file_name": path.name,
                "filesystem_read_count": 1,
                "read_started_at_utc": started,
                "read_completed_at_utc": completed,
                "bytes_read": len(raw_bytes),
                "streaming_sha256": observed_hash,
                "phase0_custody_sha256": custody["sha256"],
                "custody_hash_match": True,
                "rows_parsed": file_rows,
            }
        )

    signal_array = np.ascontiguousarray(np.stack(signals), dtype="<f8")
    labels = np.asarray([row["label_index"] for row in rows], dtype="<i8")
    multiplicities = Counter(row["exact_signal_sha256"] for row in rows)
    for row in rows:
        count = multiplicities[row["exact_signal_sha256"]]
        row["exact_signal_multiplicity"] = count
        row["unique_signal_weight"] = float(1.0 / count)
    if (
        signal_array.shape != (3150, 281)
        or len(condition_counts) != 63
        or set(condition_counts.values()) != {50}
        or len({row["sample_id"] for row in rows}) != 3150
        or len({row["repeat_id"] for row in rows}) != 3150
    ):
        raise RuntimeError("Canonical P4 evaluation population differs")
    return P4Dataset(
        rows=rows,
        signals=signal_array,
        labels=labels,
        condition_ids=[row["raw_condition_id"] for row in rows],
        sample_ids=[row["sample_id"] for row in rows],
        exact_signal_hashes=[row["exact_signal_sha256"] for row in rows],
        unique_signal_weights=np.asarray(
            [row["unique_signal_weight"] for row in rows], dtype=np.float64
        ),
        access_records=access_records,
    )


def transform_p4(
    dataset: P4Dataset, frozen_mean: np.ndarray, frozen_scale: np.ndarray
) -> np.ndarray:
    differenced = np.ascontiguousarray(
        np.diff(dataset.signals, n=1, axis=1), dtype=np.float64
    )
    return np.ascontiguousarray(
        ((differenced - frozen_mean) / frozen_scale).astype(np.float32)
    )


def _confusion(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    true_values = np.asarray(true_labels, dtype=np.int64)
    predicted_values = np.asarray(predictions, dtype=np.int64)
    if weights is None:
        matrix = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
        np.add.at(matrix, (true_values, predicted_values), 1)
    else:
        supplied = np.asarray(weights, dtype=np.float64)
        matrix = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.float64)
        np.add.at(matrix, (true_values, predicted_values), supplied)
    return matrix


def _metrics_from_confusion(matrix: np.ndarray) -> dict:
    values = np.asarray(matrix)
    total = float(values.sum())
    true_support = values.sum(axis=1).astype(np.float64)
    predicted_support = values.sum(axis=0).astype(np.float64)
    true_positive = np.diag(values).astype(np.float64)
    recall = np.divide(
        true_positive,
        true_support,
        out=np.zeros_like(true_positive),
        where=true_support > 0.0,
    )
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros_like(true_positive),
        where=predicted_support > 0.0,
    )
    denominator = 2.0 * true_positive + (predicted_support - true_positive) + (
        true_support - true_positive
    )
    f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0.0,
    )
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_f1": float(f1.mean()),
        "per_class_precision": [float(value) for value in precision],
        "per_class_recall": [float(value) for value in recall],
        "per_class_f1": [float(value) for value in f1],
        "worst_class_recall": float(recall.min()),
        "zero_recall_class_count": int(np.count_nonzero(recall == 0.0)),
        "confusion_matrix": values.tolist(),
        "total_weight": total,
    }


def classification_metrics(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict:
    return _metrics_from_confusion(_confusion(true_labels, predictions, weights))


def condition_metrics(
    true_labels: np.ndarray, logits: np.ndarray, condition_ids: list[str]
) -> tuple[dict, list[dict]]:
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for index, condition_id in enumerate(condition_ids):
        groups.setdefault(condition_id, []).append(index)
    rows = []
    block_true = []
    block_predictions = []
    for condition_id, selected in groups.items():
        labels = np.unique(true_labels[selected])
        if len(selected) != 50 or len(labels) != 1:
            raise RuntimeError(f"Incomplete P4 condition block: {condition_id}")
        mean_logits = np.asarray(logits[selected], dtype=np.float32).astype(
            np.float64
        ).mean(axis=0)
        prediction = int(np.argmax(mean_logits))
        label = int(labels[0])
        block_true.append(label)
        block_predictions.append(prediction)
        rows.append(
            {
                "raw_condition_id": condition_id,
                "sample_count": 50,
                "true_label": label,
                "predicted_label": prediction,
                "mean_logits_json": json.dumps(
                    [float(value) for value in mean_logits], separators=(",", ":")
                ),
            }
        )
    return (
        classification_metrics(
            np.asarray(block_true, dtype=np.int64),
            np.asarray(block_predictions, dtype=np.int64),
        ),
        rows,
    )


def mean_entropy(logits: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return float(
        np.mean(
            -np.sum(
                probabilities * np.log(np.clip(probabilities, 1e-300, None)),
                axis=1,
            )
        )
    )


def predict_from_logits(logits: np.ndarray) -> np.ndarray:
    """Derive predictions without touching target labels."""

    values = np.ascontiguousarray(logits, dtype=np.float32)
    if values.shape != (3150, 7):
        raise RuntimeError("P4 logits shape differs")
    return np.argmax(values, axis=1).astype(np.int64)


def evaluate_logits(
    dataset: P4Dataset, logits: np.ndarray, *, token: TargetAccessToken
) -> tuple[dict, np.ndarray, list[dict]]:
    """Score serialized predictions against target labels.

    Requires a token whose predictions have already been serialized, so target
    labels can never influence the predictions they are used to score.
    """

    token = require_token(token)
    token.release_labels(purpose="final_scoring")
    values = np.ascontiguousarray(logits, dtype=np.float32)
    if values.shape != (3150, 7):
        raise RuntimeError("P4 logits shape differs")
    predictions = np.argmax(values, axis=1).astype(np.int64)
    sample = classification_metrics(dataset.labels, predictions)
    condition, condition_rows = condition_metrics(
        dataset.labels, values, dataset.condition_ids
    )
    unique = classification_metrics(
        dataset.labels, predictions, dataset.unique_signal_weights
    )
    histogram = np.bincount(predictions, minlength=CLASS_COUNT)
    metrics = {
        "sample": sample,
        "condition": condition,
        "unique_signal_weighted": unique,
        "predicted_class_histogram": [int(value) for value in histogram],
        "dominant_predicted_class": int(np.argmax(histogram)),
        "dominant_predicted_class_fraction": float(histogram.max() / len(predictions)),
        "mean_prediction_entropy": mean_entropy(values),
        "sample_count": 3150,
        "condition_block_count": 63,
        "unique_exact_signal_count": len(set(dataset.exact_signal_hashes)),
    }
    return metrics, predictions, condition_rows


def infer_checkpoint(
    *, checkpoint_path: Path, seed: int, inputs: np.ndarray, expected_state_sha256: str
) -> np.ndarray:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = initialize_model(seed)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if model_state_sha256(model) != expected_state_sha256:
        raise RuntimeError("Frozen checkpoint model state differs")
    model.eval()
    tensor = torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.float32)).unsqueeze(1)
    logits = []
    with torch.inference_mode():
        for start in range(0, len(tensor), 256):
            logits.append(model(tensor[start : start + 256]).detach().cpu())
    return np.ascontiguousarray(torch.cat(logits, dim=0).numpy(), dtype=np.float32)


def save_prediction_bundle(
    *,
    path: Path,
    dataset: P4Dataset,
    logits: np.ndarray,
    predictions: np.ndarray,
    token: TargetAccessToken,
) -> dict:
    """Serialize the prediction bundle and open the label-scoring capability."""

    token = require_token(token)
    np.savez(
        path,
        registry_rows=np.arange(3150, dtype=np.int64),
        true_labels=dataset.labels.astype(np.int64, copy=False),
        predictions=predictions.astype(np.int64, copy=False),
        logits=np.asarray(logits, dtype=np.float32),
        unique_signal_weights=dataset.unique_signal_weights.astype(np.float64, copy=False),
        sample_ids=np.asarray(dataset.sample_ids, dtype=str),
        condition_ids=np.asarray(dataset.condition_ids, dtype=str),
        exact_signal_hashes=np.asarray(dataset.exact_signal_hashes, dtype=str),
    )
    token.mark_predictions_serialized()
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "predictions_array_sha256": array_sha256(
            np.asarray(predictions, dtype=np.int64)
        ),
        "logits_array_sha256": array_sha256(np.asarray(logits, dtype=np.float32)),
    }


def recursively_exact(left: object, right: object) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            recursively_exact(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            recursively_exact(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=0.0, abs_tol=0.0)
    return left == right
