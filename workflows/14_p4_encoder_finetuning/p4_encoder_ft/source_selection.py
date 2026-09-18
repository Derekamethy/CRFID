"""P1--P3-only pseudo-target selection for fine-tuning hyperparameters."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from p4_factor_aware.artifacts import write_csv, write_json, write_text
from p4_factor_aware.metrics import metrics_from_predictions
from p4_factor_aware.models import FrozenC1CNN1D, extract_embeddings, model_state_sha256, preprocess
from p4_factor_aware.protocol import ProtocolViolation
from p4_linear_readout.linear import fit_linear_readout

from .constants import (
    ARMS,
    EXPERIMENT_ID,
    SOURCE_EPOCHS,
    SOURCE_EPISODE_KEYS,
    SOURCE_IMPORT_SHA256,
    SOURCE_LEARNING_RATES,
    SOURCE_PROTOCOL_SHA256,
    SOURCE_REGISTRY_SHA256,
    SOURCE_SEEDS,
    SOURCE_SELECTION_RELATIVE,
    SOURCE_SIGNALS_ARRAY_SHA256,
    SOURCE_SIGNALS_FILE_SHA256,
    SOURCE_WEIGHT_DECAYS,
)
from .finetune import derive_seed, fine_tune, predict


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def legacy_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"|")
    digest.update(",".join(str(item) for item in array.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_protocol(archive_repository: Path) -> tuple[dict, dict, Path]:
    root = archive_repository / SOURCE_SELECTION_RELATIVE
    protocol_path = root / "SOURCE_PSEUDOTARGET_PROTOCOL.json"
    imports_path = root / "SOURCE_CHECKPOINT_IMPORT_MANIFEST.json"
    if sha256_file(protocol_path) != SOURCE_PROTOCOL_SHA256 or sha256_file(imports_path) != SOURCE_IMPORT_SHA256:
        raise ProtocolViolation("SOURCE_SELECTION_LINEAGE_HASH_MISMATCH")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    imports = json.loads(imports_path.read_text(encoding="utf-8"))
    if not protocol.get("source_only") or protocol.get("p4_raw_signal_accessed") or protocol.get("sealed_p4_query_labels_accessed"):
        raise ProtocolViolation("SOURCE_SELECTION_PROTOCOL_NOT_SOURCE_ONLY")
    if not imports.get("all_byte_identical") or not imports.get("all_held_positions_excluded"):
        raise ProtocolViolation("SOURCE_SELECTION_IMPORT_INVALID")
    if not protocol["leakage_audit"]["all_valid"]:
        raise ProtocolViolation("SOURCE_SELECTION_PROTOCOL_LEAKAGE")
    return protocol, imports, root


def _asset_maps(imports: dict, root: Path) -> tuple[dict[tuple[str, int], Path], dict[tuple[str, str], Path]]:
    checkpoints: dict[tuple[str, int], Path] = {}
    for row in imports["checkpoints"]:
        key = (row["fold_id"], int(row["seed"]))
        path = root.parent.parent / Path(row["destination_relative_path"])
        if sha256_file(path) != row["destination_sha256"] or row.get("p4_used"):
            raise ProtocolViolation("SOURCE_CHECKPOINT_HASH_OR_IDENTITY_MISMATCH")
        checkpoints[key] = path
    preprocessing: dict[tuple[str, str], Path] = {}
    for row in imports["preprocessing_assets"]:
        path = root.parent.parent / Path(row["destination_relative_path"])
        if sha256_file(path) != row["destination_sha256"]:
            raise ProtocolViolation("SOURCE_PREPROCESSING_HASH_MISMATCH")
        preprocessing[(row["fold_id"], row["role"])] = path
    if len(checkpoints) != 15:
        raise ProtocolViolation("SOURCE_CHECKPOINT_COUNT_MISMATCH")
    return checkpoints, preprocessing


def _load_source_data(protocol: dict) -> tuple[np.ndarray, list[dict[str, str]], Path, Path]:
    custody = protocol["source_data_custody"]
    signals_path = Path(custody["signals_path"])
    registry_path = Path(custody["registry_csv_path"])
    if sha256_file(signals_path) != SOURCE_SIGNALS_FILE_SHA256 or sha256_file(registry_path) != SOURCE_REGISTRY_SHA256:
        raise ProtocolViolation("SOURCE_DATA_FILE_HASH_MISMATCH")
    signals = np.load(signals_path, allow_pickle=False)
    if signals.shape != (9450, 281) or signals.dtype != np.float64 or legacy_array_sha256(signals) != SOURCE_SIGNALS_ARRAY_SHA256:
        raise ProtocolViolation("SOURCE_SIGNAL_ARRAY_IDENTITY_MISMATCH")
    with registry_path.open(encoding="utf-8", newline="") as handle:
        registry = list(csv.DictReader(handle))
    if len(registry) != 9450 or any(int(row["registry_row"]) != index for index, row in enumerate(registry)):
        raise ProtocolViolation("SOURCE_REGISTRY_STRUCTURE_MISMATCH")
    return signals, registry, signals_path, registry_path


def _load_source_model(path: Path, *, fold: str, seed: int) -> FrozenC1CNN1D:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("fold_id") != fold or int(payload.get("seed")) != seed or payload.get("p4_used"):
        raise ProtocolViolation("SOURCE_PSEUDOTARGET_CHECKPOINT_IDENTITY_MISMATCH")
    model = FrozenC1CNN1D()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if model_state_sha256(model) != payload["model_state_sha256"]:
        raise ProtocolViolation("SOURCE_PSEUDOTARGET_CHECKPOINT_STATE_MISMATCH")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return model


def _episode_rows(episode: dict) -> tuple[list[dict], list[dict]]:
    support_lookup = {row["sample_id"]: row for row in episode["support_pool_samples"]}
    support = [support_lookup[sample_id] for sample_id in episode["support_sample_ids_by_shot"]["5"]]
    query = list(episode["query_samples"])
    if len(support) != 35 or len(query) != 1400:
        raise ProtocolViolation("SOURCE_SELECTION_EPISODE_SIZE_MISMATCH")
    if set(row["raw_condition_id"] for row in support) & set(row["raw_condition_id"] for row in query):
        raise ProtocolViolation("SOURCE_SELECTION_BLOCK_OVERLAP")
    if set(row["exact_signal_sha256"] for row in support) & set(row["exact_signal_sha256"] for row in query):
        raise ProtocolViolation("SOURCE_SELECTION_SIGNAL_OVERLAP")
    return support, query


def _summary_rows(unit_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, float, int], list[dict[str, object]]] = defaultdict(list)
    for row in unit_rows:
        grouped[(str(row["arm"]), float(row["learning_rate"]), float(row["encoder_weight_decay"]), int(row["epochs"]))].append(row)
    summaries = []
    for (arm, learning_rate, weight_decay, epochs), rows in sorted(grouped.items()):
        if len(rows) != 45:
            raise ProtocolViolation("SOURCE_SELECTION_CANDIDATE_UNIT_COUNT_MISMATCH")
        held_means = []
        for position in ("P1", "P2", "P3"):
            held_means.append(float(np.mean([float(row["query_macro_f1"]) for row in rows if row["held_position"] == position])))
        class_recall = []
        for class_index in range(7):
            class_recall.append(float(np.mean([float(str(row["query_recall_by_class"]).split("|")[class_index]) for row in rows])))
        summaries.append(
            {
                "arm": arm,
                "learning_rate": learning_rate,
                "encoder_weight_decay": weight_decay,
                "epochs": epochs,
                "unit_count": len(rows),
                "minimum_held_position_mean_query_macro_f1": min(held_means),
                "overall_mean_query_macro_f1": float(np.mean([float(row["query_macro_f1"]) for row in rows])),
                "overall_mean_query_accuracy": float(np.mean([float(row["query_accuracy"]) for row in rows])),
                "minimum_class_mean_recall": min(class_recall),
                "mean_support_minus_query_macro_f1": float(np.mean([float(row["support_macro_f1"]) - float(row["query_macro_f1"]) for row in rows])),
                "all_fits_finite": all(bool(row["all_objectives_finite"]) for row in rows),
            }
        )
    return summaries


def _select(summaries: list[dict[str, object]], arm: str) -> dict[str, object]:
    rows = [row for row in summaries if row["arm"] == arm and bool(row["all_fits_finite"])]
    if len(rows) != len(SOURCE_LEARNING_RATES) * len(SOURCE_WEIGHT_DECAYS) * len(SOURCE_EPOCHS):
        raise ProtocolViolation("SOURCE_SELECTION_GRID_INCOMPLETE")
    return max(
        rows,
        key=lambda row: (
            float(row["minimum_held_position_mean_query_macro_f1"]),
            float(row["overall_mean_query_macro_f1"]),
            float(row["minimum_class_mean_recall"]),
            -float(row["mean_support_minus_query_macro_f1"]),
            -int(row["epochs"]),
            -float(row["learning_rate"]),
            float(row["encoder_weight_decay"]),
        ),
    )


def run_source_selection(*, archive_repository: str | Path, output_directory: str | Path) -> dict[str, object]:
    archive = Path(archive_repository).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "07_SOURCE_SELECTED_CONFIG.json"
    if selected_path.exists():
        raise RuntimeError("SOURCE_SELECTION_ALREADY_FROZEN")
    protocol, imports, fs4_root = _load_protocol(archive)
    checkpoints, preprocessing = _asset_maps(imports, fs4_root)
    signals, registry, signals_path, registry_path = _load_source_data(protocol)
    fold_for_position = {"P1": "S1", "P2": "S2", "P3": "S3"}
    episodes = {(row["held_position"], int(row["episode_id"])): row for row in protocol["episodes"]}
    if not set(SOURCE_EPISODE_KEYS).issubset(episodes):
        raise ProtocolViolation("SOURCE_SELECTION_EPISODE_KEY_MISMATCH")
    unit_rows: list[dict[str, object]] = []
    for held_position, episode_id in SOURCE_EPISODE_KEYS:
        episode = episodes[(held_position, episode_id)]
        support_rows, query_rows = _episode_rows(episode)
        support_indices = np.asarray([int(row["registry_row"]) for row in support_rows], dtype=np.int64)
        query_indices = np.asarray([int(row["registry_row"]) for row in query_rows], dtype=np.int64)
        support_labels = np.asarray([int(registry[index]["label_index"]) for index in support_indices], dtype=np.int64)
        query_labels = np.asarray([int(registry[index]["label_index"]) for index in query_indices], dtype=np.int64)
        fold = fold_for_position[held_position]
        mean = np.load(preprocessing[(fold, "outer_refit_mean")], allow_pickle=False)
        scale = np.load(preprocessing[(fold, "outer_refit_scale")], allow_pickle=False)
        support_inputs = preprocess(signals[support_indices], mean, scale)
        query_inputs = preprocess(signals[query_indices], mean, scale)
        for source_seed in SOURCE_SEEDS:
            source_model = _load_source_model(checkpoints[(fold, source_seed)], fold=fold, seed=source_seed)
            frozen_support = extract_embeddings(source_model, support_inputs)
            parent_head = fit_linear_readout(
                support_embeddings=frozen_support,
                support_labels=support_labels,
                anchor_weight=source_model.network[-1].weight,
                anchor_bias=source_model.network[-1].bias,
            )
            if not parent_head.diagnostics["optimizer_completed_finite"]:
                raise ProtocolViolation("SOURCE_SELECTION_PARENT_HEAD_FIT_FAILED")
            for arm in ARMS:
                for learning_rate in SOURCE_LEARNING_RATES:
                    for weight_decay in SOURCE_WEIGHT_DECAYS:
                        seed = derive_seed(arm, held_position, source_seed, episode_id, learning_rate, weight_decay, prefix="P4_ENCODER_FT_SOURCE_V1")
                        snapshots = fine_tune(
                            source_model=source_model,
                            support_inputs=support_inputs,
                            support_labels=support_labels,
                            initial_head_weight=parent_head.weight,
                            initial_head_bias=parent_head.bias,
                            arm=arm,
                            learning_rate=learning_rate,
                            encoder_weight_decay=weight_decay,
                            epochs=max(SOURCE_EPOCHS),
                            seed=seed,
                            capture_epochs=SOURCE_EPOCHS,
                        )
                        for snapshot in snapshots:
                            query_prediction, _ = predict(snapshot.model, query_inputs)
                            metrics = metrics_from_predictions(query_labels, query_prediction)
                            unit_rows.append(
                                {
                                    "arm": arm,
                                    "held_position": held_position,
                                    "fold_id": fold,
                                    "episode_id": episode_id,
                                    "episode_family": episode["family"],
                                    "episode_local": episode["local_episode"],
                                    "source_seed": source_seed,
                                    "learning_rate": learning_rate,
                                    "encoder_weight_decay": weight_decay,
                                    "epochs": snapshot.epoch,
                                    "finetuning_seed": seed,
                                    "support_rows": len(support_indices),
                                    "query_rows": len(query_indices),
                                    "support_macro_f1": snapshot.diagnostics["support_macro_f1"],
                                    "query_macro_f1": metrics["macro_f1"],
                                    "query_accuracy": metrics["accuracy"],
                                    "query_balanced_accuracy": float(np.mean(metrics["recall"])),
                                    "query_worst_class_f1": float(np.min(metrics["f1"])),
                                    "query_recall_by_class": "|".join(str(float(value)) for value in metrics["recall"]),
                                    "encoder_displacement_frobenius": snapshot.diagnostics["encoder_displacement_frobenius"],
                                    "all_objectives_finite": snapshot.diagnostics["all_objectives_finite"],
                                }
                            )
    summaries = _summary_rows(unit_rows)
    selected = {arm: _select(summaries, arm) for arm in ARMS}
    unit_fields = list(unit_rows[0])
    summary_fields = list(summaries[0])
    write_csv(output / "05_SOURCE_SELECTION_UNIT_RESULTS.csv", unit_rows, unit_fields)
    write_csv(output / "06_SOURCE_SELECTION_SUMMARY.csv", summaries, summary_fields)
    binding = {
        "phase": "SOURCE_ONLY_SELECTION_COMPLETE_BEFORE_P4_FINETUNING",
        "experiment_id": EXPERIMENT_ID,
        "p4_files_opened": 0,
        "p4_metrics_used": False,
        "source_protocol_sha256": SOURCE_PROTOCOL_SHA256,
        "source_import_manifest_sha256": SOURCE_IMPORT_SHA256,
        "source_registry_sha256": SOURCE_REGISTRY_SHA256,
        "source_signals_file_sha256": SOURCE_SIGNALS_FILE_SHA256,
        "source_signals_array_sha256": SOURCE_SIGNALS_ARRAY_SHA256,
        "source_signals_path": str(signals_path),
        "source_registry_path": str(registry_path),
        "unit_count": len(unit_rows),
        "candidate_summary_count": len(summaries),
    }
    write_json(output / "04_SOURCE_SELECTION_BINDING.json", binding)
    selection = {
        "phase": "SOURCE_ONLY_HYPERPARAMETERS_FROZEN_FOR_ALL_P4_BUDGETS",
        "experiment_id": EXPERIMENT_ID,
        "selection_population": "45 grouped pseudo-target units per candidate and arm: 3 held positions x 3 layouts x 5 source seeds",
        "selected": {arm: {key: row[key] for key in ("learning_rate", "encoder_weight_decay", "epochs", "minimum_held_position_mean_query_macro_f1", "overall_mean_query_macro_f1", "minimum_class_mean_recall", "mean_support_minus_query_macro_f1")} for arm, row in selected.items()},
        "same_selected_configuration_every_p4_budget": True,
        "p4_query_used": False,
        "p4_support_used": False,
        "unit_results_sha256": sha256_file(output / "05_SOURCE_SELECTION_UNIT_RESULTS.csv"),
        "summary_sha256": sha256_file(output / "06_SOURCE_SELECTION_SUMMARY.csv"),
    }
    write_json(selected_path, selection)
    lines = [
        "# Source-only fine-tuning hyperparameter selection",
        "",
        "Selection used only the audited P1--P3 position-held pseudo-target protocol. No P4 file, support, prediction, or metric was opened.",
        "",
        "| Arm | Learning rate | Encoder weight decay | Epochs | Worst-position Macro-F1 | Overall Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = selected[arm]
        lines.append(f"| {arm} | {row['learning_rate']:.6g} | {row['encoder_weight_decay']:.6g} | {int(row['epochs'])} | {row['minimum_held_position_mean_query_macro_f1']:.6f} | {row['overall_mean_query_macro_f1']:.6f} |")
    write_text(output / "08_SOURCE_SELECTION_REPORT.md", "\n".join(lines) + "\n")
    return {"status": "SOURCE_SELECTION_COMPLETE", "unit_count": len(unit_rows), "summary_count": len(summaries), "selected": selection["selected"]}

