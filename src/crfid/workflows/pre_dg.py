"""End-to-end Stage1-5 Pre-DG custody, retraining and comparison workflow."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..baseline.aggregation import aggregate_seed_results, compare_with_authoritative
from ..baseline.artifacts import file_manifest, write_csv, write_json
from ..baseline.evaluation import evaluate_predictions, save_predictions
from ..baseline.model import build_pre_dg_model, verify_model_contract
from ..baseline.preprocessing import TrainOnlyChannelScaler
from ..baseline.reporting import render_comparison, render_results
from ..baseline.training import (
    predict_reloaded_model,
    reload_checkpoint_model,
    select_candidate,
    train_candidate,
)
from ..data.pre_dg_loader import LoadedPreDGDataset, load_authoritative_processed_dataset
from ..data.pre_dg_registry import parse_dataset_registry, parse_split_registry
from ..data.pre_dg_schema import PreDGDatasetSchema, PreDGSplitSchema
from ..data.pre_dg_splits import ReconstructedSplit, reconstruct_authoritative_split
from ..governance.pre_dg_access import require_explicit_location
from ..governance.pre_dg_integrity import canonical_json_sha256, sha256_file
from ..protocols.pre_dg import build_protocol, protocol_sha256
from .common import WorkflowPlan, build_plan


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path.name}")
    return payload


def _relative(path: Path, repository: Path) -> str:
    return path.resolve().relative_to(repository.resolve()).as_posix()


def _environment_identity() -> dict[str, Any]:
    import pandas
    import torch

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable_name": Path(sys.executable).name,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "torch": torch.__version__,
    }


def validate_pre_dg_environment(required_verdict: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "crfid-env-check.cmd"],
        check=False,
        capture_output=True,
        text=True,
    )
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0 or output != required_verdict:
        raise RuntimeError(f"Locked Pre-DG environment gate failed: {output}")
    return {"verdict": output, "identity": _environment_identity()}


def _resolved_locations(config: dict[str, Any]) -> dict[str, Path]:
    locations = config["data_locations"]
    return {
        key: require_explicit_location(str(value), key)
        for key, value in locations.items()
    }


def _split_root_key(dataset_id: str) -> str:
    return "paper3_split_root" if dataset_id == "paper3_tyndall" else "paper4_split_root"


def _execution_shard(
    splits: dict[str, ReconstructedSplit], specification: str
) -> dict[str, ReconstructedSplit]:
    """Select whole splits for an optional compute-only worker shard."""

    try:
        index_text, count_text = specification.split("/", maxsplit=1)
        index, count = int(index_text), int(count_text)
    except (ValueError, TypeError) as exc:
        raise ValueError("Pre-DG execution shard must use zero-based INDEX/COUNT") from exc
    if count < 1 or index < 0 or index >= count:
        raise ValueError("Pre-DG execution shard is outside its declared range")
    return {
        name: split
        for position, (name, split) in enumerate(sorted(splits.items()))
        if position % count == index
    }


def _write_custody_outputs(
    output_root: Path,
    datasets: dict[str, LoadedPreDGDataset],
    splits: dict[str, ReconstructedSplit],
) -> None:
    custody = output_root / "custody"
    manifest_rows = []
    identity: dict[str, Any] = {}
    schema_report: dict[str, Any] = {}
    class_counts: dict[str, Any] = {}
    condition_counts: dict[str, Any] = {}
    for dataset_id, loaded in datasets.items():
        schema = loaded.schema
        artifacts = (
            ("processed_inputs", loaded.processed_root / "X.npy", schema.x_sha256),
            ("processed_metadata", loaded.processed_root / "metadata.csv", schema.metadata_sha256),
            ("canonical_manifest", loaded.manifest_path, schema.manifest_sha256),
        )
        for role, path, digest in artifacts:
            manifest_rows.append(
                {
                    "dataset_id": dataset_id,
                    "artifact_role": role,
                    "logical_name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": digest,
                    "verified": True,
                }
            )
        labels = loaded.metadata[schema.label_column].astype(int)
        class_counts[dataset_id] = {
            str(value): int(count)
            for value, count in labels.value_counts().sort_index().items()
        }
        condition_counts[dataset_id] = {
            column: {
                str(value): int(count)
                for value, count in loaded.metadata[column].value_counts().sort_index().items()
            }
            for column in schema.condition_columns
        }
        identity[dataset_id] = {
            "row_count": len(loaded.metadata),
            "sample_id_unique_count": int(loaded.metadata["sample_id"].nunique()),
            "signal_hash_unique_count": int(loaded.manifest["signal_hash"].nunique()),
            "raw_condition_group_count": int(
                loaded.metadata[schema.group_column].astype(str).nunique()
            ),
            "class_order": list(schema.class_order),
            "processed_shape": list(loaded.inputs.shape),
            "processed_dtype": str(loaded.inputs.dtype),
            "processed_x_sha256": schema.x_sha256,
            "processed_metadata_sha256": schema.metadata_sha256,
            "manifest_sha256": schema.manifest_sha256,
        }
        schema_report[dataset_id] = {
            "required_fields": [
                "sample_id",
                schema.label_column,
                schema.group_column,
                *schema.condition_columns,
            ],
            "input_channels": schema.input_channels,
            "input_length": schema.input_length,
            "finite": bool(np.isfinite(loaded.inputs).all()),
            "passed": True,
        }
    split_rows = []
    split_integrity = {}
    for split_name, split in sorted(splits.items()):
        split_rows.append(
            {
                "dataset_id": split.schema.dataset_id,
                "split_name": split_name,
                "split_sha256": split.schema.file_sha256,
                "split_metadata_sha256": split.schema.metadata_sha256,
                "train_count": split.schema.train_count,
                "validation_count": split.schema.validation_count,
                "test_count": split.schema.test_count,
            }
        )
        split_integrity[split_name] = split.integrity
        for role, suffix, digest in (
            ("split", ".csv", split.schema.file_sha256),
            ("split_metadata", ".json", split.schema.metadata_sha256),
        ):
            manifest_rows.append(
                {
                    "dataset_id": split.schema.dataset_id,
                    "artifact_role": role,
                    "logical_name": split_name + suffix,
                    "size_bytes": "",
                    "sha256": digest,
                    "verified": True,
                }
            )
    write_csv(custody / "raw_or_processed_manifest.csv", manifest_rows)
    write_json(custody / "dataset_identity.json", identity)
    write_json(custody / "schema_report.json", schema_report)
    write_csv(custody / "split_manifest.csv", split_rows)
    write_json(custody / "split_integrity.json", split_integrity)
    write_json(custody / "class_counts.json", class_counts)
    write_json(custody / "condition_counts.json", condition_counts)
    (custody / "PRE_DG_CUSTODY_REPORT.md").write_text(
        "# Pre-DG custody report\n\n"
        "Two final Stage1-5 single-dataset processed inputs and all 18 official "
        "raw-condition splits were hash-verified. Dataset, class, sample, group "
        "and split identities passed. Train/validation/test raw-condition group "
        "overlap is zero for every split.\n",
        encoding="utf-8",
    )


def validate_pre_dg_inputs(
    dataset_schemas: dict[str, PreDGDatasetSchema],
    split_schemas: tuple[PreDGSplitSchema, ...],
    locations: dict[str, Path],
    output_root: Path,
) -> tuple[dict[str, LoadedPreDGDataset], dict[str, ReconstructedSplit]]:
    datasets: dict[str, LoadedPreDGDataset] = {}
    for dataset_id, schema in dataset_schemas.items():
        prefix = "paper3" if dataset_id == "paper3_tyndall" else "paper4"
        datasets[dataset_id] = load_authoritative_processed_dataset(
            schema,
            locations[f"{prefix}_processed_root"],
            locations[f"{prefix}_manifest_path"],
        )
    splits: dict[str, ReconstructedSplit] = {}
    for schema in split_schemas:
        root = locations[_split_root_key(schema.dataset_id)]
        splits[schema.split_name] = reconstruct_authoritative_split(
            root / schema.file_name,
            root / Path(schema.file_name).with_suffix(".json"),
            schema,
            datasets[schema.dataset_id].metadata,
            datasets[schema.dataset_id].schema.group_column,
        )
    _write_custody_outputs(output_root, datasets, splits)
    return datasets, splits


def reconstruct_pre_dg_split(
    split_schema: PreDGSplitSchema,
    location: Path,
    dataset: LoadedPreDGDataset,
) -> ReconstructedSplit:
    return reconstruct_authoritative_split(
        location / split_schema.file_name,
        location / Path(split_schema.file_name).with_suffix(".json"),
        split_schema,
        dataset.metadata,
        dataset.schema.group_column,
    )


def _freeze_protocol(
    repository: Path,
    output_root: Path,
    config_directory: Path,
    protocol: dict[str, Any],
    environment: dict[str, Any],
) -> str:
    freeze = output_root / "protocol_freeze"
    write_json(freeze / "protocol.json", protocol)
    digest = protocol_sha256(protocol)
    (freeze / "protocol.sha256").write_text(digest + "\n", encoding="ascii")
    code_paths = [
        "src/crfid/data/pre_dg_registry.py",
        "src/crfid/data/pre_dg_loader.py",
        "src/crfid/data/pre_dg_schema.py",
        "src/crfid/data/pre_dg_splits.py",
        "src/crfid/protocols/pre_dg.py",
        "src/crfid/governance/pre_dg_access.py",
        "src/crfid/governance/pre_dg_integrity.py",
        "src/crfid/governance/pre_dg_claims.py",
        "src/crfid/baseline/pre_dg.py",
        "src/crfid/baseline/model.py",
        "src/crfid/baseline/preprocessing.py",
        "src/crfid/baseline/training.py",
        "src/crfid/baseline/evaluation.py",
        "src/crfid/baseline/aggregation.py",
        "src/crfid/baseline/artifacts.py",
        "src/crfid/baseline/reporting.py",
        "src/crfid/workflows/pre_dg.py",
        "workflows/01_pre_dg_baseline/run.py",
    ]
    code_rows = [
        {"relative_path": path, "sha256": sha256_file(repository / path)}
        for path in code_paths
    ]
    config_rows = [
        {
            "relative_path": _relative(path, repository),
            "sha256": sha256_file(path),
        }
        for path in sorted(config_directory.glob("*.yaml"))
    ]
    write_csv(freeze / "code_manifest.csv", code_rows)
    write_csv(freeze / "config_manifest.csv", config_rows)
    source_split = output_root / "custody" / "split_manifest.csv"
    (freeze / "split_manifest.csv").write_bytes(source_split.read_bytes())
    write_json(freeze / "environment.json", environment)
    (freeze / "PRE_DG_PROTOCOL_FREEZE.md").write_text(
        "# Pre-DG protocol freeze\n\n"
        f"Protocol SHA-256: `{digest}`.\n\n"
        "Data, split, preprocessing, model, seed, optimization, selection and "
        "comparison policies were frozen before retraining.\n",
        encoding="utf-8",
    )
    return digest


def _prepare_comparison(output_root: Path, evaluation: dict[str, Any]) -> None:
    comparison = output_root / "comparison"
    write_json(
        comparison / "authoritative_reference.json",
        {
            "historical_verdict": "STAGE1_5_FREEZE_READY",
            "run_count": len(evaluation["authoritative_runs"]),
            "runs": evaluation["authoritative_runs"],
        },
    )
    write_json(comparison / "comparison_policy.json", evaluation["comparison_policy"])


def _labels_for_indices(
    loaded: LoadedPreDGDataset, indices: np.ndarray
) -> np.ndarray:
    order = {value: index for index, value in enumerate(loaded.schema.class_order)}
    values = loaded.metadata.iloc[indices][loaded.schema.label_column].astype(int)
    encoded = values.map(order).to_numpy(dtype=np.int64)
    if np.any(encoded < 0):
        raise ValueError("Pre-DG label mapping failed")
    return encoded


def _execute_all_runs(
    repository: Path,
    output_root: Path,
    datasets: dict[str, LoadedPreDGDataset],
    splits: dict[str, ReconstructedSplit],
    canonical: dict[str, Any],
    model_config: dict[str, Any],
    resume: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    # Match the authoritative Stage 5 runner's CPU thread cap.
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    execution = output_root / "execution"
    selected_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    training = model_config["training"]
    for split_name, reconstructed in sorted(splits.items()):
        dataset_id = reconstructed.schema.dataset_id
        loaded = datasets[dataset_id]
        indices = reconstructed.indices
        scaler = TrainOnlyChannelScaler.fit(
            loaded.inputs[indices["train"]], partition="train"
        )
        state_path = (
            execution
            / "preprocessing_states"
            / dataset_id
            / f"{split_name}.npz"
        )
        preprocessing_sha256 = scaler.save(state_path)
        reloaded_scaler = TrainOnlyChannelScaler.load(state_path)
        transformed = {
            partition: reloaded_scaler.transform(loaded.inputs[indices[partition]])
            for partition in ("train", "validation", "test")
        }
        if not np.array_equal(
            scaler.transform(loaded.inputs[indices["test"][: min(16, len(indices["test"]))]]),
            reloaded_scaler.transform(
                loaded.inputs[indices["test"][: min(16, len(indices["test"]))]]
            ),
        ):
            raise ValueError("Reloaded preprocessing transform changed")
        labels = {
            partition: _labels_for_indices(loaded, indices[partition])
            for partition in ("train", "validation", "test")
        }
        for seed in canonical["random_seeds"]:
            candidates = []
            for dropout in model_config["model"]["dropout_candidates"]:
                tag = str(dropout).replace(".", "p")
                base = f"{dataset_id}__{split_name}__seed{seed}__drop{tag}"
                checkpoint_path = (
                    Path(canonical["output_path"])
                    / "execution"
                    / "checkpoints"
                    / dataset_id
                    / split_name
                    / f"seed{seed}_drop{tag}.pt"
                )
                history_path = (
                    Path(canonical["output_path"])
                    / "execution"
                    / "training_histories"
                    / dataset_id
                    / split_name
                    / f"seed{seed}_drop{tag}.json"
                )
                record_path = (
                    Path(canonical["output_path"])
                    / "execution"
                    / "candidate_records"
                    / dataset_id
                    / split_name
                    / f"seed{seed}_drop{tag}.json"
                )
                candidate = train_candidate(
                    dataset_id=dataset_id,
                    split_name=split_name,
                    seed=int(seed),
                    dropout=float(dropout),
                    class_count=loaded.schema.class_count,
                    train_inputs=transformed["train"],
                    train_labels=labels["train"],
                    validation_inputs=transformed["validation"],
                    validation_labels=labels["validation"],
                    training_config=training,
                    checkpoint_path=checkpoint_path,
                    history_path=history_path,
                    record_path=record_path,
                    resume=resume,
                )
                candidates.append(candidate)
                candidate_rows.append(
                    {
                        "run_id": base,
                        "dataset_id": dataset_id,
                        "split_name": split_name,
                        "seed": seed,
                        "dropout": dropout,
                        "validation_accuracy": candidate.validation_accuracy,
                        "validation_loss": candidate.validation_loss,
                        "best_epoch": candidate.best_epoch,
                        "epochs_ran": candidate.epochs_ran,
                        "checkpoint_path": candidate.checkpoint_path.replace("\\", "/"),
                        "checkpoint_sha256": candidate.checkpoint_sha256,
                        "model_state_sha256": candidate.model_state_sha256,
                    }
                )
                print(
                    json.dumps(
                        {
                            "completed_candidate": base,
                            "validation_accuracy": candidate.validation_accuracy,
                            "best_epoch": candidate.best_epoch,
                        }
                    ),
                    flush=True,
                )
            selected = select_candidate(candidates)
            model = reload_checkpoint_model(selected, loaded.schema.class_count)
            verify_model_contract(model, loaded.schema.class_count)
            predicted = predict_reloaded_model(
                model,
                transformed["test"],
                labels["test"],
                int(training["batch_size"]),
            )
            metrics = evaluate_predictions(
                predicted["truth"],
                predicted["prediction"],
                class_count=loaded.schema.class_count,
            )
            prediction_path = (
                execution
                / "predictions"
                / dataset_id
                / split_name
                / f"seed{seed}.npz"
            )
            prediction_sha256 = save_predictions(
                prediction_path,
                reconstructed.sample_ids["test"],
                predicted["truth"],
                predicted["prediction"],
                predicted["logits"],
            )
            confusion_path = (
                execution
                / "confusion_matrices"
                / dataset_id
                / split_name
                / f"seed{seed}.json"
            )
            write_json(confusion_path, metrics["confusion_matrix"])
            run_id = f"{dataset_id}__{split_name}__seed{seed}"
            class_counts = Counter(int(value) for value in labels["test"])
            row = {
                "run_id": run_id,
                "dataset_id": dataset_id,
                "split_name": split_name,
                "seed": int(seed),
                "selected_dropout": selected.dropout,
                "best_epoch": selected.best_epoch,
                "epochs_ran": selected.epochs_ran,
                "train_count": len(indices["train"]),
                "validation_count": len(indices["validation"]),
                "test_count": len(indices["test"]),
                "test_class_counts": json.dumps(dict(sorted(class_counts.items()))),
                "checkpoint_path": selected.checkpoint_path.replace("\\", "/"),
                "checkpoint_sha256": selected.checkpoint_sha256,
                "model_state_sha256": selected.model_state_sha256,
                "preprocessing_path": _relative(state_path, repository),
                "preprocessing_sha256": preprocessing_sha256,
                "prediction_path": _relative(prediction_path, repository),
                "prediction_sha256": prediction_sha256,
                "training_history_path": selected.history_path.replace("\\", "/"),
                "confusion_matrix_path": _relative(confusion_path, repository),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "per_class_recall": metrics["per_class_recall"],
                "confusion_matrix": metrics["confusion_matrix"],
            }
            selected_rows.append(row)
            for class_index, recall in enumerate(metrics["per_class_recall"]):
                recall_rows.append(
                    {
                        "run_id": run_id,
                        "dataset_id": dataset_id,
                        "split_name": split_name,
                        "seed": seed,
                        "class_index": class_index,
                        "class_label": loaded.schema.class_order[class_index],
                        "recall": recall,
                    }
                )
            print(
                json.dumps(
                    {
                        "completed_selected_run": run_id,
                        "selected_dropout": selected.dropout,
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                    }
                ),
                flush=True,
            )
    flat_rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"per_class_recall", "confusion_matrix"}
        }
        for row in selected_rows
    ]
    write_csv(execution / "execution_manifest.csv", flat_rows)
    write_csv(execution / "per_run_metrics.csv", flat_rows)
    write_csv(execution / "candidate_execution_manifest.csv", candidate_rows)
    write_csv(execution / "class_recall.csv", recall_rows)
    return selected_rows, candidate_rows


def _finalize(
    repository: Path,
    output_root: Path,
    protocol_digest: str,
    selected_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    execution = output_root / "execution"
    comparison_root = output_root / "comparison"
    final = output_root / "final"
    aggregate = aggregate_seed_results(selected_rows)
    write_json(execution / "aggregate_metrics.json", aggregate)
    comparisons, comparison = compare_with_authoritative(
        selected_rows,
        aggregate,
        evaluation["authoritative_runs"],
        evaluation["comparison_policy"],
    )
    write_csv(comparison_root / "reproduced_vs_authoritative.csv", comparisons)
    write_json(comparison_root / "comparison_result.json", comparison)
    render_comparison(comparison_root / "REPRODUCTION_COMPARISON.md", comparison)
    render_results(execution / "PRE_DG_RESULTS.md", aggregate, comparison)
    verdict = (
        "PASS_PRE_DG_REPRODUCED_WITH_NUMERICAL_TOLERANCE"
        if comparison["numerical_tolerance_passed"]
        else "FAIL_PRE_DG_RESULT_REPRODUCTION_MISMATCH"
    )
    integrity = {
        "protocol_sha256": protocol_digest,
        "dataset_count": 2,
        "split_count": 18,
        "authoritative_seed_count": 3,
        "selected_run_count": len(selected_rows),
        "candidate_training_count": len(candidate_rows),
        "all_checkpoints_saved_reloaded_and_state_verified": True,
        "all_predictions_serialized_and_hashed": True,
        "all_preprocessing_states_train_only_and_reloaded": True,
        "numerical_tolerance_passed": comparison["numerical_tolerance_passed"],
        "passed": comparison["numerical_tolerance_passed"],
    }
    write_json(execution / "PRE_DG_INTEGRITY.json", integrity)
    final_metrics = {
        "grouped_random_headline": aggregate["grouped_random_headline"],
        "split_results": aggregate["split_results"],
    }
    write_json(final / "final_metrics.json", final_metrics)
    evidence_levels = ["MODEL_RETRAINING_REPRODUCTION"]
    if comparison["numerical_tolerance_passed"]:
        evidence_levels.append("TOLERANCE_MATCH")
    final_verdict = {
        "verdict": verdict,
        "historical_verdict": "STAGE1_5_FREEZE_READY",
        "evidence_levels": evidence_levels,
        "exact_data_split_protocol_identity": True,
        "exact_historical_checkpoint_or_prediction_match": False,
        "historical_checkpoint_prediction_limitation": "not retained",
        "ready_for_independent_read_only_audit": comparison[
            "numerical_tolerance_passed"
        ],
    }
    write_json(final / "final_verdict.json", final_verdict)
    (final / "PRE_DG_BASELINE_REPORT.md").write_text(
        "# Pre-DG Baseline report\n\n"
        f"Verdict: `{verdict}`.\n\n"
        "The final Stage1-5 Paper3 and Paper4 DeepCNN baselines were retrained "
        "for all 18 grouped-random and leave-domain splits and seeds 42/43/44. "
        "Every selected checkpoint was reloaded before prediction. This is "
        "non-strict single-dataset evidence, not fair zero-shot DG.\n",
        encoding="utf-8",
    )
    omitted = []
    for directory, reason in (
        (execution / "checkpoints", "large model checkpoint"),
        (execution / "predictions", "large prediction/logit artifact"),
    ):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                omitted.append(
                    {
                        "relative_path": _relative(path, repository),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "omission_reason": reason,
                    }
                )
    write_csv(final / "OMITTED_LARGE_ARTIFACTS.csv", omitted)
    public_rows = file_manifest(
        output_root,
        omit_directories=("checkpoints", "predictions"),
    )
    public_rows = [
        row
        for row in public_rows
        if "/checkpoints/" not in row["relative_path"]
        and "/predictions/" not in row["relative_path"]
        and not row["relative_path"].endswith("PUBLIC_ARTIFACT_MANIFEST.csv")
    ]
    write_csv(final / "PUBLIC_ARTIFACT_MANIFEST.csv", public_rows)
    return final_verdict


def package_pre_dg_review_material(output_root: Path) -> dict[str, Any]:
    final = output_root / "final"
    required = [
        final / "final_verdict.json",
        final / "final_metrics.json",
        final / "PRE_DG_BASELINE_REPORT.md",
        final / "PUBLIC_ARTIFACT_MANIFEST.csv",
        final / "OMITTED_LARGE_ARTIFACTS.csv",
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Pre-DG review material is incomplete: {missing}")
    return {"required_files": len(required), "status": "PASS"}


def run(
    config: dict[str, Any],
    *,
    execute: bool = False,
    repository_root: str | Path | None = None,
    resume: bool = True,
) -> WorkflowPlan | dict[str, Any]:
    plan = build_plan(config, "PRE_DG_STAGE1_5_SINGLE_DATASET_BASELINES", execute)
    if not execute:
        return plan
    repository = Path(repository_root or Path.cwd()).resolve()
    if Path.cwd().resolve() != repository:
        raise ValueError("Run the Pre-DG workflow from the canonical repository root")
    output_root = (repository / str(config["output_path"])).resolve()
    output_root.relative_to(repository)
    config_directory = (repository / str(config["config_directory"])).resolve()
    data_authority = _load_mapping(config_directory / "data_authority.yaml")
    split_config = _load_mapping(config_directory / "split.yaml")
    model_config = _load_mapping(config_directory / "model.yaml")
    evaluation = _load_mapping(config_directory / "evaluation.yaml")
    dataset_schemas = parse_dataset_registry(data_authority)
    split_schemas = parse_split_registry(split_config)
    environment = validate_pre_dg_environment(config["required_environment_verdict"])
    locations = _resolved_locations(config)
    datasets, splits = validate_pre_dg_inputs(
        dataset_schemas, split_schemas, locations, output_root
    )
    protocol = build_protocol(
        config, data_authority, split_config, model_config, evaluation
    )
    protocol_digest = _freeze_protocol(
        repository, output_root, config_directory, protocol, environment
    )
    _prepare_comparison(output_root, evaluation)
    shard_specification = os.environ.get("CRFID_PRE_DG_EXECUTION_SHARD")
    if shard_specification:
        shard_splits = _execution_shard(splits, shard_specification)
        selected_rows, candidate_rows = _execute_all_runs(
            repository,
            output_root,
            datasets,
            shard_splits,
            config,
            model_config,
            resume,
        )
        return {
            "status": "PRE_DG_EXECUTION_SHARD_COMPLETE",
            "execution_shard": shard_specification,
            "selected_run_count": len(selected_rows),
            "candidate_training_count": len(candidate_rows),
        }
    selected_rows, candidate_rows = _execute_all_runs(
        repository,
        output_root,
        datasets,
        splits,
        config,
        model_config,
        resume,
    )
    final_verdict = _finalize(
        repository,
        output_root,
        protocol_digest,
        selected_rows,
        candidate_rows,
        evaluation,
    )
    package = package_pre_dg_review_material(output_root)
    return {
        "status": final_verdict["verdict"],
        "protocol_sha256": protocol_digest,
        "selected_run_count": len(selected_rows),
        "candidate_training_count": len(candidate_rows),
        "review_material": package,
    }
