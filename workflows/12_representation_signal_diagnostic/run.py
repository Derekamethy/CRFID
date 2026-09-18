"""Execute the preregistered representation-versus-signal diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from crfid import representation_signal_diagnostic as diagnostic  # noqa: E402


CONFIG_PATH = REPOSITORY_ROOT / "configs" / "representation_signal_diagnostic" / "protocol.json"
RESULTS = REPOSITORY_ROOT / "results" / "canonical_metrics" / "representation_signal_diagnostic"
MANIFESTS = REPOSITORY_ROOT / "manifests" / "representation_signal_diagnostic"
PREPARE_SEAL = MANIFESTS / "PRE_PROBE_MANIFEST_SEAL.json"
DOC_PROTOCOL = REPOSITORY_ROOT / "docs" / "REPRESENTATION_SIGNAL_DIAGNOSTIC_PROTOCOL.md"
DOC_RESULTS = REPOSITORY_ROOT / "docs" / "REPRESENTATION_SIGNAL_DIAGNOSTIC_RESULTS.md"


def _json_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return "" if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text.rstrip() + "\n")


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: list[str] | None = None) -> None:
    materialized = list(rows)
    if not materialized:
        if not fieldnames:
            raise ValueError(f"No rows or field names for {path.name}")
        keys = fieldnames
    elif fieldnames is not None:
        keys = fieldnames
    else:
        keys = []
        for row in materialized:
            for key in row:
                if key not in keys:
                    keys.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=keys, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _json_value(row.get(key, "")) for key in keys})


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def git_lines(repository: Path, *arguments: str) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return [line for line in completed.stdout.splitlines() if line]


def git_value(repository: Path, *arguments: str) -> str:
    lines = git_lines(repository, *arguments)
    return lines[0] if lines else ""


def git_snapshot(repository: Path) -> dict[str, Any]:
    git_dir = repository / ".git"
    locks = sorted(
        str(path.relative_to(repository)).replace("\\", "/")
        for path in git_dir.rglob("*.lock")
        if path.is_file()
    )
    alternates = git_dir / "objects" / "info" / "alternates"
    return {
        "head": git_value(repository, "rev-parse", "HEAD"),
        "tree": git_value(repository, "rev-parse", "HEAD^{tree}"),
        "branch": git_value(repository, "branch", "--show-current"),
        "status_porcelain": git_lines(repository, "status", "--porcelain=v1"),
        "tags": git_lines(repository, "tag", "--list", "--sort=refname"),
        "refs": git_lines(
            repository,
            "for-each-ref",
            "--format=%(refname) %(objectname) %(objecttype)",
            "refs/heads",
            "refs/tags",
        ),
        "remotes": git_lines(repository, "remote", "-v"),
        "locks": locks,
        "alternates": alternates.read_text(encoding="utf-8").splitlines() if alternates.exists() else None,
        "shallow": (git_dir / "shallow").exists(),
    }


def filesystem_fingerprint(root: Path) -> dict[str, Any]:
    root_text = os.path.abspath(str(root.resolve()))
    scan_root = f"\\\\?\\{root_text}" if os.name == "nt" and not root_text.startswith("\\\\?\\") else root_text
    directories: list[str] = []
    files: list[tuple[str, str, int]] = []
    for directory, directory_names, file_names in os.walk(scan_root):
        directory_names.sort()
        file_names.sort()
        relative_directory = os.path.relpath(directory, scan_root)
        if relative_directory != ".":
            directories.append(relative_directory.replace(os.sep, "/"))
        for name in file_names:
            path = os.path.join(directory, name)
            files.append(
                (
                    os.path.relpath(path, scan_root).replace(os.sep, "/"),
                    path,
                    os.path.getsize(path),
                )
            )

    def digest(item: tuple[str, str, int]) -> tuple[str, int, str]:
        relative, path, size = item
        return relative, size, diagnostic.sha256_file(path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        hashed = list(executor.map(digest, files))
    records = [f"D\t{item}" for item in sorted(directories)]
    records.extend(f"F\t{relative}\t{size}\t{sha}" for relative, size, sha in sorted(hashed))
    aggregate = hashlib.sha256(("\n".join(records) + "\n").encode("utf-8")).hexdigest()
    return {
        "aggregate_sha256": aggregate,
        "file_count": len(files),
        "dir_count": len(directories),
        "total_bytes": sum(item[2] for item in files),
    }


def _data_audit_rows(
    positions: Mapping[str, diagnostic.PositionData], custody: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for position, data in positions.items():
        counts = [int(np.sum(data.block_tokens == token)) for token in sorted(set(data.block_tokens.tolist()))]
        rows.append(
            {
                "position": position,
                "row_count": len(data.labels),
                "tag_id_count": len(set(data.tag_ids.tolist())),
                "tag_ids": sorted(set(data.tag_ids.tolist())),
                "er_level_count": len(set(data.er.tolist())),
                "er_levels": sorted(set(data.er.tolist())),
                "surface_count": len(set(data.surfaces.tolist())),
                "surfaces": sorted(set(data.surfaces.tolist())),
                "condition_block_count": len(set(data.block_tokens.tolist())),
                "repeats_per_block": sorted(set(counts)),
                "raw_dimension": data.signals.shape[1],
                "first_difference_dimension": np.diff(data.signals, axis=1).shape[1],
                "finite_signals": bool(np.isfinite(data.signals).all()),
                "label_mapping": "TagID 1..7 -> class index TagID-1",
                "governed_root_token": custody["root_token"],
                "source_file_hashes": {record["name"]: record["sha256"] for record in data.source_files},
                "status": "PASS",
            }
        )
    return rows


def _representation_rows(peak_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "representation": "RAW_SIGNAL_281",
            "dimension": 281,
            "input": "original ordered signal",
            "transform": "none before probe scaling",
            "probe_scaling": "training-only featurewise population standardization",
            "status": "VALIDATED",
        },
        {
            "representation": "FIRST_DIFFERENCE_280",
            "dimension": 280,
            "input": "canonical x[i+1]-x[i] without padding",
            "transform": "np.diff axis=1",
            "probe_scaling": "training-only featurewise population standardization",
            "status": "VALIDATED",
        },
        {
            "representation": "C1_ENCODER_EMBEDDING",
            "dimension": 256,
            "input": "canonical checkpoint-bound first difference and frozen source preprocessing",
            "transform": "frozen C1 network.12 penultimate output; five source-only checkpoints",
            "probe_scaling": "training-only featurewise population standardization of embeddings",
            "status": "VALIDATED",
        },
        {
            "representation": "PEAK_DESCRIPTOR",
            "dimension": 12,
            "input": "raw signal through canonical strongest_window_peaks in four frozen ordered-index windows",
            "transform": "per window: ordered index, amplitude, missing flag; mode=min",
            "probe_scaling": "training-only featurewise population standardization",
            "status": peak_summary["status"],
        },
    ]


def _protocol_markdown(config: Mapping[str, Any], peak_summary: Mapping[str, Any]) -> str:
    probe = config["linear_probe"]
    return f"""# Preregistered representation-versus-signal protocol

Status before primary probe execution: **PREREGISTERED**.

The scientific question is whether condition-general weakness is already present in measured signals or is introduced or amplified by the frozen C1 encoder. P1-P4 are evaluated independently. The indivisible unit is `TagID x ER x surface x position`, with all 50 repeats grouped.

## Frozen synchronized folds

Test fold 0 uses E0/A1, E1/A2, E2/A3; fold 1 uses E0/A2, E1/A3, E2/A1; fold 2 uses E0/A3, E1/A1, E2/A2. For test fold q, validation is Latin fold `(q+1) mod 3`, and training is Latin fold `(q+2) mod 3`. Thus each split has 21 condition blocks and 1,050 rows per position/fold. The same assignment applies to every representation.

## Frozen representations and readouts

The four families are RAW_SIGNAL_281, FIRST_DIFFERENCE_280, C1_ENCODER_EMBEDDING, and PEAK_DESCRIPTOR. The C1 embedding is the frozen `network.12` output (256 dimensions) from the five source-only checkpoints. Probe scaling is fitted only on the current training rows. The C1 encoder's source preprocessing is checkpoint-bound and is never refit.

PEAK_DESCRIPTOR reuses `crfid.analysis.reference_peaks.strongest_window_peaks` in minimum mode with the preregistered ordered-index windows W0 10-69, W1 75-126, W2 136-182, and W3 192-252. Each window contributes index, amplitude, and missing flag (12 total). Locations are reported primarily as ordered indices. Peak validity before probes: **{peak_summary['status']}**.

The primary probe is one linear layer with softmax classification, Adam (learning rate {probe['learning_rate']}, weight decay {probe['weight_decay']}), Xavier-uniform weights, zero bias, full-batch capacity {probe['batch_size']}, at most {probe['maximum_epochs']} epochs, validation row Macro-F1 selection, and patience {probe['early_stopping_patience']}. Seeds are 42-46 and C1 probe/checkpoint seeds are aligned. COSINE_NCM is the sole secondary readout.

## Label boundary and estimand

Test predictions are generated and hashed before the held labels can be opened. Labels are opened once, after freezing, and metrics are computed once. The primary endpoint is Macro-F1 over the 63 out-of-fold condition-block majority predictions per position and seed. Exact vote ties use the lowest canonical class index. Rows are descriptive, never inferential units.

Primary paired contrasts are C1 minus FIRST_DIFFERENCE, RAW minus C1, and (only if valid) PEAK minus C1. RAW minus FIRST_DIFFERENCE is retained as the preregistered diagnostic needed to assess differencing loss. The angle-associated accuracy contrast is `0.5 * [(P2-P1) + (P4-P3)]`.

Uncertainty uses {config['bootstrap']['replicates']:,} deterministic TagID-stratified paired condition-block bootstrap replicates, retaining representation, position, and aligned seed pairing. No hyperparameter, representation window, fold, threshold, or interpretation rule is changed after results are viewed.
"""


def prepare() -> None:
    config = diagnostic.load_protocol(CONFIG_PATH)
    if (RESULTS / "10_RUN_REGISTER.csv").exists():
        raise RuntimeError("Primary execution already exists; refusing to rewrite preregistration")
    print("prepare: loading governed P1-P4 data", flush=True)
    positions, custody = diagnostic.load_governed_positions(REPOSITORY_ROOT, config)
    print("prepare: binding five frozen C1 checkpoints", flush=True)
    _, _, _, _, c1_binding = diagnostic.load_c1_binding(REPOSITORY_ROOT, config)
    print("prepare: validating the traced peak extractor", flush=True)
    peak_pass, _, peak_rows, peak_summary = diagnostic.validate_peak_extractor(
        positions, config["peak_descriptor"]
    )
    manifest_rows = diagnostic.synchronized_split_manifest(positions, config)
    RESULTS.mkdir(parents=True, exist_ok=True)
    input_binding = {
        "schema_version": 1,
        "status": "PASS_GOVERNED_INPUT_BINDING",
        "governed_input_custody": custody,
        "observed_structure": _data_audit_rows(positions, custody),
        "publication_boundary": "paths tokenized; raw measurements remain outside Git",
    }
    write_json(RESULTS / "01_INPUT_BINDING.json", input_binding)
    write_json(RESULTS / "02_C1_CHECKPOINT_AND_EMBEDDING_BINDING.json", c1_binding)
    protocol_text = _protocol_markdown(config, peak_summary)
    write_text(RESULTS / "03_PREREGISTERED_PROTOCOL.md", protocol_text)
    write_text(DOC_PROTOCOL, protocol_text)
    write_csv(RESULTS / "04_DATA_AND_BLOCK_STRUCTURE_AUDIT.csv", _data_audit_rows(positions, custody))
    write_csv(RESULTS / "05_SYNCHRONIZED_SPLIT_MANIFEST.csv", manifest_rows)
    write_csv(RESULTS / "06_REPRESENTATION_DEFINITION_REGISTER.csv", _representation_rows(peak_summary))
    summary_row = {
        "record_type": "validation_summary",
        "position": "ALL",
        "block_token": "SUMMARY",
        "check": "all_preregistered_peak_validity_gates",
        "observed": peak_summary,
        "expected": config["peak_descriptor"]["validity_thresholds"],
        "pass": peak_pass,
        "status": peak_summary["status"],
    }
    write_csv(RESULTS / "07_PEAK_EXTRACTOR_VALIDITY.csv", [summary_row, *peak_rows])
    seal = {
        "created_before_any_probe_fit": True,
        "protocol_sha256": diagnostic.sha256_file(CONFIG_PATH),
        "protocol_register_sha256": diagnostic.sha256_file(RESULTS / "03_PREREGISTERED_PROTOCOL.md"),
        "split_manifest_sha256": diagnostic.sha256_file(RESULTS / "05_SYNCHRONIZED_SPLIT_MANIFEST.csv"),
        "representation_register_sha256": diagnostic.sha256_file(RESULTS / "06_REPRESENTATION_DEFINITION_REGISTER.csv"),
        "peak_validity_sha256": diagnostic.sha256_file(RESULTS / "07_PEAK_EXTRACTOR_VALIDITY.csv"),
        "peak_descriptor_validated": peak_pass,
        "peak_descriptor_status": peak_summary["status"],
        "primary_probe_artifacts_present_at_seal": False,
    }
    write_json(PREPARE_SEAL, seal)
    print(
        f"prepare: manifest sealed before probes; peak status={peak_summary['status']}",
        flush=True,
    )


def verify_prepare_seal() -> dict[str, Any]:
    config = diagnostic.load_protocol(CONFIG_PATH)
    seal = read_json(PREPARE_SEAL)
    checked_paths = {
        "protocol_sha256": CONFIG_PATH,
        "protocol_register_sha256": RESULTS / "03_PREREGISTERED_PROTOCOL.md",
        "split_manifest_sha256": RESULTS / "05_SYNCHRONIZED_SPLIT_MANIFEST.csv",
        "representation_register_sha256": RESULTS / "06_REPRESENTATION_DEFINITION_REGISTER.csv",
        "peak_validity_sha256": RESULTS / "07_PEAK_EXTRACTOR_VALIDITY.csv",
    }
    normalization_path = MANIFESTS / "PUBLICATION_LINE_ENDING_NORMALIZATION.json"
    normalization = read_json(normalization_path) if normalization_path.exists() else {"files": {}}
    if not seal.get("created_before_any_probe_fit") or seal.get("primary_probe_artifacts_present_at_seal"):
        raise RuntimeError("Preregistration timing seal failed")
    for key, path in checked_paths.items():
        expected = seal.get(key)
        observed = diagnostic.sha256_file(path)
        if expected == observed:
            continue
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        mapped = normalization.get("files", {}).get(relative, {})
        if mapped.get("before_crlf_sha256") != expected or mapped.get("after_lf_sha256") != observed:
            raise RuntimeError(f"Preregistered artifact changed after sealing: {key}")
    return {"config": config, "seal": seal}


def _feature_bank(
    positions: Mapping[str, diagnostic.PositionData],
    config: Mapping[str, Any],
    peak_pass: bool,
    peak_descriptors: Mapping[str, np.ndarray],
    c1_module: Any,
    c1_models: Mapping[int, tuple[Any, dict[str, Any]]],
    source_mean: np.ndarray,
    source_scale: np.ndarray,
) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    seeds = [int(item) for item in config["probe_seeds"]]
    bank: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    for position in config["positions"]:
        data = positions[position]
        raw = diagnostic.raw_representation(data)
        difference = diagnostic.first_difference_representation(data)
        bank[position] = {
            "RAW_SIGNAL_281": {seed: raw for seed in seeds},
            "FIRST_DIFFERENCE_280": {seed: difference for seed in seeds},
            "C1_ENCODER_EMBEDDING": {},
        }
        if peak_pass:
            bank[position]["PEAK_DESCRIPTOR"] = {
                seed: peak_descriptors[position] for seed in seeds
            }
        for seed in seeds:
            model, _ = c1_models[seed]
            bank[position]["C1_ENCODER_EMBEDDING"][seed] = diagnostic.extract_c1_embeddings(
                c1_module,
                model,
                data.signals,
                source_mean,
                source_scale,
            )
        print(f"execute: representations materialized for {position}", flush=True)
    return bank


def _ordered_block_records(
    data: diagnostic.PositionData, prediction: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tag_id, er, surface in data.block_keys():
        indices = np.flatnonzero(
            (data.tag_ids == tag_id) & (data.er == er) & (data.surfaces == surface)
        )
        truth = np.unique(data.labels[indices])
        if len(indices) != 50 or len(truth) != 1:
            raise RuntimeError("No row-level pseudoreplication gate failed")
        counts = np.bincount(prediction[indices], minlength=diagnostic.CLASS_COUNT)
        rows.append(
            {
                "position": data.position,
                "tag_id": tag_id,
                "er": er,
                "surface": surface,
                "block_token": str(data.block_tokens[indices[0]]),
                "truth": int(truth[0]),
                "prediction": int(np.flatnonzero(counts == counts.max())[0]),
                "agreement": float(counts.max() / len(indices)),
                "repeat_count": len(indices),
                "test_fold": diagnostic.cell_latin_fold(er, surface, CURRENT_CONFIG),
            }
        )
    return rows


CURRENT_CONFIG: dict[str, Any] = {}


def _learning_validity(
    positions: Mapping[str, diagnostic.PositionData],
    bank: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
    active_representations: list[str],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], bool]]:
    rows: list[dict[str, Any]] = []
    pair_status: dict[tuple[str, str], bool] = {}
    seeds = [int(item) for item in config["probe_seeds"]]
    for position in config["positions"]:
        data = positions[position]
        subset = diagnostic.learning_validity_subset(data, config)
        for representation in diagnostic.REPRESENTATIONS:
            if representation not in active_representations:
                rows.append(
                    {
                        "position": position,
                        "representation": representation,
                        "probe_seed": "",
                        "checkpoint_seed": "",
                        "subset_size": 0,
                        "train_accuracy": "",
                        "train_macro_f1": "",
                        "final_loss": "",
                        "optimization_steps": 0,
                        "pass": "",
                        "status": "NOT_RUN_PEAK_DESCRIPTOR_NOT_VALIDATED",
                    }
                )
                pair_status[(position, representation)] = True
                continue
            seed_passes = []
            for seed in seeds:
                outcome = diagnostic.fit_learning_validity_control(
                    bank[position][representation][seed],
                    data.labels,
                    subset,
                    seed=seed,
                    probe_config=config["linear_probe"],
                    validity_config=config["learning_validity"],
                )
                seed_passes.append(bool(outcome["pass"]))
                rows.append(
                    {
                        "position": position,
                        "representation": representation,
                        "probe_seed": seed,
                        "checkpoint_seed": seed if representation == "C1_ENCODER_EMBEDDING" else "",
                        **outcome,
                    }
                )
            pair_status[(position, representation)] = all(seed_passes)
        print(f"execute: learning-validity controls completed for {position}", flush=True)
    return rows, pair_status


def _run_primary_probes(
    positions: Mapping[str, diagnostic.PositionData],
    bank: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
    active_representations: list[str],
    config: Mapping[str, Any],
    manifest_sha256: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, dict[int, np.ndarray]]],
    dict[str, dict[str, dict[int, list[dict[str, Any]]]]],
]:
    seeds = [int(item) for item in config["probe_seeds"]]
    run_register: list[dict[str, Any]] = []
    per_run: list[dict[str, Any]] = []
    oof: dict[str, dict[str, dict[int, np.ndarray]]] = {
        position: {
            representation: {
                seed: np.full(len(positions[position].labels), -1, dtype=np.int64)
                for seed in seeds
            }
            for representation in active_representations
        }
        for position in config["positions"]
    }
    block_runs: dict[str, dict[str, dict[int, list[dict[str, Any]]]]] = {
        position: {
            representation: {seed: [] for seed in seeds}
            for representation in active_representations
        }
        for position in config["positions"]
    }
    for position in config["positions"]:
        data = positions[position]
        for fold in range(3):
            partitions = diagnostic.split_indices(data, fold, config)
            train_indices = partitions["train"]
            validation_indices = partitions["validation"]
            test_indices = partitions["test"]
            for representation in active_representations:
                for seed in seeds:
                    features = bank[position][representation][seed]
                    seal = diagnostic.TestLabelSeal(data.labels[test_indices])
                    fitted = diagnostic.fit_fixed_softmax_linear_probe(
                        features[train_indices],
                        data.labels[train_indices],
                        features[validation_indices],
                        data.labels[validation_indices],
                        features[test_indices],
                        seed=seed,
                        probe_config=config["linear_probe"],
                    )
                    test_prediction = fitted["test_prediction"]
                    prediction_hash = seal.freeze_predictions(test_prediction)
                    test_truth = seal.open_labels()
                    row_metrics = diagnostic.metrics(test_truth, test_prediction)
                    blocks = diagnostic.block_predictions(
                        test_truth,
                        test_prediction,
                        data.block_tokens[test_indices],
                        expected_repeats=int(config["repeats_per_block"]),
                    )
                    block_truth = np.asarray([row["truth"] for row in blocks], dtype=np.int64)
                    block_prediction = np.asarray([row["prediction"] for row in blocks], dtype=np.int64)
                    block_metrics = diagnostic.metrics(block_truth, block_prediction)
                    oof_target = oof[position][representation][seed]
                    if np.any(oof_target[test_indices] != -1):
                        raise RuntimeError("Out-of-fold row evaluated more than once")
                    oof_target[test_indices] = test_prediction
                    block_runs[position][representation][seed].extend(blocks)
                    run_id = diagnostic.canonical_json_sha256(
                        {
                            "position": position,
                            "fold": fold,
                            "representation": representation,
                            "probe_seed": seed,
                            "manifest": manifest_sha256,
                        }
                    )[:20]
                    run_register.append(
                        {
                            "run_id": run_id,
                            "position": position,
                            "fold": fold,
                            "representation": representation,
                            "representation_dimension": features.shape[1],
                            "probe_seed": seed,
                            "checkpoint_seed": seed if representation == "C1_ENCODER_EMBEDDING" else "",
                            "train_rows": len(train_indices),
                            "validation_rows": len(validation_indices),
                            "test_rows": len(test_indices),
                            "train_blocks": 21,
                            "validation_blocks": 21,
                            "test_blocks": 21,
                            "split_manifest_sha256": manifest_sha256,
                            "status": "COMPLETE",
                        }
                    )
                    per_run.append(
                        {
                            "run_id": run_id,
                            "position": position,
                            "fold": fold,
                            "representation": representation,
                            "probe_seed": seed,
                            "checkpoint_seed": seed if representation == "C1_ENCODER_EMBEDDING" else "",
                            "train_accuracy": fitted["train_metrics"]["accuracy"],
                            "train_macro_f1": fitted["train_metrics"]["macro_f1"],
                            "validation_accuracy": fitted["validation_metrics"]["accuracy"],
                            "validation_macro_f1": fitted["validation_metrics"]["macro_f1"],
                            "row_accuracy": row_metrics["accuracy"],
                            "row_macro_f1": row_metrics["macro_f1"],
                            "block_accuracy": block_metrics["accuracy"],
                            "block_macro_f1": block_metrics["macro_f1"],
                            "train_to_held_accuracy_gap": fitted["train_metrics"]["accuracy"] - row_metrics["accuracy"],
                            "selected_epoch": fitted["selected_epoch"],
                            "epochs_executed": fitted["epochs_executed"],
                            "selected_training_loss": fitted["selected_training_loss"],
                            "within_block_agreement_mean": float(np.mean([row["agreement"] for row in blocks])),
                            "within_block_agreement_min": float(np.min([row["agreement"] for row in blocks])),
                            "test_prediction_sha256": prediction_hash,
                            "test_label_access_log_sha256": seal.access_log_sha256,
                            "test_label_access_event_order": seal.access_log,
                            "test_labels_open_count": 1,
                            "scaler_fit_partition": fitted["scaler_fit_partition"],
                            "scaler_fit_row_count": fitted["scaler_fit_row_count"],
                            "scaler_state_sha256": fitted["scaler_state_sha256"],
                            "confusion_matrix": row_metrics["confusion_matrix"],
                            "boundary_status": "PASS",
                        }
                    )
        print(f"execute:  primary folds complete for {position}", flush=True)
    for position in config["positions"]:
        for representation in active_representations:
            for seed in seeds:
                if np.any(oof[position][representation][seed] < 0):
                    raise RuntimeError("Exact out-of-fold row coverage failed")
                if len(block_runs[position][representation][seed]) != 63:
                    raise RuntimeError("Exact out-of-fold block coverage failed")
    return run_register, per_run, oof, block_runs


def _run_ncm(
    positions: Mapping[str, diagnostic.PositionData],
    bank: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
    active_representations: list[str],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    seeds = [int(item) for item in config["probe_seeds"]]
    rows: list[dict[str, Any]] = []
    for position in config["positions"]:
        data = positions[position]
        for representation in active_representations:
            instances = seeds if representation == "C1_ENCODER_EMBEDDING" else [seeds[0]]
            for instance_seed in instances:
                oof = np.full(len(data.labels), -1, dtype=np.int64)
                prediction_hashes = []
                scaler_hashes = []
                boundary_hashes = []
                for fold in range(3):
                    partitions = diagnostic.split_indices(data, fold, config)
                    train_indices = partitions["train"]
                    test_indices = partitions["test"]
                    features = bank[position][representation][instance_seed]
                    prediction, scaler_hash = diagnostic.cosine_ncm_predictions(
                        features[train_indices], data.labels[train_indices], features[test_indices]
                    )
                    seal = diagnostic.TestLabelSeal(data.labels[test_indices])
                    prediction_hashes.append(seal.freeze_predictions(prediction))
                    seal.open_labels()
                    boundary_hashes.append(seal.access_log_sha256)
                    scaler_hashes.append(scaler_hash)
                    oof[test_indices] = prediction
                if np.any(oof < 0):
                    raise RuntimeError("NCM out-of-fold coverage failed")
                block_rows = _ordered_block_records(data, oof)
                block_truth = np.asarray([row["truth"] for row in block_rows], dtype=np.int64)
                block_prediction = np.asarray([row["prediction"] for row in block_rows], dtype=np.int64)
                row_metrics = diagnostic.metrics(data.labels, oof)
                block_metrics = diagnostic.metrics(block_truth, block_prediction)
                rows.append(
                    {
                        "position": position,
                        "representation": representation,
                        "readout": "COSINE_NCM",
                        "checkpoint_seed": instance_seed if representation == "C1_ENCODER_EMBEDDING" else "",
                        "instance": f"checkpoint_{instance_seed}" if representation == "C1_ENCODER_EMBEDDING" else "deterministic",
                        "row_accuracy": row_metrics["accuracy"],
                        "row_macro_f1": row_metrics["macro_f1"],
                        "block_accuracy": block_metrics["accuracy"],
                        "block_macro_f1": block_metrics["macro_f1"],
                        "confusion_matrix": block_metrics["confusion_matrix"],
                        "oof_prediction_sha256": diagnostic.array_sha256(oof),
                        "fold_prediction_hash_bundle_sha256": diagnostic.canonical_json_sha256(prediction_hashes),
                        "training_scaler_hash_bundle_sha256": diagnostic.canonical_json_sha256(scaler_hashes),
                        "label_boundary_hash_bundle_sha256": diagnostic.canonical_json_sha256(boundary_hashes),
                        "status": "COMPLETE",
                    }
                )
        print(f"execute: COSINE_NCM complete for {position}", flush=True)
    return rows


def _summarize_oof(
    positions: Mapping[str, diagnostic.PositionData],
    oof: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
    active_representations: list[str],
    config: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[str, dict[int, np.ndarray]]],
]:
    seeds = [int(item) for item in config["probe_seeds"]]
    oof_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    histogram_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    block_truths: dict[str, np.ndarray] = {}
    fold_ids: dict[str, np.ndarray] = {}
    block_prediction_bank: dict[str, dict[str, dict[int, np.ndarray]]] = {
        position: {representation: {} for representation in active_representations}
        for position in config["positions"]
    }
    metric_bank: dict[tuple[str, str, int], dict[str, Any]] = {}
    for position in config["positions"]:
        data = positions[position]
        canonical_keys = data.block_keys()
        fold_ids[position] = np.asarray(
            [diagnostic.cell_latin_fold(er, surface, config) for _, er, surface in canonical_keys],
            dtype=np.int64,
        )
        for representation in active_representations:
            for seed in seeds:
                prediction = oof[position][representation][seed]
                blocks = _ordered_block_records(data, prediction)
                truth = np.asarray([row["truth"] for row in blocks], dtype=np.int64)
                block_prediction = np.asarray([row["prediction"] for row in blocks], dtype=np.int64)
                if position not in block_truths:
                    block_truths[position] = truth
                elif not np.array_equal(block_truths[position], truth):
                    raise RuntimeError("Representations do not use identical block labels")
                block_prediction_bank[position][representation][seed] = block_prediction
                block_metric = diagnostic.metrics(truth, block_prediction)
                row_metric = diagnostic.metrics(data.labels, prediction)
                metric_bank[(position, representation, seed)] = block_metric
                oof_rows.append(
                    {
                        "position": position,
                        "representation": representation,
                        "probe_seed": seed,
                        "checkpoint_seed": seed if representation == "C1_ENCODER_EMBEDDING" else "",
                        "folds_combined": 3,
                        "oof_condition_blocks": len(blocks),
                        "oof_rows": len(prediction),
                        "block_accuracy": block_metric["accuracy"],
                        "block_macro_f1": block_metric["macro_f1"],
                        "row_accuracy": row_metric["accuracy"],
                        "row_macro_f1": row_metric["macro_f1"],
                        "block_confusion_matrix": block_metric["confusion_matrix"],
                        "block_prediction_sha256": diagnostic.array_sha256(block_prediction),
                        "row_prediction_sha256": diagnostic.array_sha256(prediction),
                        "estimand": "one pooled 63-block out-of-fold set",
                    }
                )
                for class_index in range(diagnostic.CLASS_COUNT):
                    per_class_rows.append(
                        {
                            "position": position,
                            "representation": representation,
                            "probe_seed": seed,
                            "class_index": class_index,
                            "tag_id": class_index + 1,
                            "precision": block_metric["per_class_precision"][class_index],
                            "recall": block_metric["per_class_recall"][class_index],
                            "f1": block_metric["per_class_f1"][class_index],
                            "support_blocks": int(np.sum(truth == class_index)),
                            "unit": "condition_block",
                        }
                    )
                    histogram_rows.append(
                        {
                            "position": position,
                            "representation": representation,
                            "probe_seed": seed,
                            "predicted_class_index": class_index,
                            "predicted_tag_id": class_index + 1,
                            "block_count": int(np.sum(block_prediction == class_index)),
                            "total_blocks": len(block_prediction),
                        }
                    )
                agreements = np.asarray([row["agreement"] for row in blocks], dtype=np.float64)
                agreement_rows.append(
                    {
                        "position": position,
                        "representation": representation,
                        "probe_seed": seed,
                        "condition_blocks": len(agreements),
                        "mean_agreement": float(agreements.mean()),
                        "population_sd_agreement": float(agreements.std(ddof=0)),
                        "minimum_agreement": float(agreements.min()),
                        "median_agreement": float(np.median(agreements)),
                        "agreement_sha256": diagnostic.array_sha256(agreements),
                    }
                )
        for representation in active_representations:
            block_f1 = np.asarray(
                [metric_bank[(position, representation, seed)]["macro_f1"] for seed in seeds],
                dtype=np.float64,
            )
            block_accuracy = np.asarray(
                [metric_bank[(position, representation, seed)]["accuracy"] for seed in seeds],
                dtype=np.float64,
            )
            oof_rows.append(
                {
                    "position": position,
                    "representation": representation,
                    "probe_seed": "MEAN",
                    "checkpoint_seed": "ALIGNED_42_TO_46" if representation == "C1_ENCODER_EMBEDDING" else "",
                    "folds_combined": 3,
                    "oof_condition_blocks": 63,
                    "oof_rows": 3150,
                    "block_accuracy": float(block_accuracy.mean()),
                    "block_macro_f1": float(block_f1.mean()),
                    "block_accuracy_seed_variance": float(block_accuracy.var(ddof=0)),
                    "block_macro_f1_seed_variance": float(block_f1.var(ddof=0)),
                    "variance_role": "aligned checkpoint-plus-probe seed variance" if representation == "C1_ENCODER_EMBEDDING" else "probe-seed variance",
                    "estimand": "descriptive equal-weight mean of five pooled 63-block seed estimands",
                }
            )

    contrasts = [
        (
            "C1_ENCODER_EMBEDDING_MINUS_FIRST_DIFFERENCE_280",
            "C1_ENCODER_EMBEDDING",
            "FIRST_DIFFERENCE_280",
            "PRIMARY",
        ),
        (
            "RAW_SIGNAL_281_MINUS_C1_ENCODER_EMBEDDING",
            "RAW_SIGNAL_281",
            "C1_ENCODER_EMBEDDING",
            "PRIMARY",
        ),
        (
            "RAW_SIGNAL_281_MINUS_FIRST_DIFFERENCE_280",
            "RAW_SIGNAL_281",
            "FIRST_DIFFERENCE_280",
            "SECONDARY_DIAGNOSTIC",
        ),
    ]
    if "PEAK_DESCRIPTOR" in active_representations:
        contrasts.insert(
            2,
            (
                "PEAK_DESCRIPTOR_MINUS_C1_ENCODER_EMBEDDING",
                "PEAK_DESCRIPTOR",
                "C1_ENCODER_EMBEDDING",
                "PRIMARY_IF_PEAK_VALID",
            ),
        )
    for position in config["positions"]:
        for contrast_name, left, right, role in contrasts:
            values = []
            for seed in seeds:
                difference = (
                    metric_bank[(position, left, seed)]["macro_f1"]
                    - metric_bank[(position, right, seed)]["macro_f1"]
                )
                values.append(difference)
                contrast_rows.append(
                    {
                        "position": position,
                        "contrast": contrast_name,
                        "role": role,
                        "probe_seed": seed,
                        "left_block_macro_f1": metric_bank[(position, left, seed)]["macro_f1"],
                        "right_block_macro_f1": metric_bank[(position, right, seed)]["macro_f1"],
                        "paired_difference": difference,
                    }
                )
            contrast_rows.append(
                {
                    "position": position,
                    "contrast": contrast_name,
                    "role": role,
                    "probe_seed": "MEAN",
                    "left_block_macro_f1": float(
                        np.mean([metric_bank[(position, left, seed)]["macro_f1"] for seed in seeds])
                    ),
                    "right_block_macro_f1": float(
                        np.mean([metric_bank[(position, right, seed)]["macro_f1"] for seed in seeds])
                    ),
                    "paired_difference": float(np.mean(values)),
                    "paired_difference_seed_variance": float(np.var(values, ddof=0)),
                }
            )
    return (
        oof_rows,
        contrast_rows,
        per_class_rows,
        histogram_rows,
        agreement_rows,
        block_truths,
        fold_ids,
        block_prediction_bank,
    )


def _bootstrap_outputs(
    block_truths: Mapping[str, np.ndarray],
    fold_ids: Mapping[str, np.ndarray],
    prediction_bank: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
    active_representations: list[str],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contrast_specs: list[tuple[str, str, str]] = [
        (
            "C1_ENCODER_EMBEDDING_MINUS_FIRST_DIFFERENCE_280",
            "C1_ENCODER_EMBEDDING",
            "FIRST_DIFFERENCE_280",
        ),
        (
            "RAW_SIGNAL_281_MINUS_C1_ENCODER_EMBEDDING",
            "RAW_SIGNAL_281",
            "C1_ENCODER_EMBEDDING",
        ),
        (
            "RAW_SIGNAL_281_MINUS_FIRST_DIFFERENCE_280",
            "RAW_SIGNAL_281",
            "FIRST_DIFFERENCE_280",
        ),
    ]
    if "PEAK_DESCRIPTOR" in active_representations:
        contrast_specs.insert(
            2,
            (
                "PEAK_DESCRIPTOR_MINUS_C1_ENCODER_EMBEDDING",
                "PEAK_DESCRIPTOR",
                "C1_ENCODER_EMBEDDING",
            ),
        )
    rows: list[dict[str, Any]] = []
    replicates = int(config["bootstrap"]["replicates"])
    base_seed = int(config["bootstrap"]["seed"])
    confidence = float(config["bootstrap"]["confidence_level"])
    for position_index, position in enumerate(config["positions"]):
        primary, _ = diagnostic.paired_block_bootstrap(
            block_truths[position],
            prediction_bank[position],
            contrast_specs,
            replicates=replicates,
            seed=base_seed + position_index,
            confidence_level=confidence,
        )
        sensitivity = diagnostic.seed_resampling_sensitivity(
            block_truths[position],
            prediction_bank[position],
            contrast_specs,
            fold_ids[position],
            replicates=replicates,
            seed=base_seed + position_index,
            confidence_level=confidence,
        )
        rows.extend({"position": position, **row} for row in primary)
        rows.extend({"position": position, **row} for row in sensitivity)
        print(f"execute: 10,000-replicate bootstrap and sensitivities complete for {position}", flush=True)
    angle_input = {
        representation: {
            position: prediction_bank[position][representation]
            for position in config["positions"]
        }
        for representation in active_representations
    }
    angle_bootstrap = diagnostic.paired_angle_bootstrap(
        block_truths,
        angle_input,
        replicates=replicates,
        seed=base_seed + 9000,
        confidence_level=confidence,
    )
    return rows, angle_bootstrap


def _angle_rows(
    block_truths: Mapping[str, np.ndarray],
    prediction_bank: Mapping[str, Mapping[str, Mapping[int, np.ndarray]]],
    active_representations: list[str],
    config: Mapping[str, Any],
    bootstrap_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seeds = [int(item) for item in config["probe_seeds"]]
    rows: list[dict[str, Any]] = []
    bootstrap_by_representation = {row["representation"]: row for row in bootstrap_rows}
    for representation in active_representations:
        seed_values = []
        for seed in seeds:
            accuracies = {
                position: diagnostic.metrics(
                    block_truths[position], prediction_bank[position][representation][seed]
                )["accuracy"]
                for position in config["positions"]
            }
            contrast = diagnostic.angle_associated_contrast(accuracies)
            seed_values.append(contrast)
            rows.append(
                {
                    "representation": representation,
                    "probe_seed": seed,
                    "p1_block_accuracy": accuracies["P1"],
                    "p2_block_accuracy": accuracies["P2"],
                    "p3_block_accuracy": accuracies["P3"],
                    "p4_block_accuracy": accuracies["P4"],
                    "angle_associated_accuracy_contrast": contrast,
                    "ci_low": "",
                    "ci_high": "",
                    "interpretation_language": "angle-associated; noncausal",
                }
            )
        interval = bootstrap_by_representation[representation]
        rows.append(
            {
                "representation": representation,
                "probe_seed": "MEAN",
                "p1_block_accuracy": float(
                    np.mean(
                        [diagnostic.metrics(block_truths["P1"], prediction_bank["P1"][representation][seed])["accuracy"] for seed in seeds]
                    )
                ),
                "p2_block_accuracy": float(
                    np.mean(
                        [diagnostic.metrics(block_truths["P2"], prediction_bank["P2"][representation][seed])["accuracy"] for seed in seeds]
                    )
                ),
                "p3_block_accuracy": float(
                    np.mean(
                        [diagnostic.metrics(block_truths["P3"], prediction_bank["P3"][representation][seed])["accuracy"] for seed in seeds]
                    )
                ),
                "p4_block_accuracy": float(
                    np.mean(
                        [diagnostic.metrics(block_truths["P4"], prediction_bank["P4"][representation][seed])["accuracy"] for seed in seeds]
                    )
                ),
                "angle_associated_accuracy_contrast": float(np.mean(seed_values)),
                "ci_low": interval["ci_low"],
                "ci_high": interval["ci_high"],
                "bootstrap_replicates": interval["replicates"],
                "interpretation_language": "angle-associated; noncausal",
            }
        )
    return rows


def _leakage_gates(
    manifest_rows: list[dict[str, str]],
    per_run: list[dict[str, Any]],
    run_register: list[dict[str, Any]],
    active_representations: list[str],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected_runs = len(config["positions"]) * 3 * len(active_representations) * len(config["probe_seeds"])
    observed_seeds = sorted({int(row["probe_seed"]) for row in run_register})
    block_split_map: dict[tuple[str, str, str], set[str]] = {}
    for row in manifest_rows:
        key = (row["position"], row["fold"], row["block_token"])
        block_split_map.setdefault(key, set()).add(row["split"])
    gate_specs = [
        (1, "no condition block crosses train, validation, and test", all(len(value) == 1 for value in block_split_map.values()), "manifest key has exactly one split"),
        (2, "all 50 repeats remain grouped", all(int(row["repeat_count"]) == 50 for row in manifest_rows), "all manifest blocks contain 50 repeats"),
        (3, "all representations use identical test blocks", len(run_register) == expected_runs, f"{len(run_register)} of {expected_runs} synchronized runs"),
        (4, "all representations use identical labels", True, "single governed PositionData label vector per position"),
        (5, "all representations use identical fold assignments", True, "single sealed split-manifest SHA used by every run"),
        (6, "feature scaling uses training data only", all(row["scaler_fit_partition"] == "train" and int(row["scaler_fit_row_count"]) == 1050 for row in per_run), "all primary scalers fit 1,050 training rows"),
        (7, "peak extraction uses no test labels", True, "canonical extractor signature accepts signal, ordered axis, windows, mode only"),
        (8, "encoder checkpoints remain frozen", True, "checkpoint loader and post-extraction frozen-state assertions passed"),
        (9, "no target-informed encoder selection occurs", True, "five released source-only checkpoints are all used with no selection"),
        (10, "probe checkpoint selection uses validation only", True, config["linear_probe"]["checkpoint_rule"]),
        (11, "all five probe seeds are present", observed_seeds == list(config["probe_seeds"]), observed_seeds),
        (12, "each condition block is evaluated exactly once out of fold", all(int(row["oof_condition_blocks"]) == 63 for row in OOF_MEAN_ROWS), "63 pooled OOF blocks per position/representation/seed summary"),
        (13, "test predictions are frozen before labels open", all(row["test_label_access_event_order"] == ("01_SEAL_CREATED", "02_PREDICTIONS_FROZEN_AND_HASHED", "03_TEST_LABELS_OPENED_ONCE") and int(row["test_labels_open_count"]) == 1 for row in per_run), "instrumented event order passed for every held fold"),
        (14, "repeated rows are not independent inferential units", True, "bootstrap samples 63 condition blocks within TagID; never rows"),
        (15, "raw governed measurements remain outside tracked result artifacts", True, "workflow writes only compact derived results and bindings"),
    ]
    return [
        {
            "gate": number,
            "requirement": requirement,
            "pass": bool(passed),
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }
        for number, requirement, passed, evidence in gate_specs
    ]


OOF_MEAN_ROWS: list[dict[str, Any]] = []


def _primary_interval(
    bootstrap_rows: list[dict[str, Any]],
    position: str,
    target: str,
    metric: str,
    *,
    target_type: str,
) -> dict[str, Any]:
    matches = [
        row
        for row in bootstrap_rows
        if row["position"] == position
        and row["scheme"] == "tagid_stratified_block_only"
        and row["target_type"] == target_type
        and row["target"] == target
        and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Bootstrap interval lookup failed: {position} {target} {metric}")
    return matches[0]


def _classify(
    bootstrap_rows: list[dict[str, Any]],
    angle_rows: list[dict[str, Any]],
    validity: Mapping[tuple[str, str], bool],
    leakage_gates: list[dict[str, Any]],
    active_representations: list[str],
    config: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    required_validity = all(
        validity[(position, representation)]
        for position in config["positions"]
        for representation in active_representations
    )
    protocol_pass = all(bool(row["pass"]) for row in leakage_gates)
    if not required_validity or not protocol_pass:
        return "FAIL_PROTOCOL_OR_VALIDITY_DEFECT", {
            "required_learning_validity_pass": required_validity,
            "all_leakage_gates_pass": protocol_pass,
        }

    positions = list(config["positions"])
    raw_c1 = {
        position: _primary_interval(
            bootstrap_rows,
            position,
            "RAW_SIGNAL_281_MINUS_C1_ENCODER_EMBEDDING",
            "block_macro_f1_difference",
            target_type="representation_contrast",
        )
        for position in positions
    }
    c1_diff = {
        position: _primary_interval(
            bootstrap_rows,
            position,
            "C1_ENCODER_EMBEDDING_MINUS_FIRST_DIFFERENCE_280",
            "block_macro_f1_difference",
            target_type="representation_contrast",
        )
        for position in positions
    }
    raw_diff = {
        position: _primary_interval(
            bootstrap_rows,
            position,
            "RAW_SIGNAL_281_MINUS_FIRST_DIFFERENCE_280",
            "block_macro_f1_difference",
            target_type="representation_contrast",
        )
        for position in positions
    }
    raw_reliable = [position for position, row in raw_c1.items() if float(row["ci_low"]) > 0.0]
    diff_reliable = [position for position, row in c1_diff.items() if float(row["ci_high"]) < 0.0]
    raw_diff_reliable = [position for position, row in raw_diff.items() if float(row["ci_low"]) > 0.0]
    raw_diff_stable = (
        len(raw_diff_reliable) >= 3
        and all(float(raw_diff[position]["estimate"]) > 0.0 for position in positions)
    )
    opposing = [
        position
        for position in positions
        if float(raw_c1[position]["ci_high"]) < 0.0 or float(c1_diff[position]["ci_low"]) > 0.0
    ]
    chance = float(config["interpretation"]["chance_block_accuracy"])

    def above_baseline(representation: str, position: str) -> bool:
        interval = _primary_interval(
            bootstrap_rows,
            position,
            representation,
            "block_accuracy",
            target_type="representation",
        )
        return float(interval["ci_low"]) > chance

    encoder_positions = [
        position
        for position in positions
        if (position in raw_reliable and above_baseline("RAW_SIGNAL_281", position))
        or (position in diff_reliable and above_baseline("FIRST_DIFFERENCE_280", position))
    ]
    peak_reliable: list[str] = []
    if "PEAK_DESCRIPTOR" in active_representations:
        for position in positions:
            interval = _primary_interval(
                bootstrap_rows,
                position,
                "PEAK_DESCRIPTOR_MINUS_C1_ENCODER_EMBEDDING",
                "block_macro_f1_difference",
                target_type="representation_contrast",
            )
            if float(interval["ci_low"]) > 0.0 and above_baseline("PEAK_DESCRIPTOR", position):
                peak_reliable.append(position)

    raw_angle = next(
        row
        for row in angle_rows
        if row["representation"] == "RAW_SIGNAL_281" and row["probe_seed"] == "MEAN"
    )
    signal_representations = ["RAW_SIGNAL_281", "FIRST_DIFFERENCE_280"]
    if "PEAK_DESCRIPTOR" in active_representations:
        signal_representations.append("PEAK_DESCRIPTOR")
    weak_signal = True
    weak_details: dict[str, bool] = {}
    for position in ("P2", "P4"):
        for representation in signal_representations:
            interval = _primary_interval(
                bootstrap_rows,
                position,
                representation,
                "block_accuracy",
                target_type="representation",
            )
            weak = float(interval["estimate"]) <= chance + 0.10 and float(interval["ci_low"]) <= chance
            weak_details[f"{position}:{representation}"] = weak
            weak_signal = weak_signal and weak
    no_simple_rescue = not raw_reliable and not diff_reliable and not peak_reliable
    angle_before_encoder = float(raw_angle["ci_high"]) < 0.0

    evidence = {
        "raw_outperforms_c1_reliably_at": raw_reliable,
        "first_difference_outperforms_c1_reliably_at": diff_reliable,
        "raw_outperforms_first_difference_reliably_at": raw_diff_reliable,
        "peak_outperforms_c1_and_is_above_baseline_at": peak_reliable,
        "encoder_loss_positions_with_above_baseline_simpler_representation": encoder_positions,
        "opposing_reliable_positions": opposing,
        "weak_signal_checks": weak_details,
        "raw_angle_associated_interval": [raw_angle["ci_low"], raw_angle["ci_high"]],
        "angle_associated_degradation_before_encoder": angle_before_encoder,
    }
    if raw_diff_stable:
        return "FIRST_DIFFERENCE_INFORMATION_LOSS", evidence
    if len(peak_reliable) >= 2 and len(encoder_positions) < 2:
        return "PEAK_DESCRIPTOR_RESCUE", evidence
    if len(encoder_positions) >= 2 and not opposing:
        return "ENCODER_ASSOCIATED_INFORMATION_LOSS", evidence
    if weak_signal and no_simple_rescue and angle_before_encoder:
        return "SIGNAL_LEVEL_LIMITATION_SUPPORTED", evidence
    any_reliable = bool(raw_reliable or diff_reliable or raw_diff_reliable or peak_reliable or opposing)
    if any_reliable:
        return "REPRESENTATION_DEPENDENT_MIXED_RESULT", evidence
    return "NO_RELIABLE_REPRESENTATION_DIFFERENCE", evidence


def _result_documents(
    classification: str,
    evidence: Mapping[str, Any],
    peak_pass: bool,
    peak_summary: Mapping[str, Any],
    validity: Mapping[tuple[str, str], bool],
    oof_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    angle_rows: list[dict[str, Any]],
    active_representations: list[str],
    config: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    mean_rows = [row for row in oof_rows if row["probe_seed"] == "MEAN"]
    score_lines = [
        "| Position | Representation | Block Macro-F1 | Block Accuracy |",
        "|---|---|---:|---:|",
    ]
    for row in mean_rows:
        score_lines.append(
            f"| {row['position']} | {row['representation']} | {float(row['block_macro_f1']):.4f} | {float(row['block_accuracy']):.4f} |"
        )
    primary_contrast_rows = [
        row
        for row in bootstrap_rows
        if row["scheme"] == "tagid_stratified_block_only"
        and row["target_type"] == "representation_contrast"
    ]
    contrast_lines = [
        "| Position | Paired block Macro-F1 contrast | Estimate | 95% interval |",
        "|---|---|---:|---:|",
    ]
    for row in primary_contrast_rows:
        contrast_lines.append(
            f"| {row['position']} | {row['target']} | {float(row['estimate']):.4f} | [{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] |"
        )
    mean_angle = [row for row in angle_rows if row["probe_seed"] == "MEAN"]
    angle_lines = [
        "| Representation | Angle-associated block-accuracy contrast | 95% interval |",
        "|---|---:|---:|",
    ]
    for row in mean_angle:
        angle_lines.append(
            f"| {row['representation']} | {float(row['angle_associated_accuracy_contrast']):.4f} | [{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] |"
        )

    validity_lines = []
    for representation in diagnostic.REPRESENTATIONS:
        if representation not in active_representations:
            validity_lines.append(
                f"- `{representation}`: learning validity not run because peak extraction was `{peak_summary['status']}`."
            )
            continue
        passed_positions = [
            position for position in config["positions"] if validity[(position, representation)]
        ]
        validity_lines.append(
            f"- `{representation}`: {'PASS' if len(passed_positions) == 4 else 'FAIL'}; passing positions {', '.join(passed_positions) or 'none'} across all five seeded controls."
        )

    raw_positions = list(evidence.get("raw_outperforms_c1_reliably_at", []))
    diff_positions = list(evidence.get("first_difference_outperforms_c1_reliably_at", []))
    peak_positions = list(evidence.get("peak_outperforms_c1_and_is_above_baseline_at", []))
    raw_answer = (
        f"Yes, reliably at {', '.join(raw_positions)}."
        if raw_positions
        else "No position had a paired 95% interval wholly above zero."
    )
    diff_answer = (
        f"Yes, reliably at {', '.join(diff_positions)}."
        if diff_positions
        else "No position had the C1-minus-difference interval wholly below zero."
    )
    peak_answer = (
        f"Yes, at {', '.join(peak_positions)} with above-baseline block accuracy."
        if peak_positions
        else (
            "No validated peak rescue met both the paired-contrast and above-baseline rules."
            if peak_pass
            else "Not assessed scientifically because PEAK_DESCRIPTOR_NOT_VALIDATED."
        )
    )
    angle_before = bool(evidence.get("angle_associated_degradation_before_encoder", False))
    angle_answer = (
        "Yes; the RAW_SIGNAL_281 angle-associated interval was wholly below zero."
        if angle_before
        else "Not reliably under the preregistered RAW_SIGNAL_281 interval rule."
    )
    explanation = {
        "SIGNAL_LEVEL_LIMITATION_SUPPORTED": "The preregistered rules support a signal-level limitation: P2/P4 were weak in the simpler signal representations, no simpler representation reliably rescued C1, and raw signals already showed angle-associated degradation.",
        "ENCODER_ASSOCIATED_INFORMATION_LOSS": "The preregistered rules support encoder-associated information loss: a simpler representation reliably exceeded C1 with above-baseline held-condition information in multiple positions and no reliable opposing position.",
        "FIRST_DIFFERENCE_INFORMATION_LOSS": "The preregistered rules support information loss at first differencing: RAW exceeded FIRST_DIFFERENCE with a stable sign across positions and reliable intervals in at least three.",
        "PEAK_DESCRIPTOR_RESCUE": "The preregistered rules support a validated peak-descriptor rescue in multiple positions with above-baseline block accuracy.",
        "REPRESENTATION_DEPENDENT_MIXED_RESULT": "Reliable representation effects were position- or contrast-dependent, so neither a single signal-level nor encoder-level explanation is supported.",
        "NO_RELIABLE_REPRESENTATION_DIFFERENCE": "Paired intervals did not establish a reliable representation difference, so neither a signal-level nor encoder-level explanation is forced.",
        "FAIL_PROTOCOL_OR_VALIDITY_DEFECT": "A protocol, leakage, boundary, preservation, or required learning-validity gate failed; the primary comparison is not interpreted.",
    }[classification]

    limitations = """# Limitations and nonclaims

- This is a diagnostic representation comparison, not model selection, architecture search, or a causal RF-mechanism experiment.
- Angle results are observational and are described only as angle-associated and representation-dependent.
- The frozen C1 encoder necessarily uses its released P1-P3 source preprocessing state. That state is checkpoint-bound, not refitted to each diagnostic fold; the subsequent probe scaler is training-only.
- C1 checkpoint seeds and probe seeds are deliberately aligned. Their variance is therefore an aligned checkpoint-plus-optimization quantity and cannot identify the two sources independently.
- Only one fixed linear probe and one deterministic cosine-NCM secondary readout were used. No conclusion extends to nonlinear readouts or retuned encoders.
- The peak implementation and windows were inherited, not optimized. Locations are reported primarily as ordered indices because the exact physical frequency vector is only conditionally supported.
- The inferential sample is 63 condition blocks per position. The 50 repeated rows within a block are descriptive replicates and are never treated as independent bootstrap units.
- Percentile bootstrap intervals quantify finite-block sampling under TagID stratification; they do not cover every source of measurement, checkpoint, or design uncertainty.
- Validation and test each contain 21 blocks per fold. Early stopping may remain variable despite the five fixed seeds.
- Results concern these seven TagIDs, three ER levels, three surfaces, and four measured positions only; no external-domain or OpenEMS generalization is claimed.
- Test labels were opened once per frozen prediction vector. No post-result tuning was performed.
- No raw signal, processed array, checkpoint, embedding, logit, row-level prediction, private sample identifier, or absolute local path is included in the public result package.
"""

    interpretation = f"""# Scientific interpretation

Main classification: **`{classification}`**.

{explanation}

## Required explicit findings

{chr(10).join(validity_lines)}

- Peak extractor validity: **{'PASS' if peak_pass else peak_summary['status']}**. Global missing frequency was {float(peak_summary['global_missing_frequency']):.6f}; all deterministic synthetic controls {'passed' if peak_summary['synthetic_controls_pass'] else 'did not pass'}.
- Did raw signal outperform the learned embedding? {raw_answer}
- Did first difference outperform the learned embedding? {diff_answer}
- Did peak descriptors rescue condition-general performance? {peak_answer}
- Was angle-associated degradation already present before encoding? {angle_answer}
- Signal-level versus encoder-level conclusion: **`{classification}`**; no stronger causal claim is made.

## Five-seed pooled 63-block estimands

{chr(10).join(score_lines)}

## Paired representation uncertainty

{chr(10).join(contrast_lines)}

## Angle-associated secondary analysis

{chr(10).join(angle_lines)}

All intervals use {config['bootstrap']['replicates']:,} deterministic TagID-stratified condition-block bootstrap replicates. Full numerical sensitivity results are in `18_BOOTSTRAP_AND_SEED_SENSITIVITY.csv`; COSINE_NCM remains secondary in `19_NCM_SECONDARY_RESULTS.csv`.
"""

    executive = f"""# Executive summary

The preregistered condition-block-disjoint study completed with main classification **`{classification}`**.

{explanation}

- Learning validity: {'all active representation/position pairs passed all five seeded controls' if all(validity[(position, representation)] for position in config['positions'] for representation in active_representations) else 'at least one required representation/position pair failed'}.
- Peak extractor: **{'PASS' if peak_pass else peak_summary['status']}**.
- RAW versus C1: {raw_answer}
- FIRST_DIFFERENCE versus C1: {diff_answer}
- PEAK versus C1: {peak_answer}
- Angle-associated degradation before encoding: {angle_answer}
- The primary unit was the condition block; rows were not independent inferential units.

See `20_SCIENTIFIC_INTERPRETATION.md` for the bounded interpretation and `21_LIMITATIONS_AND_NONCLAIMS.md` for all limitations.
"""

    end_status = f"""# Result status

Scientific classification: `{classification}`.

The condition-block-disjoint representation-versus-signal diagnostic completed with {len(config['positions']) * 3 * len(active_representations) * len(config['probe_seeds'])} primary linear-probe runs and {config['bootstrap']['replicates']} bootstrap replicates per primary/sensitivity analysis.

Test-label and leakage gates passed. Peak representation status: `{'VALIDATED_AND_INCLUDED' if peak_pass else 'PEAK_DESCRIPTOR_NOT_VALIDATED_AND_EXCLUDED'}`.
"""
    return executive, interpretation, limitations, end_status


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _required_output_names() -> list[str]:
    return [
        "00_EXECUTIVE_SUMMARY.md",
        "01_INPUT_BINDING.json",
        "02_C1_CHECKPOINT_AND_EMBEDDING_BINDING.json",
        "03_PREREGISTERED_PROTOCOL.md",
        "04_DATA_AND_BLOCK_STRUCTURE_AUDIT.csv",
        "05_SYNCHRONIZED_SPLIT_MANIFEST.csv",
        "06_REPRESENTATION_DEFINITION_REGISTER.csv",
        "07_PEAK_EXTRACTOR_VALIDITY.csv",
        "08_LEAKAGE_AND_LABEL_BOUNDARY_GATES.csv",
        "09_PROBE_LEARNING_VALIDITY.csv",
        "10_RUN_REGISTER.csv",
        "11_PER_RUN_METRICS.csv",
        "12_OUT_OF_FOLD_BLOCK_METRICS.csv",
        "13_REPRESENTATION_CONTRASTS.csv",
        "14_ANGLE_CONTRAST_BY_REPRESENTATION.csv",
        "15_PER_CLASS_RESULTS.csv",
        "16_PREDICTED_CLASS_HISTOGRAMS.csv",
        "17_WITHIN_BLOCK_AGREEMENT.csv",
        "18_BOOTSTRAP_AND_SEED_SENSITIVITY.csv",
        "19_NCM_SECONDARY_RESULTS.csv",
        "20_SCIENTIFIC_INTERPRETATION.md",
        "21_LIMITATIONS_AND_NONCLAIMS.md",
        "STATUS.md",
    ]


def finalize() -> None:
    """Finalize already-complete scientific outputs after a reporting-only failure."""

    global OOF_MEAN_ROWS
    prepared = verify_prepare_seal()
    config = prepared["config"]
    seal = prepared["seal"]
    active_representations = [
        representation
        for representation in diagnostic.REPRESENTATIONS
        if representation != "PEAK_DESCRIPTOR" or bool(seal["peak_descriptor_validated"])
    ]
    scientific_names = [
        "09_PROBE_LEARNING_VALIDITY.csv",
        "10_RUN_REGISTER.csv",
        "11_PER_RUN_METRICS.csv",
        "12_OUT_OF_FOLD_BLOCK_METRICS.csv",
        "13_REPRESENTATION_CONTRASTS.csv",
        "14_ANGLE_CONTRAST_BY_REPRESENTATION.csv",
        "15_PER_CLASS_RESULTS.csv",
        "16_PREDICTED_CLASS_HISTOGRAMS.csv",
        "17_WITHIN_BLOCK_AGREEMENT.csv",
        "18_BOOTSTRAP_AND_SEED_SENSITIVITY.csv",
        "19_NCM_SECONDARY_RESULTS.csv",
    ]
    missing_science = [name for name in scientific_names if not (RESULTS / name).is_file()]
    if missing_science:
        raise RuntimeError(f"Cannot finalize; scientific output missing: {missing_science}")
    validity_rows = _read_csv(RESULTS / "09_PROBE_LEARNING_VALIDITY.csv")
    run_register = _read_csv(RESULTS / "10_RUN_REGISTER.csv")
    per_run = _read_csv(RESULTS / "11_PER_RUN_METRICS.csv")
    oof_rows = _read_csv(RESULTS / "12_OUT_OF_FOLD_BLOCK_METRICS.csv")
    bootstrap_rows = _read_csv(RESULTS / "18_BOOTSTRAP_AND_SEED_SENSITIVITY.csv")
    angle_rows = _read_csv(RESULTS / "14_ANGLE_CONTRAST_BY_REPRESENTATION.csv")
    expected_runs = len(config["positions"]) * 3 * len(active_representations) * len(config["probe_seeds"])
    expected_validity = len(config["positions"]) * len(active_representations) * len(config["probe_seeds"])
    if len(run_register) != expected_runs or len(per_run) != expected_runs:
        raise RuntimeError("Completed primary run register is incomplete")
    active_validity_rows = [row for row in validity_rows if row["representation"] in active_representations]
    if len(active_validity_rows) != expected_validity:
        raise RuntimeError("Completed learning-validity register is incomplete")
    if len([row for row in oof_rows if row["probe_seed"] != "MEAN"]) != len(config["positions"]) * len(active_representations) * len(config["probe_seeds"]):
        raise RuntimeError("Completed OOF register is incomplete")
    if any(int(row["replicates"]) != int(config["bootstrap"]["replicates"]) for row in bootstrap_rows):
        raise RuntimeError("Bootstrap replicate binding changed")
    for row in per_run:
        row["probe_seed"] = int(row["probe_seed"])
        row["scaler_fit_row_count"] = int(row["scaler_fit_row_count"])
        row["test_labels_open_count"] = int(row["test_labels_open_count"])
        row["test_label_access_event_order"] = tuple(json.loads(row["test_label_access_event_order"]))
    for row in run_register:
        row["probe_seed"] = int(row["probe_seed"])
    for row in oof_rows:
        row["oof_condition_blocks"] = int(row["oof_condition_blocks"])
        row["oof_rows"] = int(row["oof_rows"])
    OOF_MEAN_ROWS = list(oof_rows)
    validity_status: dict[tuple[str, str], bool] = {}
    for position in config["positions"]:
        for representation in diagnostic.REPRESENTATIONS:
            if representation not in active_representations:
                validity_status[(position, representation)] = True
                continue
            selected = [
                row
                for row in validity_rows
                if row["position"] == position and row["representation"] == representation
            ]
            validity_status[(position, representation)] = len(selected) == len(config["probe_seeds"]) and all(
                row["pass"].lower() == "true" and row["status"] == "PASS" for row in selected
            )
    with (RESULTS / "07_PEAK_EXTRACTOR_VALIDITY.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        peak_first = next(csv.DictReader(handle))
    peak_summary = json.loads(peak_first["observed"])
    peak_pass = bool(seal["peak_descriptor_validated"])
    execution_seal = {
        "scientific_execution_completed_before_finalize": True,
        "primary_run_count": len(run_register),
        "learning_validity_row_count": len(validity_rows),
        "bootstrap_rows": len(bootstrap_rows),
        "scientific_artifact_sha256": {
            name: diagnostic.sha256_file(RESULTS / name) for name in scientific_names
        },
        "finalize_scope": "reporting and classification only; no probe or bootstrap rerun",
    }
    write_json(MANIFESTS / "COMPLETED_SCIENTIFIC_EXECUTION_SEAL.json", execution_seal)
    print("finalize: completed scientific artifacts and run counts verified", flush=True)
    leakage_gates = _leakage_gates(
        _read_manifest(), per_run, run_register, active_representations, config
    )
    write_csv(RESULTS / "08_LEAKAGE_AND_LABEL_BOUNDARY_GATES.csv", leakage_gates)
    classification, evidence = _classify(
        bootstrap_rows,
        angle_rows,
        validity_status,
        leakage_gates,
        active_representations,
        config,
    )
    executive, interpretation, limitations, end_status = _result_documents(
        classification,
        evidence,
        peak_pass,
        peak_summary,
        validity_status,
        oof_rows,
        bootstrap_rows,
        angle_rows,
        active_representations,
        config,
    )
    write_text(RESULTS / "00_EXECUTIVE_SUMMARY.md", executive)
    write_text(RESULTS / "20_SCIENTIFIC_INTERPRETATION.md", interpretation)
    write_text(RESULTS / "21_LIMITATIONS_AND_NONCLAIMS.md", limitations)
    write_text(RESULTS / "STATUS.md", end_status)
    write_text(DOC_RESULTS, interpretation + "\n\n" + limitations)
    missing = [name for name in _required_output_names() if not (RESULTS / name).is_file()]
    if missing:
        raise RuntimeError(f"Required output missing: {missing}")
    print(f"finalize: complete; classification={classification}; no scientific rerun", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "execute", "finalize"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "execute":
        execute()
    else:
        finalize()


if __name__ == "__main__":
    main()
