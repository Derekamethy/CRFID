"""Train five final P1-P3 models and freeze every target-evaluation input."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crfid.strict_runtime.hashing import array_sha256, canonical_json_sha256, hash_lines, sha256_file
from crfid.strict_runtime.neutral_data import PartitionData
from crfid.strict_runtime.neutral_model import initialize_model, model_state_sha256, parameter_count
from crfid.strict_runtime.phase3b_execution import _criterion, _erm_train_epoch, _optimizer, load_checkpoint_model
from crfid.strict_runtime.scale_policy import CANONICAL_MODE, apply_scale_policy, fallback_report


SOURCE_SELECTION = PROJECT_ROOT / "outputs" / "strict_dg" / "source_selection"
SOURCE_INPUTS = PROJECT_ROOT / "outputs" / "strict_dg" / "source_inputs"
FROZEN = PROJECT_ROOT / "outputs" / "strict_dg" / "frozen_release"
CONFIG = PROJECT_ROOT / "configs" / "strict_dg" / "canonical.yaml"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_record(path: Path) -> dict:
    return {"relative_path": path.relative_to(PROJECT_ROOT).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_manifest(path: Path, files: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        for item in sorted(files):
            writer.writerow(file_record(item))


def main() -> int:
    # Write protection: a completed frozen release is never silently replaced.
    # A new release must be produced under a new versioned directory.
    existing = FROZEN / "FROZEN_RELEASE_MANIFEST.json"
    if existing.exists() and os.environ.get("CRFID_STRICT_DG_NEW_RELEASE_ID", "").strip() == "":
        raise RuntimeError(
            f"A frozen Strict-DG release already exists at {FROZEN}. Refusing to overwrite "
            "frozen scientific artifacts. To produce a new release, set "
            "CRFID_STRICT_DG_NEW_RELEASE_ID to a new release identifier and direct the "
            "output to a new versioned directory."
        )

    recipe_source = SOURCE_SELECTION / "selected_recipe.json"
    expected_recipe_hash = (SOURCE_SELECTION / "selected_recipe.sha256").read_text(encoding="ascii").split()[0]
    if sha256_file(recipe_source) != expected_recipe_hash:
        raise RuntimeError("Selected recipe changed before final source training")
    recipe = json.loads(recipe_source.read_text(encoding="utf-8"))
    if recipe["selected_candidate"] != "C1_FIRST_DIFFERENCE_ERM_1DCNN":
        raise RuntimeError("Source selector did not reproduce the required candidate")

    FROZEN.mkdir(parents=True, exist_ok=True)
    recipe_path = FROZEN / "recipe.json"
    shutil.copy2(recipe_source, recipe_path)
    (FROZEN / "recipe.sha256").write_text(expected_recipe_hash + "  recipe.json\n", encoding="ascii")
    if sha256_file(recipe_path) != expected_recipe_hash:
        raise RuntimeError("Frozen recipe copy hash mismatch")

    signals = np.load(SOURCE_INPUTS / "source_signals_float64.npy", allow_pickle=False)
    labels = np.load(SOURCE_INPUTS / "source_labels_int64.npy", allow_pickle=False)
    if signals.shape != (9450, 281) or signals.dtype.str != "<f8" or labels.dtype.str != "<i8":
        raise RuntimeError("Final source custody mismatch")
    differenced = np.ascontiguousarray(np.diff(signals, axis=1), dtype=np.float64)
    mean = np.ascontiguousarray(differenced.mean(axis=0, dtype=np.float64))
    raw_scale = np.ascontiguousarray(differenced.std(axis=0, ddof=0, dtype=np.float64))
    # Canonical degenerate-scale rule, stated once and sealed in the resolved
    # release specification. See crfid.strict_runtime.scale_policy.
    scale = apply_scale_policy(raw_scale, mode=CANONICAL_MODE)
    inputs = np.ascontiguousarray(((differenced - mean) / scale).astype(np.float32))

    preprocessing_dir = FROZEN / "preprocessing"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    mean_path = preprocessing_dir / "mean_float64.npy"
    scale_path = preprocessing_dir / "scale_float64.npy"
    np.save(mean_path, mean, allow_pickle=False)
    np.save(scale_path, scale, allow_pickle=False)
    preprocessing_semantic = {
        "schema_version": 1,
        "candidate_id": recipe["selected_candidate"],
        "fit_domains": ["P1", "P2", "P3"],
        "fit_sample_count": int(len(signals)),
        "operation_order": ["first_difference_without_padding", "featurewise_population_standardization", "float32_model_conversion"],
        "input_length": 281,
        "output_length": 280,
        "ddof": 0,
        "minimum_scale": 1e-12,
        "minimum_scale_replacement": 1.0,
        "zero_scale_replacement_count": int(np.count_nonzero(raw_scale < 1e-12)),
        "mean_array_sha256": array_sha256(mean),
        "scale_array_sha256": array_sha256(scale),
        "source_signal_file_sha256": sha256_file(SOURCE_INPUTS / "source_signals_float64.npy"),
        "target_used_for_fit": False,
        "target_refit_permitted": False,
    }
    preprocessing_state = {**preprocessing_semantic, "state_sha256": canonical_json_sha256(preprocessing_semantic)}
    state_path = preprocessing_dir / "state.json"
    write_json(state_path, preprocessing_state)

    partition = PartitionData(
        registry_rows=np.arange(len(labels), dtype=np.int64),
        inputs=inputs,
        labels=np.ascontiguousarray(labels, dtype=np.int64),
        sample_ids=[f"source_{index:05d}" for index in range(len(labels))],
        condition_ids=["source"] * len(labels),
        exact_signal_hashes=[""] * len(labels),
        unique_signal_weights=np.ones(len(labels), dtype=np.float64),
    )
    base_config = {
        "optimizer": recipe["optimizer"],
        "loss": {"weight": None, "label_smoothing": 0.0},
        "training": {"batch_size": recipe["batch_size"], "inner_stage_offset": 0, "outer_stage_offset": 100000},
    }
    checkpoints = []
    training_records = []
    checkpoint_dir = FROZEN / "checkpoints"
    for seed in recipe["seeds"]:
        epochs = int(recipe["final_epochs_by_seed"][str(seed)])
        model = initialize_model(int(seed))
        if parameter_count(model) != 142855:
            raise RuntimeError("Model parameter count changed")
        initial_hash = model_state_sha256(model)
        optimizer = _optimizer(model, base_config)
        criterion = _criterion(base_config)
        losses = []
        for epoch in range(1, epochs + 1):
            loss, _ = _erm_train_epoch(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                partition=partition,
                fold_id="ALL_P1_P2_P3",
                seed=int(seed),
                epoch=epoch,
                stage="outer_refit",
                base_config=base_config,
                initial_hash=initial_hash,
                candidate_id=recipe["selected_candidate"],
            )
            losses.append(float(loss["total_loss"]))
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        checkpoint = checkpoint_dir / f"seed_{seed}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "method_id": recipe["selected_candidate"],
            "seed": int(seed),
            "epochs": epochs,
            "recipe_sha256": expected_recipe_hash,
            "preprocessing_state_sha256": preprocessing_state["state_sha256"],
            "model_state_sha256": model_state_sha256(state),
            "model_state_dict": state,
            "source_domains": ["P1", "P2", "P3"],
            "target_used": False,
        }
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(checkpoint)
        loaded_model, loaded_payload = load_checkpoint_model(checkpoint, int(seed))
        if loaded_payload["recipe_sha256"] != expected_recipe_hash or model_state_sha256(loaded_model) != payload["model_state_sha256"]:
            raise RuntimeError(f"Checkpoint reload failed: seed {seed}")
        record = {**file_record(checkpoint), "seed": int(seed), "epochs": epochs, "model_state_sha256": payload["model_state_sha256"], "reload_verified": True}
        checkpoints.append(record)
        training_records.append({"seed": int(seed), "epochs": epochs, "initial_model_state_sha256": initial_hash, "final_loss": losses[-1], "checkpoint_sha256": record["sha256"]})
        print(json.dumps({"event": "final_source_model_complete", "seed": seed, "epochs": epochs, "checkpoint_sha256": record["sha256"]}, sort_keys=True), flush=True)

    code_files = sorted((PROJECT_ROOT / "src" / "crfid" / "strict_runtime").glob("*.py")) + [PROJECT_ROOT / "src" / "crfid" / "protocols" / "strict_dg.py", PROJECT_ROOT / "src" / "crfid" / "workflows" / "strict_dg.py", PROJECT_ROOT / "scripts" / "reproduce_strict_dg_source_selection.py", Path(__file__).resolve()]
    write_manifest(FROZEN / "code_manifest.csv", code_files)
    write_manifest(FROZEN / "config_manifest.csv", [CONFIG, SOURCE_SELECTION / "candidate_registry.json", SOURCE_SELECTION / "source_selection_decision.json"])
    write_manifest(FROZEN / "split_manifest.csv", [SOURCE_SELECTION / "source_fold_manifest.csv"])
    environment = {
        "python": sys.version.split()[0],
        "executable": "canonical_crfid_environment/Scripts/python.exe",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "recipe_sha256": expected_recipe_hash,
    }
    write_json(FROZEN / "environment.json", environment)
    authorization = {
        "schema_version": 1,
        "authorized": True,
        "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
        "recipe_sha256": expected_recipe_hash,
        "preprocessing_state_sha256": preprocessing_state["state_sha256"],
        "checkpoint_count": len(checkpoints),
        "all_checkpoint_reloads_verified": all(row["reload_verified"] for row in checkpoints),
        "source_selection_unit_count": 60,
        "source_selection_reproduced": True,
        "target_access_before_authorization": False,
        "prohibited_actions": ["training_on_target", "optimizer_step_on_target", "target_normalization_refit", "recipe_change", "seed_selection"],
    }
    authorization["authorization_sha256"] = canonical_json_sha256(authorization)
    write_json(FROZEN / "target_access_authorization.json", authorization)
    manifest = {
        "schema_version": 1,
        "status": "FROZEN_AND_TARGET_AUTHORIZED",
        "recipe": file_record(recipe_path),
        "recipe_sha256": expected_recipe_hash,
        "preprocessing": [file_record(mean_path), file_record(scale_path), file_record(state_path)],
        "checkpoints": checkpoints,
        "training_records": training_records,
        "code_manifest": file_record(FROZEN / "code_manifest.csv"),
        "config_manifest": file_record(FROZEN / "config_manifest.csv"),
        "split_manifest": file_record(FROZEN / "split_manifest.csv"),
        "environment": file_record(FROZEN / "environment.json"),
        "authorization": file_record(FROZEN / "target_access_authorization.json"),
        "target_accessed": False,
    }
    manifest["manifest_semantic_sha256"] = canonical_json_sha256(manifest)
    write_json(FROZEN / "FROZEN_RELEASE_MANIFEST.json", manifest)
    print(json.dumps({"status": "PASS_FROZEN", "recipe_sha256": expected_recipe_hash, "checkpoint_count": len(checkpoints), "target_authorized": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
