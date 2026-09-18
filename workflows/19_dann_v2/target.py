"""Strict two-stage P4 prediction freeze and one-time DANN v2 scoring."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from metrics import (
    aggregate_blocks,
    classification_metrics,
    stratified_paired_block_bootstrap,
    uniform_random_block_reference,
)
from protocol import ARM_A0, ARM_A1, ARM_A2, ARMS
from training import infer, load_final_model


METADATA_COLUMNS = ("A3", "A2", "A1", "P4", "P3", "P2", "P1", "ER", "TagID")
SIGNAL_COLUMNS = tuple(str(index) for index in range(281))
EXPECTED_HEADER = METADATA_COLUMNS + SIGNAL_COLUMNS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _canonical_files(p4_directory: Path) -> tuple[Path, ...]:
    paths = tuple(
        Path(p4_directory).resolve() / f"A{surface}_P4.csv"
        for surface in (1, 2, 3)
    )
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(
            "BLOCKED_GOVERNED_INPUTS: expected A1_P4.csv, A2_P4.csv, A3_P4.csv"
        )
    return paths


def load_target_features_without_labels(p4_directory: Path) -> dict:
    """Parse signals and opaque block IDs without reading TagID or ER."""

    signals = []
    block_ids = []
    access = []
    for path in _canonical_files(p4_directory):
        started = utc_now()
        file_hash = sha256_file(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != EXPECTED_HEADER:
                raise RuntimeError(f"Unexpected governed P4 schema: {path.name}")
            expected_surface = path.stem.split("_")[0]
            rows = 0
            for row_index, row in enumerate(reader):
                active_surface = [
                    name for name in ("A1", "A2", "A3") if float(row[name]) == 1.0
                ]
                active_position = [
                    name
                    for name in ("P1", "P2", "P3", "P4")
                    if float(row[name]) == 1.0
                ]
                if active_surface != [expected_surface] or active_position != ["P4"]:
                    raise RuntimeError("P4 feature/domain custody mismatch")
                # TagID and ER are intentionally not indexed before prediction freeze.
                signal = np.fromiter(
                    (float(row[column]) for column in SIGNAL_COLUMNS),
                    dtype=np.float64,
                    count=281,
                )
                if signal.shape != (281,) or not np.isfinite(signal).all():
                    raise RuntimeError("Invalid P4 signal")
                signals.append(signal)
                block_ids.append(f"{path.name}:opaque_block_{row_index // 50:02d}")
                rows += 1
        if rows != 1050:
            raise RuntimeError(f"Unexpected P4 row count: {path.name}/{rows}")
        access.append(
            {
                "stage": "FEATURES_ONLY_BEFORE_PREDICTION_FREEZE",
                "file_name": path.name,
                "sha256": file_hash,
                "rows": rows,
                "started_at_utc": started,
                "completed_at_utc": utc_now(),
                "tagid_column_accessed": False,
                "er_column_accessed": False,
            }
        )
    values = np.ascontiguousarray(np.vstack(signals), dtype=np.float64)
    blocks = np.asarray(block_ids, dtype=str)
    if (
        values.shape != (3150, 281)
        or len(np.unique(blocks)) != 63
        or set(Counter(blocks).values()) != {50}
    ):
        raise RuntimeError("P4 feature/block custody mismatch")
    return {"signals": values, "block_ids": blocks, "access": access}


def freeze_target_predictions(
    *,
    p4_directory: Path,
    checkpoint_root: Path,
    preprocessing_mean: Path,
    preprocessing_scale: Path,
    selection_path: Path,
    output_root: Path,
) -> dict:
    if not selection_path.is_file():
        raise RuntimeError("Final DANN v2 lambda is not frozen")
    selection_hash = sha256_file(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_lambda = float(selection["selected_lambda"])
    target = load_target_features_without_labels(p4_directory)
    mean = np.load(preprocessing_mean, allow_pickle=False)
    scale = np.load(preprocessing_scale, allow_pickle=False)
    inputs = np.ascontiguousarray(
        ((np.diff(target["signals"], axis=1) - mean) / scale).astype(np.float32)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    slugs = {ARM_A0: "a0", ARM_A1: "a1", ARM_A2: "a2"}
    for arm in ARMS:
        for seed in (42, 43, 44, 45, 46):
            checkpoint = checkpoint_root / slugs[arm] / f"seed_{seed}.pt"
            model, payload = load_final_model(checkpoint)
            expected_lambda = 0.0 if arm == ARM_A0 else selected_lambda
            if float(payload["lambda_max"]) != expected_lambda:
                raise RuntimeError("Final checkpoint lambda differs from frozen selection")
            output = infer(model, inputs)
            predictions = np.argmax(output["tag_logits"], axis=1).astype(np.int64)
            path = output_root / "prediction_bundles" / slugs[arm] / f"seed_{seed}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                path,
                predictions=predictions,
                logits=output["tag_logits"],
                opaque_block_ids=target["block_ids"],
            )
            records.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "lambda_max": float(payload["lambda_max"]),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "checkpoint_model_state_sha256": payload["model_state_sha256"],
                    "prediction_bundle_sha256": sha256_file(path),
                    "prediction_array_sha256": array_sha256(predictions),
                    "logit_array_sha256": array_sha256(output["tag_logits"]),
                    "prediction_bundle": str(path),
                    "prediction_frozen_before_label_access": True,
                }
            )
    manifest = {
        "schema_version": 2,
        "status": "P4_PREDICTIONS_FROZEN_LABELS_SEALED",
        "selection_file_sha256": selection_hash,
        "selected_lambda": selected_lambda,
        "feature_access": target["access"],
        "records": records,
        "prediction_count": len(records),
        "labels_accessed": False,
        "frozen_at_utc": utc_now(),
    }
    manifest_path = output_root / "P4_PREDICTION_FREEZE.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "P4_PREDICTION_FREEZE.sha256").write_text(
        sha256_file(manifest_path) + "  P4_PREDICTION_FREEZE.json\n",
        encoding="ascii",
    )
    return manifest


def load_target_labels_after_freeze(
    p4_directory: Path, freeze_manifest: Path
) -> dict:
    if not freeze_manifest.is_file():
        raise RuntimeError("FAIL_P4_LABEL_BOUNDARY: prediction-freeze manifest is absent")
    manifest = json.loads(freeze_manifest.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "P4_PREDICTIONS_FROZEN_LABELS_SEALED"
        or manifest.get("prediction_count") != 15
    ):
        raise RuntimeError("FAIL_P4_LABEL_BOUNDARY: prediction freeze is incomplete")
    for record in manifest["records"]:
        path = Path(record["prediction_bundle"])
        if sha256_file(path) != record["prediction_bundle_sha256"]:
            raise RuntimeError("FAIL_P4_LABEL_BOUNDARY: frozen prediction identity changed")
    labels = []
    ers = []
    surfaces = []
    block_ids = []
    access = []
    for path in _canonical_files(p4_directory):
        started = utc_now()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != EXPECTED_HEADER:
                raise RuntimeError("P4 schema changed after prediction freeze")
            expected_surface = path.stem.split("_")[0]
            rows = 0
            for row_index, row in enumerate(reader):
                tag_id = int(float(row["TagID"]))
                er = int(float(row["ER"]))
                if tag_id not in range(1, 8) or er not in {0, 1, 2}:
                    raise RuntimeError("Invalid P4 label metadata")
                labels.append(tag_id - 1)
                ers.append(er)
                surfaces.append(expected_surface)
                block_ids.append(f"{path.name}:opaque_block_{row_index // 50:02d}")
                rows += 1
        access.append(
            {
                "stage": "LABELS_OPENED_AFTER_ALL_PREDICTIONS_FROZEN",
                "file_name": path.name,
                "sha256": sha256_file(path),
                "rows": rows,
                "started_at_utc": started,
                "completed_at_utc": utc_now(),
                "tagid_column_accessed": True,
                "er_column_accessed": True,
            }
        )
    labels_array = np.asarray(labels, dtype=np.int64)
    blocks_array = np.asarray(block_ids, dtype=str)
    seen = set()
    for block in np.unique(blocks_array):
        selected = np.flatnonzero(blocks_array == block)
        keys = {
            (int(labels_array[index]), int(ers[index]), surfaces[index])
            for index in selected
        }
        if len(selected) != 50 or len(keys) != 1:
            raise RuntimeError("P4 opaque block does not map to one condition")
        seen.update(keys)
    if len(seen) != 63 or len(labels_array) != 3150:
        raise RuntimeError("P4 label/block structure differs")
    return {
        "labels": labels_array,
        "ers": np.asarray(ers, dtype=np.int64),
        "surfaces": np.asarray(surfaces, dtype=str),
        "block_ids": blocks_array,
        "access": access,
    }


def canonical_p4_uncertainty(
    block_predictions: dict[str, np.ndarray],
    block_true: np.ndarray,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict]:
    truth = np.asarray(block_true, dtype=np.int64)
    by_arm = {
        arm: np.asarray(block_predictions[arm], dtype=np.int64) for arm in ARMS
    }
    contrasts = (
        ("A2-A0", ARM_A0, ARM_A2),
        ("A2-A1", ARM_A1, ARM_A2),
        ("A1-A0", ARM_A0, ARM_A1),
    )
    uncertainty = []
    for name, reference, treatment in contrasts:
        for metric in ("macro_f1", "accuracy"):
            for resample_seeds in (False, True):
                interval = stratified_paired_block_bootstrap(
                    by_arm[reference],
                    by_arm[treatment],
                    truth,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed,
                    resample_training_seeds=resample_seeds,
                    metric=metric,
                )
                uncertainty.append(
                    {
                        "contrast": name,
                        "endpoint": f"P4_block_{metric}",
                        "family": (
                            "tagid_stratified_block_plus_training_seed"
                            if resample_seeds
                            else "tagid_stratified_block_only"
                        ),
                        "point_estimate": interval["point_estimate"],
                        "lower_95": interval["interval"][0],
                        "upper_95": interval["interval"][1],
                        "replicates": interval["replicates"],
                        "rng_seed": int(bootstrap_seed),
                        "interval_definition": "percentile_95",
                        "stratification": "TagID_condition_blocks",
                    }
                )
    return uncertainty


def score_target_once(
    *,
    p4_directory: Path,
    freeze_manifest: Path,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20_260_809,
    chance_replicates: int = 10_000,
    chance_seed: int = 20_260_810,
) -> dict:
    manifest = json.loads(freeze_manifest.read_text(encoding="utf-8"))
    target = load_target_labels_after_freeze(p4_directory, freeze_manifest)
    by_arm: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    seed_metrics = []
    per_class_rows = []
    for record in manifest["records"]:
        arm = record["arm"]
        with np.load(record["prediction_bundle"], allow_pickle=False) as bundle:
            predictions = np.asarray(bundle["predictions"], dtype=np.int64)
            block_ids = np.asarray(bundle["opaque_block_ids"], dtype=str)
        if not np.array_equal(block_ids, target["block_ids"]):
            raise RuntimeError("Frozen P4 opaque block order changed")
        row_metrics = classification_metrics(target["labels"], predictions, class_count=7)
        blocks = aggregate_blocks(
            target["labels"], predictions, block_ids, class_count=7
        )
        scored = {
            "arm": arm,
            "seed": int(record["seed"]),
            "block_predictions": blocks["block_predicted"],
            "block_true": blocks["block_true"],
            "row_metrics": row_metrics,
            "block_metrics": blocks["metrics"],
        }
        by_arm[arm].append(scored)
        seed_metrics.append(
            {
                "arm": arm,
                "seed": int(record["seed"]),
                "p4_row_macro_f1": row_metrics["macro_f1"],
                "p4_row_accuracy": row_metrics["accuracy"],
                "p4_block_macro_f1": blocks["metrics"]["macro_f1"],
                "p4_block_accuracy": blocks["metrics"]["accuracy"],
                "row_confusion_matrix": row_metrics["confusion_matrix"],
                "block_confusion_matrix": blocks["metrics"]["confusion_matrix"],
            }
        )
        for class_index in range(7):
            per_class_rows.append(
                {
                    "dataset": "P4",
                    "arm": arm,
                    "seed": int(record["seed"]),
                    "class_index": class_index,
                    "f1": row_metrics["per_class_f1"][class_index],
                    "precision": row_metrics["per_class_precision"][class_index],
                    "recall": row_metrics["per_class_recall"][class_index],
                }
            )
    for arm in ARMS:
        by_arm[arm].sort(key=lambda row: row["seed"])
        if len(by_arm[arm]) != 5:
            raise RuntimeError(f"Incomplete P4 arm: {arm}")
    contrasts = (
        ("A2-A0", ARM_A0, ARM_A2),
        ("A2-A1", ARM_A1, ARM_A2),
        ("A1-A0", ARM_A0, ARM_A1),
    )
    paired = []
    for name, reference, treatment in contrasts:
        for left, right in zip(by_arm[reference], by_arm[treatment], strict=True):
            paired.append(
                {
                    "contrast": name,
                    "seed": left["seed"],
                    "p4_block_macro_f1": right["block_metrics"]["macro_f1"]
                    - left["block_metrics"]["macro_f1"],
                    "p4_block_accuracy": right["block_metrics"]["accuracy"]
                    - left["block_metrics"]["accuracy"],
                    "p4_row_macro_f1": right["row_metrics"]["macro_f1"]
                    - left["row_metrics"]["macro_f1"],
                    "p4_row_accuracy": right["row_metrics"]["accuracy"]
                    - left["row_metrics"]["accuracy"],
                }
            )
    block_predictions = {
        arm: np.vstack([row["block_predictions"] for row in by_arm[arm]])
        for arm in ARMS
    }
    block_true = by_arm[ARM_A0][0]["block_true"]
    uncertainty = canonical_p4_uncertainty(
        block_predictions,
        block_true,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    chance = uniform_random_block_reference(
        block_true,
        replicates=chance_replicates,
        seed=chance_seed,
        class_count=7,
    )
    return {
        "schema_version": 2,
        "label_access": target["access"],
        "seed_metrics": seed_metrics,
        "per_class": per_class_rows,
        "paired": paired,
        "uncertainty": uncertainty,
        "chance_reference": chance,
        "block_predictions": {
            arm: values.tolist() for arm, values in block_predictions.items()
        },
        "block_true": block_true.tolist(),
        "uncertainty_protocol": {
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
            "interval": "percentile_95",
            "p4_stratification": "TagID_condition_blocks",
            "contrast_specific_seed_offsets": False,
        },
        "p4_runs_completed": 15,
        "scored_once": True,
        "row_counts_used_as_independent_inference_units": False,
    }
