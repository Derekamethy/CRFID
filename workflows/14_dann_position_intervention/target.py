"""Strict two-stage P4 feature release, prediction freeze, and one-time scoring."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from metrics import aggregate_blocks, classification_metrics, safe_correlation, stratified_paired_block_bootstrap
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
    paths = tuple(Path(p4_directory).resolve() / f"A{surface}_P4.csv" for surface in (1, 2, 3))
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("BLOCKED_GOVERNED_INPUTS: expected A1_P4.csv, A2_P4.csv, A3_P4.csv")
    return paths


def load_target_features_without_labels(p4_directory: Path) -> dict:
    """Parse only signals, one-hot domain metadata, and opaque contiguous block IDs."""

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
            rows = 0
            expected_surface = path.stem.split("_")[0]
            for row_index, row in enumerate(reader):
                active_surface = [name for name in ("A1", "A2", "A3") if float(row[name]) == 1.0]
                active_position = [name for name in ("P1", "P2", "P3", "P4") if float(row[name]) == 1.0]
                if active_surface != [expected_surface] or active_position != ["P4"]:
                    raise RuntimeError("P4 feature/domain custody mismatch")
                # Deliberately do not read row['TagID'] or row['ER'] here. Opaque
                # contiguous groups are sufficient to freeze row and block predictions.
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
    if values.shape != (3150, 281) or len(np.unique(blocks)) != 63 or set(Counter(blocks).values()) != {50}:
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
        raise RuntimeError("Final nonzero lambda is not frozen")
    selection_hash = sha256_file(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_lambda = float(selection["selected_nonzero_lambda"])
    target = load_target_features_without_labels(p4_directory)
    mean = np.load(preprocessing_mean, allow_pickle=False)
    scale = np.load(preprocessing_scale, allow_pickle=False)
    differenced = np.diff(target["signals"], axis=1)
    inputs = np.ascontiguousarray(((differenced - mean) / scale).astype(np.float32))
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for method in ("ERM", "DANN"):
        for seed in (42, 43, 44, 45, 46):
            checkpoint = checkpoint_root / method.lower() / f"seed_{seed}.pt"
            model, payload = load_final_model(checkpoint)
            if method == "DANN" and float(payload["lambda_max"]) != selected_lambda:
                raise RuntimeError("Final DANN checkpoint lambda differs from frozen selection")
            output = infer(model, inputs)
            predictions = np.argmax(output["tag_logits"], axis=1).astype(np.int64)
            path = output_root / "prediction_bundles" / method.lower() / f"seed_{seed}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                path,
                predictions=predictions,
                logits=output["tag_logits"],
                opaque_block_ids=target["block_ids"],
            )
            records.append(
                {
                    "method": method,
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
        "schema_version": 1,
        "status": "P4_PREDICTIONS_FROZEN_LABELS_SEALED",
        "selection_file_sha256": selection_hash,
        "selected_nonzero_lambda": selected_lambda,
        "feature_access": target["access"],
        "records": records,
        "prediction_count": len(records),
        "labels_accessed": False,
        "frozen_at_utc": utc_now(),
    }
    manifest_path = output_root / "P4_PREDICTION_FREEZE.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sidecar = output_root / "P4_PREDICTION_FREEZE.sha256"
    sidecar.write_text(sha256_file(manifest_path) + "  P4_PREDICTION_FREEZE.json\n", encoding="ascii")
    return manifest


def load_target_labels_after_freeze(p4_directory: Path, freeze_manifest: Path) -> dict:
    if not freeze_manifest.is_file():
        raise RuntimeError("FAIL_P4_LABEL_BOUNDARY: prediction-freeze manifest is absent")
    manifest = json.loads(freeze_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "P4_PREDICTIONS_FROZEN_LABELS_SEALED" or manifest.get("prediction_count") != 10:
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
        keys = {(int(labels_array[index]), int(ers[index]), surfaces[index]) for index in selected}
        if len(selected) != 50 or len(keys) != 1:
            raise RuntimeError("P4 opaque block does not map to one TagID x ER x surface unit")
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


def score_target_once(
    *,
    p4_directory: Path,
    freeze_manifest: Path,
    output_root: Path,
    bootstrap_replicates: int = 10_000,
) -> dict:
    manifest = json.loads(freeze_manifest.read_text(encoding="utf-8"))
    target = load_target_labels_after_freeze(p4_directory, freeze_manifest)
    by_method: dict[str, list[dict]] = {"ERM": [], "DANN": []}
    compact_seed_rows = []
    per_class_rows = []
    histograms = []
    for record in manifest["records"]:
        with np.load(record["prediction_bundle"], allow_pickle=False) as bundle:
            predictions = np.asarray(bundle["predictions"], dtype=np.int64)
            block_ids = np.asarray(bundle["opaque_block_ids"], dtype=str)
        if not np.array_equal(block_ids, target["block_ids"]):
            raise RuntimeError("Frozen P4 opaque block order changed")
        row_metrics = classification_metrics(target["labels"], predictions, class_count=7)
        blocks = aggregate_blocks(target["labels"], predictions, block_ids, class_count=7)
        agreements = [row["within_block_agreement"] for row in blocks["rows"]]
        scored = {
            "method": record["method"],
            "seed": int(record["seed"]),
            "predictions": predictions,
            "block_predictions": blocks["block_predicted"],
            "block_true": blocks["block_true"],
            "row_metrics": row_metrics,
            "block_metrics": blocks["metrics"],
            "within_block_agreement_mean": float(np.mean(agreements)),
        }
        by_method[record["method"]].append(scored)
        compact_seed_rows.append(
            {
                "method": record["method"],
                "seed": int(record["seed"]),
                "block_macro_f1": blocks["metrics"]["macro_f1"],
                "block_accuracy": blocks["metrics"]["accuracy"],
                "row_macro_f1": row_metrics["macro_f1"],
                "row_accuracy": row_metrics["accuracy"],
                "within_block_agreement_mean": float(np.mean(agreements)),
                "confusion_matrix": blocks["metrics"]["confusion_matrix"],
            }
        )
        for class_index in range(7):
            per_class_rows.append(
                {
                    "method": record["method"],
                    "seed": int(record["seed"]),
                    "class_index": class_index,
                    "precision": row_metrics["per_class_precision"][class_index],
                    "recall": row_metrics["per_class_recall"][class_index],
                    "f1": row_metrics["per_class_f1"][class_index],
                }
            )
        histograms.append(
            {
                "method": record["method"],
                "seed": int(record["seed"]),
                "counts": np.bincount(predictions, minlength=7).tolist(),
            }
        )
    for method in by_method:
        by_method[method].sort(key=lambda row: row["seed"])
    erm_blocks = np.vstack([row["block_predictions"] for row in by_method["ERM"]])
    dann_blocks = np.vstack([row["block_predictions"] for row in by_method["DANN"]])
    true_blocks = by_method["ERM"][0]["block_true"]
    block_only = stratified_paired_block_bootstrap(
        erm_blocks,
        dann_blocks,
        true_blocks,
        replicates=bootstrap_replicates,
        resample_training_seeds=False,
    )
    block_plus_seed = stratified_paired_block_bootstrap(
        erm_blocks,
        dann_blocks,
        true_blocks,
        replicates=bootstrap_replicates,
        seed=20_260_805,
        resample_training_seeds=True,
    )
    accuracy_block_only = stratified_paired_block_bootstrap(
        erm_blocks,
        dann_blocks,
        true_blocks,
        replicates=bootstrap_replicates,
        seed=20_260_808,
        resample_training_seeds=False,
        metric="accuracy",
    )
    accuracy_block_plus_seed = stratified_paired_block_bootstrap(
        erm_blocks,
        dann_blocks,
        true_blocks,
        replicates=bootstrap_replicates,
        seed=20_260_809,
        resample_training_seeds=True,
        metric="accuracy",
    )
    paired_rows = []
    p4_changes = []
    for erm, dann in zip(by_method["ERM"], by_method["DANN"]):
        changes = int(np.count_nonzero(erm["predictions"] != dann["predictions"]))
        block_changes = int(np.count_nonzero(erm["block_predictions"] != dann["block_predictions"]))
        contrast = dann["block_metrics"]["macro_f1"] - erm["block_metrics"]["macro_f1"]
        p4_changes.append(contrast)
        paired_rows.append(
            {
                "seed": erm["seed"],
                "block_macro_f1_change": contrast,
                "block_accuracy_change": dann["block_metrics"]["accuracy"] - erm["block_metrics"]["accuracy"],
                "row_prediction_changes": changes,
                "block_prediction_changes": block_changes,
            }
        )
    leave_one_out = []
    for omitted in range(5):
        retained = [value for index, value in enumerate(p4_changes) if index != omitted]
        leave_one_out.append({"omitted_seed": by_method["ERM"][omitted]["seed"], "mean_change": float(np.mean(retained))})
    seed_agreement = []
    for method, matrix in (("ERM", erm_blocks), ("DANN", dann_blocks)):
        agreements = []
        for block_index in range(matrix.shape[1]):
            counts = np.bincount(matrix[:, block_index], minlength=7)
            agreements.append(float(counts.max() / matrix.shape[0]))
        seed_agreement.append(
            {
                "method": method,
                "mean_block_seed_agreement": float(np.mean(agreements)),
                "minimum_block_seed_agreement": float(np.min(agreements)),
            }
        )
    return {
        "label_access": target["access"],
        "seed_metrics": compact_seed_rows,
        "per_class": per_class_rows,
        "histograms": histograms,
        "paired": paired_rows,
        "block_only_bootstrap": block_only,
        "block_plus_training_seed_bootstrap": block_plus_seed,
        "accuracy_block_only_bootstrap": accuracy_block_only,
        "accuracy_block_plus_training_seed_bootstrap": accuracy_block_plus_seed,
        "leave_one_training_seed_out": leave_one_out,
        "seed_agreement": seed_agreement,
        "erm_block_predictions": erm_blocks.tolist(),
        "dann_block_predictions": dann_blocks.tolist(),
        "block_true": true_blocks.tolist(),
        "scored_once": True,
        "row_counts_used_as_independent_inference_units": False,
    }
