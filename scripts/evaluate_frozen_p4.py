"""One-time, evaluation-only P4 command for the frozen strict-DG release."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crfid.governance.strict_target_authorization import authorize_target_access
from crfid.protocols.strict_dg import StrictDGProtocol
from crfid.strict_runtime.hashing import array_sha256, canonical_json_sha256, sha256_file
from crfid.strict_runtime.target_evaluation import (
    evaluate_logits,
    infer_checkpoint,
    load_p4_once,
    predict_from_logits,
    save_prediction_bundle,
    transform_p4,
)


FROZEN = PROJECT_ROOT / "outputs" / "strict_dg" / "frozen_release"
OUTPUT = PROJECT_ROOT / "outputs" / "strict_dg" / "final_p4"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p4-custody-manifest", required=True, type=Path)
    parser.add_argument("--source-schema", required=True, type=Path)
    parser.add_argument("--authoritative-seed-results", required=True, type=Path)
    args = parser.parse_args()

    authorization_path = FROZEN / "target_access_authorization.json"
    authorization = read_json(authorization_path)
    release = read_json(FROZEN / "FROZEN_RELEASE_MANIFEST.json")
    recipe_hash = (FROZEN / "recipe.sha256").read_text(encoding="ascii").split()[0]
    if not authorization.get("authorized") or authorization["checkpoint_count"] != 5:
        raise RuntimeError("Target access is not authorized")
    if sha256_file(FROZEN / "recipe.json") != recipe_hash or authorization["recipe_sha256"] != recipe_hash:
        raise RuntimeError("Frozen recipe changed")
    checkpoint_hashes = {int(row["seed"]): row for row in release["checkpoints"]}
    for seed, row in checkpoint_hashes.items():
        path = FROZEN / "checkpoints" / f"seed_{seed}.pt"
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"Frozen checkpoint changed: {seed}")

    # Machine-enforced target authorization. This re-derives the recipe,
    # preprocessing-state and all five checkpoint digests from disk, drives the
    # StrictDGProtocol gate, and mints the only object that unlocks the target
    # loader. No boolean flag and no default authorization exists.
    protocol = StrictDGProtocol()
    token = authorize_target_access(frozen_release_directory=FROZEN, protocol=protocol)

    # This acceptance rule is serialized before the first numerical P4 load.
    # It is written outside the frozen release directory so that evaluation
    # never adds unmanifested files to a frozen scientific release.
    tolerance = {
        "schema_version": 1,
        "declared_before_p4_numerical_load": True,
        "prediction_array_expectation": "exact_sha256_identity",
        "logit_array_expectation": "exact_sha256_identity",
        "metric_absolute_tolerance": 0.0,
        "metric_relative_tolerance": 0.0,
        "mismatch_action": "FAIL_STRICT_DG_FINAL_TARGET_REPRODUCTION_MISMATCH",
    }
    tolerance_path = OUTPUT / "reproduction_tolerance.json"
    write_json(tolerance_path, tolerance)

    custody = read_json(args.p4_custody_manifest)
    schema = read_json(args.source_schema)
    p4 = load_p4_once(custody_manifest=custody, source_config=schema, token=token, phase=6)
    mean = np.load(FROZEN / "preprocessing" / "mean_float64.npy", allow_pickle=False)
    scale = np.load(FROZEN / "preprocessing" / "scale_float64.npy", allow_pickle=False)
    preprocessing_before = (sha256_file(FROZEN / "preprocessing" / "mean_float64.npy"), sha256_file(FROZEN / "preprocessing" / "scale_float64.npy"))
    inputs = transform_p4(p4, mean, scale)
    if inputs.shape != (3150, 280) or inputs.dtype != np.float32:
        raise RuntimeError("Frozen P4 transform custody mismatch")

    with args.authoritative_seed_results.open("r", encoding="utf-8", newline="") as handle:
        authoritative = {int(row["seed"]): row for row in csv.DictReader(handle)}
    seed_rows = []
    class_rows = []
    prediction_rows = []
    exact_matches = []
    for seed in sorted(checkpoint_hashes):
        checkpoint = FROZEN / "checkpoints" / f"seed_{seed}.pt"
        record = checkpoint_hashes[seed]
        logits = infer_checkpoint(checkpoint_path=checkpoint, seed=seed, inputs=inputs, expected_state_sha256=record["model_state_sha256"])
        # Predictions are derived and serialized before any target label is
        # released, so labels cannot influence the predictions they score.
        predictions = predict_from_logits(logits)
        prediction_path = OUTPUT / "predictions" / f"seed_{seed}.npz"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = save_prediction_bundle(path=prediction_path, dataset=p4, logits=logits, predictions=predictions, token=token)
        metrics, scored_predictions, _ = evaluate_logits(p4, logits, token=token)
        if not np.array_equal(scored_predictions, predictions):
            raise RuntimeError(f"Scored predictions differ from the serialized bundle: seed {seed}")
        sample = metrics["sample"]
        row = {
            "seed": seed,
            "epochs": int(record["epochs"]),
            "accuracy": sample["accuracy"],
            "macro_f1": sample["macro_f1"],
            "condition_accuracy": metrics["condition"]["accuracy"],
            "condition_macro_f1": metrics["condition"]["macro_f1"],
            "unique_signal_weighted_accuracy": metrics["unique_signal_weighted"]["accuracy"],
            "unique_signal_weighted_macro_f1": metrics["unique_signal_weighted"]["macro_f1"],
            "worst_class_recall": sample["worst_class_recall"],
            "zero_recall_class_count": sample["zero_recall_class_count"],
            "prediction_array_sha256": bundle["predictions_array_sha256"],
            "logits_array_sha256": bundle["logits_array_sha256"],
        }
        expected = authoritative[seed]
        exact = (
            row["prediction_array_sha256"] == expected["p4_prediction_array_sha256"]
            and row["logits_array_sha256"] == expected["p4_logits_array_sha256"]
            and row["accuracy"] == float(expected["sample_accuracy"])
            and row["macro_f1"] == float(expected["sample_macro_f1"])
        )
        row["authoritative_exact_match"] = exact
        exact_matches.append(exact)
        seed_rows.append(row)
        for class_index, recall in enumerate(sample["per_class_recall"]):
            class_rows.append({"seed": seed, "class_index": class_index, "recall": recall})
        confusion_path = OUTPUT / "confusion_matrices" / f"seed_{seed}.json"
        write_json(confusion_path, {"seed": seed, "sample": sample["confusion_matrix"], "condition": metrics["condition"]["confusion_matrix"], "unique_signal_weighted": metrics["unique_signal_weighted"]["confusion_matrix"]})
        prediction_rows.append({"seed": seed, "relative_path": prediction_path.relative_to(PROJECT_ROOT).as_posix(), "size_bytes": prediction_path.stat().st_size, "file_sha256": sha256_file(prediction_path), "predictions_array_sha256": bundle["predictions_array_sha256"], "logits_array_sha256": bundle["logits_array_sha256"]})
        print(json.dumps({"event": "p4_seed_complete", "seed": seed, "accuracy": row["accuracy"], "macro_f1": row["macro_f1"], "exact_match": exact}, sort_keys=True), flush=True)

    write_csv(OUTPUT / "per_seed_metrics.csv", seed_rows)
    write_csv(OUTPUT / "class_recall.csv", class_rows)
    write_csv(OUTPUT / "prediction_manifest.csv", prediction_rows)
    metrics_to_aggregate = ["accuracy", "macro_f1", "condition_accuracy", "condition_macro_f1", "unique_signal_weighted_accuracy", "unique_signal_weighted_macro_f1"]
    aggregate = {
        "schema_version": 1,
        "seed_count": 5,
        "equal_seed_weight": True,
        "ddof": 0,
        "metrics": {
            metric: {"mean": float(statistics.fmean(row[metric] for row in seed_rows)), "population_sd": float(statistics.pstdev(row[metric] for row in seed_rows))}
            for metric in metrics_to_aggregate
        },
        "all_per_seed_predictions_and_metrics_exactly_match_authoritative": all(exact_matches),
        "recipe_sha256": recipe_hash,
    }
    write_json(OUTPUT / "aggregate_metrics.json", aggregate)

    preprocessing_after = (sha256_file(FROZEN / "preprocessing" / "mean_float64.npy"), sha256_file(FROZEN / "preprocessing" / "scale_float64.npy"))
    checkpoint_after = {seed: sha256_file(FROZEN / "checkpoints" / f"seed_{seed}.pt") for seed in checkpoint_hashes}
    integrity = {
        "schema_version": 1,
        "status": "PASS" if all(exact_matches) else "FAIL",
        "recipe_sha256": recipe_hash,
        "authorization_sha256": sha256_file(authorization_path),
        "tolerance_declaration_sha256": sha256_file(tolerance_path),
        "p4_file_access_count": 3,
        "p4_files": [{"file_name": row["file_name"], "sha256": row["streaming_sha256"], "bytes_read": row["bytes_read"], "rows_parsed": row["rows_parsed"]} for row in p4.access_records],
        "p4_accessed_only_after_freeze": True,
        "p4_training_steps": 0,
        "optimizer_instances_created": 0,
        "target_preprocessing_fit_count": 0,
        "preprocessing_unchanged": preprocessing_before == preprocessing_after,
        "checkpoints_unchanged": all(checkpoint_after[seed] == checkpoint_hashes[seed]["sha256"] for seed in checkpoint_hashes),
        "all_authoritative_exact_matches": all(exact_matches),
        "prediction_manifest_sha256": sha256_file(OUTPUT / "prediction_manifest.csv"),
        "aggregate_metrics_sha256": sha256_file(OUTPUT / "aggregate_metrics.json"),
        "further_target_informed_development_permitted": False,
        "target_access_machine_enforced": True,
        "target_access_token": token.as_record(),
    }
    integrity["integrity_semantic_sha256"] = canonical_json_sha256(integrity)
    write_json(OUTPUT / "FINAL_P4_INTEGRITY.json", integrity)
    report = (
        "# Final frozen P4 evaluation\n\n"
        "Five independently frozen source-only models were evaluated without fitting, calibration, adaptation, optimizer steps, or checkpoint changes.\n\n"
        f"Accuracy: `{aggregate['metrics']['accuracy']['mean']}` ± `{aggregate['metrics']['accuracy']['population_sd']}` (population SD).\n\n"
        f"Macro-F1: `{aggregate['metrics']['macro_f1']['mean']}` ± `{aggregate['metrics']['macro_f1']['population_sd']}` (population SD).\n\n"
        f"Exact authoritative prediction, logit, Accuracy, and Macro-F1 match for all seeds: `{all(exact_matches)}`.\n"
    )
    (OUTPUT / "FINAL_P4_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": integrity["status"], "accuracy": aggregate["metrics"]["accuracy"], "macro_f1": aggregate["metrics"]["macro_f1"], "exact_match": all(exact_matches)}, sort_keys=True))
    return 0 if all(exact_matches) else 2


if __name__ == "__main__":
    raise SystemExit(main())
