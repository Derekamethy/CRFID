"""Recompute and freeze the 60-unit strict source-only candidate selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crfid.strict_runtime.hashing import canonical_json_sha256, sha256_file
from crfid.strict_runtime.phase3b_execution import CANDIDATE_IDS, FOLDS, SEEDS
from crfid.strict_runtime.phase3b_summary import (
    candidate_comparison,
    collect_runs,
    deterministic_ranking,
    fold_summaries,
    public_seed_rows,
    write_csv,
)


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "strict_dg" / "source_selection"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_with_hash(source: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise RuntimeError(f"Copy hash mismatch: {destination}")
    return {"relative_path": destination.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256_file(destination), "size_bytes": destination.stat().st_size}


def build_fold_manifest(split_path: Path, destination: Path) -> None:
    with split_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts: dict[tuple[str, str], int] = {}
    seen: dict[str, set[int]] = {fold: set() for fold in FOLDS}
    for row in rows:
        key = (row["fold_id"], row["partition"])
        counts[key] = counts.get(key, 0) + 1
        index = int(row["registry_row"])
        if index in seen[row["fold_id"]]:
            raise RuntimeError(f"Fold leakage or duplicate assignment: {row['fold_id']}/{index}")
        seen[row["fold_id"]].add(index)
    expected = {"inner_train": 4200, "inner_validation": 2100, "outer_held": 3150}
    for fold in FOLDS:
        if len(seen[fold]) != 9450:
            raise RuntimeError(f"Incomplete fold: {fold}")
        for part, count in expected.items():
            if counts.get((fold, part)) != count:
                raise RuntimeError(f"Unexpected {fold}/{part} count")
    copy_with_hash(split_path, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-root", required=True, type=Path)
    parser.add_argument("--candidate-registry", required=True, type=Path)
    parser.add_argument("--selection-rule", required=True, type=Path)
    parser.add_argument("--source-split", required=True, type=Path)
    args = parser.parse_args()

    registry = read_json(args.candidate_registry)
    selection_rule = read_json(args.selection_rule)
    declared = tuple(row["candidate_id"] for row in registry["candidates"])
    if declared != CANDIDATE_IDS or tuple(selection_rule["candidate_ids"]) != CANDIDATE_IDS:
        raise RuntimeError("Frozen candidate registry or selector identity mismatch")
    if registry.get("p4_evidence_used") or selection_rule.get("p4_metrics_permitted"):
        raise RuntimeError("Target-informed registry or selector is prohibited")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    build_fold_manifest(args.source_split, OUTPUT_ROOT / "source_fold_manifest.csv")
    copy_with_hash(args.candidate_registry, OUTPUT_ROOT / "candidate_registry.json")

    records = collect_runs(args.execution_root)
    folds = fold_summaries(records)
    comparisons = candidate_comparison(records, folds)
    decision = deterministic_ranking(comparisons, selection_rule)
    winner = decision["provisional_source_only_winner"]
    selected = [row for row in records if row["candidate_id"] == winner]
    epochs_by_seed = {
        str(seed): int(statistics.median(row["selected_inner_epoch"] for row in selected if row["seed"] == seed))
        for seed in SEEDS
    }
    if len(selected) != 15:
        raise RuntimeError("Selected candidate does not have 15 fold/seed units")

    write_csv(OUTPUT_ROOT / "per_run_metrics.csv", public_seed_rows(records))
    write_csv(OUTPUT_ROOT / "per_fold_metrics.csv", folds)
    write_csv(OUTPUT_ROOT / "candidate_aggregate_metrics.csv", comparisons)
    decision.update(
        {
            "status": "SOURCE_ONLY_SELECTION_REPRODUCED",
            "selected_candidate": winner,
            "final_epochs_by_seed": epochs_by_seed,
            "canonical_unit_count": len(records),
            "candidate_count": len(CANDIDATE_IDS),
            "fold_count": len(FOLDS),
            "seed_count": len(SEEDS),
            "source_prediction_metrics_recomputed": True,
            "target_accessed": False,
        }
    )
    write_json(OUTPUT_ROOT / "source_selection_decision.json", decision)

    recipe = {
        "schema_version": 1,
        "protocol": "strict_source_only_domain_generalization",
        "selected_candidate": winner,
        "representation": "first_difference",
        "ordered_input_length": 281,
        "transformed_input_length": 280,
        "class_order": list(range(7)),
        "source_domains": ["P1", "P2", "P3"],
        "target_domain": "P4",
        "normalization": {"fit": "source_training_partition_only", "ddof": 0, "minimum_scale": 1e-12, "minimum_scale_replacement": 1.0, "dtype_fit": "float64", "dtype_model": "float32", "target_refit": False},
        "architecture": {"name": "NeutralSourceOnlyCNN1D", "channels": [64, 128, 256], "kernels": [7, 5, 3], "group_norm_groups": 8, "embedding_dimension": 256, "parameter_count": 142855},
        "initialization": {"conv_linear_weight": "kaiming_uniform_a_0_fan_in_relu", "conv_linear_bias": "uniform_inverse_sqrt_fan_in", "group_norm_weight": "ones", "group_norm_bias": "zeros"},
        "optimizer": {"name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001, "betas": [0.9, 0.999], "epsilon": 1e-8, "amsgrad": False, "foreach": False, "fused": False},
        "loss": "unweighted_cross_entropy",
        "batch_size": 256,
        "seeds": list(SEEDS),
        "final_epochs_by_seed": epochs_by_seed,
        "prediction": {"logit_dtype": "float32", "rule": "int64_argmax_lowest_index_tie"},
        "source_selection": {"unit_count": 60, "folds": list(FOLDS), "selected_without_target": True, "decision_sha256": sha256_file(OUTPUT_ROOT / "source_selection_decision.json")},
        "target_policy": {"access_before_freeze": False, "training_or_update_on_target": False, "normalization_refit": False},
        "frequency_axis_status": "PHYSICAL_FREQUENCY_AXIS_PROVENANCE_UNRESOLVED",
    }
    recipe_path = OUTPUT_ROOT / "selected_recipe.json"
    write_json(recipe_path, recipe)
    recipe_hash = sha256_file(recipe_path)
    (OUTPUT_ROOT / "selected_recipe.sha256").write_text(recipe_hash + "  selected_recipe.json\n", encoding="ascii")

    predictions_root = OUTPUT_ROOT / "predictions"
    for record in records:
        source = record["manifest_path"].parent / "outer_held_predictions.npz"
        destination = predictions_root / record["candidate_id"] / record["fold_id"] / f"seed_{record['seed']}.npz"
        copy_with_hash(source, destination)
    checkpoint_root = OUTPUT_ROOT / "checkpoints"
    for record in selected:
        source_record = record["manifest"]["checkpoints"]["outer"]
        source = Path(source_record["path"])
        destination = checkpoint_root / winner / record["fold_id"] / f"seed_{record['seed']}.pt"
        copy_with_hash(source, destination)
    preprocessing_source = args.execution_root / "preprocessing" / winner
    if preprocessing_source.is_dir():
        for source in preprocessing_source.rglob("*"):
            if source.is_file() and source.suffix == ".npy":
                copy_with_hash(source, OUTPUT_ROOT / "preprocessing_state" / source.relative_to(preprocessing_source))

    report = (
        "# Strict source-only selection report\n\n"
        f"All 60 candidate/fold/seed source prediction bundles were hash-verified and their metrics independently recomputed.\n\n"
        f"Selected candidate: `{winner}`.\n\n"
        f"Final epochs by seed: `{json.dumps(epochs_by_seed, sort_keys=True)}`.\n\n"
        f"Recipe SHA-256: `{recipe_hash}`. P4 was not accessed.\n"
    )
    (OUTPUT_ROOT / "SOURCE_SELECTION_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "PASS", "runs": len(records), "selected_candidate": winner, "recipe_sha256": recipe_hash, "epochs": epochs_by_seed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
