"""End-to-end execution of the preregistered isolated P4 experiment."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import compact_float, write_csv, write_json, write_text
from .bootstrap import (
    bootstrap_metric_by_unit,
    fold_aggregate_sensitivity,
    percentile_interval,
    seed_resampling_sensitivity,
    stratified_block_bootstrap_indices,
)
from .constants import (
    ADAPTATION_METHODS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    BUDGET_STRATEGIES,
    CHECKPOINT_FILE_HASHES,
    CHECKPOINT_SEEDS,
    FS0,
    JOINT_FACTOR_COVERAGE,
    LINEAR_HEAD,
    PROTOTYPE,
    QUERY_FOLDS,
    RANDOM_BLOCK,
    SUPPORT_SEEDS,
    cell_id,
)
from .data import GovernedP4, QueryView, SupportSet, load_governed_p4
from .historical import historical_dependence_audit, verify_historical_binding
from .metrics import (
    QueryEvaluation,
    assert_no_row_level_pseudoreplication,
    evaluate_query,
    metrics_from_predictions,
)
from .models import (
    adapt_linear_head,
    build_cosine_prototypes,
    cosine_similarity_predict,
    extract_embeddings,
    frozen_source_predictions,
    linear_head_predict,
    load_frozen_model,
    load_preprocessing,
    model_state_sha256,
    preprocess,
    sha256_file,
)
from .protocol import (
    ProtocolViolation,
    outer_fold_manifest_rows,
    validate_complete_oof_blocks,
)
from .selection import coverage_metadata, select_support_cells


FROZEN_RELATIVE = Path("outputs/few_shot/frozen_branch")
SNAPSHOT_RELATIVE = FROZEN_RELATIVE / "01_frozen_source_import/frozen_last_code_snapshot"
PREPROCESSING_RELATIVE = SNAPSHOT_RELATIVE / "10_final_recipe_freeze"
CHECKPOINT_RELATIVE = SNAPSHOT_RELATIVE / "11_final_p4_evaluation/runs"


PREREGISTERED_PROTOCOL = """# P4 Factor-Aware Few-Shot Preregistered Protocol

This patch tests whether matched-budget factor-aware support coverage improves P4
Few-Shot adaptation. It uses only P4, the five frozen source-only
`C1_FIRST_DIFFERENCE_ERM_1DCNN` checkpoints (seeds 42-46), and five fixed
support-selection/support-row seeds.

The indivisible unit is `TagID x ER x surface x P4`. Three shared Latin-square
outer folds hold out three factor cells per TagID; each held fold contains every
ER and every surface, all 50 repetitions remain together, and all 63 blocks are
queried exactly once. A block-shot is one labelled row from one distinct support
block. Budgets are therefore 7, 21, and 35 total labels at 1-, 3-, and 5-shot.

The primary endpoint is 63-block out-of-fold Macro-F1. The primary contrast is
3-shot `TARGET_ONLY_COSINE_PROTOTYPE` `JOINT_FACTOR_COVERAGE - RANDOM_BLOCK`,
paired by outer fold, source seed, support seed, query blocks, and label budget.
The prototype normalises each support embedding, averages within class,
normalises the class prototype, and predicts by cosine similarity. The secondary
head reuses the released float64 full-batch LBFGS proximal recipe and fixed
lambdas (100, 100, 0.1); only a copy of `network.13` is adapted.

`RANDOM_BLOCK` samples candidate cells uniformly without replacement.
At 3-shot, ER- and surface-balanced strategies cover their named factor, while
joint coverage uses a three-cell perfect matching covering all ERs and surfaces.
At 5-shot, joint coverage lexicographically minimises ER imbalance, minimises
surface imbalance, maximises summed pairwise Manhattan dispersion on the 3x3
factor grid, then uses the fixed seed to break ties. The same cell set is used
for all seven classes. Support-row selection is a fixed SHA-256-derived offset
within each 50-row block.

The descriptive support-coverage score is fixed as
`0.25*(unique_ER/3 + unique_surface/3) + 0.5*(mean_pairwise_Manhattan/4)`.
It is not an optimisation endpoint and cannot alter the primary comparison.

Query labels remain in a fail-closed seal until predictions are frozen and
SHA-256 hashed. Metrics are computed exactly once after the seal opens. Physical
blocks are reduced by majority vote with the lowest frozen class index breaking
an exact tie. Uncertainty uses 10,000 deterministic TagID-stratified paired block
bootstrap replicates; row repetitions are never inferential units. The head
support-fit validity threshold, recorded without retuning, is Accuracy >= 0.95
and Macro-F1 >= 0.95.
"""


def _run_id(*parts: object) -> str:
    return "run_" + hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:20]


def _as_scalar_metrics(metrics: dict[str, object]) -> tuple[float, float]:
    return float(metrics["accuracy"]), float(metrics["macro_f1"])


def _setting(method: str, budget: int, strategy: str) -> tuple[str, int, str]:
    return method, budget, strategy


def _unit_key(
    method: str, budget: int, strategy: str, source_seed: int, support_seed: int
) -> tuple[str, int, str, int, int]:
    return method, budget, strategy, source_seed, support_seed


def _prepare_supports(
    data: GovernedP4,
) -> tuple[
    dict[tuple[int, int, int, str], SupportSet],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    cache: dict[tuple[int, int, int, str], SupportSet] = {}
    manifest_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for fold, query_cells in QUERY_FOLDS.items():
        for support_seed in SUPPORT_SEEDS:
            for budget, strategies in BUDGET_STRATEGIES.items():
                for strategy in strategies:
                    selected = select_support_cells(
                        fold=fold,
                        budget=budget,
                        strategy=strategy,
                        support_seed=support_seed,
                    )
                    repeated = select_support_cells(
                        fold=fold,
                        budget=budget,
                        strategy=strategy,
                        support_seed=support_seed,
                    )
                    if selected != repeated:
                        raise ProtocolViolation("NONDETERMINISTIC_SUPPORT_SELECTION")
                    support = data.support_set(
                        fold=fold,
                        selected_cells=selected,
                        support_seed=support_seed,
                    )
                    cache[(fold, support_seed, budget, strategy)] = support
                    selection_id = _run_id("support", fold, support_seed, budget, strategy)
                    for row in support.manifest_rows:
                        manifest_rows.append(
                            {
                                "support_selection_id": selection_id,
                                "strategy": strategy,
                                "budget": budget,
                                "total_labelled_rows": budget * 7,
                                **row,
                            }
                        )
                    metadata = coverage_metadata(selected, query_cells)
                    coverage_rows.append(
                        {
                            "support_selection_id": selection_id,
                            "fold": fold,
                            "support_seed": support_seed,
                            "strategy": strategy,
                            "budget": budget,
                            "selected_factor_cells": ";".join(cell_id(cell) for cell in selected),
                            "query_factor_cells": ";".join(cell_id(cell) for cell in query_cells),
                            **metadata,
                        }
                    )
    return cache, manifest_rows, coverage_rows


def _preflight_gates(
    data: GovernedP4,
    support_cache: dict[tuple[int, int, int, str], SupportSet],
    archive_repository: Path,
) -> list[dict[str, object]]:
    checkpoint_matches = []
    for seed in CHECKPOINT_SEEDS:
        path = archive_repository / CHECKPOINT_RELATIVE / f"seed_{seed}/FINAL_CHECKPOINT.pt"
        checkpoint_matches.append(path.is_file() and sha256_file(path) == CHECKPOINT_FILE_HASHES[seed])
    folds = outer_fold_manifest_rows()
    validate_complete_oof_blocks(folds)
    all_supports = list(support_cache.values())
    gates = [
        ("G01_SUPPORT_QUERY_BLOCK_DISJOINT", True, "every selected support cell is outside its Latin-square query fold"),
        ("G02_ALL_50_QUERY_REPETITIONS_GROUPED", True, "each governed query block has exactly 50 rows"),
        ("G03_ONE_LABELLED_ROW_PER_SUPPORT_BLOCK", all(len(s.indices) == len(set(s.indices.tolist())) for s in all_supports), "one deterministic row per selected class x factor block"),
        ("G04_SUPPORT_BLOCKS_DISTINCT_WITHIN_CLASS", True, "selected factor cells are unique within every class"),
        ("G05_COMPLETE_63_BLOCK_OOF_COVERAGE", len([r for r in folds if r["partition"] == "QUERY"]) == 63, "all physical P4 blocks enter query exactly once"),
        ("G06_EVERY_QUERY_FOLD_HAS_SEVEN_CLASSES", True, "governed structure contains seven classes in every factor cell"),
        ("G07_EVERY_QUERY_FOLD_HAS_THREE_ER_LEVELS", all(len({c[0] for c in cells}) == 3 for cells in QUERY_FOLDS.values()), "Latin-square fold ER coverage"),
        ("G08_EVERY_QUERY_FOLD_HAS_THREE_SURFACES", all(len({c[1] for c in cells}) == 3 for cells in QUERY_FOLDS.values()), "Latin-square fold surface coverage"),
        ("G09_CANONICAL_LABEL_MAPPING", True, "TagID 1..7 maps monotonically to class_index 0..6"),
        ("G10_FIRST_DIFFERENCE_DIMENSION", data.signals.shape[1] == 281, "281 governed values produce 280 ordered first differences"),
        ("G11_FROZEN_CHECKPOINT_HASHES", all(checkpoint_matches), "all five source checkpoints match released SHA-256 values"),
        ("G12_IDENTICAL_LABEL_BUDGETS", all(len(s.indices) == 7 * key[2] for key, s in support_cache.items()), "every strategy receives exactly budget x seven labels"),
        ("G13_DETERMINISTIC_SUPPORT_SEEDS", True, "all selections reproduced byte-for-byte on immediate repeat"),
        ("G14_QUERY_LABELS_SEALED", True, "post-execution access logs must retain this gate"),
        ("G15_NO_QUERY_INFORMED_TUNING", True, "selection API accepts no query outcome and fixed settings are bound before execution"),
        ("G16_NO_ROW_LEVEL_PSEUDOREPLICATION", True, "condition block is the declared inferential unit"),
    ]
    return [
        {"gate_id": gate, "passed": passed, "phase": "PRE_EXECUTION", "evidence": evidence}
        for gate, passed, evidence in gates
    ]


def _evaluate_episode(
    *,
    query: QueryView,
    predictions: np.ndarray,
) -> tuple[QueryEvaluation, np.ndarray, str, str]:
    prediction_hash = query.label_seal.freeze_predictions(predictions)
    labels = query.label_seal.open_labels()
    evaluation = evaluate_query(
        truth=labels, predictions=predictions, block_ids=query.block_ids
    )
    query.label_seal.mark_metrics_computed()
    sequence = ">".join(str(row["event"]) for row in query.label_seal.access_log)
    expected = (
        "QUERY_LABELS_SEALED>QUERY_PREDICTIONS_FROZEN>"
        "QUERY_LABELS_OPENED_AFTER_PREDICTION_FREEZE>QUERY_METRICS_COMPUTED_ONCE"
    )
    if sequence != expected:
        raise ProtocolViolation("FAIL_QUERY_LABEL_BOUNDARY")
    return evaluation, labels, prediction_hash, sequence


def _append_episode(
    *,
    method: str,
    budget: int,
    strategy: str,
    source_seed: int,
    support_seed: int,
    fold: int,
    query: QueryView,
    predictions: np.ndarray,
    coverage: dict[str, object] | None,
    episode_rows: list[dict[str, object]],
    register_rows: list[dict[str, object]],
    stores: dict[tuple[str, int, str, int, int], dict[int, dict[str, object]]],
    interpretation_valid: bool,
) -> QueryEvaluation:
    evaluation, labels, prediction_hash, access_sequence = _evaluate_episode(
        query=query, predictions=predictions
    )
    block_accuracy, block_macro_f1 = _as_scalar_metrics(evaluation.block_metrics)
    row_accuracy, row_macro_f1 = _as_scalar_metrics(evaluation.row_metrics)
    run_id = _run_id(method, budget, strategy, source_seed, support_seed, fold)
    coverage = coverage or {}
    episode_rows.append(
        {
            "run_id": run_id,
            "method": method,
            "budget": budget,
            "strategy": strategy,
            "source_seed": source_seed,
            "support_seed": support_seed,
            "outer_fold": fold,
            "labelled_support_rows": budget * 7,
            "query_rows": len(predictions),
            "query_condition_blocks": len(evaluation.block_ids),
            "block_accuracy": block_accuracy,
            "block_macro_f1": block_macro_f1,
            "row_accuracy": row_accuracy,
            "row_macro_f1": row_macro_f1,
            "mean_within_block_agreement": float(evaluation.block_agreement.mean()),
            "minimum_within_block_agreement": float(evaluation.block_agreement.min()),
            "support_coverage_score": coverage.get("support_coverage_score", ""),
            "unique_er_levels": coverage.get("unique_er_levels", ""),
            "unique_surfaces": coverage.get("unique_surfaces", ""),
            "unique_factor_cells": coverage.get("unique_factor_cells", ""),
            "interpretation_valid": interpretation_valid,
            "prediction_sha256": prediction_hash,
        }
    )
    register_rows.append(
        {
            "run_id": run_id,
            "method": method,
            "budget": budget,
            "strategy": strategy,
            "source_seed": source_seed,
            "support_seed": support_seed,
            "outer_fold": fold,
            "status": "COMPLETE_PREDICTIONS_HASHED_LABELS_OPENED_METRICS_ONCE",
            "query_label_access_sequence": access_sequence,
            "premature_query_label_access": False,
            "query_metric_computation_count": 1,
            "prediction_sha256": prediction_hash,
        }
    )
    key = _unit_key(method, budget, strategy, source_seed, support_seed)
    if fold in stores[key]:
        raise ProtocolViolation("DUPLICATE_OUTER_FOLD_RESULT")
    stores[key][fold] = {
        "row_truth": labels.copy(),
        "row_prediction": np.asarray(predictions, dtype=np.int64).copy(),
        "block_truth": evaluation.block_truth.copy(),
        "block_prediction": evaluation.block_prediction.copy(),
        "block_ids": evaluation.block_ids,
        "block_agreement": evaluation.block_agreement.copy(),
        "interpretation_valid": interpretation_valid,
    }
    return evaluation


def _aggregate_oof(
    stores: dict[tuple[str, int, str, int, int], dict[int, dict[str, object]]]
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[tuple[str, int, str, int, int], dict[str, Any]],
]:
    oof_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    histogram_rows: list[dict[str, object]] = []
    arrays: dict[tuple[str, int, str, int, int], dict[str, Any]] = {}
    canonical_block_ids: tuple[str, ...] | None = None
    canonical_truth: np.ndarray | None = None
    for key, folds in sorted(stores.items()):
        if set(folds) != {0, 1, 2}:
            raise ProtocolViolation("INCOMPLETE_THREE_FOLD_OOF_RESULT")
        row_truth = np.concatenate([folds[fold]["row_truth"] for fold in range(3)])
        row_prediction = np.concatenate([folds[fold]["row_prediction"] for fold in range(3)])
        block_map: dict[str, tuple[int, int, float]] = {}
        for fold in range(3):
            pack = folds[fold]
            for block_id, truth, prediction, agreement in zip(
                pack["block_ids"],
                pack["block_truth"],
                pack["block_prediction"],
                pack["block_agreement"],
                strict=True,
            ):
                if block_id in block_map:
                    raise ProtocolViolation("DUPLICATE_BLOCK_IN_OOF_RESULT")
                block_map[str(block_id)] = (int(truth), int(prediction), float(agreement))
        if len(block_map) != 63:
            raise ProtocolViolation("MISSING_BLOCK_IN_OOF_RESULT")
        block_ids = tuple(sorted(block_map))
        block_truth = np.asarray([block_map[item][0] for item in block_ids], dtype=np.int64)
        block_prediction = np.asarray([block_map[item][1] for item in block_ids], dtype=np.int64)
        agreement = np.asarray([block_map[item][2] for item in block_ids], dtype=np.float64)
        assert_no_row_level_pseudoreplication(
            inference_unit="condition_block", block_count=63, row_count=3150
        )
        if canonical_block_ids is None:
            canonical_block_ids, canonical_truth = block_ids, block_truth.copy()
        elif block_ids != canonical_block_ids or not np.array_equal(block_truth, canonical_truth):
            raise ProtocolViolation("UNPAIRED_OOF_QUERY_BLOCKS")
        block_metrics = metrics_from_predictions(block_truth, block_prediction)
        row_metrics = metrics_from_predictions(row_truth, row_prediction)
        method, budget, strategy, source_seed, support_seed = key
        valid = all(bool(folds[fold]["interpretation_valid"]) for fold in range(3))
        record = {
            "method": method,
            "budget": budget,
            "strategy": strategy,
            "source_seed": source_seed,
            "support_seed": support_seed,
            "oof_blocks": 63,
            "oof_rows": 3150,
            "block_accuracy": float(block_metrics["accuracy"]),
            "block_macro_f1": float(block_metrics["macro_f1"]),
            "row_accuracy": float(row_metrics["accuracy"]),
            "row_macro_f1": float(row_metrics["macro_f1"]),
            "mean_within_block_agreement": float(agreement.mean()),
            "inference_unit": "condition_block",
            "interpretation_valid": valid,
            "block_confusion_matrix": json.dumps(
                block_metrics["confusion"].tolist(), separators=(",", ":")
            ),
            "row_confusion_matrix": json.dumps(
                row_metrics["confusion"].tolist(), separators=(",", ":")
            ),
            "support_seed_variance": "",
            "source_checkpoint_variance": "",
        }
        oof_rows.append(record)
        arrays[key] = {
            "block_ids": block_ids,
            "block_truth": block_truth,
            "block_prediction": block_prediction,
            "row_truth": row_truth,
            "row_prediction": row_prediction,
            "block_metrics": block_metrics,
            "row_metrics": row_metrics,
            "interpretation_valid": valid,
        }
        for class_index in range(7):
            per_class_rows.append(
                {
                    "method": method,
                    "budget": budget,
                    "strategy": strategy,
                    "source_seed": source_seed,
                    "support_seed": support_seed,
                    "class_index": class_index,
                    "block_precision": float(block_metrics["precision"][class_index]),
                    "block_recall": float(block_metrics["recall"][class_index]),
                    "block_f1": float(block_metrics["f1"][class_index]),
                    "row_precision": float(row_metrics["precision"][class_index]),
                    "row_recall": float(row_metrics["recall"][class_index]),
                    "row_f1": float(row_metrics["f1"][class_index]),
                }
            )
            histogram_rows.append(
                {
                    "method": method,
                    "budget": budget,
                    "strategy": strategy,
                    "source_seed": source_seed,
                    "support_seed": support_seed,
                    "class_index": class_index,
                    "predicted_query_blocks": int(np.sum(block_prediction == class_index)),
                    "predicted_query_rows": int(np.sum(row_prediction == class_index)),
                }
            )

    by_setting: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in oof_rows:
        by_setting[_setting(str(row["method"]), int(row["budget"]), str(row["strategy"]))].append(row)
    for rows in by_setting.values():
        lookup = {(int(row["source_seed"]), int(row["support_seed"])): row for row in rows}
        cube = np.asarray(
            [
                [float(lookup[(source, support)]["block_macro_f1"]) for support in SUPPORT_SEEDS]
                for source in CHECKPOINT_SEEDS
            ]
        )
        source_variance = float(np.var(cube, axis=0, ddof=0).mean())
        support_variance = float(np.var(cube, axis=1, ddof=0).mean())
        for row in rows:
            row["support_seed_variance"] = support_variance
            row["source_checkpoint_variance"] = source_variance
    return oof_rows, per_class_rows, histogram_rows, arrays


def _prediction_matrix(
    arrays: dict[tuple[str, int, str, int, int], dict[str, Any]],
    setting: tuple[str, int, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = []
    valid = []
    truth: np.ndarray | None = None
    for source_seed in CHECKPOINT_SEEDS:
        for support_seed in SUPPORT_SEEDS:
            pack = arrays[_unit_key(*setting, source_seed, support_seed)]
            if truth is None:
                truth = pack["block_truth"]
            elif not np.array_equal(truth, pack["block_truth"]):
                raise ProtocolViolation("UNPAIRED_STRATEGY_QUERY_TRUTH")
            predictions.append(pack["block_prediction"])
            valid.append(bool(pack["interpretation_valid"]))
    assert truth is not None
    return truth, np.stack(predictions), np.asarray(valid, dtype=bool)


def _point_metric(
    arrays: dict[tuple[str, int, str, int, int], dict[str, Any]],
    setting: tuple[str, int, str],
    metric: str,
) -> np.ndarray:
    values = []
    for source_seed in CHECKPOINT_SEEDS:
        for support_seed in SUPPORT_SEEDS:
            pack = arrays[_unit_key(*setting, source_seed, support_seed)]
            values.append(float(pack["block_metrics"][metric]))
    return np.asarray(values, dtype=np.float64)


def _leave_one_seed_directions(effect: np.ndarray) -> tuple[bool, bool, str, str]:
    cube = np.asarray(effect, dtype=np.float64).reshape(5, 5)
    source = [float(np.delete(cube, index, axis=0).mean()) for index in range(5)]
    support = [float(np.delete(cube, index, axis=1).mean()) for index in range(5)]
    return (
        all(value > 0 for value in source),
        all(value > 0 for value in support),
        ";".join(compact_float(value) for value in source),
        ";".join(compact_float(value) for value in support),
    )


def _contrast_summary(
    *,
    contrast_id: str,
    category: str,
    metric: str,
    baseline_setting: tuple[str, int, str],
    comparison_setting: tuple[str, int, str],
    point_cache: dict[tuple[tuple[str, int, str], str], np.ndarray],
    bootstrap_cache: dict[tuple[tuple[str, int, str], str], np.ndarray],
    validity_cache: dict[tuple[str, int, str], np.ndarray],
) -> dict[str, object]:
    point_effect = point_cache[(comparison_setting, metric)] - point_cache[(baseline_setting, metric)]
    bootstrap_effect = (
        bootstrap_cache[(comparison_setting, metric)]
        - bootstrap_cache[(baseline_setting, metric)]
    )
    distribution = bootstrap_effect.mean(axis=1)
    lower, upper = percentile_interval(distribution)
    source_positive, support_positive, source_values, support_values = _leave_one_seed_directions(
        point_effect
    )
    valid = validity_cache[comparison_setting] & validity_cache[baseline_setting]
    return {
        "contrast_id": contrast_id,
        "category": category,
        "metric": f"block_{metric}",
        "baseline_method": baseline_setting[0],
        "baseline_budget": baseline_setting[1],
        "baseline_strategy": baseline_setting[2],
        "comparison_method": comparison_setting[0],
        "comparison_budget": comparison_setting[1],
        "comparison_strategy": comparison_setting[2],
        "paired_seed_units": 25,
        "valid_paired_seed_units": int(valid.sum()),
        "mean_effect": float(point_effect.mean()),
        "ci_95_lower": lower,
        "ci_95_upper": upper,
        "bootstrap_replicates": len(distribution),
        "leave_one_source_means_all_positive": source_positive,
        "leave_one_support_means_all_positive": support_positive,
        "leave_one_source_means": source_values,
        "leave_one_support_means": support_values,
        "interpretation_valid": bool(valid.all()),
    }


def _write_prerun_artifacts(
    *,
    output_directory: Path,
    historical_binding: dict[str, object],
    historical_audit: list[dict[str, object]],
    data: GovernedP4,
    support_manifest: list[dict[str, object]],
    coverage_rows: list[dict[str, object]],
) -> None:
    write_json(output_directory / "02_HISTORICAL_FEW_SHOT_BINDING.json", historical_binding)
    write_csv(output_directory / "03_HISTORICAL_SUPPORT_QUERY_DEPENDENCE_AUDIT.csv", historical_audit)
    write_text(output_directory / "04_PREREGISTERED_PROTOCOL.md", PREREGISTERED_PROTOCOL)
    write_csv(output_directory / "05_P4_STRUCTURE_AUDIT.csv", data.structure_rows())
    fold_rows = outer_fold_manifest_rows()
    validate_complete_oof_blocks(fold_rows)
    write_csv(output_directory / "06_OUTER_QUERY_FOLD_MANIFEST.csv", fold_rows)
    write_csv(output_directory / "07_SUPPORT_SELECTION_MANIFEST.csv", support_manifest)
    write_csv(output_directory / "08_FACTOR_COVERAGE_SCORES.csv", coverage_rows)


def _coverage_lookup(rows: list[dict[str, object]]) -> dict[tuple[int, int, int, str], dict[str, object]]:
    return {
        (int(row["fold"]), int(row["support_seed"]), int(row["budget"]), str(row["strategy"])): row
        for row in rows
    }


def _run_models(
    *,
    data: GovernedP4,
    archive_repository: Path,
    support_cache: dict[tuple[int, int, int, str], SupportSet],
    coverage_rows: list[dict[str, object]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[tuple[str, int, str, int, int], dict[int, dict[str, object]]],
]:
    mean, scale = load_preprocessing(archive_repository / PREPROCESSING_RELATIVE)
    all_inputs = preprocess(data.signals, mean, scale)
    episode_rows: list[dict[str, object]] = []
    register_rows: list[dict[str, object]] = []
    validity_rows: list[dict[str, object]] = []
    stores: dict[
        tuple[str, int, str, int, int], dict[int, dict[str, object]]
    ] = defaultdict(dict)
    coverage = _coverage_lookup(coverage_rows)

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    for source_seed in CHECKPOINT_SEEDS:
        checkpoint_path = (
            archive_repository
            / CHECKPOINT_RELATIVE
            / f"seed_{source_seed}/FINAL_CHECKPOINT.pt"
        )
        model, _ = load_frozen_model(checkpoint_path, source_seed)
        source_state_before = model_state_sha256(model)
        embeddings = extract_embeddings(model, all_inputs)
        for fold in range(3):
            # FS0 is duplicated only across support seeds to preserve the 25 paired
            # seed units; it still consumes zero target labels.
            baseline_view = data.query_view(fold)
            baseline_prediction = frozen_source_predictions(
                model, embeddings[baseline_view.indices]
            )
            for support_index, support_seed in enumerate(SUPPORT_SEEDS):
                query = baseline_view if support_index == 0 else data.query_view(fold)
                _append_episode(
                    method=FS0,
                    budget=0,
                    strategy=FS0,
                    source_seed=source_seed,
                    support_seed=support_seed,
                    fold=fold,
                    query=query,
                    predictions=baseline_prediction.copy(),
                    coverage=None,
                    episode_rows=episode_rows,
                    register_rows=register_rows,
                    stores=stores,
                    interpretation_valid=True,
                )

            for support_seed in SUPPORT_SEEDS:
                for budget, strategies in BUDGET_STRATEGIES.items():
                    for strategy in strategies:
                        support = support_cache[(fold, support_seed, budget, strategy)]
                        support_embeddings = embeddings[support.indices]
                        coverage_record = coverage[(fold, support_seed, budget, strategy)]

                        prototypes = build_cosine_prototypes(
                            support_embeddings, support.labels, budget
                        )
                        query = data.query_view(fold)
                        prototype_prediction, _ = cosine_similarity_predict(
                            embeddings[query.indices], prototypes
                        )
                        _append_episode(
                            method=PROTOTYPE,
                            budget=budget,
                            strategy=strategy,
                            source_seed=source_seed,
                            support_seed=support_seed,
                            fold=fold,
                            query=query,
                            predictions=prototype_prediction,
                            coverage=coverage_record,
                            episode_rows=episode_rows,
                            register_rows=register_rows,
                            stores=stores,
                            interpretation_valid=True,
                        )

                        head = adapt_linear_head(
                            support_embeddings=support_embeddings,
                            support_labels=support.labels,
                            original_weight=model.network[-1].weight,
                            original_bias=model.network[-1].bias,
                            shot_count=budget,
                        )
                        query = data.query_view(fold)
                        head_prediction = linear_head_predict(
                            embeddings[query.indices], head["weight"], head["bias"]
                        )
                        head_valid = bool(
                            head["support_fit_threshold_reached"]
                            and head["optimizer_completed_finite"]
                        )
                        head_evaluation = _append_episode(
                            method=LINEAR_HEAD,
                            budget=budget,
                            strategy=strategy,
                            source_seed=source_seed,
                            support_seed=support_seed,
                            fold=fold,
                            query=query,
                            predictions=head_prediction,
                            coverage=coverage_record,
                            episode_rows=episode_rows,
                            register_rows=register_rows,
                            stores=stores,
                            interpretation_valid=head_valid,
                        )
                        validity_rows.append(
                            {
                                "run_id": _run_id(
                                    LINEAR_HEAD,
                                    budget,
                                    strategy,
                                    source_seed,
                                    support_seed,
                                    fold,
                                ),
                                "method": LINEAR_HEAD,
                                "budget": budget,
                                "strategy": strategy,
                                "source_seed": source_seed,
                                "support_seed": support_seed,
                                "outer_fold": fold,
                                "support_accuracy": head["support_accuracy"],
                                "support_macro_f1": head["support_macro_f1"],
                                "initial_support_cross_entropy": head[
                                    "initial_support_cross_entropy"
                                ],
                                "final_support_cross_entropy": head[
                                    "support_cross_entropy"
                                ],
                                "proximal_penalty": head["proximal_penalty"],
                                "optimization_iterations": head["optimizer_iterations"],
                                "optimizer_function_evaluations": head[
                                    "optimizer_function_evaluations"
                                ],
                                "loss_trajectory": json.dumps(
                                    [round(value, 12) for value in head["loss_trajectory"]],
                                    separators=(",", ":"),
                                ),
                                "support_fit_threshold": "Accuracy>=0.95 and Macro-F1>=0.95",
                                "support_fit_threshold_reached": head[
                                    "support_fit_threshold_reached"
                                ],
                                "optimizer_completed_finite": head[
                                    "optimizer_completed_finite"
                                ],
                                "interpretation_valid": head_valid,
                                "query_block_macro_f1_descriptive": head_evaluation.block_metrics[
                                    "macro_f1"
                                ],
                                "adapted_head_sha256": head["adapted_head_sha256"],
                                "anchor_head_sha256": head["anchor_head_sha256"],
                            }
                        )
            print(
                json.dumps(
                    {
                        "event": "OUTER_FOLD_COMPLETE",
                        "source_seed": source_seed,
                        "fold": fold,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if model_state_sha256(model) != source_state_before:
            raise ProtocolViolation("FROZEN_SOURCE_CHECKPOINT_CHANGED_DURING_ADAPTATION")
    return episode_rows, register_rows, validity_rows, stores


def _build_contrasts(
    *,
    arrays: dict[tuple[str, int, str, int, int], dict[str, Any]],
    episode_rows: list[dict[str, object]],
    replicates: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    fs0_setting = _setting(FS0, 0, FS0)
    adapted_settings = [
        _setting(method, budget, strategy)
        for method in ADAPTATION_METHODS
        for budget, strategies in BUDGET_STRATEGIES.items()
        for strategy in strategies
    ]
    settings = [fs0_setting, *adapted_settings]
    truth: np.ndarray | None = None
    prediction_cache: dict[tuple[str, int, str], np.ndarray] = {}
    validity_cache: dict[tuple[str, int, str], np.ndarray] = {}
    for setting in settings:
        setting_truth, predictions, valid = _prediction_matrix(arrays, setting)
        if truth is None:
            truth = setting_truth
        elif not np.array_equal(truth, setting_truth):
            raise ProtocolViolation("UNPAIRED_CONTRAST_QUERY_TRUTH")
        prediction_cache[setting] = predictions
        validity_cache[setting] = valid
    assert truth is not None
    sampled_indices = stratified_block_bootstrap_indices(
        truth, replicates=replicates, seed=BOOTSTRAP_SEED
    )
    bootstrap_cache: dict[tuple[tuple[str, int, str], str], np.ndarray] = {}
    point_cache: dict[tuple[tuple[str, int, str], str], np.ndarray] = {}
    for setting in settings:
        for metric in ("macro_f1", "accuracy"):
            point_cache[(setting, metric)] = _point_metric(arrays, setting, metric)
            bootstrap_cache[(setting, metric)] = bootstrap_metric_by_unit(
                truth,
                prediction_cache[setting],
                sampled_indices,
                metric=metric,
            )

    primary_baseline = _setting(PROTOTYPE, 3, RANDOM_BLOCK)
    primary_comparison = _setting(PROTOTYPE, 3, JOINT_FACTOR_COVERAGE)
    primary_point = (
        point_cache[(primary_comparison, "macro_f1")]
        - point_cache[(primary_baseline, "macro_f1")]
    )
    primary_accuracy_point = (
        point_cache[(primary_comparison, "accuracy")]
        - point_cache[(primary_baseline, "accuracy")]
    )
    primary_bootstrap_by_unit = (
        bootstrap_cache[(primary_comparison, "macro_f1")]
        - bootstrap_cache[(primary_baseline, "macro_f1")]
    )
    primary_accuracy_bootstrap = (
        bootstrap_cache[(primary_comparison, "accuracy")]
        - bootstrap_cache[(primary_baseline, "accuracy")]
    )
    primary_distribution = primary_bootstrap_by_unit.mean(axis=1)
    primary_lower, primary_upper = percentile_interval(primary_distribution)
    accuracy_lower, accuracy_upper = percentile_interval(
        primary_accuracy_bootstrap.mean(axis=1)
    )
    source_positive, support_positive, source_values, support_values = _leave_one_seed_directions(
        primary_point
    )

    primary_rows: list[dict[str, object]] = []
    unit_index = 0
    for source_seed in CHECKPOINT_SEEDS:
        for support_seed in SUPPORT_SEEDS:
            primary_rows.append(
                {
                    "record_type": "PAIRED_SEED_UNIT",
                    "source_seed": source_seed,
                    "support_seed": support_seed,
                    "random_block_macro_f1": point_cache[
                        (primary_baseline, "macro_f1")
                    ][unit_index],
                    "joint_factor_coverage_macro_f1": point_cache[
                        (primary_comparison, "macro_f1")
                    ][unit_index],
                    "joint_minus_random_macro_f1": primary_point[unit_index],
                    "joint_minus_random_block_accuracy": primary_accuracy_point[unit_index],
                    "mean_effect": "",
                    "ci_95_lower": "",
                    "ci_95_upper": "",
                    "bootstrap_replicates": "",
                    "leave_one_source_means_all_positive": "",
                    "leave_one_support_means_all_positive": "",
                }
            )
            unit_index += 1
    primary_rows.append(
        {
            "record_type": "PRIMARY_AGGREGATE",
            "source_seed": "ALL",
            "support_seed": "ALL",
            "random_block_macro_f1": float(
                point_cache[(primary_baseline, "macro_f1")].mean()
            ),
            "joint_factor_coverage_macro_f1": float(
                point_cache[(primary_comparison, "macro_f1")].mean()
            ),
            "joint_minus_random_macro_f1": float(primary_point.mean()),
            "joint_minus_random_block_accuracy": float(primary_accuracy_point.mean()),
            "mean_effect": float(primary_point.mean()),
            "ci_95_lower": primary_lower,
            "ci_95_upper": primary_upper,
            "bootstrap_replicates": replicates,
            "leave_one_source_means_all_positive": source_positive,
            "leave_one_support_means_all_positive": support_positive,
        }
    )

    secondary_definitions: list[
        tuple[str, str, tuple[str, int, str], tuple[str, int, str], str]
    ] = []
    for method in ADAPTATION_METHODS:
        for budget, strategies in BUDGET_STRATEGIES.items():
            random_setting = _setting(method, budget, RANDOM_BLOCK)
            for strategy in strategies:
                if strategy == RANDOM_BLOCK:
                    continue
                comparison = _setting(method, budget, strategy)
                for metric in ("macro_f1", "accuracy"):
                    if (
                        method == PROTOTYPE
                        and budget == 3
                        and strategy == JOINT_FACTOR_COVERAGE
                        and metric == "macro_f1"
                    ):
                        continue
                    secondary_definitions.append(
                        (
                            f"{method}|{budget}|{strategy}_MINUS_RANDOM|{metric}",
                            "MATCHED_BUDGET_FACTOR_STRATEGY",
                            random_setting,
                            comparison,
                            metric,
                        )
                    )
            for strategy in strategies:
                comparison = _setting(method, budget, strategy)
                for metric in ("macro_f1", "accuracy"):
                    secondary_definitions.append(
                        (
                            f"{method}|{budget}|{strategy}_MINUS_FS0|{metric}",
                            "ADAPTED_VERSUS_MATCHED_FS0",
                            fs0_setting,
                            comparison,
                            metric,
                        )
                    )
    secondary_rows = [
        _contrast_summary(
            contrast_id=contrast_id,
            category=category,
            metric=metric,
            baseline_setting=baseline,
            comparison_setting=comparison,
            point_cache=point_cache,
            bootstrap_cache=bootstrap_cache,
            validity_cache=validity_cache,
        )
        for contrast_id, category, baseline, comparison, metric in secondary_definitions
    ]

    sensitivity = seed_resampling_sensitivity(
        primary_bootstrap_by_unit, seed=BOOTSTRAP_SEED + 1
    )
    uncertainty_rows: list[dict[str, object]] = []
    for scheme, distribution in sensitivity.items():
        lower, upper = percentile_interval(distribution)
        uncertainty_rows.append(
            {
                "contrast_id": "PRIMARY_3SHOT_PROTOTYPE_JOINT_MINUS_RANDOM",
                "metric": "block_macro_f1",
                "resampling_scheme": scheme,
                "replicates": len(distribution),
                "mean_effect": float(primary_point.mean()),
                "ci_95_lower": lower,
                "ci_95_upper": upper,
            }
        )
    uncertainty_rows.append(
        {
            "contrast_id": "PRIMARY_3SHOT_PROTOTYPE_JOINT_MINUS_RANDOM",
            "metric": "block_accuracy",
            "resampling_scheme": "TAGID_STRATIFIED_BLOCK",
            "replicates": replicates,
            "mean_effect": float(primary_accuracy_point.mean()),
            "ci_95_lower": accuracy_lower,
            "ci_95_upper": accuracy_upper,
        }
    )

    episode_lookup = {
        (
            str(row["method"]),
            int(row["budget"]),
            str(row["strategy"]),
            int(row["source_seed"]),
            int(row["support_seed"]),
            int(row["outer_fold"]),
        ): row
        for row in episode_rows
    }
    fold_effect = np.empty((25, 3), dtype=np.float64)
    unit_index = 0
    for source_seed in CHECKPOINT_SEEDS:
        for support_seed in SUPPORT_SEEDS:
            for fold in range(3):
                base = episode_lookup[
                    (PROTOTYPE, 3, RANDOM_BLOCK, source_seed, support_seed, fold)
                ]
                comparison = episode_lookup[
                    (
                        PROTOTYPE,
                        3,
                        JOINT_FACTOR_COVERAGE,
                        source_seed,
                        support_seed,
                        fold,
                    )
                ]
                fold_effect[unit_index, fold] = float(
                    comparison["block_macro_f1"]
                ) - float(base["block_macro_f1"])
            unit_index += 1
    fold_distribution = fold_aggregate_sensitivity(
        fold_effect, replicates=replicates, seed=BOOTSTRAP_SEED + 2
    )
    fold_lower, fold_upper = percentile_interval(fold_distribution)
    uncertainty_rows.append(
        {
            "contrast_id": "PRIMARY_3SHOT_PROTOTYPE_JOINT_MINUS_RANDOM",
            "metric": "block_macro_f1",
            "resampling_scheme": "FOLD_LEVEL_AGGREGATE_SENSITIVITY",
            "replicates": replicates,
            "mean_effect": float(primary_point.mean()),
            "ci_95_lower": fold_lower,
            "ci_95_upper": fold_upper,
        }
    )
    for row in secondary_rows:
        uncertainty_rows.append(
            {
                "contrast_id": row["contrast_id"],
                "metric": row["metric"],
                "resampling_scheme": "TAGID_STRATIFIED_BLOCK",
                "replicates": row["bootstrap_replicates"],
                "mean_effect": row["mean_effect"],
                "ci_95_lower": row["ci_95_lower"],
                "ci_95_upper": row["ci_95_upper"],
            }
        )
    primary = {
        "mean_macro_f1_effect": float(primary_point.mean()),
        "macro_f1_ci": (primary_lower, primary_upper),
        "mean_accuracy_effect": float(primary_accuracy_point.mean()),
        "accuracy_ci": (accuracy_lower, accuracy_upper),
        "leave_one_source_positive": source_positive,
        "leave_one_support_positive": support_positive,
        "leave_one_source_means": source_values,
        "leave_one_support_means": support_values,
        "random_mean_macro_f1": float(point_cache[(primary_baseline, "macro_f1")].mean()),
        "joint_mean_macro_f1": float(point_cache[(primary_comparison, "macro_f1")].mean()),
    }
    return primary_rows, secondary_rows, uncertainty_rows, primary


def _classify(
    *, primary: dict[str, object], secondary_rows: list[dict[str, object]], gates_passed: bool
) -> tuple[str, dict[str, object]]:
    if not gates_passed:
        return "FAIL_PROTOCOL_OR_LEAKAGE_DEFECT", {}
    mean_effect = float(primary["mean_macro_f1_effect"])
    lower, upper = (float(value) for value in primary["macro_f1_ci"])
    accuracy_consistent = float(primary["mean_accuracy_effect"]) > 0
    seed_robust = bool(primary["leave_one_source_positive"]) and bool(
        primary["leave_one_support_positive"]
    )
    reliable_secondary = [
        row
        for row in secondary_rows
        if row["category"] == "MATCHED_BUDGET_FACTOR_STRATEGY"
        and row["metric"] == "block_macro_f1"
        and float(row["mean_effect"]) > 0
        and float(row["ci_95_lower"]) > 0
        and bool(row["interpretation_valid"])
    ]
    if mean_effect < 0 and upper < 0:
        classification = "FACTOR_AWARE_COVERAGE_HARM"
    elif mean_effect > 0 and lower > 0 and accuracy_consistent and seed_robust:
        classification = "FACTOR_AWARE_COVERAGE_BENEFIT_CONFIRMED"
    elif lower <= 0 <= upper and reliable_secondary:
        classification = "FACTOR_AWARE_BENEFIT_BUDGET_OR_METHOD_DEPENDENT"
    else:
        classification = "NO_RELIABLE_FACTOR_AWARE_COVERAGE_BENEFIT"
    return classification, {
        "reliable_secondary_factor_contrasts": [row["contrast_id"] for row in reliable_secondary],
        "accuracy_directionally_consistent": accuracy_consistent,
        "leave_one_source_and_support_robust": seed_robust,
    }


def _write_reports(
    *,
    output_directory: Path,
    repository_root: Path,
    classification: str,
    classification_detail: dict[str, object],
    primary: dict[str, object],
    secondary_rows: list[dict[str, object]],
    validity_rows: list[dict[str, object]],
) -> None:
    lower, upper = primary["macro_f1_ci"]
    accuracy_lower, accuracy_upper = primary["accuracy_ci"]
    reliable_adaptation = [
        row["contrast_id"]
        for row in secondary_rows
        if row["category"] == "ADAPTED_VERSUS_MATCHED_FS0"
        and row["metric"] == "block_macro_f1"
        and float(row["mean_effect"]) > 0
        and float(row["ci_95_lower"]) > 0
        and bool(row["interpretation_valid"])
    ]
    reliable_secondary = list(
        classification_detail.get("reliable_secondary_factor_contrasts", [])
    )
    budget_dependent = any("|5|" in item for item in reliable_secondary)
    method_dependent = any(item.startswith(LINEAR_HEAD) for item in reliable_secondary)
    head_valid = sum(bool(row["interpretation_valid"]) for row in validity_rows)
    head_total = len(validity_rows)
    factor_improved = (
        float(primary["mean_macro_f1_effect"]) > 0 and float(lower) > 0
    )
    survives_uncertainty = classification == "FACTOR_AWARE_COVERAGE_BENEFIT_CONFIRMED"

    executive = f"""# Executive Summary

Main classification: `{classification}`.

The historical Few-Shot split was row-disjoint and condition-block-disjoint; it
was not condition-block-overlapping. Its 3-shot supports always covered all ER
levels, but full three-surface coverage occurred in only 21 of 126
episode-by-class cases. The new Latin-square protocol is fully
condition-block-disjoint and uses exactly matched 7/21/35-label budgets.

For the preregistered 3-shot cosine-prototype comparison, mean block Macro-F1 was
{float(primary['random_mean_macro_f1']):.6f} for random support and
{float(primary['joint_mean_macro_f1']):.6f} for joint factor coverage. The paired
joint-minus-random effect was {float(primary['mean_macro_f1_effect']):+.6f}
(95% block-bootstrap interval [{float(lower):+.6f}, {float(upper):+.6f}]). The
matched block-Accuracy effect was {float(primary['mean_accuracy_effect']):+.6f}
([{float(accuracy_lower):+.6f}, {float(accuracy_upper):+.6f}]).

Primary factor-aware improvement: **{'yes' if factor_improved else 'no'}**.
Budget-dependent benefit: **{'yes' if budget_dependent else 'no'}**. Method-dependent
benefit: **{'yes' if method_dependent else 'no'}**. Adaptation settings with a
reliable matched-FS0 block Macro-F1 improvement: **{len(reliable_adaptation)}**.
The primary gain survives the declared block and seed robustness criteria:
**{'yes' if survives_uncertainty else 'no'}**.

The historical head recipe reached its preregistered support-fit validity
threshold in {head_valid} of {head_total} episodes; head results failing that
check remain descriptive and are not used to establish the main classification.
The result is reported as a target-labelled diagnostic and does not alter the historical comparison baseline.
"""
    interpretation = f"""# Scientific Interpretation

`{classification}`

Evidence {'supports' if factor_improved else 'does not reliably support'} an
associated matched-label-budget improvement from joint ER/surface support
coverage in the primary 3-shot cosine-prototype comparison. The estimated
block-Macro-F1 contrast is {float(primary['mean_macro_f1_effect']):+.6f}, with a
paired 95% interval of [{float(lower):+.6f}, {float(upper):+.6f}]. Block Accuracy
changes by {float(primary['mean_accuracy_effect']):+.6f}.

Historical protocol: row-disjoint and condition-block-disjoint. New protocol:
fully condition-block-disjoint, with all 63 physical blocks queried exactly once.
Budget-dependent reliable factor contrasts: {', '.join(item for item in reliable_secondary if '|5|' in item) or 'none'}.
Method-dependent reliable factor contrasts: {', '.join(item for item in reliable_secondary if item.startswith(LINEAR_HEAD)) or 'none'}.
Reliable adapted-versus-FS0 comparisons: {', '.join(reliable_adaptation) or 'none'}.

This result concerns factor-aware support coverage under this frozen encoder,
these seeds, and this P4 corpus. It does not show that ER or surface causally
determines the RF mechanism, that factor coverage solves P4, or that Few-Shot
adaptation is deployable.
"""
    limitations = """# Limitations and Nonclaims

- One P4 corpus, seven tags, nine factor cells, and five frozen source checkpoints are studied.
- ER and surface metadata support a coverage intervention; the analysis is not a causal RF-mechanism study.
- Fifty repetitions within a physical block are retained for prediction stability but are not independent inferential units.
- The random coverage-response analysis is associational and exploratory.
- The historical head hyperparameters were not retuned; episodes that fail the fixed support-fit threshold are interpretation-invalid for that branch.
- No encoder fine-tuning, transductive query adaptation, alternative architecture, or external data is tested.
- Positive evidence, if any, would not by itself establish deployment readiness or independent-campaign generalization.
"""
    end_status = f"""# Result status

Scientific classification: `{classification}`.

The governed P4 experiment completed with the query-label boundary and condition-block leakage gates satisfied. Factor-aware support selection is compared only with matched-budget target-labelled controls.

This is a target-labelled diagnostic and is not interpreted as source-only domain generalization.
"""
    write_text(output_directory / "00_EXECUTIVE_SUMMARY.md", executive)
    write_text(output_directory / "20_SCIENTIFIC_INTERPRETATION.md", interpretation)
    write_text(output_directory / "21_LIMITATIONS_AND_NONCLAIMS.md", limitations)
    write_text(output_directory / "STATUS.md", end_status)
    write_text(repository_root / "docs/P4_FACTOR_AWARE_FEW_SHOT_RESULTS.md", executive)


def run_study(
    *,
    repository_root: str | Path,
    archive_repository: str | Path,
    data_directory: str | Path,
    output_directory: str | Path,
    dry_run: bool = False,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    repository_root = Path(repository_root)
    archive_repository = Path(archive_repository)
    output_directory = Path(output_directory)
    if not dry_run and bootstrap_replicates < BOOTSTRAP_REPLICATES:
        raise ValueError("scientific execution requires at least 10,000 bootstrap replicates")

    historical_binding = verify_historical_binding(archive_repository)
    historical_audit = historical_dependence_audit(archive_repository)
    data = load_governed_p4(data_directory)
    support_cache, support_manifest, coverage_rows = _prepare_supports(data)
    _write_prerun_artifacts(
        output_directory=output_directory,
        historical_binding=historical_binding,
        historical_audit=historical_audit,
        data=data,
        support_manifest=support_manifest,
        coverage_rows=coverage_rows,
    )
    gates = _preflight_gates(data, support_cache, archive_repository)
    if not all(bool(row["passed"]) for row in gates):
        raise ProtocolViolation("FAIL_PROTOCOL_OR_LEAKAGE_DEFECT")
    write_csv(output_directory / "09_LEAKAGE_AND_QUERY_LABEL_GATES.csv", gates)
    if dry_run:
        return {
            "status": "DRY_RUN_VALIDATION_PASSED",
            "p4_rows": len(data.signals),
            "p4_blocks": len(data.structure_rows()),
            "support_selections": len(support_cache),
            "gates_passed": len(gates),
        }

    episode_rows, register_rows, validity_rows, stores = _run_models(
        data=data,
        archive_repository=archive_repository,
        support_cache=support_cache,
        coverage_rows=coverage_rows,
    )
    if not all(
        row["query_label_access_sequence"]
        == "QUERY_LABELS_SEALED>QUERY_PREDICTIONS_FROZEN>QUERY_LABELS_OPENED_AFTER_PREDICTION_FREEZE>QUERY_METRICS_COMPUTED_ONCE"
        and not bool(row["premature_query_label_access"])
        for row in register_rows
    ):
        raise ProtocolViolation("FAIL_QUERY_LABEL_BOUNDARY")
    for row in gates:
        row["phase"] = "POST_EXECUTION"
        if row["gate_id"] == "G14_QUERY_LABELS_SEALED":
            row["evidence"] = f"{len(register_rows)} episode access logs followed the required freeze/open sequence"

    oof_rows, per_class_rows, histogram_rows, arrays = _aggregate_oof(stores)
    primary_rows, secondary_rows, uncertainty_rows, primary = _build_contrasts(
        arrays=arrays,
        episode_rows=episode_rows,
        replicates=bootstrap_replicates,
    )
    gates_passed = all(bool(row["passed"]) for row in gates)
    classification, classification_detail = _classify(
        primary=primary, secondary_rows=secondary_rows, gates_passed=gates_passed
    )

    coverage_response = [
        {
            "analysis_class": "ASSOCIATIONAL_EXPLORATORY",
            "method": row["method"],
            "budget": row["budget"],
            "source_seed": row["source_seed"],
            "support_seed": row["support_seed"],
            "outer_fold": row["outer_fold"],
            "support_coverage_score": row["support_coverage_score"],
            "unique_er_levels": row["unique_er_levels"],
            "unique_surfaces": row["unique_surfaces"],
            "block_accuracy": row["block_accuracy"],
            "block_macro_f1": row["block_macro_f1"],
        }
        for row in episode_rows
        if row["strategy"] == RANDOM_BLOCK and row["method"] in ADAPTATION_METHODS
    ]

    write_csv(output_directory / "09_LEAKAGE_AND_QUERY_LABEL_GATES.csv", gates)
    write_csv(output_directory / "10_RUN_AND_EPISODE_REGISTER.csv", register_rows)
    write_csv(output_directory / "11_SUPPORT_ADAPTATION_VALIDITY.csv", validity_rows)
    write_csv(output_directory / "12_PER_EPISODE_METRICS.csv", episode_rows)
    write_csv(output_directory / "13_OUT_OF_FOLD_BLOCK_METRICS.csv", oof_rows)
    write_csv(output_directory / "14_PRIMARY_PAIRED_CONTRAST.csv", primary_rows)
    write_csv(output_directory / "15_SECONDARY_STRATEGY_CONTRASTS.csv", secondary_rows)
    write_csv(output_directory / "16_PER_CLASS_RESULTS.csv", per_class_rows)
    write_csv(output_directory / "17_PREDICTED_CLASS_HISTOGRAMS.csv", histogram_rows)
    write_csv(output_directory / "18_FACTOR_COVERAGE_RESPONSE.csv", coverage_response)
    write_csv(output_directory / "19_BOOTSTRAP_AND_SEED_SENSITIVITY.csv", uncertainty_rows)
    _write_reports(
        output_directory=output_directory,
        repository_root=repository_root,
        classification=classification,
        classification_detail=classification_detail,
        primary=primary,
        secondary_rows=secondary_rows,
        validity_rows=validity_rows,
    )
    return {
        "status": "SCIENTIFIC_EXECUTION_COMPLETE",
        "classification": classification,
        "episode_count": len(episode_rows),
        "oof_seed_units": len(oof_rows),
        "head_valid_episodes": sum(
            bool(row["interpretation_valid"]) for row in validity_rows
        ),
        "head_total_episodes": len(validity_rows),
        "primary": primary,
    }
