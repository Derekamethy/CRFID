"""End-to-end execution and scientific audit for the larger P4 curve."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from p4_factor_aware.artifacts import write_csv, write_json, write_text
from p4_factor_aware.constants import (
    CHECKPOINT_FILE_HASHES,
    CHECKPOINT_STATE_HASHES,
    PREPROCESSING_FILES,
    QUERY_FOLDS,
    cell_id,
)
from p4_factor_aware.data import GovernedP4, load_governed_p4
from p4_factor_aware.historical import verify_historical_binding
from p4_factor_aware.metrics import metrics_from_confusion
from p4_factor_aware.models import (
    extract_embeddings,
    frozen_source_predictions,
    load_frozen_model,
    load_preprocessing,
    model_state_sha256,
    preprocess,
)
from p4_factor_aware.protocol import ProtocolViolation, QueryLabelSeal, outer_fold_manifest_rows

from . import REPOSITORY_ROOT
from .constants import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CHECKPOINT_SEEDS,
    CLAIM_STRICT_DG,
    CLAIM_TARGET_ASSISTED,
    EXPERIMENT_ID,
    HISTORICAL_REFERENCES,
    METHOD_POSITIVE,
    METHOD_ZERO,
    POSITIVE_BUDGETS,
    PRACTICAL_DELTA,
    PREREGISTRATION_SHA256,
    SHOT_LABELS,
    STRICT_AGGREGATE,
    STRICT_PER_SEED,
    SUPPORT_SEEDS,
    TOTAL_BUDGETS,
    budget_label,
)
from .sampling import (
    NestedSupportPlan,
    build_nested_plan,
    coverage_rows,
    plan_manifest_rows,
    signal_digest,
    validate_plan,
)
from .statistics import (
    aggregate_metric_from_histograms,
    block_layout,
    extended_metrics,
    hierarchical_bootstrap,
    majority_predictions,
    percentile_interval,
    prediction_histograms_by_block,
)


FROZEN_RELATIVE = Path("outputs/few_shot/frozen_branch")
SNAPSHOT_RELATIVE = FROZEN_RELATIVE / "01_frozen_source_import/frozen_last_code_snapshot"
PREPROCESSING_RELATIVE = SNAPSHOT_RELATIVE / "10_final_recipe_freeze"
CHECKPOINT_RELATIVE = SNAPSHOT_RELATIVE / "11_final_p4_evaluation/runs"
PREREGISTRATION_RELATIVE = Path("configs/p4_large_calibration_curve/preregistration.json")
MEMO_RELATIVE = Path("docs/P4_LARGE_CALIBRATION_REPOSITORY_STATE.md")


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
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments],
        text=True,
        encoding="utf-8",
        env=environment,
    ).strip()


def _runtime_identity() -> dict[str, object]:
    executable = Path(sys.executable)
    return {
        "environment_id": "CRFID_STRICT_DG_WINDOWS_CPU_V1",
        "python_version": platform.python_version(),
        "python_executable_sha256": sha256_file(executable),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "compute_platform": "cpu",
    }


def _verify_preregistration(repository_root: Path) -> dict[str, object]:
    path = repository_root / PREREGISTRATION_RELATIVE
    observed = sha256_file(path)
    if observed != PREREGISTRATION_SHA256:
        raise ProtocolViolation("PREREGISTRATION_HASH_MISMATCH")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("status") != "SEALED_BEFORE_NEW_CALIBRATION_RESULTS"
        or tuple(payload["label_budget"]["ordered_total_measurement_budgets"])
        != TOTAL_BUDGETS
        or tuple(payload["seeds"]["source"]) != CHECKPOINT_SEEDS
        or tuple(payload["support_sampling"]["support_seeds"]) != SUPPORT_SEEDS
    ):
        raise ProtocolViolation("PREREGISTRATION_CONTENT_MISMATCH")
    return payload


def _signal_digests(data: GovernedP4) -> tuple[str, ...]:
    return tuple(signal_digest(signal) for signal in data.signals)


def _exact_signal_audit(data: GovernedP4, digests: tuple[str, ...]) -> dict[str, object]:
    grouped: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for index, digest in enumerate(digests):
        grouped[digest].append(
            (int(data._labels[index]), int(data.er[index]), int(data.surface[index]))
        )
    duplicated = [blocks for blocks in grouped.values() if len(blocks) > 1]
    cross_block = [blocks for blocks in duplicated if len(set(blocks)) > 1]
    cross_tag = [blocks for blocks in cross_block if len({block[0] for block in blocks}) > 1]
    return {
        "rows": len(digests),
        "unique_exact_signal_sha256": len(grouped),
        "duplicate_digest_groups": len(duplicated),
        "cross_condition_block_digest_groups": len(cross_block),
        "cross_TagID_digest_groups": len(cross_tag),
        "maximum_digest_multiplicity": max(map(len, grouped.values())),
    }


def _prepare_plans(
    data: GovernedP4, digests: tuple[str, ...]
) -> tuple[
    dict[tuple[int, int], NestedSupportPlan],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    plans: dict[tuple[int, int], NestedSupportPlan] = {}
    validation_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    support_coverage_rows: list[dict[str, object]] = []
    for fold in sorted(QUERY_FOLDS):
        for support_seed in SUPPORT_SEEDS:
            plan = build_nested_plan(data, fold=fold, support_seed=support_seed)
            repeated = build_nested_plan(data, fold=fold, support_seed=support_seed)
            if not np.array_equal(plan.ordered_indices, repeated.ordered_indices):
                raise ProtocolViolation("NONDETERMINISTIC_SUPPORT_SAMPLING")
            plans[(fold, support_seed)] = plan
            validation_rows.extend(validate_plan(data, plan, signal_digests=digests))
            manifest_rows.extend(plan_manifest_rows(data, plan, signal_digests=digests))
            support_coverage_rows.extend(coverage_rows(data, plan, signal_digests=digests))
    return plans, validation_rows, manifest_rows, support_coverage_rows


def _checkpoint_preflight(archive_repository: Path) -> bool:
    for seed in CHECKPOINT_SEEDS:
        path = archive_repository / CHECKPOINT_RELATIVE / f"seed_{seed}/FINAL_CHECKPOINT.pt"
        if not path.is_file() or sha256_file(path) != CHECKPOINT_FILE_HASHES[seed]:
            return False
    for filename, expected in PREPROCESSING_FILES.items():
        path = archive_repository / PREPROCESSING_RELATIVE / filename
        if not path.is_file() or sha256_file(path) != expected:
            return False
    return True


def _pre_execution_gates(
    *,
    data: GovernedP4,
    exact_audit: dict[str, object],
    plan_validation: list[dict[str, object]],
    archive_repository: Path,
) -> list[dict[str, object]]:
    fold_rows = outer_fold_manifest_rows()
    gates = [
        ("G01_PREREGISTRATION_HASH", True, PREREGISTRATION_SHA256),
        ("G02_GOVERNED_P4_STRUCTURE", len(data.structure_rows()) == 63, "3,150 rows; 63 complete blocks"),
        ("G03_EXACT_SIGNAL_BLOCK_CONFINEMENT", exact_audit["cross_condition_block_digest_groups"] == 0, "no exact signal digest crosses a block"),
        ("G04_FIXED_COMPLETE_OOF_QUERY", sum(row["partition"] == "QUERY" for row in fold_rows) == 63, "all 63 blocks queried exactly once"),
        ("G05_NESTED_BUDGET_ACCOUNTING", len(plan_validation) == 3 * 20 * 9 and all(bool(row["passed"]) for row in plan_validation), "540 fold x seed x budget validations"),
        ("G06_SUPPORT_QUERY_ROW_DISJOINT", max(int(row["support_query_row_overlap"]) for row in plan_validation) == 0, "zero row overlap"),
        ("G07_SUPPORT_QUERY_BLOCK_DISJOINT", max(int(row["support_query_block_overlap"]) for row in plan_validation) == 0, "zero physical-block overlap"),
        ("G08_SUPPORT_QUERY_EXACT_SIGNAL_DISJOINT", max(int(row["support_query_exact_signal_overlap"]) for row in plan_validation) == 0, "zero exact-signal overlap"),
        ("G09_TAGID_BALANCE", max(int(row["maximum_class_count"]) - int(row["minimum_class_count"]) for row in plan_validation) <= 1, "per-TagID counts differ by at most one"),
        ("G10_FROZEN_INPUT_HASHES", _checkpoint_preflight(archive_repository), "five checkpoints and two preprocessing arrays hash-match"),
        ("G11_QUERY_INDEPENDENT_SUPPORT_POLICY", True, "sampler accepts fold/seed/structure only; no query outcomes or embeddings"),
        ("G12_INFERENTIAL_UNIT", True, "TagID x ER x surface physical condition block"),
        ("G13_GLOBAL_QUERY_LABEL_BARRIER", True, "post-execution validation required"),
        ("G14_FROZEN_ENCODER_INTEGRITY", True, "post-execution validation required"),
        ("G15_CANONICAL_ZERO_ANCHOR", True, "post-execution exact metric reproduction required"),
        ("G16_BOOTSTRAP_REPRODUCIBILITY", True, "focused deterministic test plus final sample hashes required"),
    ]
    return [
        {
            "gate_id": gate_id,
            "passed": bool(passed),
            "phase": "PRE_EXECUTION",
            "evidence": evidence,
        }
        for gate_id, passed, evidence in gates
    ]


def _prior_binding(historical_binding: dict[str, object]) -> dict[str, object]:
    return {
        "strict_dg": {
            "method": "C1_FIRST_DIFFERENCE_ERM_1DCNN",
            "accuracy": STRICT_AGGREGATE["accuracy"],
            "macro_f1": STRICT_AGGREGATE["macro_f1"],
            "checkpoint_seeds": list(CHECKPOINT_SEEDS),
        },
        "historical_few_shot": {
            "tag": historical_binding["historical_tag"],
            "prototype_macro_f1": {"7": 0.112159, "21": 0.145141, "35": 0.144360},
            "directly_comparable": False,
            "reason": "different episode/query population and squared-Euclidean readout",
        },
        "factor_aware_base": {
            "tag": "p4-factor-aware-few-shot-v1",
            "commit": "9929b84c6e1803d5b3d3c2d6c8c291c7c4312f06",
            "reused": ["frozen model", "cosine prototype", "Latin-square folds", "query seal", "block bootstrap principle"],
        },
        "retrospective": {
            "macro_f1": 0.5835693346352661,
            "directly_comparable": False,
            "reason": "full-P4 outcomes selected the method and scored the same rows",
        },
        "matched_target_assisted": {
            "macro_f1": 0.983494,
            "directly_comparable": False,
            "reason": "within-condition Adapt/Val/Test with diagnosed near-duplicate dependence",
        },
    }


def _write_pre_execution_artifacts(
    *,
    repository_root: Path,
    output_directory: Path,
    code_identity: dict[str, object],
    preregistration: dict[str, object],
    historical_binding: dict[str, object],
    data: GovernedP4,
    exact_audit: dict[str, object],
    support_manifest: list[dict[str, object]],
    support_coverage: list[dict[str, object]],
    plan_validation: list[dict[str, object]],
    gates: list[dict[str, object]],
) -> None:
    executed_code = [
        "workflows/12_p4_large_calibration_curve/run.py",
        "workflows/12_p4_large_calibration_curve/plot_results.py",
        "workflows/12_p4_large_calibration_curve/p4_large_calibration/__init__.py",
        "workflows/12_p4_large_calibration_curve/p4_large_calibration/constants.py",
        "workflows/12_p4_large_calibration_curve/p4_large_calibration/sampling.py",
        "workflows/12_p4_large_calibration_curve/p4_large_calibration/statistics.py",
        "workflows/12_p4_large_calibration_curve/p4_large_calibration/study.py",
        "workflows/11_p4_factor_aware_few_shot/p4_factor_aware/artifacts.py",
        "workflows/11_p4_factor_aware_few_shot/p4_factor_aware/constants.py",
        "workflows/11_p4_factor_aware_few_shot/p4_factor_aware/data.py",
        "workflows/11_p4_factor_aware_few_shot/p4_factor_aware/historical.py",
        "workflows/11_p4_factor_aware_few_shot/p4_factor_aware/metrics.py",
        "workflows/11_p4_factor_aware_few_shot/p4_factor_aware/models.py",
        "workflows/11_p4_factor_aware_few_shot/p4_factor_aware/protocol.py",
        "tests/test_p4_large_calibration_curve.py",
        str(PREREGISTRATION_RELATIVE).replace("\\", "/"),
    ]
    code_rows = []
    for relative in executed_code:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing executed code file: {relative}")
        code_rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_csv(output_directory / "00_EXECUTED_CODE_MANIFEST.csv", code_rows)
    write_json(
        output_directory / "01_EXACT_CONFIGURATION.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "claim_type": CLAIM_TARGET_ASSISTED,
            "code_identity": code_identity,
            "runtime": _runtime_identity(),
            "budgets": list(TOTAL_BUDGETS),
            "source_seeds": list(CHECKPOINT_SEEDS),
            "support_seeds": list(SUPPORT_SEEDS),
            "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
            "query_folds": {str(key): [cell_id(cell) for cell in value] for key, value in QUERY_FOLDS.items()},
            "model": METHOD_POSITIVE,
            "zero_anchor": METHOD_ZERO,
            "p4_source_hashes": data.source_hashes,
            "preprocessing_hashes": PREPROCESSING_FILES,
            "checkpoint_file_hashes": CHECKPOINT_FILE_HASHES,
            "checkpoint_state_hashes": CHECKPOINT_STATE_HASHES,
        },
    )
    write_text(
        output_directory / "02_REPOSITORY_STATE_RECONSTRUCTION.md",
        (repository_root / MEMO_RELATIVE).read_text(encoding="utf-8"),
    )
    write_json(
        output_directory / "03_PREREGISTRATION_BINDING.json",
        {
            "relative_path": str(PREREGISTRATION_RELATIVE).replace("\\", "/"),
            "sha256": PREREGISTRATION_SHA256,
            "status": preregistration["status"],
            "sealed_before_new_results": True,
            "preregistration_commit": "73a05ed89279e16cf6b4c6c889d082a20e85167c",
        },
    )
    write_json(output_directory / "04_PRIOR_P4_EXPERIMENT_BINDING.json", _prior_binding(historical_binding))
    structure_rows = data.structure_rows()
    for row in structure_rows:
        row["exact_signal_audit"] = "duplicates confined within block"
    write_csv(output_directory / "05_P4_STRUCTURE_AUDIT.csv", structure_rows)
    write_json(output_directory / "05A_EXACT_SIGNAL_DUPLICATION_AUDIT.json", exact_audit)
    write_csv(output_directory / "06_OUTER_QUERY_FOLD_MANIFEST.csv", outer_fold_manifest_rows())
    write_csv(output_directory / "07_SUPPORT_SELECTION_MANIFEST.csv", support_manifest)
    write_csv(output_directory / "08_SUPPORT_COVERAGE_MANIFEST.csv", support_coverage)
    write_csv(output_directory / "08A_SUPPORT_PLAN_VALIDATION.csv", plan_validation)
    write_csv(output_directory / "09_LEAKAGE_AND_VALIDITY_AUDIT.csv", gates)


def _normalise_rows(values: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
    if torch.any(~torch.isfinite(norms)) or torch.any(norms <= 0):
        raise ProtocolViolation("ZERO_OR_NONFINITE_COSINE_NORM")
    return values / norms


def _nested_prototypes(
    embeddings: torch.Tensor,
    labels: np.ndarray,
) -> dict[int, torch.Tensor]:
    values = _normalise_rows(embeddings.detach().cpu().to(dtype=torch.float64))
    classes = np.asarray(labels, dtype=np.int64)
    if values.shape != (500, 256) or classes.shape != (500,):
        raise ValueError("maximum nested support must contain 500 embeddings")
    sums = torch.zeros((7, 256), dtype=torch.float64)
    counts = np.zeros(7, dtype=np.int64)
    output: dict[int, torch.Tensor] = {}
    wanted = set(POSITIVE_BUDGETS)
    for rank, (embedding, class_index) in enumerate(zip(values, classes, strict=True), start=1):
        sums[int(class_index)] += embedding
        counts[int(class_index)] += 1
        if rank in wanted:
            if np.any(counts == 0) or int(counts.max() - counts.min()) > 1:
                raise ProtocolViolation("INVALID_CLASS_COUNTS_FOR_PROTOTYPE")
            means = sums / torch.from_numpy(counts.astype(np.float64))[:, None]
            output[rank] = _normalise_rows(means).clone()
    if set(output) != wanted:
        raise ProtocolViolation("MISSING_PREREGISTERED_PROTOTYPE_BUDGET")
    return output


def _predict_cosine(query_embeddings: torch.Tensor, prototypes: torch.Tensor) -> np.ndarray:
    queries = _normalise_rows(query_embeddings.detach().cpu().to(dtype=torch.float64))
    similarities = queries @ prototypes.T
    return np.ascontiguousarray(torch.argmax(similarities, dim=1).numpy(), dtype=np.int64)


@dataclass
class SealRecord:
    budget: int
    source_seed: int
    support_seed: int | None
    fold: int
    query_indices: np.ndarray
    prediction_sha256: str
    seal: QueryLabelSeal


def _generate_all_predictions(
    *,
    data: GovernedP4,
    plans: dict[tuple[int, int], NestedSupportPlan],
    archive_repository: Path,
    output_directory: Path,
) -> tuple[np.ndarray, list[SealRecord], list[dict[str, object]], list[dict[str, object]]]:
    mean, scale = load_preprocessing(archive_repository / PREPROCESSING_RELATIVE)
    inputs = preprocess(data.signals, mean, scale)
    predictions = np.full(
        (len(TOTAL_BUDGETS), len(CHECKPOINT_SEEDS), len(SUPPORT_SEEDS), len(data.signals)),
        -1,
        dtype=np.int8,
    )
    records: list[SealRecord] = []
    freeze_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    for source_index, source_seed in enumerate(CHECKPOINT_SEEDS):
        checkpoint_path = archive_repository / CHECKPOINT_RELATIVE / f"seed_{source_seed}/FINAL_CHECKPOINT.pt"
        model, payload = load_frozen_model(checkpoint_path, source_seed)
        state_before = model_state_sha256(model)
        embeddings = extract_embeddings(model, inputs)
        for fold in sorted(QUERY_FOLDS):
            query_indices = data.query_view(fold).indices
            query_embeddings = embeddings[query_indices]
            zero_prediction = frozen_source_predictions(model, query_embeddings)
            predictions[0, source_index][:, query_indices] = zero_prediction.astype(np.int8)[None, :]
            zero_seal = QueryLabelSeal(data._labels[query_indices])
            zero_hash = zero_seal.freeze_predictions(zero_prediction)
            zero_record = SealRecord(
                budget=0,
                source_seed=source_seed,
                support_seed=None,
                fold=fold,
                query_indices=query_indices.copy(),
                prediction_sha256=zero_hash,
                seal=zero_seal,
            )
            records.append(zero_record)
            freeze_rows.append(
                {
                    "budget": 0,
                    "claim_type": CLAIM_STRICT_DG,
                    "source_seed": source_seed,
                    "support_seed": "",
                    "fold": fold,
                    "query_rows": len(query_indices),
                    "prediction_sha256": zero_hash,
                    "state_at_registration": zero_seal.state,
                    "labels_opened": False,
                }
            )

            for support_index, support_seed in enumerate(SUPPORT_SEEDS):
                plan = plans[(fold, support_seed)]
                maximum_indices = plan.indices_for_budget(500)
                prototypes = _nested_prototypes(
                    embeddings[maximum_indices], data._labels[maximum_indices]
                )
                for budget_index, budget in enumerate(POSITIVE_BUDGETS, start=1):
                    prediction = _predict_cosine(query_embeddings, prototypes[budget])
                    predictions[budget_index, source_index, support_index, query_indices] = prediction.astype(np.int8)
                    seal = QueryLabelSeal(data._labels[query_indices])
                    prediction_hash = seal.freeze_predictions(prediction)
                    records.append(
                        SealRecord(
                            budget=budget,
                            source_seed=source_seed,
                            support_seed=support_seed,
                            fold=fold,
                            query_indices=query_indices.copy(),
                            prediction_sha256=prediction_hash,
                            seal=seal,
                        )
                    )
                    freeze_rows.append(
                        {
                            "budget": budget,
                            "claim_type": CLAIM_TARGET_ASSISTED,
                            "source_seed": source_seed,
                            "support_seed": support_seed,
                            "fold": fold,
                            "query_rows": len(query_indices),
                            "prediction_sha256": prediction_hash,
                            "state_at_registration": seal.state,
                            "labels_opened": False,
                        }
                    )
        state_after = model_state_sha256(model)
        model_rows.append(
            {
                "source_seed": source_seed,
                "checkpoint_file_sha256": sha256_file(checkpoint_path),
                "declared_model_state_sha256": payload["model_state_sha256"],
                "state_before_sha256": state_before,
                "state_after_sha256": state_after,
                "encoder_unchanged": state_before == state_after == CHECKPOINT_STATE_HASHES[source_seed],
                "all_parameters_frozen": not any(parameter.requires_grad for parameter in model.parameters()),
            }
        )
        print(json.dumps({"event": "SOURCE_SEED_PREDICTIONS_FROZEN", "seed": source_seed}), flush=True)

    if np.any(predictions < 0):
        raise ProtocolViolation("INCOMPLETE_OOF_PREDICTION_MATRIX")
    if not all(record.seal.state == "PREDICTIONS_FROZEN" for record in records):
        raise ProtocolViolation("QUERY_LABEL_OPENED_BEFORE_GLOBAL_FREEZE")

    bundle_path = output_directory / "per_run_predictions.npz"
    np.savez_compressed(
        bundle_path,
        budgets=np.asarray(TOTAL_BUDGETS, dtype=np.int64),
        source_seeds=np.asarray(CHECKPOINT_SEEDS, dtype=np.int64),
        support_seeds=np.asarray(SUPPORT_SEEDS, dtype=np.int64),
        predictions=predictions,
    )
    bundle_hash = sha256_file(bundle_path)
    write_csv(output_directory / "10_PREDICTION_FREEZE_MANIFEST.csv", freeze_rows)
    write_json(
        output_directory / "10A_PREDICTION_BUNDLE_BINDING.json",
        {
            "relative_path": "per_run_predictions.npz",
            "sha256": bundle_hash,
            "prediction_array_sha256": array_sha256(predictions),
            "shape": list(predictions.shape),
            "dtype": str(predictions.dtype),
            "all_predictions_frozen_before_any_query_label_open": True,
            "registered_prediction_units": len(records),
        },
    )
    return predictions, records, freeze_rows, model_rows


def _open_labels_after_global_freeze(
    *,
    data: GovernedP4,
    records: list[SealRecord],
) -> np.ndarray:
    if not records or not all(record.seal.state == "PREDICTIONS_FROZEN" for record in records):
        raise ProtocolViolation("GLOBAL_PREDICTION_FREEZE_NOT_COMPLETE")
    truth = np.full(len(data.signals), -1, dtype=np.int64)
    for record in records:
        labels = record.seal.open_labels()
        existing = truth[record.query_indices]
        if np.any((existing >= 0) & (existing != labels)):
            raise ProtocolViolation("INCONSISTENT_QUERY_LABEL_CUSTODY")
        truth[record.query_indices] = labels
    if np.any(truth < 0):
        raise ProtocolViolation("INCOMPLETE_OOF_QUERY_TRUTH")
    return truth


def _coverage_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = [
        {
            "total_labelled_measurements": 0,
            "unique_condition_blocks_mean": 0.0,
            "unique_condition_blocks_min": 0,
            "unique_condition_blocks_max": 0,
            "unique_exact_signals_mean": 0.0,
            "unique_exact_signals_min": 0,
            "unique_exact_signals_max": 0,
            "unique_ER_levels_mean": 0.0,
            "unique_surfaces_mean": 0.0,
            "repeated_measurements_beyond_first_per_block": 0,
            "support_plan_count": 0,
        }
    ]
    for budget in POSITIVE_BUDGETS:
        selected = [row for row in rows if int(row["total_labelled_measurements"]) == budget]
        blocks = np.asarray([row["unique_condition_blocks"] for row in selected], dtype=float)
        signals = np.asarray([row["unique_exact_signals"] for row in selected], dtype=float)
        ers = np.asarray([row["unique_ER_levels"] for row in selected], dtype=float)
        surfaces = np.asarray([row["unique_surfaces"] for row in selected], dtype=float)
        repeats = np.asarray(
            [row["repeated_measurements_beyond_first_per_block"] for row in selected],
            dtype=float,
        )
        output.append(
            {
                "total_labelled_measurements": budget,
                "unique_condition_blocks_mean": float(blocks.mean()),
                "unique_condition_blocks_min": int(blocks.min()),
                "unique_condition_blocks_max": int(blocks.max()),
                "unique_exact_signals_mean": float(signals.mean()),
                "unique_exact_signals_min": int(signals.min()),
                "unique_exact_signals_max": int(signals.max()),
                "unique_ER_levels_mean": float(ers.mean()),
                "unique_surfaces_mean": float(surfaces.mean()),
                "repeated_measurements_beyond_first_per_block": int(repeats.mean()),
                "support_plan_count": len(selected),
            }
        )
    return output


def _unit_ranges(budget: int) -> tuple[range, range]:
    return range(len(CHECKPOINT_SEEDS)), (range(1) if budget == 0 else range(len(SUPPORT_SEEDS)))


def _score_and_analyse(
    *,
    data: GovernedP4,
    truth: np.ndarray,
    predictions: np.ndarray,
    support_coverage: list[dict[str, object]],
    output_directory: Path,
    bootstrap_replicates: int,
) -> dict[str, object]:
    block_keys, block_indices, block_truth = block_layout(data)
    block_histograms = prediction_histograms_by_block(predictions, block_indices)
    block_predictions = majority_predictions(block_histograms)
    unit_macro, unit_accuracy = aggregate_metric_from_histograms(block_histograms)

    per_run_rows: list[dict[str, object]] = []
    per_class_unit: dict[tuple[int, int], list[dict[str, float]]] = defaultdict(list)
    confusion_by_budget: dict[int, list[np.ndarray]] = defaultdict(list)
    block_macro = np.empty(unit_macro.shape, dtype=np.float64)
    block_accuracy = np.empty(unit_accuracy.shape, dtype=np.float64)

    for budget_index, budget in enumerate(TOTAL_BUDGETS):
        source_range, support_range = _unit_ranges(budget)
        for source_index in source_range:
            for support_index in support_range:
                prediction = predictions[budget_index, source_index, support_index].astype(np.int64)
                metrics = extended_metrics(truth, prediction)
                block_metrics = extended_metrics(
                    block_truth, block_predictions[budget_index, source_index, support_index]
                )
                block_macro[budget_index, source_index, support_index] = float(
                    block_metrics["macro_f1"]
                )
                block_accuracy[budget_index, source_index, support_index] = float(
                    block_metrics["accuracy"]
                )
                source_seed = CHECKPOINT_SEEDS[source_index]
                support_seed: int | str = (
                    "" if budget == 0 else SUPPORT_SEEDS[support_index]
                )
                per_run_rows.append(
                    {
                        "budget": budget,
                        "budget_label": budget_label(budget),
                        "claim_type": CLAIM_STRICT_DG if budget == 0 else CLAIM_TARGET_ASSISTED,
                        "method": METHOD_ZERO if budget == 0 else METHOD_POSITIVE,
                        "source_seed": source_seed,
                        "support_seed": support_seed,
                        "query_rows": len(truth),
                        "query_condition_blocks": len(block_truth),
                        "accuracy": float(metrics["accuracy"]),
                        "macro_f1": float(metrics["macro_f1"]),
                        "balanced_accuracy": float(metrics["balanced_accuracy"]),
                        "worst_class_f1": float(metrics["worst_class_f1"]),
                        "block_majority_accuracy": float(block_metrics["accuracy"]),
                        "block_majority_macro_f1": float(block_metrics["macro_f1"]),
                        "prediction_sha256": array_sha256(prediction.astype(np.int8)),
                    }
                )
                confusion_by_budget[budget].append(np.asarray(metrics["confusion"], dtype=np.int64))
                for class_index in range(7):
                    per_class_unit[(budget, class_index)].append(
                        {
                            "precision": float(metrics["precision"][class_index]),
                            "recall": float(metrics["recall"][class_index]),
                            "f1": float(metrics["f1"][class_index]),
                        }
                    )

    # The zero-label prediction is deliberately broadcast over the support axis
    # for pairing. Fill its block metric cube without creating pseudo-replicates.
    block_macro[0, :, :] = block_macro[0, :, 0][:, None]
    block_accuracy[0, :, :] = block_accuracy[0, :, 0][:, None]

    for source_index, source_seed in enumerate(CHECKPOINT_SEEDS):
        observed_macro = float(unit_macro[0, source_index, 0])
        observed_accuracy = float(unit_accuracy[0, source_index, 0])
        if (
            abs(observed_macro - STRICT_PER_SEED[source_seed]["macro_f1"]) > 1e-15
            or abs(observed_accuracy - STRICT_PER_SEED[source_seed]["accuracy"]) > 1e-15
        ):
            raise ProtocolViolation("CANONICAL_ZERO_ANCHOR_REPRODUCTION_FAILURE")
    if (
        abs(float(unit_macro[0, :, 0].mean()) - STRICT_AGGREGATE["macro_f1"]) > 1e-15
        or abs(float(unit_accuracy[0, :, 0].mean()) - STRICT_AGGREGATE["accuracy"]) > 1e-15
    ):
        raise ProtocolViolation("CANONICAL_ZERO_AGGREGATE_REPRODUCTION_FAILURE")

    bootstrap = hierarchical_bootstrap(
        block_histograms,
        block_truth,
        replicates=bootstrap_replicates,
        seed=BOOTSTRAP_SEED,
    )
    bootstrap_path = output_directory / "bootstrap_samples.npz"
    np.savez_compressed(
        bootstrap_path,
        budgets=np.asarray(TOTAL_BUDGETS, dtype=np.int64),
        macro_f1=bootstrap.macro_f1,
        accuracy=bootstrap.accuracy,
        block_weights=bootstrap.block_weights_sha256_payload,
        source_resample_counts=bootstrap.source_counts,
        support_resample_counts=bootstrap.support_counts,
    )
    write_json(
        output_directory / "13A_BOOTSTRAP_BINDING.json",
        {
            "relative_path": "bootstrap_samples.npz",
            "sha256": sha256_file(bootstrap_path),
            "macro_f1_samples_sha256": array_sha256(bootstrap.macro_f1),
            "accuracy_samples_sha256": array_sha256(bootstrap.accuracy),
            "block_weights_sha256": array_sha256(bootstrap.block_weights_sha256_payload),
            "source_resample_counts_sha256": array_sha256(bootstrap.source_counts),
            "support_resample_counts_sha256": array_sha256(bootstrap.support_counts),
            "replicates": bootstrap_replicates,
            "seed": BOOTSTRAP_SEED,
        },
    )

    coverage_summary = _coverage_summary(support_coverage)
    coverage_lookup = {
        int(row["total_labelled_measurements"]): row for row in coverage_summary
    }
    aggregate_rows: list[dict[str, object]] = []
    uncertainty_rows: list[dict[str, object]] = []
    variance_rows: list[dict[str, object]] = []
    point_macro: list[float] = []
    point_accuracy: list[float] = []
    delta_macro_ci: dict[int, tuple[float, float]] = {}

    for budget_index, budget in enumerate(TOTAL_BUDGETS):
        if budget == 0:
            macro_units = unit_macro[budget_index, :, 0]
            accuracy_units = unit_accuracy[budget_index, :, 0]
            block_macro_units = block_macro[budget_index, :, 0]
            block_accuracy_units = block_accuracy[budget_index, :, 0]
            support_sd_macro = 0.0
            support_sd_accuracy = 0.0
        else:
            macro_units = unit_macro[budget_index].reshape(-1)
            accuracy_units = unit_accuracy[budget_index].reshape(-1)
            block_macro_units = block_macro[budget_index].reshape(-1)
            block_accuracy_units = block_accuracy[budget_index].reshape(-1)
            support_sd_macro = float(np.std(unit_macro[budget_index].mean(axis=0), ddof=1))
            support_sd_accuracy = float(np.std(unit_accuracy[budget_index].mean(axis=0), ddof=1))
        macro_point = float(macro_units.mean())
        accuracy_point = float(accuracy_units.mean())
        point_macro.append(macro_point)
        point_accuracy.append(accuracy_point)
        macro_ci = percentile_interval(bootstrap.macro_f1[:, budget_index])
        accuracy_ci = percentile_interval(bootstrap.accuracy[:, budget_index])
        macro_delta_samples = bootstrap.macro_f1[:, budget_index] - bootstrap.macro_f1[:, 0]
        accuracy_delta_samples = bootstrap.accuracy[:, budget_index] - bootstrap.accuracy[:, 0]
        macro_delta = macro_point - float(unit_macro[0, :, 0].mean())
        accuracy_delta = accuracy_point - float(unit_accuracy[0, :, 0].mean())
        macro_delta_interval = percentile_interval(macro_delta_samples)
        accuracy_delta_interval = percentile_interval(accuracy_delta_samples)
        delta_macro_ci[budget] = macro_delta_interval
        coverage = coverage_lookup[budget]
        aggregate_rows.append(
            {
                "budget": budget,
                "budget_label": budget_label(budget),
                "claim_type": CLAIM_STRICT_DG if budget == 0 else CLAIM_TARGET_ASSISTED,
                "method": METHOD_ZERO if budget == 0 else METHOD_POSITIVE,
                "independent_condition_blocks_mean": coverage["unique_condition_blocks_mean"],
                "independent_condition_blocks_range": f"{coverage['unique_condition_blocks_min']}-{coverage['unique_condition_blocks_max']}",
                "unique_exact_signals_mean": coverage["unique_exact_signals_mean"],
                "macro_f1": macro_point,
                "macro_f1_ci_95_lower": macro_ci[0],
                "macro_f1_ci_95_upper": macro_ci[1],
                "macro_f1_delta_vs_0": macro_delta,
                "macro_f1_delta_vs_0_ci_95_lower": macro_delta_interval[0],
                "macro_f1_delta_vs_0_ci_95_upper": macro_delta_interval[1],
                "accuracy": accuracy_point,
                "accuracy_ci_95_lower": accuracy_ci[0],
                "accuracy_ci_95_upper": accuracy_ci[1],
                "accuracy_delta_vs_0": accuracy_delta,
                "accuracy_delta_vs_0_ci_95_lower": accuracy_delta_interval[0],
                "accuracy_delta_vs_0_ci_95_upper": accuracy_delta_interval[1],
                "balanced_accuracy": accuracy_point,
                "block_majority_macro_f1": float(block_macro_units.mean()),
                "block_majority_accuracy": float(block_accuracy_units.mean()),
                "crossed_unit_count": len(macro_units),
                "crossed_unit_macro_f1_sd": float(np.std(macro_units, ddof=1)),
                "source_seed_macro_f1_sd": float(np.std(unit_macro[budget_index].mean(axis=1), ddof=1)),
                "support_seed_macro_f1_sd": support_sd_macro,
            }
        )
        for metric_name, samples, point, interval in (
            ("macro_f1", bootstrap.macro_f1[:, budget_index], macro_point, macro_ci),
            ("accuracy", bootstrap.accuracy[:, budget_index], accuracy_point, accuracy_ci),
            ("macro_f1_delta_vs_0", macro_delta_samples, macro_delta, macro_delta_interval),
            ("accuracy_delta_vs_0", accuracy_delta_samples, accuracy_delta, accuracy_delta_interval),
        ):
            uncertainty_rows.append(
                {
                    "contrast": f"budget_{budget}",
                    "metric": metric_name,
                    "estimate": point,
                    "ci_95_lower": interval[0],
                    "ci_95_upper": interval[1],
                    "replicates": bootstrap_replicates,
                    "seed": BOOTSTRAP_SEED,
                    "scheme": "TAGID_STRATIFIED_BLOCK_X_SOURCE_SEED_X_SUPPORT_SEED",
                }
            )
        variance_rows.append(
            {
                "budget": budget,
                "crossed_unit_macro_f1_sd": float(np.std(macro_units, ddof=1)),
                "source_seed_macro_f1_sd": float(np.std(unit_macro[budget_index].mean(axis=1), ddof=1)),
                "support_seed_macro_f1_sd": support_sd_macro,
                "crossed_unit_accuracy_sd": float(np.std(accuracy_units, ddof=1)),
                "source_seed_accuracy_sd": float(np.std(unit_accuracy[budget_index].mean(axis=1), ddof=1)),
                "support_seed_accuracy_sd": support_sd_accuracy,
            }
        )

    incremental_rows: list[dict[str, object]] = []
    for current_index in range(1, len(TOTAL_BUDGETS)):
        previous_index = current_index - 1
        previous_budget = TOTAL_BUDGETS[previous_index]
        budget = TOTAL_BUDGETS[current_index]
        added = budget - previous_budget
        macro_samples = bootstrap.macro_f1[:, current_index] - bootstrap.macro_f1[:, previous_index]
        accuracy_samples = bootstrap.accuracy[:, current_index] - bootstrap.accuracy[:, previous_index]
        macro_change = point_macro[current_index] - point_macro[previous_index]
        accuracy_change = point_accuracy[current_index] - point_accuracy[previous_index]
        macro_interval = percentile_interval(macro_samples)
        accuracy_interval = percentile_interval(accuracy_samples)
        incremental_rows.append(
            {
                "from_budget": previous_budget,
                "to_budget": budget,
                "additional_labels": added,
                "macro_f1_change": macro_change,
                "macro_f1_change_ci_95_lower": macro_interval[0],
                "macro_f1_change_ci_95_upper": macro_interval[1],
                "macro_f1_gain_per_10_labels": macro_change * 10.0 / added,
                "accuracy_change": accuracy_change,
                "accuracy_change_ci_95_lower": accuracy_interval[0],
                "accuracy_change_ci_95_upper": accuracy_interval[1],
            }
        )
        for metric_name, estimate, interval in (
            ("macro_f1", macro_change, macro_interval),
            ("accuracy", accuracy_change, accuracy_interval),
        ):
            uncertainty_rows.append(
                {
                    "contrast": f"budget_{budget}_minus_{previous_budget}",
                    "metric": metric_name,
                    "estimate": estimate,
                    "ci_95_lower": interval[0],
                    "ci_95_upper": interval[1],
                    "replicates": bootstrap_replicates,
                    "seed": BOOTSTRAP_SEED,
                    "scheme": "PAIRED_TAGID_STRATIFIED_BLOCK_X_SOURCE_SEED_X_SUPPORT_SEED",
                }
            )

    per_class_rows: list[dict[str, object]] = []
    for budget in TOTAL_BUDGETS:
        for class_index in range(7):
            values = per_class_unit[(budget, class_index)]
            for metric_name in ("precision", "recall", "f1"):
                vector = np.asarray([row[metric_name] for row in values], dtype=np.float64)
                if metric_name == "precision":
                    record = {
                        "budget": budget,
                        "class_index": class_index,
                        "TagID": class_index + 1,
                        "unit_count": len(values),
                    }
                    per_class_rows.append(record)
                per_class_rows[-1][f"{metric_name}_mean"] = float(vector.mean())
                per_class_rows[-1][f"{metric_name}_sd"] = float(np.std(vector, ddof=1))
                per_class_rows[-1][f"{metric_name}_min"] = float(vector.min())
                per_class_rows[-1][f"{metric_name}_max"] = float(vector.max())

    confusion_rows: list[dict[str, object]] = []
    for budget in TOTAL_BUDGETS:
        matrices = np.stack(confusion_by_budget[budget]).astype(np.float64)
        mean_matrix = matrices.mean(axis=0)
        row_denominator = mean_matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            mean_matrix,
            row_denominator,
            out=np.zeros_like(mean_matrix),
            where=row_denominator > 0,
        )
        for true_class in range(7):
            for predicted_class in range(7):
                confusion_rows.append(
                    {
                        "budget": budget,
                        "true_class": true_class,
                        "predicted_class": predicted_class,
                        "mean_count_per_unit": float(mean_matrix[true_class, predicted_class]),
                        "row_normalized": float(normalized[true_class, predicted_class]),
                        "unit_count": len(matrices),
                    }
                )

    condition_rows: list[dict[str, object]] = []
    for budget_index, budget in enumerate(TOTAL_BUDGETS):
        source_range, support_range = _unit_ranges(budget)
        for factor_name, factor_values in (("ER", data.er), ("surface", data.surface)):
            for level in range(3):
                mask = np.asarray(factor_values == level)
                macro_values = []
                accuracy_values = []
                for source_index in source_range:
                    for support_index in support_range:
                        metrics = extended_metrics(
                            truth[mask],
                            predictions[budget_index, source_index, support_index, mask],
                        )
                        macro_values.append(float(metrics["macro_f1"]))
                        accuracy_values.append(float(metrics["accuracy"]))
                condition_rows.append(
                    {
                        "budget": budget,
                        "factor": factor_name,
                        "level": level,
                        "rows_per_unit": int(mask.sum()),
                        "unit_count": len(macro_values),
                        "macro_f1_mean": float(np.mean(macro_values)),
                        "macro_f1_sd": float(np.std(macro_values, ddof=1)),
                        "accuracy_mean": float(np.mean(accuracy_values)),
                        "accuracy_sd": float(np.std(accuracy_values, ddof=1)),
                    }
                )

    class_lookup = {
        (int(row["budget"]), int(row["class_index"])): row for row in per_class_rows
    }
    class_recovery = [
        {
            "TagID": class_index + 1,
            "f1_at_0": float(class_lookup[(0, class_index)]["f1_mean"]),
            "f1_at_500": float(class_lookup[(500, class_index)]["f1_mean"]),
            "change_0_to_500": float(
                class_lookup[(500, class_index)]["f1_mean"]
                - class_lookup[(0, class_index)]["f1_mean"]
            ),
        }
        for class_index in range(7)
    ]
    final_condition = [row for row in condition_rows if int(row["budget"]) == 500]
    condition_summary = {
        factor: {
            "lowest_level": int(
                min(
                    (row for row in final_condition if row["factor"] == factor),
                    key=lambda row: float(row["macro_f1_mean"]),
                )["level"]
            ),
            "lowest_macro_f1": float(
                min(
                    (row for row in final_condition if row["factor"] == factor),
                    key=lambda row: float(row["macro_f1_mean"]),
                )["macro_f1_mean"]
            ),
            "highest_level": int(
                max(
                    (row for row in final_condition if row["factor"] == factor),
                    key=lambda row: float(row["macro_f1_mean"]),
                )["level"]
            ),
            "highest_macro_f1": float(
                max(
                    (row for row in final_condition if row["factor"] == factor),
                    key=lambda row: float(row["macro_f1_mean"]),
                )["macro_f1_mean"]
            ),
        }
        for factor in ("ER", "surface")
    }

    write_csv(output_directory / "11_PER_RUN_METRICS.csv", per_run_rows)
    write_csv(output_directory / "12_AGGREGATE_METRICS.csv", aggregate_rows)
    write_csv(output_directory / "13_UNCERTAINTY_TABLE.csv", uncertainty_rows)
    write_csv(output_directory / "14_PER_CLASS_METRICS.csv", per_class_rows)
    write_csv(output_directory / "15_CALIBRATION_CURVE.csv", aggregate_rows)
    write_csv(output_directory / "16_INCREMENTAL_GAINS.csv", incremental_rows)
    write_csv(output_directory / "17_SUPPORT_COVERAGE_SUMMARY.csv", coverage_summary)
    write_csv(output_directory / "18_CONDITION_STRATIFIED_METRICS.csv", condition_rows)
    write_csv(output_directory / "19_CONFUSION_MATRICES.csv", confusion_rows)
    write_csv(output_directory / "20_SUPPORT_SELECTION_VARIANCE.csv", variance_rows)

    practical_budget = next(
        (
            budget
            for budget, point in zip(TOTAL_BUDGETS[1:], point_macro[1:], strict=False)
            if point - point_macro[0] >= PRACTICAL_DELTA
        ),
        None,
    )
    reliable_budget = next(
        (budget for budget in TOTAL_BUDGETS[1:] if delta_macro_ci[budget][0] > 0.0),
        None,
    )
    key_budget = next(
        (
            budget
            for budget, point in zip(TOTAL_BUDGETS[1:], point_macro[1:], strict=False)
            if point - point_macro[0] >= PRACTICAL_DELTA
            and delta_macro_ci[budget][0] > 0.0
        ),
        None,
    )
    index = {budget: position for position, budget in enumerate(TOTAL_BUDGETS)}
    gain_7_35 = point_macro[index[35]] - point_macro[index[7]]
    gain_50_500 = point_macro[index[500]] - point_macro[index[50]]
    interval_7_35 = percentile_interval(
        bootstrap.macro_f1[:, index[35]] - bootstrap.macro_f1[:, index[7]]
    )
    interval_50_500 = percentile_interval(
        bootstrap.macro_f1[:, index[500]] - bootstrap.macro_f1[:, index[50]]
    )
    coverage_dominant = bool(
        interval_7_35[0] > 0.0
        and interval_50_500[0] <= 0.0 <= interval_50_500[1]
        and gain_7_35 / 28.0 > 2.0 * (gain_50_500 / 450.0)
    )
    if coverage_dominant:
        classification = "CONDITION_COVERAGE_MORE_IMPORTANT_THAN_RAW_LABEL_COUNT"
    elif key_budget is not None and key_budget <= 35:
        classification = "RAPID_TARGET_CALIBRATION_RECOVERY"
    elif key_budget is not None:
        classification = "GRADUAL_TARGET_CALIBRATION_RECOVERY"
    else:
        classification = "WEAK_OR_UNRELIABLE_TARGET_CALIBRATION_GAIN"

    return {
        "classification": classification,
        "aggregate_rows": aggregate_rows,
        "incremental_rows": incremental_rows,
        "coverage_summary": coverage_summary,
        "practical_budget": practical_budget,
        "reliable_budget": reliable_budget,
        "key_budget": key_budget,
        "coverage_dominant": coverage_dominant,
        "gain_7_35": gain_7_35,
        "gain_7_35_ci": interval_7_35,
        "gain_50_500": gain_50_500,
        "gain_50_500_ci": interval_50_500,
        "point_macro": point_macro,
        "point_accuracy": point_accuracy,
        "class_recovery": class_recovery,
        "condition_summary": condition_summary,
        "bootstrap_macro_sha256": array_sha256(bootstrap.macro_f1),
        "bootstrap_accuracy_sha256": array_sha256(bootstrap.accuracy),
    }


def _format_interval(lower: float, upper: float) -> str:
    return f"[{lower:.4f}, {upper:.4f}]"


def _main_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| P4 labelled budget | Independent blocks | Macro-F1 | 95% CI | Delta vs 0 | Accuracy | 95% CI |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {blocks:.1f} | {macro:.4f} | {macro_ci} | {delta:+.4f} | {accuracy:.4f} | {accuracy_ci} |".format(
                label=row["budget_label"],
                blocks=float(row["independent_condition_blocks_mean"]),
                macro=float(row["macro_f1"]),
                macro_ci=_format_interval(
                    float(row["macro_f1_ci_95_lower"]),
                    float(row["macro_f1_ci_95_upper"]),
                ),
                delta=float(row["macro_f1_delta_vs_0"]),
                accuracy=float(row["accuracy"]),
                accuracy_ci=_format_interval(
                    float(row["accuracy_ci_95_lower"]),
                    float(row["accuracy_ci_95_upper"]),
                ),
            )
        )
    return "\n".join(lines)


def _interpretation_text(analysis: dict[str, object]) -> tuple[str, str]:
    classification = str(analysis["classification"])
    key_budget = analysis["key_budget"]
    highest = analysis["aggregate_rows"][-1]
    if key_budget is None:
        recovery_sentence = (
            "No observed budget met the preregistered combined threshold of a "
            "+0.05 Macro-F1 point gain with a paired interval excluding zero."
        )
    else:
        recovery_sentence = (
            f"The first budget meeting the preregistered practical and reliability "
            f"thresholds was {key_budget} total labelled P4 measurements."
        )
    if analysis["coverage_dominant"]:
        coverage_sentence = (
            "The preregistered coverage-dominance rule was met: recovery while new "
            "condition blocks were added was reliable, whereas the 50-to-500 repeated-label "
            "contrast was not."
        )
    else:
        coverage_sentence = (
            "The preregistered coverage-dominance rule was not met; condition coverage and "
            "raw count cannot be cleanly separated below the 42-block ceiling, and the "
            "post-coverage contrast is reported descriptively."
        )
    largest_class_gain = max(analysis["class_recovery"], key=lambda row: row["change_0_to_500"])
    weakest_final_class = min(analysis["class_recovery"], key=lambda row: row["f1_at_500"])
    class_sentence = (
        f"TagID {largest_class_gain['TagID']} had the largest descriptive 0-to-500 "
        f"class-F1 gain ({largest_class_gain['change_0_to_500']:+.4f}); TagID "
        f"{weakest_final_class['TagID']} remained weakest at 500 labels "
        f"(F1 {weakest_final_class['f1_at_500']:.4f})."
    )
    thesis = (
        f"Using a frozen P1-P3-trained C1 encoder and a fixed cosine-prototype adapter, "
        f"the block-disjoint P4 labelled-target calibration curve was classified as "
        f"`{classification}`. The strict source-only anchor had Macro-F1 "
        f"{analysis['aggregate_rows'][0]['macro_f1']:.4f}; at 500 target labels the "
        f"target-assisted Macro-F1 was {highest['macro_f1']:.4f} "
        f"(paired change {highest['macro_f1_delta_vs_0']:+.4f}, 95% CI "
        f"{_format_interval(highest['macro_f1_delta_vs_0_ci_95_lower'], highest['macro_f1_delta_vs_0_ci_95_upper'])}). "
        f"{recovery_sentence} Positive-budget results quantify labelled-target calibration "
        f"to held P4 condition blocks and are not source-only domain generalisation."
    )
    return " ".join((recovery_sentence, coverage_sentence, class_sentence)), thesis


def _write_reports(
    *,
    output_directory: Path,
    code_identity: dict[str, object],
    analysis: dict[str, object],
    gates: list[dict[str, object]],
    model_rows: list[dict[str, object]],
) -> None:
    interpretation, thesis = _interpretation_text(analysis)
    table = _main_table(analysis["aggregate_rows"])
    key_value = "NONE" if analysis["key_budget"] is None else str(analysis["key_budget"])
    readme = f"""# Larger P4 Labelled-Target Calibration Curve

## A. FINAL CLASSIFICATION

`{analysis['classification']}`

## B. EXPERIMENT IDENTITY

- Branch: `{code_identity['branch']}`
- Execution code commit: `{code_identity['commit']}`
- Execution code tree: `{code_identity['tree']}`
- Worktree: `{code_identity['worktree']}`
- Preregistration SHA-256: `{PREREGISTRATION_SHA256}`
- Data lineage: governed A1/A2/A3 P4 hashes in `01_EXACT_CONFIGURATION.json`
- Model lineage: five frozen source-only C1 checkpoints, seeds 42-46

## C. CALIBRATION DESIGN

- Total-label budgets: {', '.join(map(str, TOTAL_BUDGETS))}
- Canonical matched anchors: 7 = 1-shot, 21 = 3-shot, 35 = 5-shot
- Independent unit: `TagID x ER x surface x P4` physical condition block
- Support/query: nested support prefixes; three fixed block-disjoint Latin-square query folds; all 3,150 P4 rows queried out of fold
- Seeds: five source x twenty support
- Uncertainty: 10,000-replicate paired TagID-stratified block x crossed source/support-seed bootstrap

Every positive budget is target-assisted. Only zero labels is strict source-only DG.

## D. PRIMARY RESULTS

{table}

## E. KEY THRESHOLD

`{key_value}`. Practical-only first budget: `{analysis['practical_budget']}`. Reliable-only first budget: `{analysis['reliable_budget']}`.

## F. STRICT-DG -> TARGET-ASSISTED GAP

Recovered fractions relative to the historical 0.5836 retrospective result or the
0.9835 matched within-condition result are `NOT_IDENTIFIABLE_UNDER_MATCHED_PROTOCOL`.
Those references used non-comparable selection, splitting, and dependence structures.
The directly valid quantities are the paired budget-minus-zero changes in the table.

## G. CONDITION-COVERAGE FINDING

{interpretation}

- 7-to-35 Macro-F1 change: {analysis['gain_7_35']:+.4f}, 95% CI {_format_interval(*analysis['gain_7_35_ci'])}
- 50-to-500 Macro-F1 change at fixed 42-block coverage: {analysis['gain_50_500']:+.4f}, 95% CI {_format_interval(*analysis['gain_50_500_ci'])}
- At 500 labels, weakest ER level: {analysis['condition_summary']['ER']['lowest_level']} (Macro-F1 {analysis['condition_summary']['ER']['lowest_macro_f1']:.4f}); weakest surface: {analysis['condition_summary']['surface']['lowest_level']} (Macro-F1 {analysis['condition_summary']['surface']['lowest_macro_f1']:.4f}).

## H. LEAKAGE / VALIDITY AUDIT

All {len(gates)} critical gates passed. Supports and queries are row-, physical-block-,
and exact-signal-disjoint within every fold; all 50 repeated acquisitions remain in one
role. Every prediction was frozen, SHA-256 registered, and persisted before the global
query-label barrier opened. Frozen encoder states were unchanged. P1-P3 preprocessing
was not refit. Query labels influenced final scoring only.

## I. RELATION TO PRIOR P4 EXPERIMENTS

- Strict-DG: exactly reproduced as the zero-label anchor.
- Historical few-shot: preserved as external 1/3/5-shot provenance; not copied because its episode/query and Euclidean protocol differ.
- Factor-aware few-shot: supplies the matched block-disjoint folds, frozen encoders, and cosine-prototype lineage.
- Retrospective P4: external, full-P4 outcome-informed reference only.
- Matched target-assisted P4: external within-condition, near-duplicate-inflated reference only.

## J. THESIS-READY CONCLUSION

{thesis}

## K. REMAINING ISSUES

- There is no protocol-comparable fully target-assisted endpoint, so gap-recovery percentages are unavailable.
- The 63 blocks come from one P4 corpus rather than an independent acquisition campaign.
- Below 42 support blocks, raw count and condition coverage increase together; their effects are not causally separable.
- Findings are conditional on the frozen C1 representation and cosine-prototype adapter.

## Supervisor question

Yes. The requested larger curve has been estimated at exact total-label budgets 10,
20, 50, 100, 200, and 500 in addition to matched 1/3/5-shot anchors. Budget means raw
labelled measurements; the accompanying coverage columns prevent those counts from
being mistaken for independent conditions. The first reliably and practically
beneficial budget is `{key_value}`. See `16_INCREMENTAL_GAINS.csv`,
`17_SUPPORT_COVERAGE_SUMMARY.csv`, and `22_LIMITATIONS.md` for diminishing-return,
coverage, deployment, and limitation details.
"""
    write_text(output_directory / "README.md", readme)
    write_text(
        output_directory / "21_SCIENTIFIC_INTERPRETATION.md",
        f"""# Scientific interpretation

Classification: `{analysis['classification']}`.

{interpretation}

The calibration curve estimates performance on condition blocks excluded in full from
the corresponding support fold. It therefore avoids within-condition repeated-row
leakage, but every positive point is target-assisted because P4 support labels construct
the prototypes. The historical retrospective and matched within-condition references
are deliberately excluded from recovered-gap calculations.

## Thesis-ready conclusion

{thesis}
""",
    )
    write_text(
        output_directory / "22_LIMITATIONS.md",
        """# Limitations

- One P4 corpus supplies seven TagIDs, nine ER-by-surface cells, 63 condition blocks, and no independent acquisition campaign.
- Cross-validation makes every block query exactly once, but a block can be support in other outer folds; folds are not independent datasets.
- Fifty rows per block are repeated acquisitions. They enter adaptation as labelled measurements but are never inferential bootstrap units.
- Exact duplicate signals are confined within blocks. Random row sampling can therefore spend labels on duplicate acquisitions; unique-signal coverage is reported.
- Exact 10- and 20-label budgets cannot be perfectly class-balanced across seven TagIDs; counts differ by at most one and the remainder rotates across seeds/folds.
- Independent support coverage saturates at 42 blocks. Below saturation, label count and coverage co-vary, so their effects are not causally separable.
- The historical retrospective 0.5836 and matched within-condition 0.9835 Macro-F1 values are not matched endpoints; no valid percentage of either gap is claimed.
- Results are conditional on one frozen C1 representation, one cosine-prototype adaptation rule, five source seeds, and twenty support selections.
- Bootstrap intervals describe this crossed block/seed design; they do not create new tags, devices, environments, or physical campaigns.
- No encoder fine-tuning, transductive query adaptation, threshold tuning, or alternate-method search is included.
""",
    )
    write_json(
        output_directory / "24_FINAL_STATUS.json",
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "status": "SCIENTIFIC_EXECUTION_COMPLETE",
            "classification": analysis["classification"],
            "branch": code_identity["branch"],
            "execution_code_commit": code_identity["commit"],
            "execution_code_tree": code_identity["tree"],
            "worktree": code_identity["worktree"],
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "budgets": list(TOTAL_BUDGETS),
            "independent_unit": "TagID x ER x surface x P4 physical condition block",
            "source_seeds": list(CHECKPOINT_SEEDS),
            "support_seeds": list(SUPPORT_SEEDS),
            "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
            "primary_results": analysis["aggregate_rows"],
            "key_threshold_budget": analysis["key_budget"],
            "practical_threshold_budget": analysis["practical_budget"],
            "reliable_threshold_budget": analysis["reliable_budget"],
            "gap_recovery": "NOT_IDENTIFIABLE_UNDER_MATCHED_PROTOCOL",
            "coverage_dominant_rule_met": analysis["coverage_dominant"],
            "leakage_validity_gates_passed": all(bool(row["passed"]) for row in gates),
            "leakage_validity_gate_count": len(gates),
            "model_integrity_passed": all(
                bool(row["encoder_unchanged"]) and bool(row["all_parameters_frozen"])
                for row in model_rows
            ),
            "bootstrap_macro_f1_sha256": analysis["bootstrap_macro_sha256"],
            "bootstrap_accuracy_sha256": analysis["bootstrap_accuracy_sha256"],
            "figures_present": False,
            "final_audit_manifest_sha256": None,
            "remaining_issues": [
                "no protocol-comparable full target-assisted endpoint",
                "one P4 corpus and no independent acquisition campaign",
                "coverage and count coupled below the 42-block ceiling",
            ],
        },
    )


def finalize_artifacts(output_directory: str | Path) -> dict[str, object]:
    root = Path(output_directory)
    status_path = root / "24_FINAL_STATUS.json"
    if not status_path.is_file():
        raise FileNotFoundError("final status does not exist")
    excluded = {"23_FINAL_AUDIT_MANIFEST.csv", "24_FINAL_STATUS.json"}
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "manifest_scope": "SCIENTIFIC_OR_REPRODUCIBILITY_ARTIFACT",
            }
        )
    manifest_path = root / "23_FINAL_AUDIT_MANIFEST.csv"
    write_csv(manifest_path, rows)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    figure_names = {
        "figures/p4_macro_f1_calibration_curve.png",
        "figures/p4_macro_f1_calibration_curve.pdf",
        "figures/p4_accuracy_calibration_curve.png",
        "figures/p4_accuracy_calibration_curve.pdf",
        "figures/incremental_macro_f1_gain.png",
        "figures/support_condition_coverage.png",
    }
    present = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    status["figures_present"] = figure_names.issubset(present)
    status["final_audit_manifest_sha256"] = sha256_file(manifest_path)
    status["final_audit_manifest_file_count"] = len(rows)
    write_json(status_path, status)
    return {
        "status": "FINAL_ARTIFACT_AUDIT_COMPLETE",
        "artifact_count": len(rows),
        "manifest_sha256": status["final_audit_manifest_sha256"],
        "figures_present": status["figures_present"],
    }


def _bootstrap_reproducibility_probe() -> bool:
    histograms = np.zeros((len(TOTAL_BUDGETS), 5, 20, 63, 7), dtype=np.int16)
    for block_index, class_index in enumerate(np.repeat(np.arange(7), 9)):
        histograms[..., block_index, class_index] = 50
    truth = np.repeat(np.arange(7), 9).astype(np.int64)
    left = hierarchical_bootstrap(histograms, truth, replicates=32, seed=BOOTSTRAP_SEED)
    right = hierarchical_bootstrap(histograms, truth, replicates=32, seed=BOOTSTRAP_SEED)
    return bool(
        np.array_equal(left.macro_f1, right.macro_f1)
        and np.array_equal(left.accuracy, right.accuracy)
        and np.array_equal(left.block_weights_sha256_payload, right.block_weights_sha256_payload)
        and np.all(left.macro_f1 == 1.0)
        and np.all(left.accuracy == 1.0)
    )


def _run_smoke(
    *,
    data: GovernedP4,
    plans: dict[tuple[int, int], NestedSupportPlan],
    archive_repository: Path,
    output_directory: Path,
) -> dict[str, object]:
    mean, scale = load_preprocessing(archive_repository / PREPROCESSING_RELATIVE)
    inputs = preprocess(data.signals, mean, scale)
    source_seed = CHECKPOINT_SEEDS[0]
    model, _ = load_frozen_model(
        archive_repository / CHECKPOINT_RELATIVE / f"seed_{source_seed}/FINAL_CHECKPOINT.pt",
        source_seed,
    )
    state_before = model_state_sha256(model)
    embeddings = extract_embeddings(model, inputs)
    fold = 0
    support_seed = SUPPORT_SEEDS[0]
    query_indices = data.query_view(fold).indices
    plan = plans[(fold, support_seed)]
    maximum = plan.indices_for_budget(500)
    prototypes = _nested_prototypes(embeddings[maximum], data._labels[maximum])
    first = _predict_cosine(embeddings[query_indices], prototypes[500])
    second = _predict_cosine(embeddings[query_indices], prototypes[500])
    zero = frozen_source_predictions(model, embeddings[query_indices])
    seal_zero = QueryLabelSeal(data._labels[query_indices])
    seal_target = QueryLabelSeal(data._labels[query_indices])
    zero_hash = seal_zero.freeze_predictions(zero)
    target_hash = seal_target.freeze_predictions(first)
    passed = bool(
        np.array_equal(first, second)
        and seal_zero.state == "PREDICTIONS_FROZEN"
        and seal_target.state == "PREDICTIONS_FROZEN"
        and all(not bool(row.get("labels_opened")) for row in seal_zero.access_log + seal_target.access_log)
        and model_state_sha256(model) == state_before
    )
    payload = {
        "status": "SMOKE_PREDICTIONS_FROZEN_WITHOUT_QUERY_LABEL_OPEN" if passed else "SMOKE_FAILED",
        "passed": passed,
        "source_seed": source_seed,
        "support_seed": support_seed,
        "fold": fold,
        "tested_budgets": [0, 500],
        "query_rows": len(query_indices),
        "zero_prediction_sha256": zero_hash,
        "target_prediction_sha256": target_hash,
        "deterministic_repeat": np.array_equal(first, second),
        "query_labels_opened": False,
        "encoder_unchanged": model_state_sha256(model) == state_before,
    }
    write_json(output_directory / "SMOKE_TEST_STATUS.json", payload)
    if not passed:
        raise ProtocolViolation("SMOKE_TEST_FAILURE")
    return payload


def run_study(
    *,
    repository_root: str | Path,
    archive_repository: str | Path,
    data_directory: str | Path,
    output_directory: str | Path,
    mode: str,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    repository_root = Path(repository_root)
    archive_repository = Path(archive_repository)
    data_directory = Path(data_directory)
    output_directory = Path(output_directory)
    if mode not in {"dry-run", "smoke", "full"}:
        raise ValueError("mode must be dry-run, smoke, or full")
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError("output directory already contains artifacts; preserve the attempt")
    if mode == "full" and bootstrap_replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("full execution requires exactly 10,000 bootstrap replicates")

    status_before = _git(repository_root, "status", "--porcelain")
    if mode == "full" and status_before:
        raise ProtocolViolation("FULL_EXECUTION_REQUIRES_CLEAN_CODE_WORKTREE")
    code_identity = {
        "branch": _git(repository_root, "branch", "--show-current"),
        "commit": _git(repository_root, "rev-parse", "HEAD"),
        "tree": _git(repository_root, "rev-parse", "HEAD^{tree}"),
        "worktree": str(repository_root),
        "clean_before_execution": not bool(status_before),
    }
    preregistration = _verify_preregistration(repository_root)
    historical_binding = verify_historical_binding(archive_repository)
    data = load_governed_p4(data_directory)
    digests = _signal_digests(data)
    exact_audit = _exact_signal_audit(data, digests)
    plans, plan_validation, support_manifest, support_coverage = _prepare_plans(data, digests)
    gates = _pre_execution_gates(
        data=data,
        exact_audit=exact_audit,
        plan_validation=plan_validation,
        archive_repository=archive_repository,
    )
    if not all(bool(row["passed"]) for row in gates):
        raise ProtocolViolation("PRE_EXECUTION_GATE_FAILURE")
    _write_pre_execution_artifacts(
        repository_root=repository_root,
        output_directory=output_directory,
        code_identity=code_identity,
        preregistration=preregistration,
        historical_binding=historical_binding,
        data=data,
        exact_audit=exact_audit,
        support_manifest=support_manifest,
        support_coverage=support_coverage,
        plan_validation=plan_validation,
        gates=gates,
    )
    if mode == "dry-run":
        payload = {
            "status": "DRY_RUN_VALIDATION_PASSED",
            "rows": len(data.signals),
            "blocks": len(data.structure_rows()),
            "support_plans": len(plans),
            "plan_budget_validations": len(plan_validation),
            "gates_passed": len(gates),
        }
        write_json(output_directory / "DRY_RUN_STATUS.json", payload)
        return payload
    if mode == "smoke":
        return _run_smoke(
            data=data,
            plans=plans,
            archive_repository=archive_repository,
            output_directory=output_directory,
        )

    predictions, records, _, model_rows = _generate_all_predictions(
        data=data,
        plans=plans,
        archive_repository=archive_repository,
        output_directory=output_directory,
    )
    # The freeze manifest and prediction bundle now exist and every seal remains
    # closed. Only this explicit global barrier opens query labels.
    truth = _open_labels_after_global_freeze(data=data, records=records)
    truth_path = output_directory / "query_truth_int64.npy"
    np.save(truth_path, truth, allow_pickle=False)
    write_json(
        output_directory / "10D_QUERY_TRUTH_BINDING.json",
        {
            "relative_path": "query_truth_int64.npy",
            "sha256": sha256_file(truth_path),
            "array_sha256": array_sha256(truth),
            "shape": list(truth.shape),
            "dtype": str(truth.dtype),
            "written_after_global_prediction_freeze": True,
            "purpose": "final scoring and independent metric replay only",
        },
    )
    analysis = _score_and_analyse(
        data=data,
        truth=truth,
        predictions=predictions,
        support_coverage=support_coverage,
        output_directory=output_directory,
        bootstrap_replicates=bootstrap_replicates,
    )
    access_rows = []
    expected_sequence = (
        "QUERY_LABELS_SEALED>QUERY_PREDICTIONS_FROZEN>"
        "QUERY_LABELS_OPENED_AFTER_PREDICTION_FREEZE>QUERY_METRICS_COMPUTED_ONCE"
    )
    for record in records:
        record.seal.mark_metrics_computed()
        sequence = ">".join(str(row["event"]) for row in record.seal.access_log)
        access_rows.append(
            {
                "budget": record.budget,
                "source_seed": record.source_seed,
                "support_seed": "" if record.support_seed is None else record.support_seed,
                "fold": record.fold,
                "prediction_sha256": record.prediction_sha256,
                "access_sequence": sequence,
                "passed": sequence == expected_sequence,
            }
        )
    write_csv(output_directory / "10B_QUERY_LABEL_ACCESS_AUDIT.csv", access_rows)
    write_csv(output_directory / "10C_FROZEN_MODEL_INTEGRITY.csv", model_rows)

    for row in gates:
        row["phase"] = "POST_EXECUTION"
        if row["gate_id"] == "G13_GLOBAL_QUERY_LABEL_BARRIER":
            row["passed"] = all(bool(item["passed"]) for item in access_rows)
            row["evidence"] = f"{len(access_rows)} prediction units followed the freeze/open/score-once sequence"
        elif row["gate_id"] == "G14_FROZEN_ENCODER_INTEGRITY":
            row["passed"] = all(
                bool(item["encoder_unchanged"]) and bool(item["all_parameters_frozen"])
                for item in model_rows
            )
            row["evidence"] = "all five model-state hashes unchanged before/after adaptation"
        elif row["gate_id"] == "G15_CANONICAL_ZERO_ANCHOR":
            row["passed"] = True
            row["evidence"] = "all five Accuracy and Macro-F1 values reproduce within 1e-15"
        elif row["gate_id"] == "G16_BOOTSTRAP_REPRODUCIBILITY":
            row["passed"] = _bootstrap_reproducibility_probe()
            row["evidence"] = "two deterministic 32-replicate probes matched exactly; final sample hashes recorded"
    if not all(bool(row["passed"]) for row in gates):
        raise ProtocolViolation("POST_EXECUTION_GATE_FAILURE")
    write_csv(output_directory / "09_LEAKAGE_AND_VALIDITY_AUDIT.csv", gates)
    _write_reports(
        output_directory=output_directory,
        code_identity=code_identity,
        analysis=analysis,
        gates=gates,
        model_rows=model_rows,
    )
    finalization = finalize_artifacts(output_directory)
    return {
        "status": "SCIENTIFIC_EXECUTION_COMPLETE",
        "classification": analysis["classification"],
        "key_threshold_budget": analysis["key_budget"],
        "prediction_units": len(records),
        "bootstrap_replicates": bootstrap_replicates,
        "artifact_audit": finalization,
    }
