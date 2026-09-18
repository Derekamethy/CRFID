"""Section 6 -- composite P4 noise floor, noise-augmented retraining, one P4 read.

Three stages:

1. Estimate the P4 measurement-noise floor by composing per-condition
   repetition variances observed in P1/P2/P3, under the pre-registered
   additive-on-variance rule and its pre-declared fallback.
2. Add calibrated Gaussian noise to the P1-P3 training set so that each
   condition's repetition variance matches that estimate, then retrain five
   fresh instances of the frozen Strict-DG C1 candidate and freeze them into a
   sub-module-local release. The architecture, optimizer, loss, batching,
   shuffling, preprocessing policy, seeds and per-seed epoch counts are reused
   by import from the frozen Strict-DG runtime, not copy-pasted.
3. Evaluate the five frozen models on P4 exactly once. Target access runs
   through the existing machine-enforced gate: a TargetAccessToken is minted
   from this sub-module's own frozen release, features are released before
   predictions are serialized, and labels are released only for final scoring.

The frozen Strict-DG release is read for digest comparison only; nothing
outside 06_failure_mechanism/composition_warp is written.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import (  # noqa: E402
    ENCODING_STATES,
    POINT_COUNT,
    POSITIONS,
    PROJECT_ROOT,
    SURFACES,
    TAG_IDS,
    canonical_json_sha256,
    frozen_strict_dg_root,
    load_source,
    sha256_file,
    write_json,
)

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crfid.governance.strict_target_authorization import authorize_target_access  # noqa: E402
from crfid.protocols.strict_dg import StrictDGProtocol  # noqa: E402
from crfid.strict_runtime.hashing import array_sha256  # noqa: E402
from crfid.strict_runtime.neutral_data import PartitionData  # noqa: E402
from crfid.strict_runtime.neutral_model import (  # noqa: E402
    initialize_model,
    model_state_sha256,
    parameter_count,
)
from crfid.strict_runtime.phase3b_execution import (  # noqa: E402
    _criterion,
    _erm_train_epoch,
    _optimizer,
    load_checkpoint_model,
)
from crfid.strict_runtime.scale_policy import CANONICAL_MODE, apply_scale_policy  # noqa: E402
from crfid.strict_runtime.target_evaluation import (  # noqa: E402
    evaluate_logits,
    infer_checkpoint,
    load_p4_once,
    predict_from_logits,
    save_prediction_bundle,
    transform_p4,
)

RESULTS = HERE / "results"
EXECUTION = HERE / "execution"
LOCAL_RELEASE = EXECUTION / "frozen_release"

#: Pre-registered in preregistration.md section 3. Not revised here.
NEGATIVE_CELL_FRACTION_LIMIT = 0.01
NOISE_SEED_OFFSET = 7
NOISE_SEED_MULTIPLIER = 1_000_000

P4_METADATA_COLUMNS = ("A3", "A2", "A1", "P4", "P3", "P2", "P1", "ER", "TagID")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Stage 1 -- noise floor
# --------------------------------------------------------------------------


def condition_variances(data) -> tuple[np.ndarray, list[tuple[int, int, str]]]:
    """Per-condition repetition variance sigma^2[c, p](f), ddof = 1."""

    keys: list[tuple[int, int, str]] = []
    blocks = []
    for tag in TAG_IDS:
        for er in ENCODING_STATES:
            for surface in SURFACES:
                keys.append((tag, er, surface))
                per_position = []
                for position in POSITIONS:
                    selected = np.flatnonzero(
                        (data.tag_id == tag)
                        & (data.er == er)
                        & (data.surface == surface)
                        & (data.position == position)
                    )
                    if selected.size != 50:
                        raise RuntimeError(f"Condition block is not 50 repeats: {keys[-1]}/{position}")
                    per_position.append(data.signals[selected].var(axis=0, ddof=1))
                blocks.append(np.vstack(per_position))
    variances = np.stack(blocks)  # (63, 3, 281)
    if variances.shape != (63, 3, POINT_COUNT):
        raise RuntimeError(f"Unexpected variance tensor shape: {variances.shape}")
    return variances, keys


def compose_noise_floor(variances: np.ndarray, total_source_variance: float) -> dict:
    """Apply the pre-registered composition rule, its declared fallback, and a
    recorded numerical-validity override.

    The pre-registration fixed an automatic switch to the log-variance additive
    fallback whenever more than 1 % of cells go negative under the primary
    additive-on-variance rule. That switch fires on this data, and the resulting
    ratio estimator turns out to be numerically invalid: ``sigma2_P1`` reaches
    values many orders of magnitude below the typical repetition variance, so
    ``sigma2_P2 * sigma2_P3 / sigma2_P1`` diverges far beyond the total variance
    of the measured signal itself. A "noise floor" larger than the entire
    dynamic range of the data cannot be a noise floor.

    The override criterion below was formulated **after** seeing that diagnostic
    and is therefore a documented deviation from the pre-registration, not a
    silent substitution. Both candidate estimators are reported in full so a
    reviewer can see exactly what was rejected and why.
    """

    p1, p2, p3 = variances[:, 0], variances[:, 1], variances[:, 2]

    additive = p2 + p3 - p1
    negative_cells = int(np.count_nonzero(additive < 0.0))
    total_cells = int(additive.size)
    negative_fraction = negative_cells / total_cells
    additive_clipped = np.clip(additive, 0.0, None)

    with np.errstate(divide="ignore", invalid="ignore"):
        multiplicative = p2 * p3 / p1

    candidates = {
        "additive_on_variance": {
            "expression": "sigma2_P2 + sigma2_P3 - sigma2_P1, negatives clipped to zero",
            "values": additive_clipped,
            "raw_maximum": float(additive.max()),
            "raw_minimum": float(additive.min()),
            "negative_cells": negative_cells,
            "negative_cell_fraction": negative_fraction,
            "non_finite_cells": int(np.count_nonzero(~np.isfinite(additive))),
        },
        "log_variance_additive": {
            "expression": "sigma2_P2 * sigma2_P3 / sigma2_P1",
            "values": multiplicative,
            "raw_maximum": float(np.max(multiplicative[np.isfinite(multiplicative)]))
            if np.any(np.isfinite(multiplicative))
            else float("inf"),
            "raw_minimum": float(np.min(multiplicative[np.isfinite(multiplicative)]))
            if np.any(np.isfinite(multiplicative))
            else float("inf"),
            "negative_cells": 0,
            "negative_cell_fraction": 0.0,
            "non_finite_cells": int(np.count_nonzero(~np.isfinite(multiplicative))),
        },
    }

    preregistered_choice = (
        "log_variance_additive"
        if negative_fraction > NEGATIVE_CELL_FRACTION_LIMIT
        else "additive_on_variance"
    )

    # Recorded numerical-validity criterion: an estimated per-point noise
    # variance may not exceed the total variance of the measured source signal.
    chosen = preregistered_choice
    override = None
    candidate_maximum = candidates[preregistered_choice]["raw_maximum"]
    if (
        not np.isfinite(candidate_maximum)
        or candidate_maximum > total_source_variance
        or candidates[preregistered_choice]["non_finite_cells"] > 0
    ):
        chosen = "additive_on_variance"
        override = {
            "applied": preregistered_choice != chosen,
            "preregistered_choice": preregistered_choice,
            "chosen": chosen,
            "criterion": (
                "max estimated noise variance must not exceed the total variance of the "
                "measured source signal, and the estimate must be finite everywhere"
            ),
            "total_source_signal_variance_db2": total_source_variance,
            "rejected_estimator_maximum_db2": candidate_maximum,
            "rejected_estimator_non_finite_cells": candidates[preregistered_choice][
                "non_finite_cells"
            ],
            "criterion_formulated": "after_stage_1_diagnostics_post_hoc_documented_deviation",
        }

    estimate = np.ascontiguousarray(candidates[chosen]["values"], dtype=np.float64)
    if not np.all(np.isfinite(estimate)):
        raise RuntimeError("Composite noise floor contains non-finite values")

    if override is not None and override["applied"]:
        reason = (
            f"The pre-registered automatic switch fired: additive-on-variance produced negative "
            f"estimates in {negative_fraction:.6f} of cells, above the pre-declared "
            f"{NEGATIVE_CELL_FRACTION_LIMIT} limit. The pre-declared log-variance additive "
            f"fallback was then computed and rejected on a recorded numerical-validity criterion: "
            f"its maximum estimate is {candidate_maximum:.6g} dB^2 against a total measured source "
            f"signal variance of {total_source_variance:.6g} dB^2, because sigma2_P1 reaches values "
            f"far below the typical repetition variance and the ratio estimator diverges. An "
            f"estimated noise floor larger than the entire dynamic range of the data is not a noise "
            f"floor. The primary additive-on-variance rule was therefore used, with its "
            f"{negative_cells} negative cells clipped to zero, which is interpretable as "
            f"'the composed P4 floor does not exceed the observed floor at this point'. This is a "
            f"documented post-hoc deviation from the pre-registration, not a silent substitution; "
            f"both estimators are reported."
        )
    elif chosen == "additive_on_variance":
        reason = (
            f"Additive-on-variance produced negative estimates in {negative_fraction:.6f} of cells, "
            f"at or below the pre-declared {NEGATIVE_CELL_FRACTION_LIMIT} limit, so the primary "
            "formula was kept and the negative cells were clipped to zero."
        )
    else:
        reason = (
            f"Additive-on-variance produced negative estimates in {negative_fraction:.6f} of cells, "
            f"above the pre-declared {NEGATIVE_CELL_FRACTION_LIMIT} limit, so the pre-declared "
            "log-variance additive fallback was used and passed the numerical-validity criterion."
        )

    return {
        "estimate": estimate,
        "formula": chosen,
        "expression": candidates[chosen]["expression"],
        "reason": reason,
        "preregistered_choice": preregistered_choice,
        "override": override,
        "negative_cells": negative_cells,
        "total_cells": total_cells,
        "negative_cell_fraction": negative_fraction,
        "clipped_cells": negative_cells if chosen == "additive_on_variance" else 0,
        "candidate_diagnostics": {
            name: {key: value for key, value in entry.items() if key != "values"}
            for name, entry in candidates.items()
        },
        "total_source_signal_variance_db2": total_source_variance,
    }


# --------------------------------------------------------------------------
# Stage 2 -- augmented training
# --------------------------------------------------------------------------


def condition_index_map(data) -> tuple[np.ndarray, np.ndarray]:
    """Row -> (condition index in the 63-block order, position index)."""

    condition_of_row = np.full(len(data.signals), -1, dtype=np.int64)
    position_of_row = np.full(len(data.signals), -1, dtype=np.int64)
    index = 0
    for tag in TAG_IDS:
        for er in ENCODING_STATES:
            for surface in SURFACES:
                mask = (data.tag_id == tag) & (data.er == er) & (data.surface == surface)
                condition_of_row[mask] = index
                index += 1
    for position_index, position in enumerate(POSITIONS):
        position_of_row[data.position == position] = position_index
    if np.any(condition_of_row < 0) or np.any(position_of_row < 0):
        raise RuntimeError("Unmapped source row")
    return condition_of_row, position_of_row


def augment(data, added_variance: np.ndarray, condition_of_row, position_of_row, seed: int) -> np.ndarray:
    """Add calibrated zero-mean Gaussian noise matched to the estimated P4 floor."""

    generator = np.random.default_rng(NOISE_SEED_MULTIPLIER * seed + NOISE_SEED_OFFSET)
    standard_deviation = np.sqrt(added_variance[condition_of_row, position_of_row])
    noise = generator.standard_normal(size=data.signals.shape) * standard_deviation
    return np.ascontiguousarray(data.signals + noise, dtype=np.float64)


def fit_preprocessing(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    differenced = np.ascontiguousarray(np.diff(signals, n=1, axis=1), dtype=np.float64)
    mean = np.ascontiguousarray(differenced.mean(axis=0, dtype=np.float64))
    raw_scale = np.ascontiguousarray(differenced.std(axis=0, ddof=0, dtype=np.float64))
    scale = apply_scale_policy(raw_scale, mode=CANONICAL_MODE)
    inputs = np.ascontiguousarray(((differenced - mean) / scale).astype(np.float32))
    return mean, scale, inputs, int(np.count_nonzero(raw_scale < 1e-12))


def train_seed(*, seed: int, epochs: int, inputs: np.ndarray, labels: np.ndarray, recipe: dict) -> dict:
    base_config = {
        "optimizer": recipe["optimizer"],
        "loss": {"weight": None, "label_smoothing": 0.0},
        "training": {
            "batch_size": recipe["batch_size"],
            "inner_stage_offset": 0,
            "outer_stage_offset": 100000,
        },
    }
    partition = PartitionData(
        registry_rows=np.arange(len(labels), dtype=np.int64),
        inputs=inputs,
        labels=np.ascontiguousarray(labels, dtype=np.int64),
        sample_ids=[f"source_{index:05d}" for index in range(len(labels))],
        condition_ids=["source"] * len(labels),
        exact_signal_hashes=[""] * len(labels),
        unique_signal_weights=np.ones(len(labels), dtype=np.float64),
    )
    model = initialize_model(int(seed))
    if parameter_count(model) != 142855:
        raise RuntimeError("C1 parameter count changed")
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
        print(
            json.dumps(
                {
                    "event": "composition_warp_epoch",
                    "seed": int(seed),
                    "epoch": epoch,
                    "epochs": epochs,
                    "loss": losses[-1],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return {
        "state": state,
        "initial_model_state_sha256": initial_hash,
        "model_state_sha256": model_state_sha256(state),
        "epoch_losses": losses,
    }


# --------------------------------------------------------------------------
# Stage 3 -- one P4 evaluation
# --------------------------------------------------------------------------


def build_p4_custody(directory: Path) -> dict:
    records = []
    for surface in SURFACES:
        path = (directory / f"{surface}_P4.csv").resolve()
        if not path.is_file():
            raise RuntimeError(f"P4 measurement file is missing: {path}")
        records.append(
            {
                "domain": "P4",
                "absolute_path": str(path),
                "file_name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": 1,
        "purpose": "composition_warp_single_held_out_p4_evaluation",
        "created_at_utc": utc_now(),
        "inputs": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p4-directory",
        type=Path,
        default=Path(os.environ.get("CRFID_TYNDALL_P4_DIR", "")),
        help="Directory holding A1_P4.csv, A2_P4.csv and A3_P4.csv. "
        "Defaults to $CRFID_TYNDALL_P4_DIR. No workstation path is embedded here.",
    )
    args = parser.parse_args()
    if not args.p4_directory or not Path(args.p4_directory).is_dir():
        raise SystemExit(
            "A P4 measurement directory is required: pass --p4-directory or set CRFID_TYNDALL_P4_DIR."
        )

    recipe_path = frozen_strict_dg_root() / "recipe.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    frozen_recipe_sha256 = sha256_file(recipe_path)
    if recipe["selected_candidate"] != "C1_FIRST_DIFFERENCE_ERM_1DCNN":
        raise RuntimeError("Frozen Strict-DG recipe does not select C1")
    seeds = [int(value) for value in recipe["seeds"]]
    epochs_by_seed = {int(k): int(v) for k, v in recipe["final_epochs_by_seed"].items()}

    # ---------------- Stage 1: noise floor -------------------------------
    data = load_source()
    variances, condition_keys = condition_variances(data)
    total_source_variance = float(np.var(data.signals))
    composition = compose_noise_floor(variances, total_source_variance)
    estimate = composition["estimate"]
    print(
        json.dumps(
            {
                "event": "noise_floor_composed",
                "formula": composition["formula"],
                "preregistered_choice": composition["preregistered_choice"],
                "override_applied": bool(
                    composition["override"] and composition["override"]["applied"]
                ),
                "negative_cell_fraction": composition["negative_cell_fraction"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    condition_of_row, position_of_row = condition_index_map(data)
    # (63, 1, 281) broadcast against the (63, 3, 281) observed variances.
    added_variance = np.maximum(0.0, estimate[:, None, :] - variances)

    noise_payload = {
        "schema_version": 1,
        "sub_experiment": "02_noise_floor_composition",
        "generated_at_utc": utc_now(),
        "stage": "noise_floor_estimation",
        "p4_used": False,
        "variance_definition": {
            "grain": "per (tag_id, er, surface) condition, per source position",
            "across": "50 repetitions",
            "ddof": 1,
            "condition_count": len(condition_keys),
            "signal_point_count": POINT_COUNT,
            "units": "decibel_squared",
        },
        "composition_rule": {
            "preregistered_primary": "sigma2_P2 + sigma2_P3 - sigma2_P1",
            "preregistered_fallback": "sigma2_P2 * sigma2_P3 / sigma2_P1",
            "fallback_trigger": f"negative-cell fraction > {NEGATIVE_CELL_FRACTION_LIMIT}",
            "preregistered_choice": composition["preregistered_choice"],
            "formula_used": composition["formula"],
            "expression_used": composition["expression"],
            "justification": composition["reason"],
            "negative_cells_under_primary": composition["negative_cells"],
            "total_cells": composition["total_cells"],
            "negative_cell_fraction": composition["negative_cell_fraction"],
            "cells_clipped_to_zero": composition["clipped_cells"],
            "candidate_diagnostics": composition["candidate_diagnostics"],
            "total_source_signal_variance_db2": composition["total_source_signal_variance_db2"],
            "documented_deviation_from_preregistration": composition["override"],
        },
        "observed_source_variance": {
            position: {
                "mean": float(variances[:, index].mean()),
                "median": float(np.median(variances[:, index])),
                "maximum": float(variances[:, index].max()),
                "minimum": float(variances[:, index].min()),
            }
            for index, position in enumerate(POSITIONS)
        },
        "estimated_p4_noise_floor": {
            "mean": float(estimate.mean()),
            "median": float(np.median(estimate)),
            "maximum": float(estimate.max()),
            "minimum": float(estimate.min()),
            "mean_ratio_to_p1": float(estimate.mean() / variances[:, 0].mean()),
            "mean_ratio_to_p2": float(estimate.mean() / variances[:, 1].mean()),
            "mean_ratio_to_p3": float(estimate.mean() / variances[:, 2].mean()),
        },
        "added_noise_variance": {
            position: {
                "mean": float(added_variance[:, index].mean()),
                "maximum": float(added_variance[:, index].max()),
                "cells_requiring_no_addition": int(
                    np.count_nonzero(added_variance[:, index] <= 0.0)
                ),
            }
            for index, position in enumerate(POSITIONS)
        },
        "condition_mean_curves": {
            "signal_position": list(range(POINT_COUNT)),
            "sigma2_p1_mean_over_conditions": [float(value) for value in variances[:, 0].mean(axis=0)],
            "sigma2_p2_mean_over_conditions": [float(value) for value in variances[:, 1].mean(axis=0)],
            "sigma2_p3_mean_over_conditions": [float(value) for value in variances[:, 2].mean(axis=0)],
            "sigma2_p4_estimate_mean_over_conditions": [float(value) for value in estimate.mean(axis=0)],
        },
        "augmentation_rule": {
            "added_variance": "max(0, sigmahat2_P4[c](f) - sigma2[c,p](f))",
            "distribution": "independent zero-mean Gaussian per sample and signal position",
            "generator": "numpy.random.default_rng",
            "generator_seed_formula": f"{NOISE_SEED_MULTIPLIER} * training_seed + {NOISE_SEED_OFFSET}",
            "applied_to": "P1/P2/P3 training data only",
            "target_augmented": False,
        },
        "provenance": {
            "source_signals_sha256": data.signals_sha256,
            "source_registry_sha256": data.registry_sha256,
            "frozen_strict_dg_recipe_sha256": frozen_recipe_sha256,
        },
    }
    noise_payload["noise_floor_semantic_sha256"] = canonical_json_sha256(noise_payload)
    write_json(RESULTS / "noise_floor_estimates.json", noise_payload)

    # ---------------- Stage 2: augmented retraining ----------------------
    LOCAL_RELEASE.mkdir(parents=True, exist_ok=True)
    (LOCAL_RELEASE / "checkpoints").mkdir(parents=True, exist_ok=True)
    (LOCAL_RELEASE / "preprocessing").mkdir(parents=True, exist_ok=True)

    local_recipe = {
        "schema_version": 1,
        "protocol": "composition_warp_noise_augmented_source_only",
        "derived_from_frozen_strict_dg_recipe_sha256": frozen_recipe_sha256,
        "architecture": recipe["architecture"],
        "batch_size": recipe["batch_size"],
        "class_order": recipe["class_order"],
        "final_epochs_by_seed": recipe["final_epochs_by_seed"],
        "initialization": recipe["initialization"],
        "loss": recipe["loss"],
        "normalization": recipe["normalization"],
        "optimizer": recipe["optimizer"],
        "ordered_input_length": recipe["ordered_input_length"],
        "prediction": recipe["prediction"],
        "representation": recipe["representation"],
        "seeds": recipe["seeds"],
        "selected_candidate": recipe["selected_candidate"],
        "source_domains": recipe["source_domains"],
        "target_domain": recipe["target_domain"],
        "transformed_input_length": recipe["transformed_input_length"],
        "augmentation": noise_payload["augmentation_rule"],
        "noise_floor_semantic_sha256": noise_payload["noise_floor_semantic_sha256"],
        "target_policy": recipe["target_policy"],
    }
    local_recipe_path = LOCAL_RELEASE / "recipe.json"
    write_json(local_recipe_path, local_recipe)
    local_recipe_sha256 = sha256_file(local_recipe_path)
    (LOCAL_RELEASE / "recipe.sha256").write_text(
        local_recipe_sha256 + "  recipe.json\n", encoding="ascii"
    )

    per_seed_preprocessing: dict[int, dict] = {}
    preprocessing_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    checkpoints = []
    training_records = []
    for seed in seeds:
        epochs = epochs_by_seed[seed]
        augmented = augment(data, added_variance, condition_of_row, position_of_row, seed)
        mean, scale, inputs, degenerate = fit_preprocessing(augmented)
        if inputs.shape != (9450, 280) or inputs.dtype != np.float32:
            raise RuntimeError(f"Augmented transform custody mismatch: {inputs.shape}")
        mean_path = LOCAL_RELEASE / "preprocessing" / f"mean_float64_seed_{seed}.npy"
        scale_path = LOCAL_RELEASE / "preprocessing" / f"scale_float64_seed_{seed}.npy"
        np.save(mean_path, mean, allow_pickle=False)
        np.save(scale_path, scale, allow_pickle=False)
        preprocessing_arrays[seed] = (mean, scale)
        per_seed_preprocessing[seed] = {
            "seed": seed,
            "noise_generator_seed": NOISE_SEED_MULTIPLIER * seed + NOISE_SEED_OFFSET,
            "fit_domains": ["P1", "P2", "P3"],
            "fit_sample_count": 9450,
            "fit_on_augmented_source_only": True,
            "target_used_for_fit": False,
            "ddof": 0,
            "degenerate_scale_mode": CANONICAL_MODE,
            "zero_scale_replacement_count": degenerate,
            "mean_array_sha256": array_sha256(mean),
            "scale_array_sha256": array_sha256(scale),
            "augmented_signal_array_sha256": array_sha256(augmented),
        }

        result = train_seed(seed=seed, epochs=epochs, inputs=inputs, labels=data.labels, recipe=recipe)
        checkpoint = LOCAL_RELEASE / "checkpoints" / f"seed_{seed}.pt"
        payload = {
            "schema_version": 1,
            "method_id": recipe["selected_candidate"],
            "protocol": "composition_warp_noise_augmented_source_only",
            "seed": seed,
            "epochs": epochs,
            "recipe_sha256": local_recipe_sha256,
            "preprocessing_mean_sha256": per_seed_preprocessing[seed]["mean_array_sha256"],
            "preprocessing_scale_sha256": per_seed_preprocessing[seed]["scale_array_sha256"],
            "model_state_sha256": result["model_state_sha256"],
            "model_state_dict": result["state"],
            "source_domains": ["P1", "P2", "P3"],
            "target_used": False,
        }
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(checkpoint)
        loaded_model, loaded_payload = load_checkpoint_model(checkpoint, seed)
        if (
            loaded_payload["recipe_sha256"] != local_recipe_sha256
            or model_state_sha256(loaded_model) != result["model_state_sha256"]
        ):
            raise RuntimeError(f"Checkpoint reload failed: seed {seed}")
        checkpoints.append(
            {
                "seed": seed,
                "epochs": epochs,
                "relative_path": checkpoint.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
                "model_state_sha256": result["model_state_sha256"],
                "reload_verified": True,
            }
        )
        training_records.append(
            {
                "seed": seed,
                "epochs": epochs,
                "initial_model_state_sha256": result["initial_model_state_sha256"],
                "final_epoch_loss": result["epoch_losses"][-1],
                "epoch_losses": result["epoch_losses"],
            }
        )

    preprocessing_semantic = {
        "schema_version": 1,
        "candidate_id": recipe["selected_candidate"],
        "protocol": "composition_warp_noise_augmented_source_only",
        "per_seed_states": [per_seed_preprocessing[seed] for seed in seeds],
        "operation_order": [
            "calibrated_gaussian_noise_addition_on_source_only",
            "first_difference_without_padding",
            "featurewise_population_standardization",
            "float32_model_conversion",
        ],
        "input_length": POINT_COUNT,
        "output_length": POINT_COUNT - 1,
        "target_used_for_fit": False,
        "target_refit_permitted": False,
    }
    preprocessing_state = {
        **preprocessing_semantic,
        "state_sha256": canonical_json_sha256(preprocessing_semantic),
    }
    write_json(LOCAL_RELEASE / "preprocessing" / "state.json", preprocessing_state)

    authorization = {
        "schema_version": 1,
        "authorized": True,
        "authorized_at_utc": utc_now(),
        "recipe_sha256": local_recipe_sha256,
        "preprocessing_state_sha256": preprocessing_state["state_sha256"],
        "checkpoint_count": len(checkpoints),
        "all_checkpoint_reloads_verified": all(row["reload_verified"] for row in checkpoints),
        "scope": "composition_warp_single_held_out_p4_evaluation",
        "target_access_before_authorization": False,
        "prohibited_actions": [
            "training_on_target",
            "optimizer_step_on_target",
            "target_normalization_refit",
            "recipe_change",
            "seed_selection",
        ],
    }
    authorization["authorization_sha256"] = canonical_json_sha256(authorization)
    write_json(LOCAL_RELEASE / "target_access_authorization.json", authorization)

    manifest = {
        "schema_version": 1,
        "status": "FROZEN_AND_TARGET_AUTHORIZED",
        "protocol": "composition_warp_noise_augmented_source_only",
        "recipe_sha256": local_recipe_sha256,
        "checkpoints": checkpoints,
        "training_records": training_records,
        "target_accessed": False,
    }
    manifest["manifest_semantic_sha256"] = canonical_json_sha256(manifest)
    write_json(LOCAL_RELEASE / "FROZEN_RELEASE_MANIFEST.json", manifest)
    print(json.dumps({"event": "composition_warp_release_frozen", "checkpoints": len(checkpoints)}), flush=True)

    # ---------------- Stage 3: the single held-out P4 evaluation ---------
    tolerance = {
        "schema_version": 1,
        "declared_before_p4_numerical_load": True,
        "evaluation_count": 1,
        "target_training_steps": 0,
        "target_preprocessing_refit_count": 0,
        "note": "Diagnostic sub-module. No authoritative reproduction target exists for these models.",
    }
    write_json(RESULTS / "p4_evaluation_declaration.json", tolerance)

    protocol = StrictDGProtocol()
    token = authorize_target_access(frozen_release_directory=LOCAL_RELEASE, protocol=protocol)
    custody = build_p4_custody(Path(args.p4_directory))
    write_json(EXECUTION / "p4_custody_manifest.json", custody)
    source_schema = {
        "metadata_columns": list(P4_METADATA_COLUMNS),
        "signal": {"point_count": POINT_COUNT},
    }
    p4 = load_p4_once(custody_manifest=custody, source_config=source_schema, token=token, phase=6)

    seed_rows = []
    for record in checkpoints:
        seed = record["seed"]
        mean, scale = preprocessing_arrays[seed]
        inputs = transform_p4(p4, mean, scale)
        if inputs.shape != (3150, 280) or inputs.dtype != np.float32:
            raise RuntimeError("P4 transform custody mismatch")
        logits = infer_checkpoint(
            checkpoint_path=LOCAL_RELEASE / "checkpoints" / f"seed_{seed}.pt",
            seed=seed,
            inputs=inputs,
            expected_state_sha256=record["model_state_sha256"],
        )
        predictions = predict_from_logits(logits)
        bundle_path = EXECUTION / "predictions" / f"seed_{seed}.npz"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = save_prediction_bundle(
            path=bundle_path, dataset=p4, logits=logits, predictions=predictions, token=token
        )
        metrics, scored, _ = evaluate_logits(p4, logits, token=token)
        if not np.array_equal(scored, predictions):
            raise RuntimeError(f"Scored predictions differ from the serialized bundle: seed {seed}")
        sample = metrics["sample"]
        seed_rows.append(
            {
                "seed": seed,
                "epochs": record["epochs"],
                "accuracy": sample["accuracy"],
                "macro_f1": sample["macro_f1"],
                "condition_accuracy": metrics["condition"]["accuracy"],
                "condition_macro_f1": metrics["condition"]["macro_f1"],
                "worst_class_recall": sample["worst_class_recall"],
                "zero_recall_class_count": sample["zero_recall_class_count"],
                "per_class_recall": sample["per_class_recall"],
                "predicted_class_histogram": metrics["predicted_class_histogram"],
                "dominant_predicted_class_fraction": metrics["dominant_predicted_class_fraction"],
                "prediction_array_sha256": bundle["predictions_array_sha256"],
                "logits_array_sha256": bundle["logits_array_sha256"],
            }
        )
        print(
            json.dumps(
                {
                    "event": "composition_warp_p4_seed_complete",
                    "seed": seed,
                    "accuracy": sample["accuracy"],
                    "macro_f1": sample["macro_f1"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    strict_reference = json.loads(
        (PROJECT_ROOT / "outputs" / "strict_dg" / "final_p4" / "aggregate_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    aggregate = {
        metric: {
            "mean": float(statistics.fmean(row[metric] for row in seed_rows)),
            "population_sd": float(statistics.pstdev(row[metric] for row in seed_rows)),
        }
        for metric in ("accuracy", "macro_f1", "condition_accuracy", "condition_macro_f1")
    }

    evaluation_payload = {
        "schema_version": 1,
        "sub_experiment": "02_noise_floor_composition",
        "generated_at_utc": utc_now(),
        "stage": "single_held_out_p4_evaluation",
        "protocol": "composition_warp_noise_augmented_source_only",
        "seed_count": len(seed_rows),
        "equal_seed_weight": True,
        "ddof": 0,
        "per_seed": seed_rows,
        "aggregate": aggregate,
        "noise_floor_formula_used": composition["formula"],
        "frozen_strict_dg_c1_reference": {
            "note": (
                "Quoted inside this sub-experiment only, as the reference point the augmented "
                "retraining is measured against. It is not restated as a new result and is not "
                "inserted into any existing comparison table."
            ),
            "accuracy_mean": strict_reference["metrics"]["accuracy"]["mean"],
            "accuracy_population_sd": strict_reference["metrics"]["accuracy"]["population_sd"],
            "macro_f1_mean": strict_reference["metrics"]["macro_f1"]["mean"],
            "macro_f1_population_sd": strict_reference["metrics"]["macro_f1"]["population_sd"],
            "recipe_sha256": strict_reference["recipe_sha256"],
        },
        "delta_versus_frozen_strict_dg_c1": {
            "accuracy": aggregate["accuracy"]["mean"] - strict_reference["metrics"]["accuracy"]["mean"],
            "macro_f1": aggregate["macro_f1"]["mean"] - strict_reference["metrics"]["macro_f1"]["mean"],
        },
        "chance_level": {"accuracy": 1.0 / 7.0, "class_count": 7, "balanced": True},
        "target_access": {
            "machine_enforced": True,
            "authorized_against": "this sub-module's own frozen release",
            "frozen_strict_dg_release_modified": False,
            "p4_filesystem_read_count": len(p4.access_records),
            "p4_training_steps": 0,
            "target_preprocessing_refit_count": 0,
            "predictions_serialized_before_label_release": True,
            "token": token.as_record(),
            "p4_files": [
                {
                    "file_name": row["file_name"],
                    "sha256": row["streaming_sha256"],
                    "bytes_read": row["bytes_read"],
                    "rows_parsed": row["rows_parsed"],
                }
                for row in p4.access_records
            ],
        },
        "provenance": {
            "local_recipe_sha256": local_recipe_sha256,
            "derived_from_frozen_strict_dg_recipe_sha256": frozen_recipe_sha256,
            "preprocessing_state_sha256": preprocessing_state["state_sha256"],
            "release_manifest_semantic_sha256": manifest["manifest_semantic_sha256"],
            "source_signals_sha256": data.signals_sha256,
        },
    }
    evaluation_payload["evaluation_semantic_sha256"] = canonical_json_sha256(evaluation_payload)
    write_json(RESULTS / "retrain_eval_summary.json", evaluation_payload)

    write_report(noise_payload, evaluation_payload, training_records)
    print(
        json.dumps(
            {
                "status": "COMPOSITION_WARP_NOISE_FLOOR_EVALUATION_COMPLETE",
                "accuracy": aggregate["accuracy"],
                "macro_f1": aggregate["macro_f1"],
                "formula": composition["formula"],
            },
            sort_keys=True,
        )
    )
    return 0


def write_report(noise: dict, evaluation: dict, training_records: list[dict]) -> None:
    rule = noise["composition_rule"]
    aggregate = evaluation["aggregate"]
    reference = evaluation["frozen_strict_dg_c1_reference"]
    delta = evaluation["delta_versus_frozen_strict_dg_c1"]
    lines = [
        "# Section 6 — Composite P4 noise floor and noise-augmented retraining",
        "",
        "Sub-module: `06_failure_mechanism/composition_warp/02_noise_floor_composition`.",
        "",
        "## Question",
        "",
        "If the P4 failure were driven by the *measurement noise* that the 150 mm / 45°",
        "condition inherits from its two factor levels, then raising the source-domain",
        "noise to the composed P4 level should reproduce part of that failure — and",
        "training under it should confer some robustness. This sub-experiment composes",
        "the noise floor from P1/P2/P3 alone, retrains under it, and reads P4 once.",
        "",
        "## Stage 1 — noise floor",
        "",
        "Per-condition repetition variance `sigma2[c, p](f)` is computed across the 50",
        f"repetitions of each of the {noise['variance_definition']['condition_count']} `(tag_id, er, surface)` conditions at each source",
        f"position, with `ddof = {noise['variance_definition']['ddof']}`, over all {noise['variance_definition']['signal_point_count']} ordered signal positions.",
        "",
        "Pre-registered primary rule: `sigma2_P2 + sigma2_P3 - sigma2_P1`.",
        "Pre-declared fallback (if more than 1 % of cells go negative):",
        "`sigma2_P2 * sigma2_P3 / sigma2_P1`, additive in `log sigma^2`.",
        "",
        f"**Negative cells under the primary rule: {rule['negative_cells_under_primary']} of {rule['total_cells']} "
        f"({rule['negative_cell_fraction']:.6f}).**",
        "",
        f"**Formula used: `{rule['formula_used']}` — `{rule['expression_used']}`.**",
        "",
        rule["justification"],
        "",
    ]
    deviation = rule.get("documented_deviation_from_preregistration")
    if deviation and deviation.get("applied"):
        diagnostics = rule["candidate_diagnostics"]
        lines += [
            "> **Documented deviation from pre-registration.** The pre-registered automatic",
            f"> switch selected `{deviation['preregistered_choice']}`. That estimator was computed and then",
            "> rejected on a recorded numerical-validity criterion, and",
            f"> `{deviation['chosen']}` was used instead. The criterion is:",
            f"> *{deviation['criterion']}*. It was formulated after the Stage-1 diagnostics and is",
            "> therefore post-hoc. It is recorded here rather than applied silently, and both",
            "> estimators are reported in `results/noise_floor_estimates.json`.",
            ">",
            "> | Estimator | Max (dB²) | Min (dB²) | Negative cells | Non-finite cells |",
            "> |---|---:|---:|---:|---:|",
        ]
        for name, entry in diagnostics.items():
            lines.append(
                f"> | `{name}` | {entry['raw_maximum']:.6g} | {entry['raw_minimum']:.6g} | "
                f"{entry['negative_cells']} | {entry['non_finite_cells']} |"
            )
        lines += [
            f"> | *total measured source signal variance* | {deviation['total_source_signal_variance_db2']:.6g} | — | — | — |",
            ">",
            "> The rejected ratio estimator exceeds the total variance of the measured signal by",
            f"> a factor of about {deviation['rejected_estimator_maximum_db2'] / deviation['total_source_signal_variance_db2']:.3g},",
            "> because `sigma2_P1` reaches values far below the typical repetition variance. A noise",
            "> floor larger than the entire dynamic range of the data cannot be a noise floor.",
            "",
        ]
    lines += [
        "| Quantity | Mean (dB²) | Median (dB²) | Max (dB²) |",
        "|---|---:|---:|---:|",
    ]
    for position in POSITIONS:
        row = noise["observed_source_variance"][position]
        lines.append(
            f"| observed `sigma2` at {position} | {row['mean']:.6f} | {row['median']:.6f} | {row['maximum']:.6f} |"
        )
    floor = noise["estimated_p4_noise_floor"]
    lines += [
        f"| **composed `sigmahat2` for P4** | **{floor['mean']:.6f}** | **{floor['median']:.6f}** | **{floor['maximum']:.6f}** |",
        "",
        f"The composed P4 floor is {floor['mean_ratio_to_p1']:.4f}× the mean P1 variance, "
        f"{floor['mean_ratio_to_p2']:.4f}× P2 and {floor['mean_ratio_to_p3']:.4f}× P3.",
        "",
        "## Stage 2 — augmented retraining",
        "",
        "Independent zero-mean Gaussian noise of variance",
        "`max(0, sigmahat2_P4[c](f) - sigma2[c,p](f))` is added per sample and per signal",
        "position, so each augmented condition's repetition variance matches the composed",
        "P4 floor. Noise is applied to the **P1–P3 training data only**; P4 is never",
        "augmented and preprocessing is fitted on augmented source data only. The noise",
        f"generator seed is `{noise['augmentation_rule']['generator_seed_formula']}`, making every draw reproducible.",
        "",
        "Everything else is reused by import from the frozen Strict-DG runtime — the",
        "`NeutralSourceOnlyCNN1D` architecture (142,855 parameters), `_optimizer`,",
        "`_criterion`, `_erm_train_epoch`, and the canonical degenerate-scale policy —",
        "with hyperparameters read from `$CRFID_COMPOSITION_FROZEN_STRICT_DG/recipe.json`:",
        "AdamW lr 1e-3 / wd 1e-4, batch 256, unweighted cross-entropy, seeds 42–46 at",
        "13/13/9/10/9 epochs. Five fresh models were trained. Nothing in the frozen",
        "Strict-DG release was written to.",
        "",
        "Final-epoch training loss per seed: "
        + ", ".join(f"seed {row['seed']} `{row['final_epoch_loss']:.6f}`" for row in training_records)
        + ".",
        "",
        "## Stage 3 — the single held-out P4 evaluation",
        "",
        "Target access was machine-enforced through the existing gate. This sub-module",
        "built its own frozen release (recipe, per-seed preprocessing states, five",
        "verified checkpoints, release manifest, authorization) and minted a",
        "`TargetAccessToken` via `crfid.governance.strict_target_authorization.authorize_target_access`",
        "against **its own** release directory. P4 was read from disk exactly",
        f"{evaluation['target_access']['p4_filesystem_read_count']} times (once per surface file) through the existing",
        "`load_p4_once`. Predictions were serialized before labels were released for",
        "scoring. Target training steps: 0. Target preprocessing refits: 0.",
        "",
        "| Seed | Epochs | Accuracy | Macro-F1 | Worst-class recall | Zero-recall classes |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in evaluation["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['epochs']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | "
            f"{row['worst_class_recall']:.6f} | {row['zero_recall_class_count']} |"
        )
    lines += [
        "",
        f"**Accuracy {aggregate['accuracy']['mean']:.6f} ± {aggregate['accuracy']['population_sd']:.6f}** "
        f"(population SD, five seeds, equal weight).",
        "",
        f"**Macro-F1 {aggregate['macro_f1']['mean']:.6f} ± {aggregate['macro_f1']['population_sd']:.6f}.**",
        "",
        "### Reference point",
        "",
        reference["note"],
        "",
        "| | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
        f"| noise-augmented C1 (this sub-experiment) | {aggregate['accuracy']['mean']:.6f} ± {aggregate['accuracy']['population_sd']:.6f} | "
        f"{aggregate['macro_f1']['mean']:.6f} ± {aggregate['macro_f1']['population_sd']:.6f} |",
        f"| frozen Strict-DG C1 | {reference['accuracy_mean']:.6f} ± {reference['accuracy_population_sd']:.6f} | "
        f"{reference['macro_f1_mean']:.6f} ± {reference['macro_f1_population_sd']:.6f} |",
        f"| difference | {delta['accuracy']:+.6f} | {delta['macro_f1']:+.6f} |",
        f"| chance (7 balanced classes) | {1.0/7.0:.6f} | — |",
        "",
        "## Outputs",
        "",
        "* `results/noise_floor_estimates.json` — variance definitions, the composition rule actually used",
        "  and why, observed and composed variance summaries, the condition-averaged 281-point curves,",
        "  and the augmentation rule.",
        "* `results/retrain_eval_summary.json` — per-seed and aggregate P4 metrics, the target-access record",
        "  including the minted token and the P4 file digests, and the frozen Strict-DG reference.",
        "* `results/p4_evaluation_declaration.json` — the evaluation declaration, written before the first",
        "  numerical P4 load.",
        "* `execution/` — the sub-module's own frozen release, P4 custody manifest and per-seed prediction",
        "  bundles. Excluded from the review package: checkpoints are `.pt` and the bundles carry P4 arrays.",
        "",
        "## Scope",
        "",
        "Diagnostic and mechanistic only. This does not modify or challenge the frozen",
        "Strict-DG, Few-Shot or External-Data1 results, and these numbers are not",
        "inserted into any existing comparison table.",
    ]
    (HERE / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
