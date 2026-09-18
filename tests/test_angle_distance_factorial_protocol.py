from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from crfid.angle_distance_factorial_contrast import (
    BASE_MANIFEST,
    BASE_RESULTS,
    audit_paired_design,
    bootstrap_factor_intervals,
    build_macro_f1_rows,
    build_primary_contrast_rows,
    classify_factor_pattern,
    draw_paired_bootstrap_indices,
    factor_contrasts,
    paired_unit_id,
    prediction_arrays,
    read_csv,
    simple_effects,
)


def synthetic_predictions() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    surfaces = ("A1", "A2", "A3")
    for tag_id in range(1, 8):
        for er in range(3):
            for surface_index, surface in enumerate(surfaces):
                fold = (er + surface_index) % 3 + 1
                identifier = paired_unit_id(tag_id, er, surface)
                block_number = (tag_id - 1) * 9 + er * 3 + surface_index
                for seed_index, seed in enumerate((42, 43, 44, 45, 46)):
                    for position_index, position in enumerate(("P1", "P2", "P3", "P4")):
                        true_label = tag_id - 1
                        correct = int((block_number + seed_index + position_index) % 4 != 0)
                        rows.append(
                            {
                                "paired_unit_id": identifier,
                                "TagID": tag_id,
                                "ER": er,
                                "surface": surface,
                                "position": position,
                                "fold": fold,
                                "seed": seed,
                                "true_label": true_label,
                                "predicted_label": true_label if correct else (true_label + 1) % 7,
                                "correct": correct,
                                "sample_count": 50,
                            }
                        )
    return rows


def test_factor_coding_and_known_synthetic_effects() -> None:
    angle_only = factor_contrasts({"P1": 0.6, "P2": 0.4, "P3": 0.6, "P4": 0.4})
    assert angle_only == pytest.approx(
        {
            "angle_45_minus_0": -0.2,
            "distance_150_minus_50": 0.0,
            "angle_by_distance_interaction": 0.0,
        }
    )
    distance_only = factor_contrasts({"P1": 0.6, "P2": 0.6, "P3": 0.4, "P4": 0.4})
    assert distance_only == pytest.approx(
        {
            "angle_45_minus_0": 0.0,
            "distance_150_minus_50": -0.2,
            "angle_by_distance_interaction": 0.0,
        }
    )
    pure_interaction = factor_contrasts({"P1": 0.5, "P2": 0.5, "P3": 0.5, "P4": 0.3})
    assert pure_interaction == pytest.approx(
        {
            "angle_45_minus_0": -0.1,
            "distance_150_minus_50": -0.1,
            "angle_by_distance_interaction": -0.2,
        }
    )
    assert factor_contrasts({position: 0.5 for position in ("P1", "P2", "P3", "P4")}) == {
        "angle_45_minus_0": 0.0,
        "distance_150_minus_50": 0.0,
        "angle_by_distance_interaction": 0.0,
    }


def test_exact_simple_effect_parameterization() -> None:
    effects = simple_effects({"P1": 0.2, "P2": 0.1, "P3": 0.6, "P4": 0.3})
    assert effects == pytest.approx(
        {
            "angle_45_minus_0_at_50mm": -0.1,
            "angle_45_minus_0_at_150mm": -0.3,
            "distance_150_minus_50_at_0deg": 0.4,
            "distance_150_minus_50_at_45deg": 0.2,
        }
    )


def test_paired_block_and_seed_alignment() -> None:
    rows = synthetic_predictions()
    blocks, true, predicted, correct = prediction_arrays(rows)
    assert len(blocks) == 63
    assert true.shape == (63,)
    assert predicted.shape == (63, 5, 4)
    assert correct.shape == (63, 5, 4)
    with pytest.raises(RuntimeError, match="Missing position or seed"):
        prediction_arrays(rows[:-1])
    with pytest.raises(RuntimeError, match="Duplicate aligned prediction"):
        prediction_arrays(rows + [dict(rows[0])])


def test_tagid_stratified_bootstrap_preserves_paired_units_and_is_deterministic() -> None:
    tag_ids = np.repeat(np.arange(1, 8), 9)
    first = draw_paired_bootstrap_indices(
        tag_ids, 5, np.random.default_rng(123), resample_seeds=True
    )
    second = draw_paired_bootstrap_indices(
        tag_ids, 5, np.random.default_rng(123), resample_seeds=True
    )
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert len(first[0]) == 63
    assert len(first[1]) == 5
    assert all(np.count_nonzero(tag_ids[first[0]] == tag_id) == 9 for tag_id in range(1, 8))


def test_bootstrap_results_are_deterministic_for_fixed_seed() -> None:
    rows = synthetic_predictions()
    first = bootstrap_factor_intervals(rows, resamples=50, random_seed=900)
    second = bootstrap_factor_intervals(rows, resamples=50, random_seed=900)
    assert first == second


def test_zero_boundary_is_not_a_confirmed_interaction() -> None:
    intervals = [
        {
            "endpoint": "condition_block_accuracy",
            "method": "paired_tagid_stratified_block_bootstrap_seeds_fixed",
            "contrast": contrast,
            "estimate": estimate,
            "confidence_interval_95_low": low,
            "confidence_interval_95_high": high,
        }
        for contrast, estimate, low, high in (
            ("angle_45_minus_0", -0.11, -0.16, -0.06),
            ("distance_150_minus_50", -0.05, -0.10, 0.01),
            ("angle_by_distance_interaction", 0.11, 2.8e-17, 0.23),
        )
    ]
    simple = [
        {
            "endpoint": "condition_block_accuracy",
            "method": "paired_tagid_stratified_block_bootstrap_seeds_fixed",
            "simple_effect": name,
            "estimate": estimate,
        }
        for name, estimate in (
            ("angle_45_minus_0_at_50mm", -0.17),
            ("angle_45_minus_0_at_150mm", -0.05),
            ("distance_150_minus_50_at_0deg", -0.11),
            ("distance_150_minus_50_at_45deg", 0.01),
        )
    ]
    classification, details = classify_factor_pattern(intervals, simple)
    assert classification == "ANGLE_ASSOCIATED_DEGRADATION"
    assert details["significant_primary_contrasts"]["interaction"] is False


def test_macro_f1_estimand_is_not_the_descriptive_15_run_mean() -> None:
    predictions = synthetic_predictions()
    run_metrics = [
        {"position": position, "held_block_macro_f1": str(0.01 * fold)}
        for position in ("P1", "P2", "P3", "P4")
        for fold in range(1, 16)
    ]
    rows = build_macro_f1_rows(predictions, run_metrics)
    estimands = {row["record_type"]: row["estimand"] for row in rows}
    assert "mean_of_five_seed_specific" in estimands["position_pooled_estimand"]
    assert "not_the_pooled_estimand" in estimands["descriptive_15_run_position_mean"]


def test_no_row_level_pseudoreplication_in_primary_output() -> None:
    rows = build_primary_contrast_rows(synthetic_predictions())
    block_rows = [row for row in rows if row["record_type"] == "paired_block_seed_contrast"]
    assert len(block_rows) == 63 * 5 * 3
    assert {row["physical_block_count"] for row in block_rows} == {1}
    assert all("row_count" not in row for row in block_rows)


def test_missing_and_duplicate_cells_fail_the_paired_audit() -> None:
    manifest = read_csv(BASE_MANIFEST)
    register = read_csv(BASE_RESULTS / "05_RUN_REGISTER.csv")
    inherited_audit, _ = audit_paired_design(
        manifest, register, synthetic_predictions(), raise_on_failure=False
    )
    inherited_failures = [row for row in inherited_audit if row["status"] == "FAIL"]
    assert [(row["check"], row["details"]) for row in inherited_failures] == [
        ("train_validation_test_synchronization", "failures=492")
    ]
    p1_assignment = {
        (int(row["TagID"]), int(row["ER"]), row["surface"], int(row["fold"]), int(row["seed"])): row["split_assignment"]
        for row in manifest
        if row["position"] == "P1"
    }
    synchronized = []
    for source in manifest:
        row = dict(source)
        row["split_assignment"] = p1_assignment[
            (int(row["TagID"]), int(row["ER"]), row["surface"], int(row["fold"]), int(row["seed"]))
        ]
        synchronized.append(row)
    passed, paired_manifest = audit_paired_design(
        synchronized, register, synthetic_predictions()
    )
    assert {row["status"] for row in passed} == {"PASS"}
    assert len(paired_manifest) == 63 * 5
    missing = synchronized[1:]
    with pytest.raises(RuntimeError, match="BLOCKED_PAIRED_GOVERNED_INPUTS"):
        audit_paired_design(missing, register, synthetic_predictions())
    duplicate = synchronized + [dict(synchronized[0])]
    with pytest.raises(RuntimeError, match="BLOCKED_PAIRED_GOVERNED_INPUTS"):
        audit_paired_design(duplicate, register, synthetic_predictions())
