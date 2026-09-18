from __future__ import annotations

import pytest

from crfid.angle_distance_factorial_contrast import (
    BASE_RESULTS,
    DEFAULT_RESULTS,
    PROJECT_ROOT,
    factor_contrasts,
    read_csv,
    reconstruct_released_results,
)


def test_compact_evidence_reconstructs_all_released_core_aggregates() -> None:
    rows, status = reconstruct_released_results(BASE_RESULTS)
    assert status["status"] == "PASS_COMPACT_RESULT_RECONSTRUCTION"
    assert status["run_count"] == 60
    assert len(rows) == 4 * 4
    assert {row["status"] for row in rows} == {"PASS_EXACT_RECONSTRUCTION"}


def test_required_factorial_protocol_was_frozen_before_results() -> None:
    protocol = PROJECT_ROOT / "docs" / "ANGLE_DISTANCE_FACTORIAL_PROTOCOL.md"
    result_protocol = BASE_RESULTS.parent / "angle_distance_factorial" / "06_FACTORIAL_PROTOCOL.md"
    assert protocol.is_file()
    assert result_protocol.is_file()
    text = protocol.read_text(encoding="utf-8")
    assert "10,000 paired" in text
    assert "never independent inferential" in text


def test_all_required_factorial_outputs_exist() -> None:
    required = [
        "00_EXECUTIVE_SUMMARY.md",
        "02_FIXED_POSITION_BINDING.json",
        "03_EXISTING_RESULT_RECONSTRUCTION.csv",
        "04_PAIRED_DESIGN_AUDIT.csv",
        "05_PAIRED_UNIT_MANIFEST.csv",
        "06_FACTORIAL_PROTOCOL.md",
        "07_POSITION_METRIC_RECONSTRUCTION.csv",
        "08_PRIMARY_BLOCK_ACCURACY_CONTRASTS.csv",
        "09_SECONDARY_MACRO_F1_CONTRASTS.csv",
        "10_BOOTSTRAP_INTERVALS.csv",
        "11_SIMPLE_EFFECTS_BY_DISTANCE_AND_ANGLE.csv",
        "12_PER_CLASS_FACTOR_RESULTS.csv",
        "13_ER_SURFACE_HETEROGENEITY.csv",
        "14_SEED_AND_FOLD_CONSISTENCY.csv",
        "15_FACTORIAL_INTERPRETATION.md",
        "16_LIMITATIONS_AND_NONCLAIMS.md",
        "STATUS.md",
    ]
    assert [name for name in required if not (DEFAULT_RESULTS / name).is_file()] == []


def test_inherited_pairing_failure_and_synchronized_pairing_pass_are_explicit() -> None:
    audit = read_csv(DEFAULT_RESULTS / "04_PAIRED_DESIGN_AUDIT.csv")
    inherited_failures = [
        row for row in audit if row["design"] == "inherited_60_runs" and row["status"] == "FAIL"
    ]
    assert [(row["check"], row["details"]) for row in inherited_failures] == [
        ("train_validation_test_synchronization", "failures=492")
    ]
    synchronized = [row for row in audit if row["design"] == "case_b_synchronized_60_run_rerun"]
    assert len(synchronized) == 10
    assert {row["status"] for row in synchronized} == {"PASS"}


def test_synchronized_run_and_manifest_coverage_are_exact() -> None:
    register = read_csv(DEFAULT_RESULTS / "SYNCHRONIZED_RUN_REGISTER.csv")
    metrics = read_csv(DEFAULT_RESULTS / "SYNCHRONIZED_PER_RUN_METRICS.csv")
    assert len(register) == len({row["run_id"] for row in register}) == 60
    assert len(metrics) == len({row["run_id"] for row in metrics}) == 60
    assert {row["status"] for row in register} == {"PASS"}
    manifest = read_csv(
        PROJECT_ROOT
        / "manifests"
        / "angle_distance_factorial"
        / "synchronized_group_split_manifest.csv"
    )
    assert len(manifest) == 4 * 3 * 5 * 63
    assignments = {}
    for row in manifest:
        key = (row["TagID"], row["ER"], row["surface"], row["fold"], row["seed"])
        assignments.setdefault(key, set()).add(row["split_assignment"])
    assert len(assignments) == 3 * 5 * 63
    assert {tuple(values) for values in assignments.values()} <= {
        ("train",),
        ("validation",),
        ("test",),
    }


def test_primary_factor_estimates_recompute_from_sanitized_block_contrasts() -> None:
    rows = read_csv(DEFAULT_RESULTS / "08_PRIMARY_BLOCK_ACCURACY_CONTRASTS.csv")
    block_rows = [row for row in rows if row["record_type"] == "paired_block_seed_contrast"]
    overall = {
        row["contrast"]: float(row["estimate"])
        for row in rows
        if row["record_type"] == "overall_primary_estimand"
    }
    assert len(block_rows) == 63 * 5 * 3
    for contrast in overall:
        values = [float(row["estimate"]) for row in block_rows if row["contrast"] == contrast]
        assert sum(values) / len(values) == pytest.approx(overall[contrast])
    positions = {
        row["position"]: float(row["value"])
        for row in read_csv(DEFAULT_RESULTS / "07_POSITION_METRIC_RECONSTRUCTION.csv")
        if row["metric"] == "condition_block_accuracy"
    }
    assert factor_contrasts(positions) == pytest.approx(overall)
    intervals = {
        row["contrast"]: float(row["estimate"])
        for row in read_csv(DEFAULT_RESULTS / "10_BOOTSTRAP_INTERVALS.csv")
        if row["endpoint"] == "condition_block_accuracy"
        and row["method"] == "paired_tagid_stratified_block_bootstrap_seeds_fixed"
    }
    assert intervals == pytest.approx(overall)


def test_final_classification_respects_zero_boundary_and_seed_sensitivity() -> None:
    intervals = read_csv(DEFAULT_RESULTS / "10_BOOTSTRAP_INTERVALS.csv")
    primary = {
        row["contrast"]: row
        for row in intervals
        if row["endpoint"] == "condition_block_accuracy"
        and row["method"] == "paired_tagid_stratified_block_bootstrap_seeds_fixed"
    }
    sensitivity = {
        row["contrast"]: row
        for row in intervals
        if row["endpoint"] == "condition_block_accuracy"
        and row["method"] == "paired_tagid_stratified_block_plus_seed_bootstrap"
    }
    assert float(primary["angle_45_minus_0"]["confidence_interval_95_high"]) < 0.0
    assert abs(float(primary["angle_by_distance_interaction"]["confidence_interval_95_low"])) < 1e-12
    assert float(sensitivity["angle_by_distance_interaction"]["confidence_interval_95_low"]) < 0.0
    end = (DEFAULT_RESULTS / "STATUS.md").read_text(encoding="utf-8")
    assert "ANGLE_ASSOCIATED_DEGRADATION" in end
    assert "Interaction confirmed: no" in end
