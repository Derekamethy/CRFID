from __future__ import annotations

import json
import re
import csv
from pathlib import Path

import pytest

from crfid.domain_aware_mixup.core import (
    FINAL_P4_EVALUATION,
    FINAL_SOURCE_TRAINING,
    SOURCE_LOPO_DEVELOPMENT,
    classify_main_result,
    execution_record,
    select_alpha_source_only,
)
from crfid.domain_aware_mixup.runtime import PROJECT_ROOT


@pytest.mark.parametrize(
    "stage,fold,training,held,target",
    [
        (SOURCE_LOPO_DEVELOPMENT, "S1", "P2_P3", "P1", None),
        (SOURCE_LOPO_DEVELOPMENT, "S2", "P1_P3", "P2", None),
        (SOURCE_LOPO_DEVELOPMENT, "S3", "P1_P2", "P3", None),
        (FINAL_SOURCE_TRAINING, None, "P1_P2_P3", None, None),
        (FINAL_P4_EVALUATION, None, "P1_P2_P3", None, "P4"),
    ],
)
def test_explicit_execution_records(stage, fold, training, held, target):
    record = execution_record(stage, fold)
    assert record["training_positions"] == training
    assert record["held_source_position"] == held
    assert record["target_position"] == target


def selection_rows_with_distractors():
    rows = []
    for alpha, score in ((.2, .6), (.5, .61), (1., .59)):
        for position in ("P1", "P2", "P3"):
            for seed in (42, 43, 44, 45, 46):
                rows.append({"method": "cross_position_mixup", "alpha": alpha, "held_position": position, "held_macro_f1": score, "held_accuracy": score, "pair_coverage": 1.0})
    rows.extend({"method": "within_position_mixup", "alpha": .5, "held_position": "P4", "held_macro_f1": 1.0, "held_accuracy": 1.0, "pair_coverage": 1.0} for _ in range(15))
    return rows


def test_control_cannot_select_alpha():
    assert select_alpha_source_only(selection_rows_with_distractors())["selected_alpha"] == .5


def test_p4_cannot_select_alpha():
    receipt = select_alpha_source_only(selection_rows_with_distractors())
    assert receipt["p4_accessed"] is False


@pytest.mark.parametrize(
    "override,expected",
    [
        ({"source_guardrail_passed": False}, "DOMAIN_AWARE_MIXUP_TAGID_TRADEOFF"),
        ({"unstable": True}, "DOMAIN_AWARE_MIXUP_MIXED_OR_UNSTABLE_RESULT"),
        ({}, "DOMAIN_AWARE_MIXUP_NO_RELIABLE_BENEFIT"),
        ({"intervention_valid": False}, "FAIL_MIXUP_INTERVENTION_VALIDITY"),
        ({"protocol_valid": False}, "FAIL_PROTOCOL_OR_LABEL_BOUNDARY_DEFECT"),
    ],
)
def test_remaining_main_classifications(override, expected):
    args = {
        "source_mean_interval": (-.01, .01),
        "source_worst_interval": (-.01, .01),
        "p4_cross_vs_erm_interval": (-.01, .01),
        "p4_cross_vs_within_interval": (-.01, .01),
        "p4_within_vs_erm_interval": (-.01, .01),
        "source_guardrail_passed": True,
        "diagnostic_classification": "CROSS_POSITION_GEOMETRY_NOT_IMPROVED",
        "unstable": False,
        "intervention_valid": True,
        "protocol_valid": True,
    }
    args.update(override)
    assert classify_main_result(**args) == expected


def test_patch_layout_is_additive():
    for relative in (
        "workflows/17_domain_aware_mixup/run.py",
        "configs/domain_aware_mixup/canonical.json",
        "src/crfid/domain_aware_mixup/core.py",
        "src/crfid/domain_aware_mixup/runtime.py",
        "docs",
        "manifests/domain_aware_mixup",
        "results/canonical_metrics/domain_aware_mixup",
    ):
        assert (PROJECT_ROOT / relative).exists()


def test_runtime_code_has_no_personal_absolute_path():
    text = "\n".join((PROJECT_ROOT / relative).read_text(encoding="utf-8") for relative in (
        "src/crfid/domain_aware_mixup/core.py",
        "src/crfid/domain_aware_mixup/runtime.py",
        "configs/domain_aware_mixup/canonical.json",
    ))
    assert not re.search(r"[A-Za-z]:[\\/]Users[\\/]", text)


@pytest.mark.parametrize("placeholder", ["<GOVERNED_DATA_ROOT>", "<LOCKED_CRFID_PYTHON>", "<EXTERNAL_RUNTIME_ROOT>"])
def test_sanitized_runtime_placeholders(placeholder):
    assert placeholder in (PROJECT_ROOT / "src/crfid/domain_aware_mixup/runtime.py").read_text(encoding="utf-8")


def test_no_authoritative_remote_dependency_in_patch():
    assert not (PROJECT_ROOT / ".git/objects/info/alternates").is_file()


def test_mixed_embeddings_not_physical_claim():
    config = json.loads((PROJECT_ROOT / "configs/domain_aware_mixup/canonical.json").read_text(encoding="utf-8"))
    assert config["mixup"]["location"] == "penultimate_encoder_embedding"
    assert config["mixup"]["input_space_mixup"] is False


def test_governed_pairing_failure_receipt_is_exact():
    path = PROJECT_ROOT / "results/canonical_metrics/domain_aware_mixup/06_DATA_AND_PAIRING_STRUCTURE_AUDIT.csv"
    if not path.is_file():
        pytest.skip("governed preflight has not run")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    inner_cross = [row for row in rows if row["partition"] == "inner_train" and row["method"] == "cross_position_mixup"]
    assert len(inner_cross) == 3
    assert all(int(row["eligible_partner_count"]) == 2100 and int(row["training_parent_count"]) == 4200 for row in inner_cross)
    assert all(float(row["pair_coverage"]) == 0.5 for row in inner_cross)


def test_blocked_receipt_proves_no_p4_access():
    path = PROJECT_ROOT / "manifests/domain_aware_mixup/pairing_coverage_failure.json"
    if not path.is_file():
        pytest.skip("governed preflight has not run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["p4_artifacts_opened"] is False
    assert payload["p4_labels_opened"] is False
