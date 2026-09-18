from __future__ import annotations

import csv

import numpy as np
import pytest

from crfid.groupdro_worst_source.core import (
    GroupDROProtocolError,
    P4LabelSeal,
    classify_interpretation,
    deterministic_block_majority_vote,
    paired_source_worst_bootstrap,
    paired_tagid_stratified_block_bootstrap,
    select_eta_source_only,
    source_worst_position_metrics,
)


def _source_rows() -> list[dict[str, float | int | str]]:
    rows = []
    scores = {0.01: 0.55, 0.05: 0.56, 0.10: 0.555, 0.20: 0.53}
    for eta, base in scores.items():
        for fold, position in (("S1", "P1"), ("S2", "P2"), ("S3", "P3")):
            for seed in (42, 43, 44, 45, 46):
                adjustment = 0.003 if position == "P1" else 0.0
                rows.append(
                    {
                        "eta": eta,
                        "fold_id": fold,
                        "held_position": position,
                        "seed": seed,
                        "held_macro_f1": base + adjustment,
                        "held_accuracy": base + 0.1,
                        "max_q_weight": 0.6,
                    }
                )
    return rows


def test_source_only_eta_selector_and_worst_position_endpoint() -> None:
    decision = select_eta_source_only(_source_rows())
    assert decision["selected_eta"] == 0.05
    assert not decision["selection_input_p4_accessed"]
    selected = [row for row in _source_rows() if row["eta"] == decision["selected_eta"]]
    endpoint = source_worst_position_metrics(selected)
    assert endpoint["worst_position_macro_f1"] == pytest.approx(0.56)


def test_paired_source_bootstrap_preserves_fold_seed_pairing() -> None:
    paired = [
        {
            "fold_id": fold,
            "seed": seed,
            "erm_macro_f1": 0.50,
            "groupdro_macro_f1": 0.60,
        }
        for fold in ("S1", "S2", "S3")
        for seed in (42, 43, 44, 45, 46)
    ]
    result = paired_source_worst_bootstrap(paired, resamples=500, seed=11)
    assert result["point_estimate"] == pytest.approx(0.10)
    assert result["interval_95"][0] > 0
    leave_fold_out = paired_source_worst_bootstrap(
        paired,
        resamples=100,
        seed=11,
        fold_ids=("S1", "S2"),
    )
    assert leave_fold_out["active_folds"] == ["S1", "S2"]
    assert leave_fold_out["point_estimate"] == pytest.approx(0.10)


def test_deterministic_block_majority_vote_and_tie_breaking() -> None:
    labels = np.asarray([1] * 50 + [0] * 50, dtype=np.int64)
    predictions = np.asarray([1] * 25 + [2] * 25 + [0] * 50, dtype=np.int64)
    metrics, rows = deterministic_block_majority_vote(labels, predictions, ["b"] * 50 + ["a"] * 50)
    assert [row["condition_block_index"] for row in rows] == [0, 1]
    assert rows[1]["predicted_label"] == 1
    assert metrics["accuracy"] == pytest.approx(1.0)


def _write_sealed_p4(root) -> None:
    np.save(root / "p4_signals_float64.npy", np.zeros((3150, 281), dtype="<f8"), allow_pickle=False)
    with (root / "p4_unlabelled_registry.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "raw_condition_id", "position"])
        writer.writeheader()
        for block in range(63):
            for repeat in range(50):
                writer.writerow({"sample_id": f"s_{block}_{repeat}", "raw_condition_id": f"c_{block}", "position": "P4"})
    np.save(root / "p4_labels_int64.npy", np.repeat(np.arange(7, dtype="<i8"), 450), allow_pickle=False)


def test_p4_label_seal_freezes_predictions_before_opening_labels(tmp_path) -> None:
    _write_sealed_p4(tmp_path)
    seal = P4LabelSeal(tmp_path)
    with pytest.raises(GroupDROProtocolError, match="FAIL_P4_LABEL_BOUNDARY"):
        seal.open_labels()
    data = seal.load_unlabelled()
    assert data.signals.shape == (3150, 281)
    frozen = seal.freeze_predictions({"erm_seed_42": np.zeros(3150, dtype=np.int64)}, tmp_path / "predictions.npz")
    labels = seal.open_labels()
    assert labels.shape == (3150,)
    assert frozen["prediction_bundle_sha256"]
    assert [row["labels_accessed"] for row in seal.access_log] == [False, False, True]


def test_paired_block_bootstrap_and_no_row_level_pseudoreplication() -> None:
    labels = np.repeat(np.arange(7, dtype=np.int64), 9)
    erm = np.tile(labels, (5, 1))
    groupdro = np.tile(labels, (5, 1))
    groupdro[:, labels == 0] = 1
    result = paired_tagid_stratified_block_bootstrap(labels, erm, groupdro, resamples=300, seed=7)
    assert result["resamples"] == 300
    assert len(result["seed_wise_contrasts"]) == 5
    seed_resampled = paired_tagid_stratified_block_bootstrap(labels, erm, groupdro, resamples=100, seed=7, resample_seeds=True)
    assert seed_resampled["resample_training_seeds"]
    with pytest.raises(ValueError):
        paired_tagid_stratified_block_bootstrap(labels, erm[0], groupdro[0])


def test_interpretation_classes_cover_source_p4_and_protocol_outcomes() -> None:
    assert classify_interpretation(governed_inputs_available=False, protocol_defect=False, source_worst_interval=None, p4_interval=None, source_mean_contrast=None, source_mean_interval=None, p4_point_contrast=None, unstable=False) == "BLOCKED_GOVERNED_INPUTS"
    assert classify_interpretation(governed_inputs_available=True, protocol_defect=True, source_worst_interval=None, p4_interval=None, source_mean_contrast=None, source_mean_interval=None, p4_point_contrast=None, unstable=False) == "FAIL_PROTOCOL_OR_LABEL_BOUNDARY_DEFECT"
    assert classify_interpretation(governed_inputs_available=True, protocol_defect=False, source_worst_interval=(0.01, 0.2), p4_interval=(0.02, 0.3), source_mean_contrast=0.0, source_mean_interval=(-0.01, 0.01), p4_point_contrast=0.1, unstable=False) == "GROUPDRO_SOURCE_ROBUSTNESS_AND_P4_BENEFIT_CONFIRMED"
    assert classify_interpretation(governed_inputs_available=True, protocol_defect=False, source_worst_interval=(-0.1, 0.1), p4_interval=(-0.3, -0.02), source_mean_contrast=0.0, source_mean_interval=(-0.01, 0.01), p4_point_contrast=-0.1, unstable=False) == "GROUPDRO_HARMS_P4_TRANSFER"
