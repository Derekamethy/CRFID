"""Regression tests over the released Strict-DG artifacts.

These tests validate the artifacts that are actually shipped. They recompute
every value from the released prediction bundles and hash tables rather than
comparing a file to a hash generated in the same run, and they require no raw
measurement data and no retraining.

The external audit found that no test anywhere touched ``outputs/strict_dg``.
This module closes that gap.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pytest

from crfid.governance.strict_release_manifest import verify_release_manifests
from crfid.governance.strict_target_authorization import (
    TargetAccessToken,
    TargetAuthorizationError,
    authorize_target_access,
    require_token,
)
from crfid.protocols.strict_dg import StrictDGProtocol
from crfid.strict_runtime.scale_policy import (
    CANONICAL_MODE,
    MODE_CLIP_AT_MINIMUM,
    MODE_REPLACE_WITH_ONE,
    apply_scale_policy,
)


ROOT = Path(__file__).resolve().parents[2]
SD = ROOT / "outputs" / "strict_dg"
FROZEN = SD / "frozen_release"
FINAL = SD / "final_p4"
SELECTION = SD / "source_selection"
RELEASE = SD / "canonical_release_20260727"

CANDIDATES = (
    "C0_NEUTRAL_ERM_1DCNN",
    "C1_FIRST_DIFFERENCE_ERM_1DCNN",
    "C2_SOURCE_EUCLIDEAN_NCM_READOUT",
    "C3_SOURCE_CORAL_ERM_1DCNN",
)
FOLDS = ("S1", "S2", "S3")
SEEDS = (42, 43, 44, 45, 46)
CLASS_COUNT = 7

EXPECTED_ACCURACY = 0.15663492063492063
EXPECTED_ACCURACY_SD = 0.030516243521601576
EXPECTED_MACRO_F1 = 0.11223088363457531
EXPECTED_MACRO_F1_SD = 0.01895556992674854
HISTORICAL_RECIPE_SHA256 = "4805f8040f3609dbf4a02553b15c64471f64e3d2a9bc7a6c1068a6342c29d1b6"

pytestmark = pytest.mark.skipif(
    not FINAL.is_dir() or not SELECTION.is_dir(),
    reason="Strict-DG released artifacts are not present in this checkout",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def confusion(true_labels, predictions, weights=None):
    matrix = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.float64)
    if weights is None:
        for a, b in zip(true_labels, predictions):
            matrix[a, b] += 1.0
    else:
        for a, b, w in zip(true_labels, predictions, weights):
            matrix[a, b] += float(w)
    return matrix


def metrics_from(matrix):
    tp = np.diag(matrix).astype(np.float64)
    row, col, total = matrix.sum(axis=1), matrix.sum(axis=0), matrix.sum()
    recall = np.array([tp[i] / row[i] if row[i] > 0 else 0.0 for i in range(CLASS_COUNT)])
    f1 = np.array(
        [
            (2.0 * tp[i]) / (2.0 * tp[i] + (col[i] - tp[i]) + (row[i] - tp[i]))
            if (2.0 * tp[i] + (col[i] - tp[i]) + (row[i] - tp[i])) > 0
            else 0.0
            for i in range(CLASS_COUNT)
        ]
    )
    return {
        "accuracy": float(tp.sum() / total) if total else 0.0,
        "macro_f1": float(f1.mean()),
        "per_class_recall": [float(v) for v in recall],
        "worst_class_recall": float(recall.min()),
        "zero_recall_class_count": int(np.count_nonzero(recall == 0.0)),
    }


def condition_metrics(true_labels, logits, condition_ids):
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for index, cid in enumerate(condition_ids):
        groups.setdefault(str(cid), []).append(index)
    block_true, block_pred = [], []
    for indices in groups.values():
        labels = np.unique(true_labels[indices])
        assert len(labels) == 1
        block_true.append(int(labels[0]))
        block_pred.append(int(np.argmax(logits[indices].astype(np.float64).mean(axis=0))))
    return metrics_from(confusion(np.asarray(block_true), np.asarray(block_pred)))


# --------------------------------------------------------------------------
# source folds and splits
# --------------------------------------------------------------------------
def _registry():
    with (SD / "source_inputs" / "CANONICAL_SOURCE_REGISTRY.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def _splits():
    parts = defaultdict(lambda: defaultdict(list))
    with (SELECTION / "source_fold_manifest.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parts[row["fold_id"]][row["partition"]].append(int(row["registry_row"]))
    return {fold: {part: sorted(rows) for part, rows in value.items()} for fold, value in parts.items()}


def test_source_registry_identity() -> None:
    registry = _registry()
    assert len(registry) == 9450
    assert Counter(row["position"] for row in registry) == {"P1": 3150, "P2": 3150, "P3": 3150}
    assert Counter(int(row["label_index"]) for row in registry) == {index: 1350 for index in range(7)}
    assert all(int(row["tag_id"]) - 1 == int(row["label_index"]) for row in registry)
    assert not any(row["position"] == "P4" for row in registry)
    blocks = Counter(row["raw_condition_id"] for row in registry)
    assert len(blocks) == 189 and set(blocks.values()) == {50}


def test_source_folds_are_leave_one_position_out_with_declared_counts() -> None:
    registry = _registry()
    splits = _splits()
    held_by_fold = {"S1": "P1", "S2": "P2", "S3": "P3"}
    for fold, held_position in held_by_fold.items():
        train = splits[fold]["inner_train"]
        validation = splits[fold]["inner_validation"]
        held = splits[fold]["outer_held"]
        assert (len(train), len(validation), len(held)) == (4200, 2100, 3150)
        assert held == sorted(
            int(row["registry_row"]) for row in registry if row["position"] == held_position
        )
        assert sorted(train + validation) == sorted(
            int(row["registry_row"]) for row in registry if row["position"] != held_position
        )
        assert sorted(train + validation + held) == list(range(len(registry)))


def test_no_split_overlap_of_samples_blocks_or_exact_signals() -> None:
    registry = _registry()
    splits = _splits()
    for fold in FOLDS:
        parts = {name: set(splits[fold][name]) for name in ("inner_train", "inner_validation", "outer_held")}
        names = list(parts)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                assert not parts[left] & parts[right]
                blocks_left = {registry[index]["raw_condition_id"] for index in parts[left]}
                blocks_right = {registry[index]["raw_condition_id"] for index in parts[right]}
                assert not blocks_left & blocks_right
                signals_left = {registry[index]["exact_signal_sha256"] for index in parts[left]}
                signals_right = {registry[index]["exact_signal_sha256"] for index in parts[right]}
                assert not signals_left & signals_right


# --------------------------------------------------------------------------
# the 60 source-selection units
# --------------------------------------------------------------------------
def _released_per_run():
    with (SELECTION / "per_run_metrics.csv").open(encoding="utf-8", newline="") as handle:
        return {(r["candidate_id"], r["fold_id"], int(r["seed"])): r for r in csv.DictReader(handle)}


def test_all_sixty_prediction_bundles_are_readable_and_self_consistent() -> None:
    released = _released_per_run()
    assert len(released) == 60
    for candidate in CANDIDATES:
        for fold in FOLDS:
            for seed in SEEDS:
                path = SELECTION / "predictions" / candidate / fold / f"seed_{seed}.npz"
                assert path.is_file(), path
                with np.load(path, allow_pickle=False) as bundle:
                    logits = np.asarray(bundle["logits"], dtype=np.float32)
                    saved = np.asarray(bundle["predictions"], dtype=np.int64)
                    labels = np.asarray(bundle["true_labels"], dtype=np.int64)
                assert logits.shape == (3150, CLASS_COUNT)
                assert np.array_equal(np.argmax(logits, axis=1).astype(np.int64), saved)
                assert set(np.unique(labels).tolist()) <= set(range(CLASS_COUNT))


def test_all_sixty_unit_metrics_recompute_exactly() -> None:
    released = _released_per_run()
    for key, row in released.items():
        candidate, fold, seed = key
        path = SELECTION / "predictions" / candidate / fold / f"seed_{seed}.npz"
        with np.load(path, allow_pickle=False) as bundle:
            labels = np.asarray(bundle["true_labels"], dtype=np.int64)
            logits = np.asarray(bundle["logits"], dtype=np.float32)
            conditions = np.asarray(bundle["condition_ids"], dtype=str)
            weights = np.asarray(bundle["unique_signal_weights"], dtype=np.float64)
        predictions = np.argmax(logits, axis=1).astype(np.int64)
        sample = metrics_from(confusion(labels, predictions))
        unique = metrics_from(confusion(labels, predictions, weights))
        block = condition_metrics(labels, logits, conditions)
        assert sample["accuracy"] == float(row["sample_accuracy"])
        assert sample["macro_f1"] == float(row["sample_macro_f1"])
        assert block["macro_f1"] == float(row["condition_macro_f1"])
        assert unique["macro_f1"] == float(row["unique_signal_weighted_macro_f1"])


def test_c1_is_independently_reselected_by_the_frozen_selector() -> None:
    released = _released_per_run()
    per_unit = {}
    for key in released:
        candidate, fold, seed = key
        path = SELECTION / "predictions" / candidate / fold / f"seed_{seed}.npz"
        with np.load(path, allow_pickle=False) as bundle:
            labels = np.asarray(bundle["true_labels"], dtype=np.int64)
            logits = np.asarray(bundle["logits"], dtype=np.float32)
            conditions = np.asarray(bundle["condition_ids"], dtype=str)
            weights = np.asarray(bundle["unique_signal_weights"], dtype=np.float64)
        predictions = np.argmax(logits, axis=1).astype(np.int64)
        per_unit[key] = {
            "sample": metrics_from(confusion(labels, predictions)),
            "unique": metrics_from(confusion(labels, predictions, weights)),
            "block": condition_metrics(labels, logits, conditions),
        }

    comparisons = []
    for candidate in CANDIDATES:
        units = [(candidate, fold, seed) for fold in FOLDS for seed in SEEDS]
        macro = {key: per_unit[key]["sample"]["macro_f1"] for key in units}
        fold_means = [
            statistics.fmean([macro[(candidate, fold, seed)] for seed in SEEDS]) for fold in FOLDS
        ]
        seed_means = [
            statistics.fmean([macro[(candidate, fold, seed)] for fold in FOLDS]) for seed in SEEDS
        ]
        per_class = [
            statistics.fmean([per_unit[key]["sample"]["per_class_recall"][index] for key in units])
            for index in range(CLASS_COUNT)
        ]
        comparisons.append(
            {
                "candidate_id": candidate,
                "worst": min(fold_means),
                "mean": statistics.fmean(list(macro.values())),
                "worst_class_recall": min(per_class),
                "zero_recall_frequency": sum(
                    per_unit[key]["sample"]["zero_recall_class_count"] for key in units
                )
                / (len(units) * CLASS_COUNT),
                "block": statistics.fmean([per_unit[key]["block"]["macro_f1"] for key in units]),
                "unique": statistics.fmean([per_unit[key]["unique"]["macro_f1"] for key in units]),
                "seed_sd": statistics.pstdev(seed_means),
            }
        )

    ordered = sorted(
        comparisons,
        key=lambda row: (
            -row["worst"],
            -row["mean"],
            -row["worst_class_recall"],
            row["zero_recall_frequency"],
            -row["block"],
            -row["unique"],
            row["seed_sd"],
            row["candidate_id"],
        ),
    )
    decision = json.loads((SELECTION / "source_selection_decision.json").read_text(encoding="utf-8"))
    assert ordered[0]["candidate_id"] == "C1_FIRST_DIFFERENCE_ERM_1DCNN"
    assert ordered[0]["candidate_id"] == decision["selected_candidate"]
    assert [row["candidate_id"] for row in ordered] == [row["candidate_id"] for row in decision["ranking"]]
    # decided on the first criterion, with a wide margin
    assert ordered[0]["worst"] - ordered[1]["worst"] > 0.05


def test_source_selection_declares_no_target_use_and_no_seed_selection() -> None:
    decision = json.loads((SELECTION / "source_selection_decision.json").read_text(encoding="utf-8"))
    assert decision["canonical_unit_count"] == 60
    assert decision["p4_used"] is False
    assert decision["target_accessed"] is False
    assert decision["best_seed_selection"] is False
    assert decision["all_seeds_equal_weight"] is True


def test_final_epochs_are_the_median_selected_inner_epoch_of_the_winner() -> None:
    released = _released_per_run()
    recipe = json.loads((FROZEN / "recipe.json").read_text(encoding="utf-8"))
    winner = recipe["selected_candidate"]
    for seed in SEEDS:
        folds = [int(released[(winner, fold, seed)]["selected_inner_epoch"]) for fold in FOLDS]
        assert int(statistics.median(folds)) == int(recipe["final_epochs_by_seed"][str(seed)])


# --------------------------------------------------------------------------
# frozen release identity
# --------------------------------------------------------------------------
def test_historical_recipe_identity_is_preserved() -> None:
    declared = (FROZEN / "recipe.sha256").read_text(encoding="ascii").split()[0]
    assert declared == HISTORICAL_RECIPE_SHA256
    assert sha256_file(FROZEN / "recipe.json") == HISTORICAL_RECIPE_SHA256
    recipe = json.loads((FROZEN / "recipe.json").read_text(encoding="utf-8"))
    assert recipe["class_order"] == list(range(7))
    assert recipe["seeds"] == list(SEEDS)
    assert recipe["selected_candidate"] == "C1_FIRST_DIFFERENCE_ERM_1DCNN"


def test_five_checkpoint_hashes_match_the_frozen_manifest() -> None:
    release = json.loads((FROZEN / "FROZEN_RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    declared = {int(row["seed"]): row for row in release["checkpoints"]}
    assert sorted(declared) == list(SEEDS)
    for seed, row in declared.items():
        path = FROZEN / "checkpoints" / f"seed_{seed}.pt"
        assert path.is_file()
        assert sha256_file(path) == row["sha256"]


def test_preprocessing_state_is_self_consistent_and_bound_to_every_checkpoint() -> None:
    import torch

    state = json.loads((FROZEN / "preprocessing" / "state.json").read_text(encoding="utf-8"))
    payload = {key: value for key, value in state.items() if key != "state_sha256"}
    recomputed = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert recomputed == state["state_sha256"]
    assert state["target_used_for_fit"] is False
    assert state["ddof"] == 0
    for seed in SEEDS:
        checkpoint = torch.load(
            FROZEN / "checkpoints" / f"seed_{seed}.pt", map_location="cpu", weights_only=True
        )
        assert checkpoint["preprocessing_state_sha256"] == state["state_sha256"]
        assert checkpoint["recipe_sha256"] == HISTORICAL_RECIPE_SHA256
        assert checkpoint["target_used"] is False


def test_canonical_scale_policy_reproduces_the_released_scale_bitwise() -> None:
    signals = np.load(SD / "source_inputs" / "source_signals_float64.npy", allow_pickle=False)
    diff = np.ascontiguousarray(np.diff(signals, axis=1), dtype=np.float64)
    raw = np.ascontiguousarray(diff.std(axis=0, ddof=0, dtype=np.float64))
    released = np.load(FROZEN / "preprocessing" / "scale_float64.npy", allow_pickle=False)
    assert np.array_equal(apply_scale_policy(raw, mode=CANONICAL_MODE), released)
    # the two historical modes are equivalent on this data; the fallback is unreachable
    assert float(raw.min()) > 1e-12
    assert np.array_equal(
        apply_scale_policy(raw, mode=MODE_REPLACE_WITH_ONE),
        apply_scale_policy(raw, mode=MODE_CLIP_AT_MINIMUM),
    )


# --------------------------------------------------------------------------
# final P4 results
# --------------------------------------------------------------------------
def _released_per_seed():
    with (FINAL / "per_seed_metrics.csv").open(encoding="utf-8", newline="") as handle:
        return {int(row["seed"]): row for row in csv.DictReader(handle)}


def test_five_p4_prediction_hashes_match_the_prediction_manifest() -> None:
    with (FINAL / "prediction_manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    for row in rows:
        path = ROOT / row["relative_path"]
        assert path.is_file()
        assert sha256_file(path) == row["file_sha256"]


def test_five_per_seed_metrics_recompute_exactly_from_the_released_bundles() -> None:
    released = _released_per_seed()
    assert sorted(released) == list(SEEDS)
    for seed, row in released.items():
        with np.load(FINAL / "predictions" / f"seed_{seed}.npz", allow_pickle=False) as bundle:
            labels = np.asarray(bundle["true_labels"], dtype=np.int64)
            logits = np.asarray(bundle["logits"], dtype=np.float32)
            conditions = np.asarray(bundle["condition_ids"], dtype=str)
            weights = np.asarray(bundle["unique_signal_weights"], dtype=np.float64)
            saved = np.asarray(bundle["predictions"], dtype=np.int64)
        predictions = np.argmax(logits, axis=1).astype(np.int64)
        assert np.array_equal(predictions, saved)
        sample = metrics_from(confusion(labels, predictions))
        unique = metrics_from(confusion(labels, predictions, weights))
        block = condition_metrics(labels, logits, conditions)
        assert sample["accuracy"] == float(row["accuracy"])
        assert sample["macro_f1"] == float(row["macro_f1"])
        assert sample["worst_class_recall"] == float(row["worst_class_recall"])
        assert sample["zero_recall_class_count"] == int(row["zero_recall_class_count"])
        assert block["accuracy"] == float(row["condition_accuracy"])
        assert block["macro_f1"] == float(row["condition_macro_f1"])
        assert unique["accuracy"] == float(row["unique_signal_weighted_accuracy"])
        assert unique["macro_f1"] == float(row["unique_signal_weighted_macro_f1"])


def test_aggregate_metrics_use_equal_seed_weight_and_population_sd() -> None:
    released = _released_per_seed()
    aggregate = json.loads((FINAL / "aggregate_metrics.json").read_text(encoding="utf-8"))
    assert aggregate["seed_count"] == 5
    assert aggregate["equal_seed_weight"] is True
    assert aggregate["ddof"] == 0

    accuracies = [float(released[seed]["accuracy"]) for seed in SEEDS]
    macro = [float(released[seed]["macro_f1"]) for seed in SEEDS]

    assert statistics.fmean(accuracies) == EXPECTED_ACCURACY
    assert statistics.pstdev(accuracies) == EXPECTED_ACCURACY_SD
    assert statistics.fmean(macro) == EXPECTED_MACRO_F1
    assert statistics.pstdev(macro) == EXPECTED_MACRO_F1_SD

    assert aggregate["metrics"]["accuracy"]["mean"] == EXPECTED_ACCURACY
    assert aggregate["metrics"]["accuracy"]["population_sd"] == EXPECTED_ACCURACY_SD
    assert aggregate["metrics"]["macro_f1"]["mean"] == EXPECTED_MACRO_F1
    assert aggregate["metrics"]["macro_f1"]["population_sd"] == EXPECTED_MACRO_F1_SD

    # ddof=1 must NOT reproduce the published figure
    assert statistics.stdev(accuracies) != EXPECTED_ACCURACY_SD


def test_no_seed_is_dropped_and_no_best_seed_is_chosen() -> None:
    assert len(list((FINAL / "predictions").glob("seed_*.npz"))) == 5
    assert len(list((FROZEN / "checkpoints").glob("seed_*.pt"))) == 5
    assert sorted(_released_per_seed()) == list(SEEDS)


# --------------------------------------------------------------------------
# release integrity layer
# --------------------------------------------------------------------------
def test_canonical_release_manifests_verify() -> None:
    report = verify_release_manifests(root=ROOT, release_directory=RELEASE)
    assert report["all_manifests_passed"], report["results"]
    assert report["no_unexpected_files"], report["unexpected_files_in_managed_directories"]
    assert report["verification_passed"]


def test_resolved_release_specification_matches_its_sidecar_and_seals_the_gaps() -> None:
    spec_path = RELEASE / "resolved_strict_release_specification.json"
    sidecar = RELEASE / "resolved_strict_release_specification.sha256"
    assert sha256_file(spec_path) == sidecar.read_text(encoding="ascii").split()[0]
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["historical_recipe"]["sha256"] == HISTORICAL_RECIPE_SHA256
    assert spec["training"]["outer_stage_offset"] == 100000
    assert spec["training"]["shuffle_seed_formula"]
    assert spec["architecture"]["padding"] == [3, 2, 1]
    assert spec["architecture"]["activation"] == "ReLU"
    assert spec["coral"]["alignment_weight"] == 0.1
    assert spec["normalization"]["low_variance_mode"] == CANONICAL_MODE
    assert spec["claim_boundaries"]["source_selection_units_retrained_in_v2"] == 0
    assert spec["claim_boundaries"]["final_models_retrained_in_v2"] == 5
    assert spec["claim_boundaries"]["full_60_model_retraining_from_this_package"] is False


def test_historical_manifest_status_is_disclosed_not_rewritten() -> None:
    status = json.loads((RELEASE / "historical_manifest_status.json").read_text(encoding="utf-8"))
    assert status["preserved_unchanged"] is True
    assert status["superseded_for_current_review_packaging"] is True
    assert status["provenance_assessment"]["entry_count_consistent_with_declared_producer"] is False
    assert status["supersession"]["new_manifest_claims_to_be_the_2026_07_22_manifest"] is False
    # the historical manifests themselves are still on disk
    for name in ("code_manifest.csv", "config_manifest.csv", "split_manifest.csv"):
        assert (FROZEN / name).is_file()


def test_low_variance_fallback_is_unreachable_on_real_data() -> None:
    report = json.loads((RELEASE / "low_variance_fallback_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["any_feature_below_1e-12"] is False
    assert report["summary"]["all_partitions_modes_agree_bitwise"] is True
    assert report["released_frozen_scale_check"]["canonical_mode_reproduces_released_scale_bitwise"] is True


# --------------------------------------------------------------------------
# official target-access enforcement
# --------------------------------------------------------------------------
def test_official_target_authorization_can_be_minted_from_the_frozen_release() -> None:
    token = authorize_target_access(frozen_release_directory=FROZEN, protocol=StrictDGProtocol())
    assert token.recipe_sha256 == HISTORICAL_RECIPE_SHA256
    assert sorted(token.checkpoint_sha256_by_seed) == list(SEEDS)
    assert token.token_sha256


def test_target_token_cannot_be_forged_by_direct_construction() -> None:
    with pytest.raises(TargetAuthorizationError):
        TargetAccessToken(
            schema_version=1,
            recipe_sha256=HISTORICAL_RECIPE_SHA256,
            preprocessing_state_sha256="x",
            checkpoint_sha256_by_seed={},
            authorization_file_sha256="x",
            frozen_release_directory=str(FROZEN),
            issued_at_utc="now",
        )


def test_target_access_rejects_missing_and_boolean_authorization() -> None:
    with pytest.raises(TargetAuthorizationError):
        require_token(None)
    with pytest.raises(TargetAuthorizationError):
        require_token(True)  # type: ignore[arg-type]
    with pytest.raises(TargetAuthorizationError):
        require_token({"authorized": True})  # type: ignore[arg-type]


def test_labels_are_not_released_before_predictions_are_serialized() -> None:
    token = authorize_target_access(frozen_release_directory=FROZEN, protocol=StrictDGProtocol())
    with pytest.raises(TargetAuthorizationError):
        token.release_labels(purpose="final_scoring")
    token.release_features(purpose="test")
    with pytest.raises(TargetAuthorizationError):
        token.release_labels(purpose="final_scoring")
    token.mark_predictions_serialized()
    token.release_labels(purpose="final_scoring")
    assert any(entry["resource"] == "labels" and entry["granted"] for entry in token.access_log)


def test_labels_are_never_released_for_a_non_scoring_purpose() -> None:
    token = authorize_target_access(frozen_release_directory=FROZEN, protocol=StrictDGProtocol())
    token.release_features(purpose="test")
    token.mark_predictions_serialized()
    for purpose in ("selection", "training", "adaptation"):
        with pytest.raises(TargetAuthorizationError):
            token.release_labels(purpose=purpose)


def test_authorization_bound_to_a_different_release_fails(tmp_path) -> None:
    token = authorize_target_access(frozen_release_directory=FROZEN, protocol=StrictDGProtocol())
    token.frozen_release_directory = str(tmp_path)
    with pytest.raises(TargetAuthorizationError):
        token.revalidate()


def test_authorization_fails_when_a_checkpoint_digest_changes() -> None:
    token = authorize_target_access(frozen_release_directory=FROZEN, protocol=StrictDGProtocol())
    token.checkpoint_sha256_by_seed[42] = "0" * 64
    with pytest.raises(TargetAuthorizationError):
        token.revalidate()


def test_authorization_fails_when_the_recipe_digest_changes() -> None:
    token = authorize_target_access(frozen_release_directory=FROZEN, protocol=StrictDGProtocol())
    token.recipe_sha256 = "0" * 64
    with pytest.raises(TargetAuthorizationError):
        token.revalidate()


def test_authorization_rejects_a_tampered_authorization_file(tmp_path) -> None:
    import shutil

    fake = tmp_path / "frozen_release"
    shutil.copytree(FROZEN, fake)
    payload = json.loads((fake / "target_access_authorization.json").read_text(encoding="utf-8"))
    payload["source_selection_unit_count"] = 6
    (fake / "target_access_authorization.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(TargetAuthorizationError):
        authorize_target_access(frozen_release_directory=fake, protocol=StrictDGProtocol())


def test_official_p4_loader_refuses_without_a_token() -> None:
    from crfid.strict_runtime import target_evaluation

    with pytest.raises(TypeError):
        target_evaluation.load_p4_once(custody_manifest={}, source_config={})  # type: ignore[call-arg]
    with pytest.raises(TargetAuthorizationError):
        target_evaluation.load_p4_once(custody_manifest={}, source_config={}, token=None)  # type: ignore[arg-type]


def test_no_target_assisted_artifact_is_referenced_by_strict_dg_sources() -> None:
    forbidden = ("target_assisted", "few_shot", "external_data1", "openems")
    roots = [ROOT / "src" / "crfid" / "strict_runtime", ROOT / "scripts"]
    for base in roots:
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            for name in forbidden:
                assert name not in text, f"{path} references {name}"


# --------------------------------------------------------------------------
# packaging hygiene
# --------------------------------------------------------------------------
def test_release_layer_contains_no_local_absolute_paths() -> None:
    import re

    # Assembled from fragments so that this test file does not itself contain a
    # literal that the repository-wide local-path scanner would flag.
    windows_root = r"[A-Za-z]:[\\/]{1,2}(?:" + "Users|Documents and Settings" + r")[\\/]{1,2}"
    unix_root = "/" + "home/" + r"[^/]+/"
    pattern = re.compile(windows_root + "|" + unix_root)
    for path in RELEASE.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".md", ".sha256"}:
            assert not pattern.search(path.read_text(encoding="utf-8", errors="ignore")), path
