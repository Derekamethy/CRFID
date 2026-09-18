"""Boundary tests for the external-Data1 SSL strict-DG branch.

These cover the scientific boundaries rather than incidental behaviour: label leakage,
target leakage, fold disjointness, sampler fidelity, gate ordering and write isolation.
Tests that need executed evidence skip cleanly when that evidence is not present yet, so
the file is runnable at any point in the branch's life.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest

from crfid.external_data1_ssl_strict_dg import (
    augmentations,
    carrier,
    config_validation,
    data1_corpus,
    metrics,
    model as model_module,
    objectives,
    paths,
    pretrain,
    readout,
    sealed_target,
    source_data,
    supervised,
    treatments,
)

OUTPUT = paths.BRANCH_OUTPUT_ROOT


def _load(*parts: str):
    path = OUTPUT.joinpath(*parts)
    if not path.is_file():
        pytest.skip(f"evidence not produced yet: {'/'.join(parts)}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_files(*required: Path) -> None:
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        pytest.skip(f"governed external artifacts are absent: {', '.join(missing)}")


# --------------------------------------------------------------- label and target leakage


def test_data1_loader_exposes_no_label_path():
    """No public function in the Data1 module may return a label."""

    source = inspect.getsource(data1_corpus)
    assert "return signals, record" in source
    for name, function in vars(data1_corpus).items():
        if not callable(function) or name.startswith("_"):
            continue
        annotations = getattr(function, "__annotations__", {})
        assert "label" not in str(annotations.get("return", "")).lower(), name


def test_data1_ssl_corpus_contains_only_signals():
    if not os.environ.get("CRFID_SSL_DG_DATA1_ROOT"):
        pytest.skip("CRFID_SSL_DG_DATA1_ROOT is not configured")
    corpus = data1_corpus.load_corpus()
    assert corpus.ndim == 2 and corpus.shape[1] == data1_corpus.SIGNAL_LENGTH
    built = pretrain.build_ssl_corpus(pretrain.DATA1_CORPUS, corpus[:64])
    assert not hasattr(built, "labels")
    fields = {field for field in built.__dataclass_fields__}
    assert not any("label" in field for field in fields)


def test_ssl_loss_signature_accepts_no_labels():
    parameters = set(inspect.signature(objectives.compute_loss).parameters)
    assert not any("label" in name or "target" in name for name in parameters)


def test_pretrain_signature_accepts_no_labels():
    parameters = set(inspect.signature(pretrain.pretrain).parameters)
    assert not any("label" in name for name in parameters)


def test_source_corpus_is_p1_p2_p3_only():
    _require_files(
        paths.STRICT_DG_SOURCE_SIGNALS,
        paths.STRICT_DG_SOURCE_LABELS,
        paths.STRICT_DG_SOURCE_REGISTRY,
    )
    corpus = source_data.load_source_corpus()
    assert sorted(set(corpus.domains.tolist())) == ["P1", "P2", "P3"]
    assert source_data.TARGET_DOMAIN not in set(corpus.domains.tolist())


def test_source_manifest_has_no_target_rows():
    path = OUTPUT / "01_manifests" / "06_PAPER4_SOURCE_ONLY_MANIFEST.csv"
    if not path.is_file():
        pytest.skip("Gate A has not run")
    text = path.read_text(encoding="utf-8")
    assert ",P4," not in text and not text.rstrip().endswith(",P4")


def test_data1_manifest_carries_no_label_column():
    path = OUTPUT / "01_manifests" / "05_LABEL_FREE_DATA1_MANIFEST.csv"
    if not path.is_file():
        pytest.skip("Gate A has not run")
    header = path.read_text(encoding="utf-8").splitlines()[0].lower()
    assert "label_use" in header
    assert "class" not in header and "label_value" not in header


# ----------------------------------------------------------------------------- folds


def test_source_folds_are_disjoint_and_hold_out_their_domain():
    _require_files(
        paths.STRICT_DG_SOURCE_SIGNALS,
        paths.STRICT_DG_SOURCE_LABELS,
        paths.STRICT_DG_SOURCE_REGISTRY,
        paths.STRICT_DG_LOPO_SPLITS,
    )
    corpus = source_data.load_source_corpus()
    for fold, groups in source_data.load_fold_partitions().items():
        held = set(groups["outer_held"].tolist())
        train = set(groups["inner_train"].tolist())
        validation = set(groups["inner_validation"].tolist())
        assert not (train & validation) and not (train & held) and not (validation & held)
        expected = source_data.FOLD_OUTER_DOMAIN[fold]
        assert {corpus.domains[index] for index in held} == {expected}
        assert expected not in {corpus.domains[index] for index in train | validation}


def test_source_validation_never_touches_the_target():
    _require_files(
        paths.STRICT_DG_SOURCE_SIGNALS,
        paths.STRICT_DG_SOURCE_LABELS,
        paths.STRICT_DG_SOURCE_REGISTRY,
        paths.STRICT_DG_LOPO_SPLITS,
    )
    for groups in source_data.load_fold_partitions().values():
        for indices in groups.values():
            assert indices.max() < source_data.load_source_corpus().signals.shape[0]


# --------------------------------------------------------------------------- sampler


def test_balanced_joint_sampler_respects_the_declared_ratio():
    allocation = pretrain._batch_allocation({pretrain.PAPER4_CORPUS: 0.5, pretrain.DATA1_CORPUS: 0.5}, 256)
    assert allocation == {pretrain.PAPER4_CORPUS: 128, pretrain.DATA1_CORPUS: 128}
    allocation = pretrain._batch_allocation({pretrain.PAPER4_CORPUS: 0.25, pretrain.DATA1_CORPUS: 0.75}, 256)
    assert allocation == {pretrain.PAPER4_CORPUS: 64, pretrain.DATA1_CORPUS: 192}
    assert sum(allocation.values()) == 256


def test_sampler_rejects_ratios_that_do_not_sum_to_one():
    with pytest.raises(pretrain.SslCorpusViolation):
        pretrain._batch_allocation({pretrain.PAPER4_CORPUS: 0.4, pretrain.DATA1_CORPUS: 0.4}, 256)


def test_pretrain_rejects_undeclared_corpora():
    corpus = pretrain.build_ssl_corpus(pretrain.PAPER4_CORPUS, np.random.default_rng(0).normal(size=(32, 281)))
    config = pretrain.PretrainConfig(
        objective=objectives.ObjectiveConfig(objective_id=objectives.O1_MASKED_SPAN),
        epochs=1,
        steps_per_epoch=1,
        corpus_fractions={pretrain.PAPER4_CORPUS: 0.5, pretrain.DATA1_CORPUS: 0.5},
    )
    with pytest.raises(pretrain.SslCorpusViolation):
        pretrain.pretrain({pretrain.PAPER4_CORPUS: corpus}, config, seed=42)


# --------------------------------------------------------------------- augmentations


def test_augmentations_preserve_shape_and_finiteness():
    generator = np.random.default_rng(7)
    views = generator.normal(size=(16, 1, 256)).astype(np.float32)
    augmented = augmentations.augment(views, augmentations.AugmentationConfig(), generator)
    assert augmented.shape == views.shape
    assert np.isfinite(augmented).all()
    assert augmented.dtype == np.float32


def test_augmentations_are_seed_reproducible():
    views = np.random.default_rng(1).normal(size=(8, 1, 256)).astype(np.float32)
    config = augmentations.AugmentationConfig()
    first = augmentations.augment(views, config, np.random.default_rng(11))
    second = augmentations.augment(views, config, np.random.default_rng(11))
    assert np.array_equal(first, second)


def test_span_mask_marks_contiguous_spans_only():
    generator = np.random.default_rng(3)
    mask = augmentations.contiguous_span_mask(8, 256, span_count=4, span_length=16, generator=generator)
    assert mask.shape == (8, 256)
    assert mask.any(axis=1).all()
    assert mask.sum(axis=1).max() <= 4 * 16


# ------------------------------------------------------------- carrier and padding cue


def test_no_padding_exists_and_ssl_windows_share_one_length():
    generator = np.random.default_rng(5)
    paper4 = pretrain.build_ssl_corpus("paper4_source", generator.normal(size=(32, 281)))
    data1 = pretrain.build_ssl_corpus("data1_external", generator.normal(size=(32, 1601)))
    assert paper4.difference.shape[1] == 280 and data1.difference.shape[1] == 1600
    window = carrier.SSL_WINDOW_LENGTH
    first = augmentations.random_window(paper4.difference, window, np.random.default_rng(1))
    second = augmentations.random_window(data1.difference, window, np.random.default_rng(1))
    assert first.shape[1] == second.shape[1] == window


def test_backbone_is_length_agnostic_and_canonical():
    import torch

    backbone = model_module.initialize_backbone(42)
    assert model_module.parameter_count(backbone) == model_module.CANONICAL_PARAMETER_COUNT_SEVEN_CLASS
    with torch.no_grad():
        for length in (256, 280, 1600):
            assert tuple(backbone(torch.zeros(2, 1, length)).shape) == (2, 7)


def test_backbone_initialisation_matches_the_canonical_anchor():
    from crfid.models.cnn1d import initialize_strict_model

    assert model_module.state_sha256(model_module.initialize_backbone(42)) == model_module.state_sha256(
        initialize_strict_model(42)
    )


def test_standardizer_reproduces_the_canonical_frozen_state():
    _require_files(
        paths.STRICT_DG_SOURCE_SIGNALS,
        paths.STRICT_DG_SOURCE_LABELS,
        paths.STRICT_DG_SOURCE_REGISTRY,
        paths.STRICT_DG_FROZEN_RELEASE / "preprocessing" / "mean_float64.npy",
        paths.STRICT_DG_FROZEN_RELEASE / "preprocessing" / "scale_float64.npy",
    )
    corpus = source_data.load_source_corpus()
    fitted = carrier.fit_supervised_standardizer(corpus.signals)
    directory = paths.STRICT_DG_FROZEN_RELEASE / "preprocessing"
    assert np.array_equal(fitted.mean, np.load(directory / "mean_float64.npy"))
    assert np.array_equal(fitted.scale, np.load(directory / "scale_float64.npy"))


# ------------------------------------------------------------------------ determinism


def test_final_epoch_rule_reproduces_the_canonical_recipe():
    import csv

    _require_files(
        paths.STRICT_DG_SOURCE_SELECTION / "per_run_metrics.csv",
        paths.STRICT_DG_FROZEN_RELEASE / "recipe.json",
    )

    rows = [
        row
        for row in csv.DictReader(
            (paths.STRICT_DG_SOURCE_SELECTION / "per_run_metrics.csv").open(newline="", encoding="utf-8")
        )
        if row["candidate_id"] == "C1_FIRST_DIFFERENCE_ERM_1DCNN"
    ]
    by_seed: dict[int, list[int]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(int(row["selected_inner_epoch"]))
    recipe = json.loads((paths.STRICT_DG_FROZEN_RELEASE / "recipe.json").read_text(encoding="utf-8"))
    derived = {str(seed): supervised.median_epoch(values) for seed, values in by_seed.items()}
    assert derived == {key: int(value) for key, value in recipe["final_epochs_by_seed"].items()}


def test_permutation_seed_matches_the_canonical_rule():
    assert supervised.permutation_seed(42, supervised.INNER_STAGE_OFFSET, 3) == 42_000_003
    assert supervised.permutation_seed(42, supervised.OUTER_STAGE_OFFSET, 3) == 42_100_003


def test_ssl_pretraining_is_seed_reproducible():
    corpus = pretrain.build_ssl_corpus(
        pretrain.PAPER4_CORPUS, np.random.default_rng(0).normal(size=(64, 281))
    )
    config = pretrain.PretrainConfig(
        objective=objectives.ObjectiveConfig(objective_id=objectives.O1_MASKED_SPAN),
        epochs=1,
        steps_per_epoch=2,
        batch_size=16,
        corpus_fractions={pretrain.PAPER4_CORPUS: 1.0},
    )
    first = pretrain.pretrain({pretrain.PAPER4_CORPUS: corpus}, config, seed=42)
    second = pretrain.pretrain({pretrain.PAPER4_CORPUS: corpus}, config, seed=42)
    assert first.trunk_sha256 == second.trunk_sha256


# ----------------------------------------------------------------------------- metrics


def test_macro_f1_matches_sklearn_zero_division_zero():
    from sklearn.metrics import accuracy_score, f1_score

    generator = np.random.default_rng(4)
    labels = generator.integers(0, 7, 400)
    predictions = generator.integers(0, 5, 400)  # two classes never predicted
    computed = metrics.classification_metrics(labels, predictions)
    assert computed["macro_f1"] == pytest.approx(
        f1_score(labels, predictions, average="macro", labels=list(range(7)), zero_division=0)
    )
    assert computed["accuracy"] == pytest.approx(accuracy_score(labels, predictions))


def test_argmax_uses_lowest_index_tie_breaking():
    logits = np.array([[1.0, 1.0, 0.0], [0.0, 2.0, 2.0]], dtype=np.float32)
    assert metrics.argmax_predictions(logits).tolist() == [0, 1]


# -------------------------------------------------------------------------- readouts


def test_ncm_uses_source_prototypes_only():
    generator = np.random.default_rng(2)
    embeddings = generator.normal(size=(70, 256))
    labels = np.repeat(np.arange(7), 10)
    prototypes = readout.NearestClassMean.fit(embeddings, labels, fitted_on="source")
    record = prototypes.as_record()
    assert record["target_prototypes_used"] is False
    assert record["target_support_samples_used"] is False
    assert prototypes.predict(embeddings).shape == (70,)


# ------------------------------------------------------------------ gate-E enforcement


def test_token_cannot_be_hand_constructed():
    with pytest.raises(sealed_target.TargetAuthorizationError):
        sealed_target.PreregistrationToken(
            schema_version=1,
            preregistration_sha256="0" * 64,
            checkpoint_sha256_by_unit={},
            registered_treatments=(),
            preregistration_path="x",
            checkpoint_directory="y",
            issued_at_utc="now",
        )


def _mint(tmp_path: Path, units: dict[str, str]) -> sealed_target.PreregistrationToken:
    import torch

    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    entries = []
    for unit in units:
        path = checkpoints / f"{unit}.pt"
        torch.save({"unit_id": unit}, path)
        entries.append({"unit_id": unit, "checkpoint_sha256": _sha(path)})
    payload = {
        "gate_d_status": "PASS_FINAL_PREREGISTRATION_FROZEN",
        "target_accessed_before_preregistration": False,
        "registered_treatments": sorted({unit.split("__")[0] for unit in units}),
        "evaluation_units": entries,
    }
    preregistration = tmp_path / "prereg.json"
    preregistration.write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "prereg.json.sha256").write_text(f"{_sha(preregistration)}  prereg.json\n", encoding="ascii")
    return sealed_target.authorize_target_access(
        preregistration_path=preregistration, checkpoint_directory=checkpoints
    )


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_labels_cannot_open_before_predictions_close(tmp_path):
    token = _mint(tmp_path, {"T0_MATCHED_SUPERVISED_ONLY__seed_42": ""})
    sealed = sealed_target.SealedLabels(np.zeros(4, dtype=np.int64))
    with pytest.raises(sealed_target.TargetAuthorizationError):
        sealed.open(token)
    token.release_features(purpose="test")
    with pytest.raises(sealed_target.TargetAuthorizationError):
        sealed.open(token)


def test_predictions_cannot_be_regenerated_after_label_access(tmp_path):
    unit = "T0_MATCHED_SUPERVISED_ONLY__seed_42"
    token = _mint(tmp_path, {unit: ""})
    token.release_features(purpose="test")
    token.close_predictions({unit: "d" * 64})
    sealed = sealed_target.SealedLabels(np.zeros(4, dtype=np.int64))
    sealed.open(token)
    with pytest.raises(sealed_target.TargetAuthorizationError):
        token.release_features(purpose="second_pass")


def test_labels_release_only_for_final_scoring(tmp_path):
    unit = "T0_MATCHED_SUPERVISED_ONLY__seed_42"
    token = _mint(tmp_path, {unit: ""})
    token.release_features(purpose="test")
    token.close_predictions({unit: "d" * 64})
    with pytest.raises(sealed_target.TargetAuthorizationError):
        token.release_labels(purpose="diagnostics")


def test_evaluator_rejects_unregistered_treatments(tmp_path):
    unit = "T0_MATCHED_SUPERVISED_ONLY__seed_42"
    token = _mint(tmp_path, {unit: ""})
    with pytest.raises(sealed_target.TargetAuthorizationError):
        token.require_registered("T9_UNREGISTERED", "T9_UNREGISTERED__seed_42")


def test_evaluator_rejects_changed_checkpoints(tmp_path):
    import torch

    unit = "T0_MATCHED_SUPERVISED_ONLY__seed_42"
    token = _mint(tmp_path, {unit: ""})
    torch.save({"unit_id": unit, "tampered": True}, tmp_path / "checkpoints" / f"{unit}.pt")
    with pytest.raises(sealed_target.TargetAuthorizationError):
        token.revalidate()


def test_prediction_closure_rejects_incomplete_and_extra_units(tmp_path):
    unit = "T0_MATCHED_SUPERVISED_ONLY__seed_42"
    token = _mint(tmp_path, {unit: ""})
    token.release_features(purpose="test")
    with pytest.raises(sealed_target.TargetAuthorizationError):
        token.close_predictions({})
    with pytest.raises(sealed_target.TargetAuthorizationError):
        token.close_predictions({unit: "d" * 64, "T0_MATCHED_SUPERVISED_ONLY__seed_99": "e" * 64})


def test_sealed_target_module_computes_no_metric():
    """The prediction path must not be able to score itself."""

    source = inspect.getsource(sealed_target)
    for forbidden in ("macro_f1", "accuracy_score", "f1_score", "classification_metrics"):
        assert forbidden not in source


# ------------------------------------------------------------------ write containment


def test_writes_outside_the_branch_are_refused(tmp_path):
    with pytest.raises(paths.BranchWriteViolation):
        paths.ensure_branch_output(tmp_path / "escape.json")
    with pytest.raises(paths.BranchWriteViolation):
        paths.ensure_branch_output(paths.PROJECT_ROOT / "README.md")
    with pytest.raises(paths.BranchWriteViolation):
        paths.ensure_branch_output(paths.PROJECT_ROOT / "outputs" / "strict_dg" / "x.json")
    assert paths.ensure_branch_output(paths.BRANCH_OUTPUT_ROOT / "ok.json")


# ------------------------------------------------------------------------ config gate


def test_configuration_rejects_target_and_data1_label_use():
    config = config_validation.load("branch.json")
    broken = json.loads(json.dumps(config))
    broken["target_policy"]["target_training"] = True
    with pytest.raises(config_validation.ConfigurationViolation):
        config_validation.validate(broken)
    broken = json.loads(json.dumps(config))
    broken["data1_policy"]["labels_for_supervised_pretraining"] = True
    with pytest.raises(config_validation.ConfigurationViolation):
        config_validation.validate(broken)
    broken = json.loads(json.dumps(config))
    broken["training_domains"].append("P4")
    with pytest.raises(config_validation.ConfigurationViolation):
        config_validation.validate(broken)


def test_configuration_seed_and_fold_sets_are_frozen():
    config = config_validation.load("branch.json")
    assert tuple(config["seeds"]) == treatments.SEEDS
    assert tuple(config["folds"]) == treatments.FOLDS


# -------------------------------------------------------- executed-evidence integrity


def test_every_treatment_cell_appears_exactly_once():
    preregistration = _load("04_frozen_final_configs", "11_FINAL_P4_PREREGISTRATION.json")
    units = [entry["unit_id"] for entry in preregistration["evaluation_units"]]
    assert len(units) == len(set(units))
    expected = {
        f"{treatment_id}__seed_{seed}"
        for treatment_id in preregistration["registered_treatments"]
        for seed in preregistration["seeds"]
    }
    assert set(units) == expected


def test_source_side_sweeps_have_no_missing_or_duplicate_units():
    directory = OUTPUT / "03_source_only_treatment_selection"
    if not directory.is_dir():
        pytest.skip("no treatment sweeps yet")
    found = False
    for path in sorted(directory.glob("T*.json")):
        if path.name == "T5_GATE_DECISION.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        cells = [(row["fold_id"], row["seed"]) for row in payload["units"]]
        assert len(cells) == len(set(cells)) == 15, path.name
        assert {cell[0] for cell in cells} == set(treatments.FOLDS)
        assert {cell[1] for cell in cells} == set(treatments.SEEDS)
        found = True
    if not found:
        pytest.skip("no treatment sweeps yet")


def test_preregistration_digest_matches_its_sidecar():
    path = OUTPUT / "04_frozen_final_configs" / "11_FINAL_P4_PREREGISTRATION.json"
    if not path.is_file():
        pytest.skip("Gate D has not run")
    declared = (path.parent / "11_FINAL_P4_PREREGISTRATION.json.sha256").read_text(encoding="ascii").split()[0]
    assert _sha(path) == declared


def test_frozen_checkpoints_match_the_preregistration():
    preregistration = _load("04_frozen_final_configs", "11_FINAL_P4_PREREGISTRATION.json")
    directory = OUTPUT / "05_frozen_checkpoints"
    for entry in preregistration["evaluation_units"]:
        path = directory / f"{entry['unit_id']}.pt"
        assert path.is_file()
        assert _sha(path) == entry["checkpoint_sha256"]


def test_gate_a_report_shows_zero_target_rows_and_zero_overlap():
    report = _load("00_discovery", "GATE_A_REPORT.json")["metrics"]
    assert report["target_rows_in_training_manifests"] == 0
    assert report["cross_root_exact_signal_overlap"]["exact_signal_matches"] == 0
    assert report["all_folds_disjoint"] is True
    assert report["source_domains_are_P1_P2_P3_only"] is True
