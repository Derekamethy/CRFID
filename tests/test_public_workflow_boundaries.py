from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from crfid.config import load_config
from crfid.workflows.strict_dg import PUBLIC_EVIDENCE, verify_public_evidence
from crfid import angle_distance_factorial_contrast as factorial

ROOT = Path(__file__).resolve().parents[1]


def test_public_strict_evidence_verifies_without_measurements():
    result = verify_public_evidence(PUBLIC_EVIDENCE, load_config(ROOT / "configs/strict_dg/canonical.yaml"))
    assert result["status"] == "PUBLIC_COMPACT_EVIDENCE_VERIFIED"
    assert result["checkpoint_count"] == 5
    assert result["measurements_opened"] is False
    assert result["historical_replay"] == "HISTORICAL_FULL_RELEASE_REPLAY_NOT_DISTRIBUTED"


@pytest.mark.parametrize("damage", ["metric", "checkpoint", "preprocessing", "source_ranking"])
def test_public_strict_verifier_rejects_changed_evidence(tmp_path, damage):
    root = tmp_path / "evidence"
    shutil.copytree(PUBLIC_EVIDENCE, root)
    if damage == "metric":
        path = root / "FINAL_P4_RESULTS.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["seeds"][0]["macro_f1"] = "0.99"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif damage == "checkpoint":
        path = next((root / "checkpoints").glob("*42*"))
        path.write_bytes(path.read_bytes() + b"changed")
    elif damage == "preprocessing":
        path = root / "preprocessing/mean_float64.npy"
        values = np.load(path, allow_pickle=False)
        values[0] += 1
        np.save(path, values, allow_pickle=False)
    else:
        path = root / "candidate_aggregate_metrics.csv"
        path.write_text(path.read_text(encoding="utf-8").replace("0.12386044528827292", "0.02386044528827292"), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence mismatch"):
        verify_public_evidence(root, load_config(ROOT / "configs/strict_dg/canonical.yaml"))


def test_factorial_missing_measurements_precedes_training_and_writes(tmp_path, monkeypatch):
    def unexpected(**kwargs):
        pytest.fail("Training must not start without governed inputs")
    monkeypatch.setattr(factorial, "execute_case_b_synchronized_rerun", unexpected)
    output = tmp_path / "output"
    with pytest.raises(FileNotFoundError, match="GOVERNED_MEASUREMENTS_REQUIRED"):
        factorial.run_factorial_analysis(raw_root=tmp_path, output_root=output)
    assert not output.exists()


def test_factorial_preflight_checks_scientific_identity_without_git(tmp_path, monkeypatch):
    binding = json.loads((factorial.DEFAULT_RESULTS / "02_FIXED_POSITION_BINDING.json").read_text(encoding="utf-8"))
    fingerprints = binding["synchronized_checkpoint_inference"]["source_data_fingerprints"]
    for position in factorial.POSITIONS:
        for surface in ("A1", "A2", "A3"):
            (tmp_path / f"{surface}_{position}.csv").write_text("synthetic loader fixture", encoding="utf-8")
    monkeypatch.setattr(factorial, "load_governed_position", lambda root, position: SimpleNamespace(source_data_fingerprint=fingerprints[position]))
    output = tmp_path / "output"
    assert factorial.preflight_factorial(raw_root=tmp_path, output_root=output)["run_count"] == 60
    assert not output.exists()
    monkeypatch.setattr(factorial, "load_governed_position", lambda root, position: SimpleNamespace(source_data_fingerprint="wrong"))
    with pytest.raises(ValueError, match="fingerprints differ"):
        factorial.preflight_factorial(raw_root=tmp_path, output_root=output)


def test_factorial_refuses_to_write_retained_evidence(tmp_path):
    with pytest.raises(ValueError, match="retained evidence is read-only"):
        factorial.preflight_factorial(raw_root=tmp_path, output_root=factorial.DEFAULT_RESULTS)


def _fine_tuning():
    for directory in ("11_p4_factor_aware_few_shot", "12_p4_large_calibration_curve", "13_p4_trainable_linear_readout", "14_p4_encoder_finetuning"):
        sys.path.insert(0, str(ROOT / "workflows" / directory))
    from p4_encoder_ft import study
    return study


def test_fine_tuning_missing_archive_precedes_target_reads_and_writes(tmp_path, monkeypatch):
    study = _fine_tuning()
    def unexpected(*args, **kwargs):
        pytest.fail("Target measurements must not open without parent bindings")
    monkeypatch.setattr(study, "load_governed_p4", unexpected)
    output = tmp_path / "output"
    with pytest.raises(study.ProtocolViolation, match="HISTORICAL_ARCHIVE_REQUIRED"):
        study.run_study(repository_root=ROOT, archive_repository=tmp_path, data_directory=tmp_path, output_directory=output, mode="dry-run")
    assert not output.exists()


def test_fine_tuning_dry_run_is_read_only_and_needs_no_git(tmp_path, monkeypatch):
    study = _fine_tuning()
    monkeypatch.setattr(study, "_executed_code_rows", lambda root: [])
    monkeypatch.setattr(study, "preflight", lambda **kwargs: (None, {}, [], None, None, {}, [], {}))
    output = tmp_path / "output"
    result = study.run_study(repository_root=tmp_path, archive_repository=tmp_path, data_directory=tmp_path, output_directory=output, mode="dry-run")
    assert result["status"] == "HISTORICAL_INPUTS_VERIFIED"
    assert not output.exists()
