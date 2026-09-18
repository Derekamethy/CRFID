from __future__ import annotations

from pathlib import Path

from crfid.groupdro_worst_source import runtime
from crfid.groupdro_worst_source.runtime import c1_lineage_binding, load_config


def test_c1_lineage_binding_is_canonical_and_hashes_required_inputs() -> None:
    binding = c1_lineage_binding(load_config())
    assert binding["canonical_candidate"] == "C1_FIRST_DIFFERENCE_ERM_1DCNN"
    assert binding["input_contract"] == {"ordered_points": 281, "first_difference_points": 280, "class_count": 7}
    assert len(binding["bound_files"]) >= 7
    assert all(len(row["sha256"]) == 64 for row in binding["bound_files"])


def test_dry_run_creates_required_blocked_outputs_without_p4_access(tmp_path: Path, monkeypatch) -> None:
    results_root = tmp_path / "results"
    monkeypatch.setattr(runtime, "RESULTS_ROOT", results_root)
    monkeypatch.setattr(runtime, "MANIFESTS_ROOT", tmp_path / "manifests")
    result = runtime.execute_benchmark(dry_run=True)
    assert result["status"] == "BLOCKED_GOVERNED_INPUTS"
    required = [
        "00_EXECUTIVE_SUMMARY.md", "02_C1_LINEAGE_BINDING.json",
        "03_PREREGISTERED_PROTOCOL.md", "04_DATA_AND_GROUP_STRUCTURE_AUDIT.csv", "05_SOURCE_LOPO_SPLIT_MANIFEST.csv",
        "06_GROUPDRO_IMPLEMENTATION_VALIDITY.csv", "07_ETA_GRID_REGISTER.csv", "08_LEAKAGE_AND_LABEL_BOUNDARY_GATES.csv",
        "09_DEVELOPMENT_RUN_REGISTER.csv", "10_SOURCE_HELD_POSITION_METRICS.csv", "11_GROUP_LOSS_AND_Q_TRAJECTORIES.csv",
        "12_SOURCE_WORST_GROUP_CONTRASTS.csv", "13_SOURCE_MEAN_AND_VARIANCE_RESULTS.csv", "14_POSITION_AND_TAGID_PROBE_RESULTS.csv",
        "15_SOURCE_ONLY_ETA_SELECTION.md", "16_FINAL_MODEL_REGISTER.csv", "17_P4_PREDICTION_FREEZE_AND_ACCESS_LOG.csv",
        "18_P4_BLOCK_METRICS.csv", "19_P4_PRIMARY_GROUPDRO_VS_ERM_CONTRAST.csv", "20_P4_PER_CLASS_RESULTS.csv",
        "21_PREDICTED_CLASS_HISTOGRAMS.csv", "22_BOOTSTRAP_AND_SEED_SENSITIVITY.csv", "23_SCIENTIFIC_INTERPRETATION.md",
        "24_LIMITATIONS_AND_NONCLAIMS.md", "STATUS.md",
        "27_RUNTIME_BINDING.md", "28_EXECUTION_STATE.json",
    ]
    assert all((results_root / name).is_file() for name in required)
    assert len((results_root / "09_DEVELOPMENT_RUN_REGISTER.csv").read_text(encoding="utf-8").splitlines()) == 76
    assert len((results_root / "16_FINAL_MODEL_REGISTER.csv").read_text(encoding="utf-8").splitlines()) == 11
    assert "P4_LABEL_SEAL_INITIALIZED" in (results_root / "17_P4_PREDICTION_FREEZE_AND_ACCESS_LOG.csv").read_text(encoding="utf-8")
