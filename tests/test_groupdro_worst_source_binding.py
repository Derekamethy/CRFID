from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from crfid.groupdro_worst_source import core
from crfid.groupdro_worst_source.core import GroupDROProtocolError, P4LabelSeal
from crfid.groupdro_worst_source.runtime import (
    RuntimePaths,
    execution_binding_evidence,
    governed_source_input_hashes,
)


def _write_governed_raw_p4(root) -> None:
    for surface in (1, 2, 3):
        path = root / f"A{surface}_P4.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(core.P4_RAW_HEADER)
            for block in range(21):
                tag_id = block // 3 + 1
                er = block % 3
                for _ in range(50):
                    writer.writerow(
                        [
                            int(surface == 3),
                            int(surface == 2),
                            int(surface == 1),
                            1,
                            0,
                            0,
                            0,
                            er,
                            tag_id,
                            *([0.0] * 281),
                        ]
                    )


def test_raw_p4_csv_seal_uses_only_features_before_prediction_freeze(tmp_path, monkeypatch) -> None:
    _write_governed_raw_p4(tmp_path)
    monkeypatch.setattr(
        core,
        "P4_RAW_FILE_HASHES",
        {name: core.sha256_file(tmp_path / name) for name in core.P4_RAW_FILENAMES},
    )
    seal = P4LabelSeal(tmp_path)
    with pytest.raises(GroupDROProtocolError, match="FAIL_P4_LABEL_BOUNDARY"):
        seal.open_labels()
    data = seal.load_unlabelled()
    assert data.signals.shape == (3150, 281)
    assert len(set(data.condition_ids)) == 63
    assert seal.access_log[-1]["tagid_column_accessed"] is False
    assert seal.access_log[-1]["er_column_accessed"] is False
    seal.freeze_predictions({"erm_seed_42": np.zeros(3150, dtype=np.int64)}, tmp_path / "frozen.npz")
    labels = seal.open_labels()
    assert labels.shape == (3150,)
    assert set(labels.tolist()) == set(range(7))
    assert seal.access_log[-1]["tagid_column_accessed"] is True
    assert seal.access_log[-1]["er_column_accessed"] is True


def test_binding_evidence_sanitizes_governed_paths_and_requires_source_custody(tmp_path) -> None:
    with pytest.raises(GroupDROProtocolError, match="BLOCKED_GOVERNED_INPUTS"):
        governed_source_input_hashes(tmp_path / "absent")
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    evidence = execution_binding_evidence(
        paths=RuntimePaths(
            strict_artifact_root=tmp_path / "strict",
            p4_governed_root=tmp_path / "p4",
            runtime_root=runtime_root,
        ),
        source_hashes={"source_signals_float64.npy": "a" * 64},
        p4_preflight={
            "input_mode": "GOVERNED_RAW_CSV",
            "p4_file_hashes": {"A1_P4.csv": "b" * 64},
        },
    )
    public_text = json.dumps(evidence, sort_keys=True)
    assert str(tmp_path) not in public_text
    assert "<GOVERNED_STRICT_DG_RUNTIME_ROOT>" in public_text
    assert "<LOCKED_CRIFD_PYTHON>" in public_text
    receipt = json.loads((runtime_root / "EXECUTION_BINDING_LOCAL_RECEIPT.json").read_text())
    assert receipt["strict_artifact_root"] == str((tmp_path / "strict").resolve())
