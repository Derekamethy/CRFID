from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crfid.data.pre_dg_schema import PreDGSplitSchema
from crfid.data.pre_dg_splits import reconstruct_authoritative_split
from crfid.governance.pre_dg_access import (
    require_explicit_location,
    validate_pre_dg_dataset_access,
)
from crfid.governance.pre_dg_claims import validate_pre_dg_claim
from crfid.governance.pre_dg_integrity import sha256_file


def _split_fixture(tmp_path: Path) -> tuple[Path, Path, pd.DataFrame, PreDGSplitSchema]:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "dataset_id": ["paper3_tyndall"] * 4,
            "split": ["train", "train", "val", "test"],
            "raw_sample_group_id": ["g1", "g2", "g3", "g4"],
        }
    )
    split_path = tmp_path / "split.csv"
    metadata_path = tmp_path / "split.json"
    frame.to_csv(split_path, index=False)
    metadata_path.write_text(json.dumps({"passed": True}), encoding="utf-8")
    metadata = frame.drop(columns=["split", "dataset_id"]).copy()
    schema = PreDGSplitSchema(
        dataset_id="paper3_tyndall",
        split_name="split",
        file_name="split.csv",
        file_sha256=sha256_file(split_path),
        metadata_sha256=sha256_file(metadata_path),
        train_count=2,
        validation_count=1,
        test_count=1,
    )
    return split_path, metadata_path, metadata, schema


def test_authoritative_split_reconstruction(tmp_path: Path) -> None:
    split_path, metadata_path, metadata, schema = _split_fixture(tmp_path)
    result = reconstruct_authoritative_split(
        split_path, metadata_path, schema, metadata, "raw_sample_group_id"
    )
    assert result.integrity["passed"]
    assert result.indices["train"].tolist() == [0, 1]


def test_altered_split_manifest_is_rejected(tmp_path: Path) -> None:
    split_path, metadata_path, metadata, schema = _split_fixture(tmp_path)
    split_path.write_text(split_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        reconstruct_authoritative_split(
            split_path, metadata_path, schema, metadata, "raw_sample_group_id"
        )


def test_split_group_leakage_is_rejected(tmp_path: Path) -> None:
    split_path, metadata_path, metadata, schema = _split_fixture(tmp_path)
    frame = pd.read_csv(split_path)
    frame.loc[3, "raw_sample_group_id"] = "g1"
    frame.to_csv(split_path, index=False)
    schema = PreDGSplitSchema(
        **{**schema.__dict__, "file_sha256": sha256_file(split_path)}
    )
    with pytest.raises(ValueError, match="leakage"):
        reconstruct_authoritative_split(
            split_path, metadata_path, schema, metadata, "raw_sample_group_id"
        )


def test_unrecognized_dataset_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("sample_id\nx\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="not authorized"):
        validate_pre_dg_dataset_access("unknown", tmp_path, manifest)


@pytest.mark.parametrize(
    "role", ["strict_dg", "target_assisted", "external_data1"]
)
def test_non_pre_dg_artifact_roles_are_rejected(tmp_path: Path, role: str) -> None:
    root = tmp_path / role
    root.mkdir()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("sample_id\nx\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="Rejected"):
        validate_pre_dg_dataset_access("paper3_tyndall", root, manifest)


def test_undocumented_local_fallback_is_rejected() -> None:
    with pytest.raises(ValueError, match="explicit"):
        require_explicit_location("data/guess", "paper3_processed_root")


def test_unresolved_environment_location_is_rejected() -> None:
    with pytest.raises(ValueError, match="required"):
        require_explicit_location("${MISSING}", "paper3_processed_root")


def test_pre_dg_claim_guard_rejects_strict_relabeling() -> None:
    with pytest.raises(ValueError, match="Prohibited"):
        validate_pre_dg_claim("This is strict DG.")


def test_pre_dg_claim_guard_accepts_non_strict_description() -> None:
    validate_pre_dg_claim("Historical non-strict single-dataset learnability baseline.")
