from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from crfid.baseline.aggregation import aggregate_seed_results, compare_with_authoritative
from crfid.baseline.artifacts import read_json, write_csv, write_json
from crfid.baseline.evaluation import (
    evaluate_predictions,
    load_predictions,
    save_predictions,
)
from crfid.baseline.model import (
    build_pre_dg_model,
    expected_parameter_count,
    verify_model_contract,
)
from crfid.baseline.preprocessing import (
    TrainOnlyChannelScaler,
    native_physical_two_channel,
)
from crfid.data.pre_dg_registry import parse_dataset_registry, parse_split_registry
from crfid.governance.pre_dg_integrity import canonical_json_sha256, sha256_file
from crfid.protocols.pre_dg import build_protocol, protocol_sha256
from crfid.workflows.pre_dg import _execution_shard


def _dataset_registry() -> dict:
    return {
        "datasets": {
            name: {
                "row_count": 4,
                "input_channels": 2,
                "input_length": 512,
                "class_order": order,
                "class_count": len(order),
                "label_column": "label__tag_id",
                "condition_columns": ["domain__position"],
                "group_column": "raw_sample_group_id",
                "x_sha256": "a",
                "metadata_sha256": "b",
                "manifest_sha256": "c",
                "parameter_count": expected_parameter_count(len(order)),
            }
            for name, order in (
                ("paper3_tyndall", list(range(8))),
                ("paper4_depolarizing", list(range(1, 8))),
            )
        }
    }


def test_execution_shards_keep_whole_splits_disjoint() -> None:
    splits = {f"split_{index}": object() for index in range(5)}
    first = _execution_shard(splits, "0/2")
    second = _execution_shard(splits, "1/2")
    assert not set(first).intersection(second)
    assert set(first).union(second) == set(splits)


def test_pre_dg_registry_parsing() -> None:
    parsed = parse_dataset_registry(_dataset_registry())
    assert tuple(parsed) == ("paper3_tyndall", "paper4_depolarizing")
    assert parsed["paper3_tyndall"].class_count == 8


def test_pre_dg_registry_rejects_wrong_order() -> None:
    payload = _dataset_registry()
    payload["datasets"] = dict(reversed(payload["datasets"].items()))
    with pytest.raises(ValueError, match="order"):
        parse_dataset_registry(payload)


def test_split_registry_requires_complete_authority() -> None:
    with pytest.raises(ValueError, match="18"):
        parse_split_registry(
            {
                "splits": [
                    {
                        "dataset_id": "paper3_tyndall",
                        "split_name": "one",
                        "file_name": "one.csv",
                        "file_sha256": "a",
                        "metadata_sha256": "b",
                        "counts": {"train": 1, "validation": 1, "test": 1},
                    }
                ]
            }
        )


@pytest.mark.parametrize("partition", ["validation", "test"])
def test_validation_and_test_cannot_fit_preprocessing(partition: str) -> None:
    with pytest.raises(PermissionError, match="train only"):
        TrainOnlyChannelScaler.fit(
            np.zeros((4, 2, 512), dtype=np.float32), partition=partition
        )


def test_preprocessing_fit_transform_and_zero_variance() -> None:
    values = np.zeros((4, 2, 512), dtype=np.float32)
    values[:, 0] = np.arange(4, dtype=np.float32)[:, None]
    scaler = TrainOnlyChannelScaler.fit(values, partition="train")
    transformed = scaler.transform(values)
    assert transformed.dtype == np.float32
    assert np.isfinite(transformed).all()
    assert np.array_equal(transformed[:, 1], np.zeros((4, 512), dtype=np.float32))


def test_preprocessing_serialization_is_exact(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(5, 2, 512)).astype(np.float32)
    scaler = TrainOnlyChannelScaler.fit(values, partition="train")
    path = tmp_path / "state.npz"
    digest = scaler.save(path)
    loaded = TrainOnlyChannelScaler.load(path)
    assert digest == sha256_file(path)
    assert np.array_equal(scaler.transform(values), loaded.transform(values))


def test_preprocessing_preserves_sample_order() -> None:
    values = np.zeros((3, 2, 512), dtype=np.float32)
    values[:, 0, 0] = [1, 2, 3]
    scaler = TrainOnlyChannelScaler.fit(values, partition="train")
    transformed = scaler.transform(values)
    assert np.all(np.diff(transformed[:, 0, 0]) > 0)


def test_native_physical_representation_uses_zero_compatibility_channel() -> None:
    output = native_physical_two_channel(
        np.asarray([1, 3, 5], dtype=np.float32),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        512,
    )
    assert output.shape == (2, 512)
    assert np.count_nonzero(output[1]) == 0
    assert output[0, 0] == 1 and output[0, -1] == 5


@pytest.mark.parametrize("class_count", [7, 8])
def test_model_shape_and_parameter_count(class_count: int) -> None:
    import torch

    model = build_pre_dg_model(class_count, 0.3)
    contract = verify_model_contract(model, class_count)
    assert contract["parameter_count"] == expected_parameter_count(class_count)
    assert model(torch.zeros((2, 2, 512))).shape == (2, class_count)


def test_model_rejects_non_authoritative_dropout() -> None:
    with pytest.raises(ValueError, match="dropout"):
        build_pre_dg_model(7, 0.1)


def test_prediction_serialization_and_metrics(tmp_path: Path) -> None:
    truth = np.asarray([0, 1, 1, 0])
    prediction = np.asarray([0, 1, 0, 0])
    logits = np.eye(2, dtype=np.float32)[prediction]
    path = tmp_path / "prediction.npz"
    digest = save_predictions(path, ("a", "b", "c", "d"), truth, prediction, logits)
    loaded = load_predictions(path)
    metrics = evaluate_predictions(loaded["truth"], loaded["prediction"], class_count=2)
    assert digest == sha256_file(path)
    assert metrics["accuracy"] == 0.75
    assert 0.0 < metrics["macro_f1"] < 1.0


def test_artifact_json_csv_hashing(tmp_path: Path) -> None:
    json_path = tmp_path / "item.json"
    csv_path = tmp_path / "item.csv"
    assert write_json(json_path, {"b": 2, "a": 1}) == sha256_file(json_path)
    assert read_json(json_path) == {"a": 1, "b": 2}
    assert write_csv(csv_path, [{"a": 1, "b": 2}]) == sha256_file(csv_path)


def test_seed_aggregation_and_comparison_pass() -> None:
    rows = [
        {
            "dataset_id": "paper3_tyndall",
            "split_name": "paper3_grouped_random_raw_condition",
            "seed": seed,
            "accuracy": accuracy,
            "macro_f1": accuracy - 0.01,
            "selected_dropout": 0.3,
        }
        for seed, accuracy in ((42, 0.9), (43, 0.92), (44, 0.91))
    ]
    aggregate = aggregate_seed_results(rows)
    references = [
        {
            "dataset_id": row["dataset_id"],
            "split_name": row["split_name"],
            "seed": row["seed"],
            "accuracy": row["accuracy"],
            "selected_dropout": 0.3,
        }
        for row in rows
    ]
    policy = {
        "per_run_accuracy_absolute_tolerance": 0.01,
        "per_split_mean_accuracy_absolute_tolerance": 0.01,
        "grouped_random_mean_accuracy_absolute_tolerance": 0.01,
    }
    _, comparison = compare_with_authoritative(rows, aggregate, references, policy)
    assert comparison["numerical_tolerance_passed"]


def test_comparison_detects_mismatch() -> None:
    rows = [
        {
            "dataset_id": "paper3_tyndall",
            "split_name": "paper3_grouped_random_raw_condition",
            "seed": 42,
            "accuracy": 0.5,
            "macro_f1": 0.5,
            "selected_dropout": 0.0,
        }
    ]
    aggregate = aggregate_seed_results(rows)
    reference = [{**rows[0], "accuracy": 0.9, "selected_dropout": 0.3}]
    policy = {
        "per_run_accuracy_absolute_tolerance": 0.01,
        "per_split_mean_accuracy_absolute_tolerance": 0.01,
        "grouped_random_mean_accuracy_absolute_tolerance": 0.01,
    }
    _, comparison = compare_with_authoritative(rows, aggregate, reference, policy)
    assert not comparison["numerical_tolerance_passed"]


def test_protocol_hash_detects_alteration() -> None:
    protocol = {"a": 1, "b": [2, 3]}
    digest = canonical_json_sha256(protocol)
    assert digest != canonical_json_sha256({"a": 2, "b": [2, 3]})


def test_build_protocol_has_all_explicit_stages() -> None:
    canonical = {
        "scientific_name": "x",
        "historical_verdict": "ready",
        "random_seeds": [42, 43, 44],
        "claim_boundary": "bounded",
    }
    data = {"datasets": {"paper3_tyndall": {}, "paper4_depolarizing": {}}}
    split = {"splits": [{"split_name": f"s{i}"} for i in range(18)]}
    model = {"preprocessing": {}, "model": {}, "training": {}}
    evaluation = {
        "metrics": ["Accuracy"],
        "aggregation": "mean",
        "comparison_policy": {},
    }
    protocol = build_protocol(canonical, data, split, model, evaluation)
    assert len(protocol["stage_order"]) == 15
    assert protocol_sha256(protocol)
