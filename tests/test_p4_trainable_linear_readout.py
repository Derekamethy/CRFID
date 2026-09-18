from __future__ import annotations

import csv
import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workflows/11_p4_factor_aware_few_shot"))
sys.path.insert(0, str(ROOT / "workflows/12_p4_large_calibration_curve"))
sys.path.insert(0, str(ROOT / "workflows/13_p4_trainable_linear_readout"))

from p4_factor_aware.protocol import ProtocolViolation
from p4_linear_readout.constants import (
    PARENT_ARTIFACT_HASHES,
    PARENT_PREDICTION_ARRAY_SHA256,
    PARENT_RESULTS_RELATIVE,
    POSITIVE_BUDGETS,
    PREREGISTRATION_SHA256,
    SELECTED_LAMBDA,
    SOURCE_SEEDS,
    SUPPORT_SEEDS,
    TOTAL_BUDGETS,
)
from p4_linear_readout.linear import fit_linear_readout, linear_predict
from p4_linear_readout.study import _verify_parent_artifacts, array_sha256, sha256_file


def synthetic_support(rows: int = 21) -> tuple[torch.Tensor, np.ndarray, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260809)
    labels = np.arange(rows, dtype=np.int64) % 7
    embeddings = torch.randn((rows, 256), generator=generator)
    embeddings[:, :7] += torch.eye(7)[torch.from_numpy(labels)] * 2.0
    anchor_weight = torch.randn((7, 256), generator=generator) * 0.01
    anchor_bias = torch.zeros(7)
    return embeddings, labels, anchor_weight, anchor_bias


def test_preregistration_hash_is_sealed() -> None:
    path = ROOT / "configs/p4_trainable_linear_readout/preregistration.json"
    assert sha256_file(path) == PREREGISTRATION_SHA256


def test_parent_scientific_artifacts_are_hash_bound() -> None:
    binding = _verify_parent_artifacts(ROOT)
    rows = binding["artifact_rows"]
    assert len(rows) == len(PARENT_ARTIFACT_HASHES)
    assert {row["hash_mode"] for row in rows} == {"RAW_FILE", "PATH_NORMALIZED_JSON"}


def test_parent_prediction_axes_and_array_hash() -> None:
    with np.load(ROOT / PARENT_RESULTS_RELATIVE / "per_run_predictions.npz", allow_pickle=False) as payload:
        assert tuple(payload["budgets"].tolist()) == TOTAL_BUDGETS
        assert tuple(payload["source_seeds"].tolist()) == SOURCE_SEEDS
        assert tuple(payload["support_seeds"].tolist()) == SUPPORT_SEEDS
        assert array_sha256(payload["predictions"]) == PARENT_PREDICTION_ARRAY_SHA256


def test_support_manifest_has_exact_plan_and_rank_counts() -> None:
    path = ROOT / PARENT_RESULTS_RELATIVE / "07_SUPPORT_SELECTION_MANIFEST.csv"
    counts: dict[tuple[int, int], list[int]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            counts.setdefault((int(row["fold"]), int(row["support_seed"])), []).append(int(row["selection_rank"]))
    assert len(counts) == 60
    assert all(sorted(ranks) == list(range(1, 501)) for ranks in counts.values())


def test_linear_fitter_has_no_query_parameter() -> None:
    assert not any(name.startswith("query") for name in inspect.signature(fit_linear_readout).parameters)


def test_linear_fit_is_bitwise_deterministic() -> None:
    embeddings, labels, anchor_weight, anchor_bias = synthetic_support()
    first = fit_linear_readout(
        support_embeddings=embeddings,
        support_labels=labels,
        anchor_weight=anchor_weight,
        anchor_bias=anchor_bias,
    )
    second = fit_linear_readout(
        support_embeddings=embeddings,
        support_labels=labels,
        anchor_weight=anchor_weight,
        anchor_bias=anchor_bias,
    )
    assert first.diagnostics["adapted_head_sha256"] == second.diagnostics["adapted_head_sha256"]
    assert np.array_equal(
        linear_predict(embeddings, first.weight, first.bias),
        linear_predict(embeddings, second.weight, second.bias),
    )


def test_linear_fit_does_not_mutate_inputs_or_anchor() -> None:
    embeddings, labels, anchor_weight, anchor_bias = synthetic_support()
    before = tuple(value.clone() for value in (embeddings, anchor_weight, anchor_bias))
    fit_linear_readout(
        support_embeddings=embeddings,
        support_labels=labels,
        anchor_weight=anchor_weight,
        anchor_bias=anchor_bias,
    )
    for observed, expected in zip((embeddings, anchor_weight, anchor_bias), before, strict=True):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_linear_fit_requires_every_class() -> None:
    embeddings, labels, anchor_weight, anchor_bias = synthetic_support()
    labels[labels == 6] = 5
    with pytest.raises(ProtocolViolation, match="CLASS_COUNTS"):
        fit_linear_readout(
            support_embeddings=embeddings,
            support_labels=labels,
            anchor_weight=anchor_weight,
            anchor_bias=anchor_bias,
        )


def test_linear_fit_accepts_near_balanced_budget_ten() -> None:
    embeddings, labels, anchor_weight, anchor_bias = synthetic_support(10)
    result = fit_linear_readout(
        support_embeddings=embeddings,
        support_labels=labels,
        anchor_weight=anchor_weight,
        anchor_bias=anchor_bias,
    )
    assert result.diagnostics["class_count_max"] - result.diagnostics["class_count_min"] == 1
    assert result.diagnostics["optimizer_completed_finite"]


def test_selected_lambda_is_single_and_positive() -> None:
    assert SELECTED_LAMBDA == 100.0
    assert all(budget > 0 for budget in POSITIVE_BUDGETS)


def test_prediction_shape_and_dtype() -> None:
    embeddings, labels, anchor_weight, anchor_bias = synthetic_support()
    result = fit_linear_readout(
        support_embeddings=embeddings,
        support_labels=labels,
        anchor_weight=anchor_weight,
        anchor_bias=anchor_bias,
    )
    prediction = linear_predict(embeddings, result.weight, result.bias)
    assert prediction.shape == (21,)
    assert prediction.dtype == np.int64
    assert set(prediction.tolist()) <= set(range(7))


def test_array_hash_binds_dtype_shape_and_bytes() -> None:
    values = np.arange(12, dtype=np.int8).reshape(3, 4)
    assert array_sha256(values) != array_sha256(values.astype(np.int16))
    assert array_sha256(values) != array_sha256(values.reshape(4, 3))
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(str(tuple(values.shape)).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    assert array_sha256(values) == digest.hexdigest()
