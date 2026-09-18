from __future__ import annotations

import csv
import json
import os
from functools import lru_cache
from pathlib import Path

import pytest

from crfid.external_data1.loader import load_authoritative_data1


def _roots() -> tuple[Path, Path]:
    raw_value = os.environ.get("CRFID_EXTERNAL_DATA1_ROOT")
    output_value = os.environ.get("CRFID_OUTPUT_ROOT")
    if not raw_value or not output_value:
        pytest.skip("Real Data1 roots are explicitly supplied only for custody runs")
    return Path(raw_value), Path(output_value) / "external_data1"


def _authority() -> dict[str, object]:
    path = Path("configs/external_data1/data_authority.yaml")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _loaded_data1() -> object:
    raw_root, _ = _roots()
    authority = _authority()
    return load_authoritative_data1(
        raw_root,
        expected_file_sha256=authority["expected_file_sha256"],
        expected_aggregate_sha256=authority["raw_aggregate_sha256"],
    )


def test_real_raw_manifest_and_dataset_identity() -> None:
    authority = _authority()
    loaded = _loaded_data1()
    assert len(loaded.sets) == 9
    assert sum(len(item.labels) for item in loaded.sets) == 12_250
    assert loaded.raw_aggregate_sha256 == authority["raw_aggregate_sha256"]


def test_real_frozen_split_counts_and_disjointness() -> None:
    loaded = _loaded_data1()
    assert loaded.train.sample_count == 5_600
    assert loaded.test_features.sample_count == 1_850
    assert set(loaded.train.sample_ids).isdisjoint(loaded.test_features.sample_ids)
    assert set(loaded.train.signal_sha256).isdisjoint(
        loaded.test_features.signal_sha256
    )


def test_real_ev3r_all_runs_and_predictions_reproduced() -> None:
    _, output_root = _roots()
    execution = list(
        csv.DictReader(
            (output_root / "ev3r/execution_manifest.csv").open(
                "r", encoding="utf-8", newline=""
            )
        )
    )
    comparison = list(
        csv.DictReader(
            (output_root / "comparison/prediction_comparison.csv").open(
                "r", encoding="utf-8", newline=""
            )
        )
    )
    assert len(execution) == 12
    assert len(comparison) == 12
    assert all(row["exact_prediction_match"].lower() == "true" for row in comparison)
    assert all(int(row["mismatch_count"]) == 0 for row in comparison)


def test_real_ev3r_metrics_and_historical_aggregate_reproduced() -> None:
    _, output_root = _roots()
    metric_rows = list(
        csv.DictReader(
            (output_root / "comparison/metric_comparison.csv").open(
                "r", encoding="utf-8", newline=""
            )
        )
    )
    historical = json.loads(
        (
            output_root / "comparison/historical_aggregate_verification.json"
        ).read_text(encoding="utf-8")
    )
    assert len(metric_rows) == 12
    assert all(row["tolerance_match"].lower() == "true" for row in metric_rows)
    assert historical["exact_match"] is True
    assert (
        historical["recomputed_sha256"]
        == "886698bfbc5fca7d440842ee4e2778f6ae5549db6563233c6e232ad2ff3a3cd7"
    )


def test_real_ev4_independent_verification_passes() -> None:
    _, output_root = _roots()
    integrity = json.loads(
        (output_root / "ev4/EV4_INTEGRITY.json").read_text(encoding="utf-8")
    )
    assert integrity["verdict"] in {
        "PASS_EV4_INDEPENDENT_AUDIT",
        "PASS_EV4_WITH_REQUIRED_CAVEATS",
    }
    assert integrity["prediction_recomputation_pass"] is True
    assert integrity["metric_recomputation_pass"] is True
    assert integrity["historical_aggregate_exact"] is True
