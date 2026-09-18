from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from crfid.data.tyndall import map_raw_tag_label
from crfid.fixed_position_grouped_validity import (
    DEFAULT_CONFIG,
    EXPECTED_POSITIONS,
    FixedPositionData,
    build_partitions,
    generate_split_manifest,
    split_validation_rows,
)
from crfid.strict_runtime.neutral_model import initialize_model, parameter_count


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def _position(position: str, with_signals: bool = False) -> FixedPositionData:
    rows = []
    for surface in ("A1", "A2", "A3"):
        for er in range(3):
            for tag_id in range(1, 8):
                condition_id = f"cond_{position}_{surface}_{er}_{tag_id}"
                for repeat in range(50):
                    rows.append(
                        {
                            "registry_row": len(rows),
                            "sample_id": f"sample_{position}_{surface}_{er}_{tag_id}_{repeat:02d}",
                            "raw_condition_id": condition_id,
                            "repeat_id": f"{condition_id}_r{repeat:02d}",
                            "tag_id": tag_id,
                            "label_index": map_raw_tag_label(tag_id),
                            "er": er,
                            "surface": surface,
                            "position": position,
                            "exact_signal_sha256": f"signal_{len(rows)}",
                            "unique_signal_weight": 1.0,
                        }
                    )
    signals = (
        np.arange(3150 * 281, dtype=np.float64).reshape(3150, 281) / 1000.0
        if with_signals
        else np.empty((0, 281), dtype=np.float64)
    )
    labels = np.asarray([row["label_index"] for row in rows], dtype=np.int64)
    return FixedPositionData(position, rows, signals, labels, f"fingerprint_{position}", ())


def _compatibility() -> dict:
    return {
        position: {
            "canonical_loader": "synthetic",
            "sample_identity_equal": True,
            "signals_bitwise_equal": True,
            "labels_bitwise_equal": True,
            "condition_metadata_equal": True,
        }
        for position in EXPECTED_POSITIONS
    }


def test_exact_balanced_group_manifest_and_pretraining_gates() -> None:
    config = _config()
    data = {position: _position(position) for position in EXPECTED_POSITIONS}
    manifest = generate_split_manifest(data, config)
    assert len(manifest) == 4 * 3 * 5 * 63
    gates = split_validation_rows(manifest, generate_split_manifest(data, config), config, _compatibility())
    assert [row["status"] for row in gates] == ["PASS"] * 10

    run = [
        row
        for row in manifest
        if row["position"] == "P1" and row["fold"] == 1 and row["seed"] == 42
    ]
    assert Counter(row["split_assignment"] for row in run) == {
        "train": 28,
        "validation": 14,
        "test": 21,
    }
    for split, count in (("train", 4), ("validation", 2), ("test", 3)):
        assert set(
            Counter(row["TagID"] for row in run if row["split_assignment"] == split).values()
        ) == {count}
    assert {row["row_count"] for row in run} == {50}


def test_seed_changes_only_inner_membership() -> None:
    config = _config()
    data = {position: _position(position) for position in EXPECTED_POSITIONS}
    first = generate_split_manifest(data, config)
    second = generate_split_manifest(data, config)
    assert first == second
    for position in EXPECTED_POSITIONS:
        for fold in range(1, 4):
            by_seed = {}
            for seed in config["seeds"]:
                rows = [
                    row
                    for row in first
                    if row["position"] == position
                    and row["fold"] == fold
                    and row["seed"] == seed
                ]
                by_seed[seed] = {
                    split: {
                        row["condition_block_identifier"]
                        for row in rows
                        if row["split_assignment"] == split
                    }
                    for split in ("train", "validation", "test")
                }
            assert len({frozenset(value["test"]) for value in by_seed.values()}) == 1
            assert len({frozenset(value["validation"]) for value in by_seed.values()}) > 1
            for value in by_seed.values():
                assert value["train"].isdisjoint(value["validation"])
                assert value["train"].isdisjoint(value["test"])
                assert value["validation"].isdisjoint(value["test"])


def test_partition_adapter_is_group_disjoint_and_position_isolated() -> None:
    config = _config()
    data = _position("P2", with_signals=True)
    manifest = generate_split_manifest({"P2": data}, {**config, "positions": ["P2"]})
    run = [row for row in manifest if row["fold"] == 2 and row["seed"] == 44]
    partitions, state = build_partitions(data, run)
    assert {name: len(partition.labels) for name, partition in partitions.items()} == {
        "inner_train": 1400,
        "inner_validation": 700,
        "outer_development": 2100,
        "outer_held": 1050,
    }
    assert {partition.inputs.shape[1] for partition in partitions.values()} == {280}
    assert state["mode"] == "replace_with_one"
    groups = {name: set(partition.condition_ids) for name, partition in partitions.items()}
    assert groups["inner_train"].isdisjoint(groups["inner_validation"])
    assert groups["outer_development"].isdisjoint(groups["outer_held"])
    for partition in partitions.values():
        assert {data.rows[int(index)]["position"] for index in partition.registry_rows} == {"P2"}


def test_canonical_model_optimizer_and_label_declarations_are_unchanged() -> None:
    config = _config()
    strict = json.loads((ROOT / "configs" / "strict_dg" / "canonical.yaml").read_text(encoding="utf-8"))
    assert config["method_id"] == strict["selected_candidate"]
    assert config["optimizer"] == strict["optimizer"]
    assert config["training"]["batch_size"] == strict["training"]["batch_size"]
    assert config["training"]["maximum_inner_epochs"] == strict["training"]["maximum_inner_epochs"]
    assert config["training"]["early_stopping_patience"] == strict["training"]["early_stopping_patience"]
    assert config["training"]["minimum_improvement"] == strict["training"]["minimum_improvement"]
    assert [map_raw_tag_label(tag_id) for tag_id in range(1, 8)] == strict["class_order"]
    assert parameter_count(initialize_model(42)) == strict["model"]["parameter_count"]
