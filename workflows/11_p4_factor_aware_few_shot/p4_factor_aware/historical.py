"""Read-only binding and factor-dependence audit of the frozen Few-Shot release."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

from .constants import (
    CHECKPOINT_FILE_HASHES,
    CHECKPOINT_STATE_HASHES,
    HEAD_LAMBDAS,
    HISTORICAL_BINDING_HASHES,
    HISTORICAL_PAYLOAD_MANIFEST_SHA256,
    PREPROCESSING_FILES,
    cell_id,
)
from .protocol import ProtocolViolation
from .selection import factor_count_imbalance, pairwise_dispersion


HISTORICAL_TAG = "few-shot-v2-final-release-2026-07-29"
HISTORICAL_TAG_OBJECT = "b32531e32a38cbbea57563de0a962f01fa5cb019"
HISTORICAL_TAG_COMMIT = "40693e570d4a39c15246aa865a3aa05271b9c279"
HISTORICAL_TAG_TREE = "bda85656b06f50bcf5880cef769fa0625595ed09"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *arguments: str) -> str:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments],
        text=True,
        encoding="utf-8",
        env=environment,
    ).strip()


def verify_historical_binding(archive_repository: str | Path) -> dict[str, object]:
    repository = Path(archive_repository)
    frozen = repository / "outputs/few_shot/frozen_branch"
    manifest = repository / "configs/few_shot/frozen_payload_manifest.csv"
    if not frozen.is_dir() or not manifest.is_file():
        raise FileNotFoundError("BLOCKED_GOVERNED_P4_INPUTS")
    if sha256_file(manifest) != HISTORICAL_PAYLOAD_MANIFEST_SHA256:
        raise ProtocolViolation("HISTORICAL_FEW_SHOT_MANIFEST_MISMATCH")
    with manifest.open(encoding="utf-8", newline="") as handle:
        declared = {row["relative_path"]: row for row in csv.DictReader(handle)}
    bindings = []
    for relative_path, expected_hash in HISTORICAL_BINDING_HASHES.items():
        path = frozen / relative_path
        row = declared.get(relative_path)
        observed = sha256_file(path) if path.is_file() else ""
        passed = bool(
            row
            and observed == expected_hash
            and row["sha256"] == expected_hash
            and int(row["size_bytes"]) == path.stat().st_size
        )
        if not passed:
            raise ProtocolViolation(f"HISTORICAL_FEW_SHOT_BINDING_MISMATCH:{relative_path}")
        bindings.append(
            {
                "relative_path": relative_path,
                "sha256": observed,
                "size_bytes": path.stat().st_size,
                "payload_manifest_match": True,
            }
        )

    tag_object = _git(repository, "rev-parse", HISTORICAL_TAG)
    tag_commit = _git(repository, "rev-parse", f"{HISTORICAL_TAG}^{{commit}}")
    tag_tree = _git(repository, "rev-parse", f"{HISTORICAL_TAG}^{{tree}}")
    if (tag_object, tag_commit, tag_tree) != (
        HISTORICAL_TAG_OBJECT,
        HISTORICAL_TAG_COMMIT,
        HISTORICAL_TAG_TREE,
    ):
        raise ProtocolViolation("HISTORICAL_FEW_SHOT_TAG_MISMATCH")

    head_freeze = json.loads(
        (frozen / "05_head_finetuning/fs4_source_recipe_selection/HEAD_METHOD_FREEZE.json").read_text(
            encoding="utf-8"
        )
    )
    observed_lambdas = {int(key): float(value) for key, value in head_freeze["selected_lambdas"].items()}
    optimizer = head_freeze["optimizer"]
    if observed_lambdas != HEAD_LAMBDAS or optimizer != {
        "full_batch": True,
        "history_size": 20,
        "line_search_fn": "strong_wolfe",
        "max_iter": 100,
        "name": "torch.optim.LBFGS",
        "query_metric_early_stopping": False,
        "scheduler": None,
        "tolerance_change": 1e-12,
        "tolerance_grad": 1e-09,
    }:
        raise ProtocolViolation("HISTORICAL_LINEAR_HEAD_HYPERPARAMETER_DRIFT")

    return {
        "historical_tag": {
            "name": HISTORICAL_TAG,
            "object": tag_object,
            "commit": tag_commit,
            "tree": tag_tree,
            "object_type": _git(repository, "cat-file", "-t", tag_object),
        },
        "frozen_payload_manifest": {
            "relative_path": "configs/few_shot/frozen_payload_manifest.csv",
            "sha256": sha256_file(manifest),
            "declared_file_count": len(declared),
        },
        "released_source_and_result_bindings": bindings,
        "source_method": "C1_FIRST_DIFFERENCE_ERM_1DCNN",
        "checkpoint_seeds": [42, 43, 44, 45, 46],
        "frozen_source_checkpoints": [
            {
                "seed": seed,
                "relative_path": (
                    "01_frozen_source_import/frozen_last_code_snapshot/"
                    f"11_final_p4_evaluation/runs/seed_{seed}/FINAL_CHECKPOINT.pt"
                ),
                "checkpoint_file_sha256": CHECKPOINT_FILE_HASHES[seed],
                "model_state_sha256": CHECKPOINT_STATE_HASHES[seed],
                "weights_frozen": True,
            }
            for seed in sorted(CHECKPOINT_FILE_HASHES)
        ],
        "frozen_preprocessing_files": [
            {"filename": filename, "sha256": digest}
            for filename, digest in PREPROCESSING_FILES.items()
        ],
        "label_mapping": "TagID 1..7 -> canonical class_index 0..6",
        "preprocessing": "281 raw values -> 280 first differences -> frozen P1-P3 population standardization",
        "historical_prototype": "target-only float64 mean; squared Euclidean (not reused as the new cosine method)",
        "historical_linear_head": {
            "implementation": "full-batch float64 LBFGS proximal adaptation of network.13 only",
            "selected_lambdas": {str(key): value for key, value in HEAD_LAMBDAS.items()},
            "optimizer": optimizer,
            "initialization": head_freeze["head_initialization"],
            "objective": head_freeze["objective"],
        },
        "historical_results_preserved": True,
    }


def historical_dependence_audit(archive_repository: str | Path) -> list[dict[str, object]]:
    root = Path(archive_repository) / "outputs/few_shot/frozen_branch/02_support_query_protocol"
    leakage = json.loads((root / "LEAKAGE_AUDIT.json").read_text(encoding="utf-8"))
    if not leakage.get("all_episode_shot_checks_valid") or len(leakage.get("checks", [])) != 54:
        raise ProtocolViolation("HISTORICAL_LEAKAGE_AUDIT_MISMATCH")
    rows: list[dict[str, object]] = []
    for episode_path in sorted((root / "episodes").glob("episode_*.json")):
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        query_cells = {
            (int(item["er"]), int(str(item["surface"])[1:]) - 1)
            for item in episode["query_condition_combinations"]
        }
        query_ids = set(episode["query_sample_ids"])
        pool_rows = episode["support_pool_samples"]
        for shot in (1, 3, 5):
            selected_ids = set(episode["support_sample_ids_by_shot"][str(shot)])
            selected_rows = [row for row in pool_rows if row["sample_id"] in selected_ids]
            if len(selected_rows) != 7 * shot:
                raise ProtocolViolation("HISTORICAL_SUPPORT_BUDGET_MISMATCH")
            class_cell_sets = []
            for class_index in range(7):
                class_rows = [row for row in selected_rows if int(row["label_index"]) == class_index]
                support_cells = tuple(
                    sorted(
                        {
                            (int(row["er"]), int(str(row["surface"])[1:]) - 1)
                            for row in class_rows
                        }
                    )
                )
                support_ids = {row["sample_id"] for row in class_rows}
                overlap_cells = set(support_cells) & query_cells
                class_cell_sets.append(support_cells)
                er_imbalance = factor_count_imbalance(support_cells, 0)
                surface_imbalance = factor_count_imbalance(support_cells, 1)
                rows.append(
                    {
                        "episode": int(episode["episode_id"]),
                        "shot": shot,
                        "class_index": class_index,
                        "support_rows": len(class_rows),
                        "query_rows": len(query_cells) * 50,
                        "row_overlap_count": len(support_ids & query_ids),
                        "row_disjoint": not bool(support_ids & query_ids),
                        "condition_block_overlap_count": len(overlap_cells),
                        "condition_block_relation": (
                            "condition-block-overlapping" if overlap_cells else "condition-block-disjoint"
                        ),
                        "support_unique_er_levels": len({cell[0] for cell in support_cells}),
                        "support_unique_surfaces": len({cell[1] for cell in support_cells}),
                        "support_unique_factor_cells": len(support_cells),
                        "support_er_count_imbalance": er_imbalance,
                        "support_surface_count_imbalance": surface_imbalance,
                        "support_pairwise_dispersion": pairwise_dispersion(support_cells),
                        "er_balanced_where_feasible": shot < 3 or er_imbalance <= 1,
                        "surface_balanced_where_feasible": shot < 3 or surface_imbalance <= 1,
                        "jointly_balanced_where_feasible": shot < 3
                        or (er_imbalance <= 1 and surface_imbalance <= 1),
                        "support_factor_cells": ";".join(cell_id(cell) for cell in support_cells),
                        "query_factor_cells": ";".join(cell_id(cell) for cell in sorted(query_cells)),
                        "near_duplicate_repetitions_from_same_condition_on_both_sides": bool(
                            overlap_cells
                        ),
                        "repeated_measurement_dependence": (
                            "same-condition repetitions cross split"
                            if overlap_cells
                            else "no same-condition repetitions cross split"
                        ),
                    }
                )
            if len(set(class_cell_sets)) != 1:
                raise ProtocolViolation("HISTORICAL_CLASS_SPECIFIC_FACTOR_CELL_SELECTION")
    return rows
