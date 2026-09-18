"""Read-only verification of the retained public Strict-DG evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluation.metrics import metrics_from_confusion
from ..governance.integrity import sha256_file
from ..protocols.strict_dg import StrictDGProtocol
from ..strict_runtime.hashing import array_sha256, canonical_json_sha256
from .common import WorkflowPlan, build_plan

PUBLIC_EVIDENCE = Path(__file__).resolve().parents[3] / "results/canonical_metrics/strict_dg"


def verify_public_evidence(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Check compact identities without opening measurements or omitted predictions."""
    def read(name: str) -> Any:
        path = root / name
        if not path.is_file():
            raise ValueError(f"Public Strict-DG evidence is missing: {name}")
        return json.loads(path.read_text(encoding="utf-8"))

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"Public Strict-DG evidence mismatch: {message}")

    def close(left: Any, right: Any) -> bool:
        return bool(np.isclose(float(left), float(right), rtol=0, atol=1e-12))

    final = read("FINAL_P4_RESULTS.json")
    decision = read("source_selection_decision.json")
    winner = read("winner_selection.json")
    selected = config["selected_candidate"]
    require(final["selected_candidate"] == decision["selected_candidate"] == winner["selected_candidate"] == selected, "selected candidate")
    require(decision["p4_used"] is False and decision["target_accessed"] is False and winner["p4_used"] is False, "source-only selection declaration")
    require(decision["canonical_unit_count"] == winner["unit_count"] == 60, "source unit count")
    with (root / "candidate_aggregate_metrics.csv").open(encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    require(len(candidates) == 4 and {r["candidate_id"] for r in candidates} == set(config["candidate_set"]), "candidate coverage")
    selector = decision["ordered_selector"]
    require([(r["metric"], r["operation"].replace("ascending_deterministic", "ascending")) for r in selector] == [tuple(r) for r in config["selector"]], "selector definition")

    def rank(row: dict[str, Any]) -> tuple:
        return tuple(row[r["metric"]] if r["metric"] == "candidate_id" else float(row[r["metric"]]) * (-1 if r["operation"] == "maximize" else 1) for r in selector)

    ranked = sorted(candidates, key=rank)
    require(ranked[0]["candidate_id"] == selected, "source ranking")
    require([r["candidate_id"] for r in ranked] == [r["candidate_id"] for r in decision["ranking"]], "recorded ranking")
    for observed, recorded in zip(ranked, decision["ranking"]):
        require(int(observed["canonical_unit_count"]) == 15, "candidate unit count")
        for rule in selector:
            key = rule["metric"]
            if key != "candidate_id":
                require(close(observed[key], recorded[key]), f"source aggregate {key}")

    state = read("preprocessing/state.json")
    require(canonical_json_sha256({k: v for k, v in state.items() if k != "state_sha256"}) == state["state_sha256"], "preprocessing state digest")
    require(state["fit_domains"] == ["P1", "P2", "P3"] and state["fit_sample_count"] == 9450 and state["candidate_id"] == selected, "preprocessing fit scope")
    require(state["target_used_for_fit"] is False and state["target_refit_permitted"] is False, "preprocessing target boundary")
    for name in ("mean", "scale"):
        array = np.load(root / f"preprocessing/{name}_float64.npy", allow_pickle=False)
        require(array.shape == (280,) and array.dtype == np.dtype("float64") and np.isfinite(array).all(), f"{name} shape/dtype/values")
        require(array_sha256(array) == state[f"{name}_array_sha256"], f"{name} hash")
        if name == "scale":
            require(bool(np.all(array > 0)), "positive scale")

    seeds = list(config["random_seeds"])
    require([int(r["seed"]) for r in final["seeds"]] == seeds == [42, 43, 44, 45, 46], "seed coverage")
    require(final["class_order"] == list(range(7)) and final["sample_count_per_seed"] == 3150, "class/sample contract")
    with (root / "FINAL_P4_RESULTS.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    require([int(r["seed"]) for r in csv_rows] == seeds, "CSV seed coverage")
    values: dict[str, list[float]] = {"accuracy": [], "macro_f1": []}
    for row, csv_row in zip(final["seeds"], csv_rows):
        seed = int(row["seed"])
        matrix_path = root / row["confusion_matrix_reference"]
        require(matrix_path.resolve().is_relative_to(root.resolve()), "confusion path")
        # Historical text hashes used CRLF; Git distributes LF-normalized JSON.
        raw = matrix_path.read_bytes()
        historical_text = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        require(row["confusion_matrix_sha256"] in {hashlib.sha256(raw).hexdigest(), hashlib.sha256(historical_text).hexdigest()}, f"seed {seed} confusion hash")
        matrix = np.asarray(read(row["confusion_matrix_reference"])["sample"])
        require(matrix.shape == (7, 7) and np.issubdtype(matrix.dtype, np.integer) and np.all(matrix >= 0), f"seed {seed} confusion structure")
        require(int(matrix.sum()) == row["sample_count"] == 3150 and np.all(matrix.sum(axis=1) == 450), f"seed {seed} class support")
        metrics = metrics_from_confusion(matrix)
        require(int(row["selected_epoch"]) == config["final_epochs_by_seed"][str(seed)] == decision["final_epochs_by_seed"][str(seed)], f"seed {seed} epoch")
        for key in values:
            require(close(metrics[key], row[key]) and close(metrics[key], csv_row[key]), f"seed {seed} {key}")
            values[key].append(metrics[key])
        checkpoints = list((root / "checkpoints").glob(f"*{seed}*"))
        require(len(checkpoints) == 1 and sha256_file(checkpoints[0]) == row["checkpoint_sha256"] == csv_row["checkpoint_sha256"], f"seed {seed} checkpoint hash")
    aggregate = final["aggregate"]
    require(aggregate["ddof"] == 0 and aggregate["equal_seed_weight"] is True, "aggregation convention")
    for key, numbers in values.items():
        require(close(np.mean(numbers), aggregate[f"{key}_mean"]) and close(np.std(numbers, ddof=0), aggregate[f"{key}_population_sd"]), f"aggregate {key}")
    return {
        "status": "PUBLIC_COMPACT_EVIDENCE_VERIFIED",
        "selected_candidate": selected,
        "source_selection_units": 60,
        "checkpoint_count": len(seeds),
        "text_hash_policy": "Recorded CRLF or distributed LF serialization; numerical contents verified separately",
        "aggregate": aggregate,
        "measurements_opened": False,
        "training_performed": False,
        "historical_replay": "HISTORICAL_FULL_RELEASE_REPLAY_NOT_DISTRIBUTED",
        "scope": "Retained artifact hashes, source ranking, preprocessing and confusion-derived metrics; no raw-data inference or omitted prediction replay.",
    }


def run(config: dict[str, Any], *, execute: bool = False) -> WorkflowPlan | dict[str, Any]:
    if not execute:
        return build_plan(config, StrictDGProtocol.declaration.scientific_label, execute)
    return verify_public_evidence(PUBLIC_EVIDENCE, config)
