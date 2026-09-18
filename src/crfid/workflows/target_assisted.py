"""Reproduction workflow for P4 Target-Assisted Adaptation."""

from __future__ import annotations

import csv
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from ..adaptation.adaptation_results import aggregate_metrics
from ..adaptation.readout_adaptation import load_readout_state, save_readout_state
from ..adaptation.target_assisted import execute_target_informed_ncm
from ..adaptation.target_partition import historical_full_p4_partition, overlap_report
from ..evaluation.metrics import classification_metrics
from ..governance.adaptation_integrity import (
    canonical_json_sha256,
    sha256_file,
    validate_source_release as validate_frozen_source_release,
)
from ..governance.claims import RETROSPECTIVE_CLAIM, validate_target_assisted_claim
from ..governance.target_access import TargetAccessContract
from ..protocols.common import AccessRequest, Purpose, Resource
from ..protocols.target_assisted import TargetAssistedProtocol
from .common import WorkflowPlan, build_plan


EXPECTED_KEYS = ("emb_train", "emb_test", "y_train", "y_test", "dom_train", "dom_test", "class_names")


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_contract(path: str | Path) -> TargetAccessContract:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return TargetAccessContract.from_mapping(payload["target_access"])


def validate_source_release(config: dict[str, Any]) -> dict[str, Any]:
    manifest = validate_frozen_source_release(
        config["source_release_path"], config["strict_recipe_sha256"]
    )
    _json(Path(config["output_path"]) / "source_import" / "SOURCE_IMPORT_MANIFEST.json", manifest)
    return manifest


def validate_target_inputs(config: dict[str, Any]) -> dict[str, Any]:
    bundle = Path(config["historical_embedding_path"])
    if sha256_file(bundle) != config["historical_embedding_sha256"]:
        raise ValueError("Historical embedding bundle identity changed")
    with np.load(bundle, allow_pickle=False) as payload:
        missing = sorted(set(EXPECTED_KEYS) - set(payload.files))
        if missing:
            raise ValueError(f"Historical embedding bundle is missing arrays: {missing}")
        shapes = {key: list(payload[key].shape) for key in EXPECTED_KEYS}
        if shapes["emb_train"] != [7350, 256] or shapes["emb_test"] != [3150, 256]:
            raise ValueError("Historical embedding dimensions changed")
        if not np.array_equal(np.unique(payload["y_test"]), np.arange(7)):
            raise ValueError("Target class order changed")
    return {"bundle_sha256": config["historical_embedding_sha256"], "shapes": shapes, "status": "PASS"}


def build_target_partitions(config: dict[str, Any]) -> dict[str, Any]:
    output = Path(config["output_path"]) / "target_protocol"
    with Path(config["target_data_path"]).open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["position"] == "P4"]
    if len(rows) != 3150:
        raise ValueError("Expected exactly 3150 P4 samples")
    sample_ids = [row["sample_id"] for row in rows]
    assignments = {row["sample_id"]: row for row in historical_full_p4_partition(sample_ids)}
    sample_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        sample_rows.append(
            {
                "row_index": index,
                "file": row["source_file"],
                "sample_id": row["sample_id"],
                "position": row["position"],
                "surface": row["surface_case"],
                "ER": row["ER"],
                "TagID": row["TagID"],
                "condition_group": f"{row['surface_case']}|ER={row['ER']}|TagID={row['TagID']}",
                "signal_hash": row["signal_hash"],
                **assignments[row["sample_id"]],
            }
        )
    fields = list(sample_rows[0])
    _csv(output / "target_sample_manifest.csv", fields, sample_rows)
    _csv(output / "target_partition_manifest.csv", fields, sample_rows)
    condition_counts: dict[str, int] = {}
    for row in sample_rows:
        condition_counts[row["condition_group"]] = condition_counts.get(row["condition_group"], 0) + 1
    condition_rows = [
        {"condition_group": key, "sample_count": condition_counts[key], "assigned_role": "selection_and_final_evaluation"}
        for key in sorted(condition_counts)
    ]
    _csv(output / "target_condition_manifest.csv", list(condition_rows[0]), condition_rows)
    label_rows = [
        {"sample_id": row["sample_id"], "partition": "p4_full", "purpose": "selection_and_final_scoring", "label_accessed": "true"}
        for row in sample_rows
    ]
    _csv(output / "target_label_access_log.csv", list(label_rows[0]), label_rows)
    metric_rows = [{"event": "retrospective_readout_selection", "partition": "p4_full", "sample_count": 3150, "metrics_accessed": "accuracy;macro_f1"}]
    _csv(output / "target_metric_access_log.csv", list(metric_rows[0]), metric_rows)
    report = overlap_report(
        {"selection": sample_ids, "final_evaluation": sample_ids}, overlap_allowed=True
    )
    report.update({"sample_count": len(rows), "condition_count": len(condition_rows), "independent_target_evaluation": False})
    _json(output / "target_overlap_report.json", report)
    return {"rows": rows, "sample_rows": sample_rows, "condition_rows": condition_rows, "overlap": report}


def prepare_target_assisted_method(config: dict[str, Any]) -> dict[str, Any]:
    contract = _load_contract(config["target_access_path"])
    contract.authorize_parameter_update(())
    with np.load(config["historical_embedding_path"], allow_pickle=False) as payload:
        result = execute_target_informed_ncm(payload["emb_train"], payload["y_train"], payload["emb_test"])
    state_path = Path(config["output_path"]) / "execution" / "adaptation_states" / "seed_42_source_cosine_ncm.npz"
    save_readout_state(state_path, result.prototypes, tuple(range(7)))
    return {"state_path": state_path, "prototypes": result.prototypes}


def execute_target_adaptation(config: dict[str, Any]) -> dict[str, np.ndarray]:
    contract = _load_contract(config["target_access_path"])
    protocol = TargetAssistedProtocol(contract)
    protocol.authorize(AccessRequest("P4", Resource.FEATURES, Purpose.ADAPTATION, "p4_full"))
    with np.load(config["historical_embedding_path"], allow_pickle=False) as payload:
        result = execute_target_informed_ncm(payload["emb_train"], payload["y_train"], payload["emb_test"])
    return {"prototypes": result.prototypes, "distances": result.distances, "predictions": result.predictions}


def generate_target_predictions(config: dict[str, Any], execution: dict[str, np.ndarray]) -> dict[str, str]:
    root = Path(config["output_path"]) / "execution"
    prediction_path = root / "predictions" / "seed_42.npy"
    distance_path = root / "logits_or_distances" / "seed_42_cosine_distances.npy"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    distance_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(prediction_path, execution["predictions"], allow_pickle=False)
    np.save(distance_path, execution["distances"], allow_pickle=False)
    return {
        "prediction_path": prediction_path.as_posix(),
        "prediction_sha256": sha256_file(prediction_path),
        "distance_path": distance_path.as_posix(),
        "distance_sha256": sha256_file(distance_path),
        "prototype_array_sha256": __import__("hashlib").sha256(execution["prototypes"].tobytes()).hexdigest(),
    }


def calculate_target_metrics(config: dict[str, Any], prediction: np.ndarray) -> dict[str, Any]:
    contract = _load_contract(config["target_access_path"])
    protocol = TargetAssistedProtocol(contract)
    protocol.authorize(AccessRequest("P4", Resource.LABELS, Purpose.EVALUATION, "p4_full"))
    with np.load(config["historical_embedding_path"], allow_pickle=False) as payload:
        return classification_metrics(payload["y_test"], prediction, class_count=7)


def aggregate_target_results(metrics: dict[str, Any]) -> dict[str, Any]:
    return aggregate_metrics([metrics])


def reproduce_historical_selection(config: dict[str, Any]) -> dict[str, Any]:
    contract = _load_contract(config["target_access_path"])
    TargetAssistedProtocol(contract).authorize(
        AccessRequest("P4", Resource.OUTCOMES, Purpose.SELECTION, "p4_full")
    )
    return {
        "selected_method": "source_cosine_ncm",
        "selection_type": "retrospective_full_p4_outcome_informed",
        "source_only_selector_winner": "whiten_l0.75",
        "same_p4_partition_reused_for_final_metrics": True,
        "claim_type": RETROSPECTIVE_CLAIM,
    }


def compare_with_authoritative_outputs(config: dict[str, Any], metrics: dict[str, Any], identities: dict[str, str]) -> dict[str, Any]:
    reference = config["authoritative_reference"]
    metric_match = (
        metrics["accuracy"] == reference["accuracy"]
        and abs(metrics["macro_f1"] - reference["macro_f1"]) <= float(config["macro_f1_tolerance"])
        and metrics["confusion_matrix"] == reference["confusion_matrix"]
        and metrics["per_class_recall"] == reference["per_class_recall"]
    )
    return {
        "metric_match": metric_match,
        "prediction_status": "METHOD_DERIVED_EXACT_NO_SEPARATE_AUTHORITATIVE_ARRAY_ARCHIVED",
        "distance_status": "RECOMPUTED_FLOAT64_FROM_HASHED_AUTHORITATIVE_EMBEDDINGS",
        **identities,
    }


def generate_target_assisted_report(config: dict[str, Any], metrics: dict[str, Any], comparison: dict[str, Any]) -> str:
    validate_target_assisted_claim(RETROSPECTIVE_CLAIM)
    return (
        "# P4 Target-Assisted Adaptation results\n\n"
        f"Accuracy: `{metrics['accuracy']}`  \nMacro-F1: `{metrics['macro_f1']}`\n\n"
        "Claim: `RETROSPECTIVE_P4_INFORMED_REFERENCE`. The full P4 label set influenced "
        "readout selection and was reused for final metrics; this is not an independent target evaluation.\n\n"
        f"Authoritative comparison passed: `{comparison['metric_match']}`.\n"
    )


def _freeze_protocol(config: dict[str, Any], source_manifest: dict[str, Any], partition: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["output_path"])
    freeze = root / "protocol_freeze"
    specification = {
        "schema_version": 1,
        "public_name": "P4 Target-Assisted Adaptation",
        "claim_type": RETROSPECTIVE_CLAIM,
        "source_release_recipe_sha256": config["strict_recipe_sha256"],
        "historical_embedding_bundle_sha256": config["historical_embedding_sha256"],
        "source_lineage_boundary": "current strict release is verified but did not generate the historical replay embeddings",
        "p4_sample_count": 3150,
        "p4_condition_count": len(partition["condition_rows"]),
        "target_partitions": {"adaptation": "p4_full", "selection": "p4_full", "evaluation": "p4_full"},
        "same_examples_used_for_selection_and_evaluation": True,
        "class_order": list(range(7)),
        "seeds": [42],
        "retained_methods": ["source_cosine_ncm"],
        "representation": "historical_frozen_256d_embedding",
        "prototype": "L2(class_mean(L2(source_embedding)))",
        "distance_metric": "cosine_similarity",
        "target_source_blending": "none",
        "target_prototypes": False,
        "trainable_parameters": [],
        "optimizer": None,
        "epochs": 0,
        "metric_definitions": ["sample_accuracy", "unweighted_macro_f1"],
        "aggregation": "single_seed_42_run",
        "tie_breaking": "lowest_class_index",
        "selection_rule": "retrospective P4-informed promotion of ordinary source NCM",
        "allowed_claims": [RETROSPECTIVE_CLAIM],
    }
    digest = canonical_json_sha256(specification)
    _json(freeze / "adaptation_specification.json", specification)
    (freeze / "adaptation_specification.sha256").write_text(digest + "\n", encoding="ascii")
    _json(freeze / "source_import_manifest.json", source_manifest)
    target_manifest = root / "target_protocol" / "target_partition_manifest.csv"
    (freeze / "target_partition_manifest.csv").write_bytes(target_manifest.read_bytes())
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "executable": "canonical_crfid_environment/Scripts/python.exe",
        "device": "cpu",
    }
    _json(freeze / "environment.json", environment)
    repository = Path(__file__).resolve().parents[3]
    code_paths = [
        "src/crfid/adaptation/target_assisted.py",
        "src/crfid/adaptation/prototypes.py",
        "src/crfid/adaptation/readout_adaptation.py",
        "src/crfid/adaptation/target_partition.py",
        "src/crfid/adaptation/adaptation_results.py",
        "src/crfid/protocols/target_assisted.py",
        "src/crfid/governance/target_access.py",
        "src/crfid/governance/adaptation_integrity.py",
        "src/crfid/governance/claims.py",
        "src/crfid/workflows/target_assisted.py",
        "workflows/03_target_assisted_adaptation/run.py",
    ]
    code_rows = [{"relative_path": item, "sha256": sha256_file(repository / item)} for item in code_paths]
    _csv(freeze / "code_manifest.csv", list(code_rows[0]), code_rows)
    config_paths = ["configs/target_assisted/canonical.yaml", "configs/target_assisted/target_access.yaml"]
    config_rows = [{"relative_path": item, "sha256": sha256_file(repository / item)} for item in config_paths]
    _csv(freeze / "config_manifest.csv", list(config_rows[0]), config_rows)
    (freeze / "ADAPTATION_PROTOCOL_FREEZE.md").write_text(
        "# Adaptation protocol freeze\n\nThe specification hash seals the retrospective P4-informed NCM replay before metric comparison.\n",
        encoding="utf-8",
    )
    return {"specification": specification, "sha256": digest}


def run(config: dict[str, Any], *, execute: bool = False) -> WorkflowPlan | dict[str, Any]:
    plan = build_plan(config, RETROSPECTIVE_CLAIM, execute)
    if not execute:
        return plan
    output = Path(config["output_path"])
    source_manifest = validate_source_release(config)
    target_validation = validate_target_inputs(config)
    partition = build_target_partitions(config)
    freeze = _freeze_protocol(config, source_manifest, partition)
    prepared = prepare_target_assisted_method(config)
    execution = execute_target_adaptation(config)
    identities = generate_target_predictions(config, execution)
    reloaded, order = load_readout_state(prepared["state_path"])
    if order != tuple(range(7)) or not np.array_equal(reloaded, execution["prototypes"]):
        raise ValueError("Reloaded adaptation state differs")
    metrics = calculate_target_metrics(config, execution["predictions"])
    aggregate = aggregate_target_results(metrics)
    selection = reproduce_historical_selection(config)
    comparison = compare_with_authoritative_outputs(config, metrics, identities)
    if not comparison["metric_match"]:
        raise ValueError("Reproduced metrics differ from the authoritative reference")
    report = generate_target_assisted_report(config, metrics, comparison)
    _write_final_outputs(config, source_manifest, target_validation, freeze, metrics, aggregate, selection, comparison, report)
    _write_artifact_manifest(Path(config["output_path"]))
    return {
        "status": "PASS_P4_RETROSPECTIVE_RESULT_REPRODUCED_WITH_PROTOCOL_LIMITATION",
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "prediction_sha256": identities["prediction_sha256"],
        "distance_sha256": identities["distance_sha256"],
        "adaptation_specification_sha256": freeze["sha256"],
    }


def _write_final_outputs(
    config: dict[str, Any], source: dict[str, Any], target: dict[str, Any], freeze: dict[str, Any],
    metrics: dict[str, Any], aggregate: dict[str, Any], selection: dict[str, Any], comparison: dict[str, Any], report: str,
) -> None:
    root = Path(config["output_path"])
    execution = root / "execution"
    final = root / "final"
    compare = root / "comparison"
    registry = {
        "retained": [selection],
        "decisive_target_informed_comparison": [
            {"method": "learned_head", "accuracy": 0.5561904761904762, "macro_f1": 0.5621821956995522},
            {"method": "source_cosine_ncm", "accuracy": 0.5771428571428572, "macro_f1": 0.5835693346352661},
            {"method": "t3a_confident_target_prototypes", "accuracy": 0.533968253968254, "macro_f1": 0.5421032988408008},
            {"method": "sinkhorn", "accuracy": 0.525079365079365, "macro_f1": 0.5252874012117256},
        ],
        "source_only_selector_candidate_set": [
            "head", "ncm", "ncm_cl2n", "posproto", "whiten_l0.25", "wcl2n_l0.25",
            "whiten_l0.5", "wcl2n_l0.5", "whiten_l0.75", "wcl2n_l0.75", "fuse_hn",
        ],
        "source_only_selector_winner": "whiten_l0.75",
        "source_only_selector_winner_p4_metrics": {"accuracy": 0.5355555555555556, "macro_f1": 0.5382807709220131},
        "implementation_scope": "Only the retained source cosine NCM is implemented; other candidates are lineage evidence.",
    }
    _json(execution / "method_registry.json", registry)
    run_row = {"seed": 42, "method": "source_cosine_ncm", "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}
    _csv(execution / "per_run_metrics.csv", list(run_row), [run_row])
    _csv(execution / "per_method_metrics.csv", list(run_row), [run_row])
    _json(final / "selected_method.json", selection)
    _csv(final / "per_seed_metrics.csv", list(run_row), [run_row])
    _json(final / "aggregate_metrics.json", aggregate)
    recall_rows = [{"class_index": index, "recall": value} for index, value in enumerate(metrics["per_class_recall"])]
    _csv(final / "class_recall.csv", list(recall_rows[0]), recall_rows)
    _csv(final / "confusion_matrices" / "seed_42.csv", [str(i) for i in range(7)], [dict(zip([str(i) for i in range(7)], row)) for row in metrics["confusion_matrix"]])
    prediction_manifest = [{"seed": 42, "prediction_sha256": comparison["prediction_sha256"], "distance_sha256": comparison["distance_sha256"]}]
    _csv(final / "prediction_manifest.csv", list(prediction_manifest[0]), prediction_manifest)
    (final / "TARGET_ASSISTED_RESULTS.md").write_text(report, encoding="utf-8")
    integrity = {
        "verdict": "PASS_P4_RETROSPECTIVE_RESULT_REPRODUCED_WITH_PROTOCOL_LIMITATION",
        "strict_source_import": source["verification_status"],
        "target_input_validation": target["status"],
        "specification_sha256": freeze["sha256"],
        "metric_match": comparison["metric_match"],
        "source_artifacts_modified": False,
        "independent_target_evaluation": False,
    }
    _json(final / "TARGET_ASSISTED_INTEGRITY.json", integrity)
    _json(compare / "authoritative_reference.json", config["authoritative_reference"])
    compare_rows = [
        {"metric": "accuracy", "reproduced": metrics["accuracy"], "authoritative": config["authoritative_reference"]["accuracy"], "match": True},
        {"metric": "macro_f1", "reproduced": metrics["macro_f1"], "authoritative": config["authoritative_reference"]["macro_f1"], "match": True},
    ]
    _csv(compare / "reproduced_vs_authoritative.csv", list(compare_rows[0]), compare_rows)
    (compare / "REPRODUCTION_COMPARISON.md").write_text(
        "# Reproduction comparison\n\nAccuracy, Macro-F1, class recalls, and the confusion matrix match the authoritative reference. Predictions and cosine distances were independently regenerated from the hashed embedding carrier.\n",
        encoding="utf-8",
    )


def _write_artifact_manifest(root: Path) -> None:
    destination = root / "execution" / "artifact_manifest.csv"
    rows = []
    for path in sorted((item for item in root.rglob("*") if item.is_file() and item != destination), key=lambda item: item.as_posix()):
        rows.append({"relative_path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    _csv(destination, list(rows[0]), rows)
