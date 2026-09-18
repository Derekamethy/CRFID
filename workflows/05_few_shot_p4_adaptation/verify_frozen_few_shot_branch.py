"""Branch-local verification of the migrated frozen Few-Shot P4 adaptation line.

The Few-Shot line is closed: no model is loaded, no embedding is extracted, no
prototype or adapted head is built, no optimiser or inference runs and no
prediction is regenerated. This entry point only rehashes frozen artifacts and
recomputes reported statistics from persisted per-unit records.

Usage::

    python workflows/05_few_shot_p4_adaptation/verify_frozen_few_shot_branch.py
    python workflows/05_few_shot_p4_adaptation/verify_frozen_few_shot_branch.py \
        --root <frozen-branch-root> --report <output.json>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crfid.few_shot import manifests, protocol, results  # noqa: E402
from crfid.few_shot.paths import FrozenBranch  # noqa: E402

PLACES = 12
METHOD_FRAGMENTS = {
    "FS3": "PROTOTYPES",
    "FS5": "SOURCE_ANCHORED",
}


def _episode_means(rows: list[dict[str, str]], field: str) -> dict[tuple[int, int], float]:
    grouped: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        key = (int(row["episode_id"]), int(row["shot_count"]))
        grouped.setdefault(key, []).append(float(row[field]))
    return {key: statistics.fmean(values) for key, values in grouped.items()}


def _close(observed: float, expected: float) -> bool:
    return abs(observed - expected) <= 10.0 ** (-PLACES)


def _check(report: list[dict[str, Any]], check_id: str, passed: bool, detail: str, **extra: Any) -> None:
    entry: dict[str, Any] = {"check_id": check_id, "passed": bool(passed), "detail": detail}
    entry.update(extra)
    report.append(entry)


def verify_unit_completeness(root: Path | None, frozen: protocol.FrozenProtocol) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for stage in ("FS3", "FS5"):
        rows = results.load_unit_results(stage, root)
        _check(
            checks,
            f"{stage}_unit_count",
            len(rows) == protocol.FORMAL_UNITS_PER_METHOD_LINE,
            f"{stage} carries the frozen formal evaluation-unit count",
            observed=len(rows),
            expected=protocol.FORMAL_UNITS_PER_METHOD_LINE,
        )
        zero = [row for row in rows if int(row["shot_count"]) == 0]
        adapted = [row for row in rows if int(row["shot_count"]) != 0]
        _check(
            checks,
            f"{stage}_zero_shot_unit_count",
            len(zero) == protocol.ZERO_SHOT_UNITS_PER_METHOD_LINE,
            f"{stage} zero-shot reference units",
            observed=len(zero),
            expected=protocol.ZERO_SHOT_UNITS_PER_METHOD_LINE,
        )
        _check(
            checks,
            f"{stage}_adaptation_unit_count",
            len(adapted) == protocol.ADAPTED_UNITS_PER_METHOD_LINE,
            f"{stage} adaptation units across 1, 3 and 5 shot",
            observed=len(adapted),
            expected=protocol.ADAPTED_UNITS_PER_METHOD_LINE,
        )
        _check(
            checks,
            f"{stage}_unit_ids_unique",
            len({row["unit_id"] for row in rows}) == len(rows),
            f"{stage} unit identities are unique",
        )
        seeds = sorted({int(row["checkpoint_seed"]) for row in rows})
        _check(
            checks,
            f"{stage}_all_seeds_retained",
            tuple(seeds) == frozen.checkpoint_seeds,
            f"{stage} retains every predeclared seed with no best-seed selection",
            observed=seeds,
            expected=list(frozen.checkpoint_seeds),
        )
        _check(
            checks,
            f"{stage}_query_population_constant",
            {int(row["query_sample_count"]) for row in rows} == {frozen.query_samples_per_episode},
            f"{stage} evaluates the same query population in every unit",
        )
        expected_cells = {
            (episode, seed, shot)
            for episode in range(frozen.episode_count)
            for seed in frozen.checkpoint_seeds
            for shot in frozen.shot_counts
        }
        observed_cells = {
            (int(row["episode_id"]), int(row["checkpoint_seed"]), int(row["shot_count"]))
            for row in rows
        }
        _check(
            checks,
            f"{stage}_no_missing_or_duplicate_cell",
            observed_cells == expected_cells and len(observed_cells) == len(rows),
            f"{stage} covers every episode x seed x shot cell exactly once",
            missing=len(expected_cells - observed_cells),
            unexpected=len(observed_cells - expected_cells),
        )
    fs3 = {row["unit_id"]: row for row in results.load_unit_results("FS3", root) if row["shot_count"] == "0"}
    fs5 = {row["unit_id"]: row for row in results.load_unit_results("FS5", root) if row["shot_count"] == "0"}
    _check(
        checks,
        "zero_shot_rows_identical_across_method_lines",
        fs3 == fs5,
        "the shared zero-shot reference is identical in both method lines",
    )
    return checks


def verify_aggregation(root: Path | None) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    canonical = results.load_canonical_results(root)
    unit_rows = {stage: results.load_unit_results(stage, root) for stage in ("FS3", "FS5")}

    for stage, fragment in METHOD_FRAGMENTS.items():
        rows = unit_rows[stage]
        acc = _episode_means(rows, "sample_accuracy")
        f1 = _episode_means(rows, "sample_macro_f1")
        for output in [row for row in canonical if fragment in row["method"]]:
            shot = int(output["shot_count"])
            acc_values = [acc[(episode, shot)] for episode in range(protocol.EPISODE_COUNT)]
            f1_values = [f1[(episode, shot)] for episode in range(protocol.EPISODE_COUNT)]
            ok = (
                _close(statistics.fmean(acc_values), float(output["sample_accuracy_mean"]))
                and _close(statistics.pstdev(acc_values), float(output["sample_accuracy_population_sd"]))
                and _close(statistics.fmean(f1_values), float(output["sample_macro_f1_mean"]))
                and _close(statistics.pstdev(f1_values), float(output["sample_macro_f1_population_sd"]))
            )
            _check(
                checks,
                f"canonical_row_{output['method']}_{shot}",
                ok,
                "canonical mean and population SD recompute from the 18 episode means",
                recomputed_accuracy_mean=statistics.fmean(acc_values),
                stored_accuracy_mean=float(output["sample_accuracy_mean"]),
            )

    zero_row = next(row for row in canonical if row["method"] == protocol.ZERO_SHOT_METHOD)
    zero_acc = _episode_means(unit_rows["FS3"], "sample_accuracy")
    zero_f1 = _episode_means(unit_rows["FS3"], "sample_macro_f1")
    acc_values = [zero_acc[(episode, 0)] for episode in range(protocol.EPISODE_COUNT)]
    f1_values = [zero_f1[(episode, 0)] for episode in range(protocol.EPISODE_COUNT)]
    _check(
        checks,
        "canonical_row_zero_shot",
        _close(statistics.fmean(acc_values), float(zero_row["sample_accuracy_mean"]))
        and _close(statistics.pstdev(f1_values), float(zero_row["sample_macro_f1_population_sd"])),
        "zero-shot canonical row recomputes from the 18 episode means",
    )
    _check(
        checks,
        "population_sd_is_over_episode_means",
        not _close(
            statistics.pstdev(
                float(row["sample_macro_f1"]) for row in unit_rows["FS3"] if row["shot_count"] == "0"
            ),
            float(zero_row["sample_macro_f1_population_sd"]),
        ),
        "the reported SD is the 18-episode SD, not a pooled 90-unit SD",
    )

    deltas = results.load_paired_episode_deltas(root)
    source = {
        "PROTOTYPE": (
            _episode_means(unit_rows["FS3"], "sample_accuracy"),
            _episode_means(unit_rows["FS3"], "sample_macro_f1"),
        ),
        "HEAD": (
            _episode_means(unit_rows["FS5"], "sample_accuracy"),
            _episode_means(unit_rows["FS5"], "sample_macro_f1"),
        ),
    }
    mismatched = 0
    for row in deltas:
        episode = int(row["episode_id"])
        from_acc, from_f1 = source[row["from_method"]]
        to_acc, to_f1 = source[row["to_method"]]
        expected_acc = to_acc[(episode, int(row["to_shot"]))] - from_acc[(episode, int(row["from_shot"]))]
        expected_f1 = to_f1[(episode, int(row["to_shot"]))] - from_f1[(episode, int(row["from_shot"]))]
        if not (_close(expected_acc, float(row["accuracy_delta"])) and _close(expected_f1, float(row["macro_f1_delta"]))):
            mismatched += 1
    _check(
        checks,
        "paired_episode_deltas_recompute",
        mismatched == 0 and len(deltas) == 15 * protocol.EPISODE_COUNT,
        "every paired episode delta recomputes from the episode means",
        row_count=len(deltas),
        mismatched=mismatched,
    )

    per_class = results.load_per_class_comparison(root)
    sources = {
        "SOURCE": results.load_per_class_results("FS3", root),
        "PROTOTYPE": results.load_per_class_results("FS3", root),
        "HEAD": results.load_per_class_results("FS5", root),
    }
    per_class_mismatched = 0
    for output in per_class:
        selected = [
            row
            for row in sources[output["method_short"]]
            if int(row["shot_count"]) == int(output["shot_count"])
            and int(row["class_index"]) == int(output["class_index"])
        ]
        values = [float(row["sample_recall"]) for row in selected]
        episode_values = [
            statistics.fmean(
                float(row["sample_recall"]) for row in selected if int(row["episode_id"]) == episode
            )
            for episode in range(protocol.EPISODE_COUNT)
        ]
        if not (
            len(values) == 90
            and _close(statistics.fmean(values), float(output["mean_recall"]))
            and _close(statistics.pstdev(episode_values), float(output["episode_recall_population_sd"]))
        ):
            per_class_mismatched += 1
    _check(
        checks,
        "per_class_summaries_recompute",
        per_class_mismatched == 0,
        "every per-class recall summary recomputes from persisted unit records",
        row_count=len(per_class),
        mismatched=per_class_mismatched,
    )
    return checks


def verify_claim_boundaries(root: Path | None) -> list[dict[str, Any]]:
    branch = FrozenBranch.resolve(root)
    checks: list[dict[str, Any]] = []
    canonical = results.load_canonical_results(root)
    _check(
        checks,
        "no_combined_adapter_claimed",
        all(row["deployable_combined_adapter"] == "False" for row in canonical)
        and all("HYBRID" not in row["method"] for row in canonical),
        "no deployable combined or hybrid adapter is claimed",
    )
    status = json.loads(branch.read_text(manifests.STATUS_RELATIVE_PATH))
    for key in (
        "raw_p4_numerical_access_during_fsa",
        "sealed_query_label_access_during_fsa",
        "model_loaded_or_inference_executed_during_fsa",
        "optimizer_or_prediction_execution_during_fsa",
        "embedding_extraction_or_prototype_construction_during_fsa",
        "future_writes_to_few_shot_code_authorized",
    ):
        _check(checks, f"closed_boundary_{key}", status[key] is False, f"{key} is recorded false")
    for key in (
        "all_canonical_hashes_passed",
        "all_preexisting_files_unchanged",
        "cross_method_query_identity_passed",
        "metric_and_aggregation_recomputation_passed",
        "zero_shot_equality_passed",
    ):
        _check(checks, f"sealed_assertion_{key}", status[key] is True, f"{key} is recorded true")
    _check(
        checks,
        "mandatory_test_record",
        status["mandatory_tests"]["passed"] == status["mandatory_tests"]["test_count"]
        and status["mandatory_tests"]["failed"] == 0,
        "the sealed mandatory-test record is complete",
        tests_passed=status["mandatory_tests"]["passed"],
        test_count=status["mandatory_tests"]["test_count"],
    )
    fs2 = json.loads(branch.read_text("04_prototype_adaptation/fs2_failed_attempt_01/FS2_STATUS.json"))
    fs5 = json.loads(
        branch.read_text(
            "05_head_finetuning/fs5_p4_execution/fs5_failed_stage_b_attempt_01/FAILED_FS5_STATUS.json"
        )
    )
    _check(
        checks,
        "historical_failures_preserved",
        fs2["classification"] == "FAIL_METHOD_OR_NUMERICAL_RULE_INCOMPLETE"
        and fs5["classification"] == "FAIL_PREDICTION_OR_METRIC_RECOMPUTATION",
        "both historical failed attempts are preserved with their failing classification",
    )
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=None, help="Frozen branch root (default: outputs/few_shot/frozen_branch)")
    parser.add_argument("--report", type=Path, default=None, help="Write the JSON verification report here")
    parser.add_argument("--anchors", type=Path, default=ROOT / "configs" / "few_shot" / "canonical.yaml")
    parser.add_argument("--skip-file-hashes", action="store_true", help="Skip the per-file rehash of every manifest row")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip the pre-audit baseline recheck")
    arguments = parser.parse_args(argv)

    anchor_config: dict[str, Any] = {}
    expected_aggregates: dict[str, str] = {}
    if arguments.anchors.is_file():
        anchor_config = json.loads(arguments.anchors.read_text(encoding="utf-8"))
        expected_aggregates = dict(anchor_config.get("canonical_lineage_aggregates", {}))

    branch_report = manifests.verify_branch(
        arguments.root,
        expected_aggregates=expected_aggregates,
        expected_inventory=anchor_config.get("frozen_payload_inventory"),
        verify_every_file=not arguments.skip_file_hashes,
    )
    frozen = protocol.load_protocol(arguments.root)

    payload: dict[str, Any] = {
        "verification": "FROZEN_FEW_SHOT_BRANCH_VERIFICATION",
        "execution_performed": {
            "model_loaded": False,
            "embedding_extracted": False,
            "prototype_or_head_constructed": False,
            "optimizer_or_inference_run": False,
            "prediction_regenerated": False,
            "sealed_query_label_values_read": False,
        },
        "protocol": frozen.as_dict(),
        "manifest_verification": branch_report.as_dict(),
        "checks": [],
    }
    payload["checks"].extend(verify_unit_completeness(arguments.root, frozen))
    payload["checks"].extend(verify_aggregation(arguments.root))
    payload["checks"].extend(verify_claim_boundaries(arguments.root))

    if not arguments.skip_baseline:
        baseline_count, baseline_mismatches, absent_bytecode = manifests.verify_preaudit_baseline(
            arguments.root
        )
        payload["preaudit_baseline"] = {
            "row_count": baseline_count,
            "mismatch_count": len(baseline_mismatches),
            "mismatches": baseline_mismatches[:50],
            "absent_bytecode_cache_count": len(absent_bytecode),
            "absent_bytecode_caches": absent_bytecode,
            "bytecode_cache_policy": (
                "Python bytecode caches are deterministic derivatives of preserved "
                "sources, carry no scientific content and are excluded from the "
                "byte-identity requirement; they are listed rather than ignored."
            ),
        }
        payload["checks"].append(
            {
                "check_id": "preaudit_baseline_bytes_unchanged",
                "passed": not baseline_mismatches,
                "detail": "every scientific pre-audit baseline row matches by size and SHA-256",
                "row_count": baseline_count,
                "absent_bytecode_cache_count": len(absent_bytecode),
            }
        )

    failed = [check for check in payload["checks"] if not check["passed"]]
    payload["all_checks_passed"] = branch_report.passed and not failed
    payload["failed_check_ids"] = [check["check_id"] for check in failed]
    payload["status"] = (
        "PASS_FROZEN_FEW_SHOT_BRANCH_VERIFIED"
        if payload["all_checks_passed"]
        else "FAIL_FROZEN_FEW_SHOT_BRANCH_VERIFICATION"
    )

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(text, encoding="utf-8")
    print(text)
    return 0 if payload["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
