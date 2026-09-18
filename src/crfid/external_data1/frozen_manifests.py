"""Read-only integrity verification of the migrated frozen External Data1 branch.

Nothing here loads a model, fits an estimator, runs inference, regenerates a
prediction or opens raw external signal data. Every function rehashes frozen
bytes and recomputes declared aggregates.

The branch carries seven stage manifests, one per historical stage. Each lists
``relative_path``, ``size_bytes`` and ``sha256`` for the files that stage
created, and declares an aggregate over its own rows. The aggregate recipe is
recorded inside each manifest and reimplemented in :func:`stage_aggregate`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .paths import FrozenBranch, frozen_branch_root

#: Stage identifier -> branch-relative manifest file, in execution order.
STAGE_MANIFESTS: tuple[tuple[str, str], ...] = (
    ("EV0", "EV0_ARTIFACT_HASHES.json"),
    ("EV1", "EV1_ARTIFACT_HASHES.json"),
    ("EV2_ABORTED", "EV2_ABORTED_ARTIFACT_HASHES.json"),
    ("EV2R", "EV2R_ARTIFACT_HASHES.json"),
    ("EV2R_A1", "EV2R_A1_ARTIFACT_HASHES.json"),
    ("EV3R", "EV3R_ARTIFACT_HASHES.json"),
    ("EV3R_R1", "EV3R_R1_ARTIFACT_HASHES.json"),
)

#: Stage identifier -> branch-relative status record.
STAGE_STATUS: tuple[tuple[str, str], ...] = (
    ("EV0", "EV0_STATUS.json"),
    ("EV1", "EV1_STATUS.json"),
    ("EV2_ABORTED", "EV2_ABORTED_STATUS.json"),
    ("EV2R", "EV2R_STATUS.json"),
    ("EV2R_A1", "EV2R_A1_STATUS.json"),
    ("EV3R", "EV3R_STATUS.json"),
    ("EV3R_R1", "EV3R_R1_STATUS.json"),
)

TIE_RULE_PATH = "04_protocol_preregistration/amendments/EV2R_A1/EV2R_A1_MAJORITY_TIE_RULE.json"
SPLIT_SUMMARY_PATH = "04_protocol_preregistration/splits_ev2r/EV2R_SPLIT_SUMMARY.csv"
CLASS_COVERAGE_PATH = "04_protocol_preregistration/EV2R_CLASS_COVERAGE_AUDIT.csv"
CONCLUSIONS_PATH = "06_internal_portability_benchmark/ev3r_r1/EV3R_R1_THREE_CONCLUSION_DECISIONS.json"
BLOCKED_EXECUTION_LOG = "05_external_execution/EV3R_EXECUTION_LOG.jsonl"
FINAL_TEST_ACCESS_LEDGER = "05_external_execution/EV3R_R1_TEST_ACCESS_LEDGER.jsonl"

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_aggregate(entries: Iterable[Mapping[str, Any]]) -> str:
    """Reimplement the frozen manifest aggregate recipe.

    SHA-256 over manifest-order ``relative_path<TAB>size_bytes<TAB>sha256``
    lines joined by LF with no trailing LF.
    """

    body = "\n".join(
        f"{entry['relative_path']}\t{entry['size_bytes']}\t{entry['sha256']}" for entry in entries
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass
class Check:
    check_id: str
    passed: bool
    detail: str
    observed: Any = None
    expected: Any = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "check_id": self.check_id,
            "passed": bool(self.passed),
            "detail": self.detail,
        }
        if self.observed is not None:
            payload["observed"] = self.observed
        if self.expected is not None:
            payload["expected"] = self.expected
        return payload


@dataclass
class BranchReport:
    frozen_branch_root: str
    checks: list[Check] = field(default_factory=list)
    file_mismatches: list[str] = field(default_factory=list)
    unclaimed_files: list[str] = field(default_factory=list)
    stage_aggregates: dict[str, str] = field(default_factory=dict)
    file_count: int = 0
    total_bytes: int = 0

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks) and not self.file_mismatches

    def as_dict(self) -> dict[str, Any]:
        return {
            "frozen_branch_root": self.frozen_branch_root,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "stage_aggregates": self.stage_aggregates,
            "file_mismatches": self.file_mismatches,
            "unclaimed_files": self.unclaimed_files,
            "checks": [check.as_dict() for check in self.checks],
            "all_checks_passed": self.passed,
        }


def _relative_root(root: Path) -> str:
    """Express the branch root relative to the repository when possible."""

    from .paths import repository_root

    try:
        return root.relative_to(repository_root().resolve()).as_posix()
    except ValueError:
        return root.name


def verify_branch(
    root: str | Path | None = None,
    *,
    expected_aggregates: Mapping[str, str] | None = None,
    expected_file_count: int | None = None,
    expected_total_bytes: int | None = None,
    verify_every_file: bool = True,
) -> BranchReport:
    """Verify the frozen branch against its own seals and the repository anchors."""

    branch = FrozenBranch.resolve(root)
    present: dict[str, tuple[int, str | None]] = {}
    total = 0
    for path in sorted(branch.root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(branch.root).as_posix()
        size = path.stat().st_size
        total += size
        present[relative] = (size, sha256_file(path) if verify_every_file else None)

    report = BranchReport(
        frozen_branch_root=_relative_root(branch.root),
        file_count=len(present),
        total_bytes=total,
    )

    claimed: set[str] = set()
    for stage, manifest_name in STAGE_MANIFESTS:
        claimed.add(manifest_name)
        manifest_path = branch.path(manifest_name)
        if not manifest_path.is_file():
            report.checks.append(Check(f"{stage}_manifest_present", False, f"{manifest_name} is absent"))
            continue
        blob = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = blob["entries"]
        declared = blob["aggregate_sha256"]
        report.stage_aggregates[stage] = declared

        for entry in entries:
            relative = str(entry["relative_path"]).replace("\\", "/")
            claimed.add(relative)
            observed = present.get(relative)
            if observed is None:
                report.file_mismatches.append(f"{stage}:MISSING:{relative}")
                continue
            if observed[0] != entry["size_bytes"]:
                report.file_mismatches.append(f"{stage}:SIZE:{relative}")
                continue
            if verify_every_file and observed[1] != entry["sha256"]:
                report.file_mismatches.append(f"{stage}:SHA256:{relative}")

        recomputed = stage_aggregate(entries)
        report.checks.append(
            Check(
                f"{stage}_aggregate_recomputes",
                recomputed == declared,
                f"{stage} manifest aggregate recomputes from its own rows",
                observed=recomputed,
                expected=declared,
            )
        )
        if expected_aggregates and stage in expected_aggregates:
            report.checks.append(
                Check(
                    f"{stage}_aggregate_matches_repository_anchor",
                    declared == expected_aggregates[stage],
                    f"{stage} aggregate matches the independent repository anchor",
                    observed=declared,
                    expected=expected_aggregates[stage],
                )
            )

    report.unclaimed_files = sorted(set(present) - claimed)
    report.checks.append(
        Check(
            "every_file_is_claimed_by_a_stage_manifest",
            not report.unclaimed_files,
            "no frozen file exists outside the historical stage manifests",
            observed=len(report.unclaimed_files),
            expected=0,
        )
    )
    report.checks.append(
        Check(
            "no_unexpected_extra_or_changed_file",
            not report.file_mismatches,
            "every manifest row is present at the recorded size and digest",
            observed=len(report.file_mismatches),
            expected=0,
        )
    )
    if expected_file_count is not None:
        report.checks.append(
            Check(
                "file_count_matches_repository_anchor",
                len(present) == expected_file_count,
                "frozen file count matches the independently enumerated source",
                observed=len(present),
                expected=expected_file_count,
            )
        )
    if expected_total_bytes is not None:
        report.checks.append(
            Check(
                "total_bytes_matches_repository_anchor",
                total == expected_total_bytes,
                "frozen byte total matches the independently enumerated source",
                observed=total,
                expected=expected_total_bytes,
            )
        )
    return report


def verify_stage_chronology(root: str | Path | None = None) -> list[Check]:
    """Check that the stage order recorded inside the artifacts is coherent."""

    branch = FrozenBranch.resolve(root)
    checks: list[Check] = []
    statuses = {}
    for stage, name in STAGE_STATUS:
        statuses[stage] = json.loads(branch.read_text(name))

    checks.append(
        Check(
            "blocked_stage_is_recorded_as_blocked",
            statuses["EV3R"]["classification"] == "BLOCKED_EV3R_MAJORITY_TIE_RULE_UNRESOLVED",
            "the first execution attempt is recorded as blocked, not as a negative result",
            observed=statuses["EV3R"]["classification"],
        )
    )
    checks.append(
        Check(
            "amendment_precedes_final_execution",
            statuses["EV2R_A1"]["next_authorizable_stage"].startswith("EV3R-R1"),
            "the tie-rule amendment names the final execution as its only successor",
            observed=statuses["EV2R_A1"]["next_authorizable_stage"],
        )
    )
    checks.append(
        Check(
            "final_execution_cites_amended_authority",
            statuses["EV3R_R1"]["effective_protocol_authority"] == ["EV2R", "EV2R-A1"],
            "the executed stage cites the preregistration plus its amendment as authority",
            observed=statuses["EV3R_R1"]["effective_protocol_authority"],
        )
    )

    blocked_events = [
        json.loads(line)
        for line in branch.read_text(BLOCKED_EXECUTION_LOG).splitlines()
        if line.strip()
    ]
    final_access = [
        json.loads(line)
        for line in branch.read_text(FINAL_TEST_ACCESS_LEDGER).splitlines()
        if line.strip()
    ]
    blocked_last = max(event["timestamp_utc"] for event in blocked_events)
    first_scoring = min(event["timestamp_utc"] for event in final_access)
    checks.append(
        Check(
            "no_scoring_access_before_the_block_was_resolved",
            blocked_last < first_scoring,
            "the blocked attempt ended before the first held-out scoring access",
            observed={"blocked_last": blocked_last, "first_scoring": first_scoring},
        )
    )
    return checks


def verify_blocked_stage_had_no_performance_access(root: str | Path | None = None) -> list[Check]:
    """Assert, from the frozen record, that the blocked attempt produced no result."""

    branch = FrozenBranch.resolve(root)
    status = json.loads(branch.read_text("EV3R_STATUS.json"))
    checks: list[Check] = []
    for key in (
        "model_framework_imported",
        "model_instantiated",
        "PCA_fitted",
        "logistic_regression_fitted",
        "training_executed",
        "inference_executed",
        "prediction_calculated",
        "performance_metric_calculated",
        "checkpoint_loaded",
        "test_signal_access",
        "P4_numerical_access",
    ):
        checks.append(
            Check(
                f"blocked_stage_{key}_is_false",
                status.get(key) is False,
                f"the blocked attempt records {key} as false",
                observed=status.get(key),
            )
        )
    checks.append(
        Check(
            "blocked_stage_started_no_learned_run",
            status["learned_runs_started"] == 0 and status["learned_runs_completed"] == 0,
            "no learned run was started or completed by the blocked attempt",
            observed=[status["learned_runs_started"], status["learned_runs_completed"]],
        )
    )
    created = json.loads(branch.read_text("EV3R_CREATED_FILES.json"))["files"]
    performance_bearing = [
        name
        for name in created
        if any(token in name.lower() for token in ("prediction", "checkpoint", "metric", "confusion"))
    ]
    checks.append(
        Check(
            "blocked_stage_wrote_no_performance_artifact",
            not performance_bearing,
            "the blocked attempt created no prediction, checkpoint, metric or confusion artifact",
            observed=performance_bearing,
        )
    )
    return checks


def verify_majority_tie_rule(
    root: str | Path | None = None, *, expected: Mapping[str, Any] | None = None
) -> list[Check]:
    """Recompute the tie-break decision from the frozen training class counts."""

    branch = FrozenBranch.resolve(root)
    rule = json.loads(branch.read_text(TIE_RULE_PATH))
    order = list(rule["frozen_global_class_order"])
    counts = {int(key): int(value) for key, value in rule["training_class_counts"].items()}
    maximum = max(counts[label] for label in order)
    tied = [label for label in order if counts[label] == maximum]
    selected = tied[0]

    checks = [
        Check(
            "tie_rule_selection_recomputes",
            selected == rule["selected_constant_class"],
            "the recorded constant class is the first tied class in the frozen order",
            observed=selected,
            expected=rule["selected_constant_class"],
        ),
        Check(
            "tie_rule_tied_set_recomputes",
            tied == list(rule["tied_candidate_set_T"]),
            "the recorded tied set recomputes from the training counts",
            observed=tied,
            expected=list(rule["tied_candidate_set_T"]),
        ),
        Check(
            "tie_rule_uses_training_counts_only",
            rule["majority_source"] == "frozen training labels only"
            and rule["test_information_used"] is False
            and rule["validation_information_used"] is False
            and rule["performance_information_used"] is False
            and rule["random_tie_breaking_used"] is False,
            "the tie rule declares training-only, deterministic, non-performance selection",
        ),
    ]
    if expected:
        checks.append(
            Check(
                "tie_rule_matches_repository_anchor",
                selected == expected.get("selected_class")
                and order == list(expected.get("global_class_order", []))
                and [counts[label] for label in order] == list(expected.get("training_class_counts", [])),
                "the tie rule matches the independent repository anchor",
                observed={"selected": selected, "order": order,
                          "counts": [counts[label] for label in order]},
                expected=dict(expected),
            )
        )
    return checks


def verify_conclusions(root: str | Path | None = None,
                       expected: Mapping[str, str] | None = None) -> list[Check]:
    """Check the three frozen conclusion decisions against the repository anchor."""

    branch = FrozenBranch.resolve(root)
    blob = json.loads(branch.read_text(CONCLUSIONS_PATH))
    observed = {
        "Data1_learnability": blob["Conclusion_A_Data1_learnability"]["decision"],
        "CNN_pipeline_portability": blob["Conclusion_B_CNN_pipeline_portability"]["decision"],
        "first_difference_portability": blob["Conclusion_C_first_difference_portability"]["decision"],
    }
    checks = [
        Check(
            "conclusion_rules_applied_without_modification",
            all(
                blob[key]["rule_applied_without_modification"] is True
                for key in (
                    "Conclusion_A_Data1_learnability",
                    "Conclusion_B_CNN_pipeline_portability",
                    "Conclusion_C_first_difference_portability",
                )
            ),
            "each conclusion records that its preregistered rule was applied unmodified",
        )
    ]
    if expected:
        checks.append(
            Check(
                "conclusions_match_repository_anchor",
                observed == dict(expected),
                "the frozen conclusions match the independent repository anchor",
                observed=observed,
                expected=dict(expected),
            )
        )
    return checks


def frozen_branch_available(root: str | Path | None = None) -> bool:
    """Convenience predicate mirroring :func:`crfid.external_data1.paths.is_materialised`."""

    from .paths import is_materialised

    return is_materialised(root if root is not None else frozen_branch_root())
