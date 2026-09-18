"""Frozen Few-Shot manifest, aggregate and seal verification.

The aggregate rule reproduced here is the one recorded inside every stage
manifest of the authoritative branch::

    sha256 over "<relative_path>|<size_bytes>|<sha256>\\n" rows,
    ordered by case-folded relative path

Nothing in this module loads an array, a checkpoint or a query label value. The
two sealed query-label files are reduced to a SHA-256 digest for byte-identity
proof only; their contents are never parsed, decoded or returned.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .paths import FrozenBranch

CHUNK_BYTES = 1024 * 1024

AUDIT_DIRECTORY = "06_final_comparative_audit_and_archive"
SEAL_RELATIVE_PATH = f"{AUDIT_DIRECTORY}/FINAL_FEW_SHOT_LINE_SEAL.json"
STATUS_RELATIVE_PATH = f"{AUDIT_DIRECTORY}/FINAL_FEW_SHOT_STATUS.json"
FINAL_ARTIFACT_HASHES = f"{AUDIT_DIRECTORY}/FINAL_ARTIFACT_HASHES.json"
LINEAGE_RELATIVE_PATH = f"{AUDIT_DIRECTORY}/FINAL_LINEAGE_MANIFEST.json"
PREAUDIT_BASELINE = f"{AUDIT_DIRECTORY}/FINAL_PREAUDIT_BASELINE.json"
RELEASE_INVENTORY_RELATIVE_PATH = "configs/few_shot/frozen_payload_manifest.csv"

#: Stage identifier -> branch-relative hash manifest, in canonical workflow order.
CANONICAL_STAGES: dict[str, str] = {
    "FS0": "00_governance/FS0_HASH_MANIFEST.json",
    "FS1": "02_support_query_protocol/FS1_HASH_MANIFEST.json",
    "FS2": "04_prototype_adaptation/FS2_ARTIFACT_HASHES.json",
    "FS2_HISTORICAL_FAILURE": "04_prototype_adaptation/fs2_failed_attempt_01/FS2_ARTIFACT_HASHES.json",
    "FS3": "04_prototype_adaptation/fs3_execution/FS3_ARTIFACT_HASHES.json",
    "FS4": "05_head_finetuning/fs4_source_recipe_selection/FS4_ARTIFACT_HASHES.json",
    "FS5": "05_head_finetuning/fs5_p4_execution/FS5_ARTIFACT_HASHES.json",
    "FS5R": "05_head_finetuning/fs5_p4_execution/fs5r_evaluation_recovery/FS5R_ARTIFACT_HASHES.json",
}

#: Stage identifier -> branch-relative Stage-A prediction closure gate.
PREDICTION_GATES: dict[str, str] = {
    "FS3": "04_prototype_adaptation/fs3_execution/PREDICTION_STAGE_CLOSED.json",
    "FS5": "05_head_finetuning/fs5_p4_execution/PREDICTION_STAGE_CLOSED.json",
}

#: Files whose bytes are hashed for identity but whose contents are never read.
SEALED_RELATIVE_PATHS: frozenset[str] = frozenset(
    {
        "02_support_query_protocol/sealed_query_labels/QUERY_LABELS.csv",
        "02_support_query_protocol/sealed_query_labels/QUERY_CLASS_MAPPING.json",
    }
)

#: The archived failed FS2 attempt records the pre-remediation hashes of two
#: shared implementation paths. The authoritative audit verifies those two rows
#: through the archive manifest instead of the live files, because the passing
#: FS2 remediation intentionally superseded them.
HISTORICAL_FAILURE_STAGE = "FS2_HISTORICAL_FAILURE"
SUPERSEDED_SHARED_PATHS: frozenset[str] = frozenset(
    {"scripts/run_fs2.py", "tests/test_fs2_method_freeze.py"}
)
SUPERSESSION_EVIDENCE = (
    "04_prototype_adaptation/fs2_failed_attempt_01/FS2_FAILED_ATTEMPT_ARCHIVE_MANIFEST.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_rows(rows: Iterable[Mapping[str, Any]]) -> str:
    """Reproduce the frozen stage-aggregate rule."""

    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["relative_path"]).casefold()):
        line = f"{row['relative_path']}|{row['size_bytes']}|{row['sha256']}\n"
        digest.update(line.encode())
    return digest.hexdigest()


def canonical_payload_sha256(payload: Any) -> str:
    """Reproduce the frozen seal rule over a JSON payload."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class CheckResult:
    check_id: str
    passed: bool
    detail: str
    observed: str = ""
    expected: str = ""


def describe_root(root: Path) -> str:
    """Return a portable label for a branch root.

    Roots inside the repository are reported relative to it. Roots outside are
    reduced to a non-machine-specific label so verification reports never carry
    an absolute filesystem path; the concrete path belongs in a custody register.
    """

    from .paths import repository_root

    try:
        return root.relative_to(repository_root()).as_posix()
    except ValueError:
        return f"EXTERNAL_BRANCH_ROOT:{root.name}"


@dataclass
class BranchVerification:
    root: str
    checks: list[CheckResult] = field(default_factory=list)
    file_mismatches: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks) and not self.file_mismatches

    def add(self, check_id: str, passed: bool, detail: str, observed: str = "", expected: str = "") -> None:
        self.checks.append(CheckResult(check_id, passed, detail, observed, expected))

    def as_dict(self) -> dict[str, Any]:
        return {
            "frozen_branch_root": self.root,
            "all_checks_passed": self.passed,
            "check_count": len(self.checks),
            "failed_check_count": sum(1 for check in self.checks if not check.passed),
            "file_mismatch_count": len(self.file_mismatches),
            "file_mismatches": self.file_mismatches[:50],
            "checks": [
                {
                    "check_id": check.check_id,
                    "passed": check.passed,
                    "detail": check.detail,
                    "observed": check.observed,
                    "expected": check.expected,
                }
                for check in self.checks
            ],
        }


@dataclass
class InventoryVerification:
    row_count: int
    total_bytes: int
    aggregate_sha256: str
    manifest_sha256: str
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.missing and not self.extra and not self.mismatches


def verify_release_inventory(
    root: str | Path | None = None,
    *,
    manifest_path: str | Path | None = None,
    verify_hashes: bool = True,
) -> InventoryVerification:
    """Require exact equality with the canonical 5,807-file release inventory."""

    from .paths import repository_root

    branch = FrozenBranch.resolve(root)
    inventory_path = (
        Path(manifest_path)
        if manifest_path is not None
        else repository_root() / RELEASE_INVENTORY_RELATIVE_PATH
    )
    with inventory_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    declared: dict[str, Mapping[str, Any]] = {}
    mismatches: list[str] = []
    for row in rows:
        relative_path = str(row["relative_path"])
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts or relative_path in declared:
            mismatches.append(f"invalid_or_duplicate_manifest_path:{relative_path}")
            continue
        declared[relative_path] = row

    actual = {
        path.relative_to(branch.root).as_posix(): path
        for path in branch.root.rglob("*")
        if path.is_file()
    }
    missing = sorted(set(declared) - set(actual), key=str.casefold)
    extra = sorted(set(actual) - set(declared), key=str.casefold)
    for relative_path in sorted(set(declared) & set(actual), key=str.casefold):
        row = declared[relative_path]
        path = actual[relative_path]
        if path.stat().st_size != int(row["size_bytes"]):
            mismatches.append(f"size:{relative_path}")
        elif verify_hashes and sha256_file(path) != row["sha256"]:
            mismatches.append(f"sha256:{relative_path}")

    return InventoryVerification(
        row_count=len(rows),
        total_bytes=sum(int(row["size_bytes"]) for row in rows),
        aggregate_sha256=aggregate_rows(rows),
        manifest_sha256=sha256_file(inventory_path),
        missing=missing,
        extra=extra,
        mismatches=mismatches,
    )


def _load_json(branch: FrozenBranch, relative_path: str) -> dict[str, Any]:
    return json.loads(branch.read_text(relative_path))


def _manifest_rows(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = manifest.get("files")
    if rows is None:
        rows = manifest.get("archived_files")
    if rows is None:
        raise ValueError("Manifest carries neither 'files' nor 'archived_files'")
    return list(rows)


def verify_branch(
    root: str | Path | None = None,
    *,
    expected_aggregates: Mapping[str, str] | None = None,
    expected_inventory: Mapping[str, Any] | None = None,
    verify_every_file: bool = True,
) -> BranchVerification:
    """Verify the frozen branch against its own sealed manifests.

    ``expected_aggregates`` supplies an independent anchor (for example from the
    repository configuration) so a tampered seal file cannot certify itself.
    """

    branch = FrozenBranch.resolve(root)
    report = BranchVerification(root=describe_root(branch.root))

    seal = _load_json(branch, SEAL_RELATIVE_PATH)
    payload = seal["seal_payload"]
    report.add(
        "seal_payload_hash",
        canonical_payload_sha256(payload) == seal["seal_sha256"],
        "final line seal recomputes from its payload",
        canonical_payload_sha256(payload),
        seal["seal_sha256"],
    )
    lineage_aggregates = dict(payload["canonical_lineage_aggregates"])

    if expected_aggregates:
        for stage, expected in expected_aggregates.items():
            report.add(
                f"anchor_{stage}",
                lineage_aggregates.get(stage) == expected,
                f"sealed aggregate for {stage} matches the independent repository anchor",
                str(lineage_aggregates.get(stage)),
                expected,
            )

    inventory = verify_release_inventory(root, verify_hashes=verify_every_file)
    if expected_inventory:
        report.add(
            "release_inventory_manifest_anchor",
            inventory.manifest_sha256 == expected_inventory["manifest_sha256"],
            "release inventory matches the independent repository anchor",
            inventory.manifest_sha256,
            str(expected_inventory["manifest_sha256"]),
        )
        report.add(
            "release_inventory_declared_shape",
            inventory.row_count == int(expected_inventory["file_count"])
            and inventory.total_bytes == int(expected_inventory["total_bytes"])
            and inventory.aggregate_sha256 == expected_inventory["aggregate_sha256"],
            "release inventory count, bytes and aggregate match independent anchors",
            f"{inventory.row_count}|{inventory.total_bytes}|{inventory.aggregate_sha256}",
            (
                f"{expected_inventory['file_count']}|{expected_inventory['total_bytes']}|"
                f"{expected_inventory['aggregate_sha256']}"
            ),
        )
    report.add(
        "release_inventory_exact_file_set",
        not inventory.missing and not inventory.extra,
        "actual frozen file set equals the manifest-declared file set",
        f"missing={len(inventory.missing)};extra={len(inventory.extra)}",
        "missing=0;extra=0",
    )
    report.add(
        "release_inventory_file_integrity",
        not inventory.mismatches,
        "every inventory row matches by size and, when enabled, SHA-256",
        f"mismatches={len(inventory.mismatches)}",
        "mismatches=0",
    )
    report.file_mismatches.extend(f"release_inventory:{item}" for item in inventory.missing)
    report.file_mismatches.extend(f"release_inventory:extra:{item}" for item in inventory.extra)
    report.file_mismatches.extend(f"release_inventory:{item}" for item in inventory.mismatches)

    lineage = _load_json(branch, LINEAGE_RELATIVE_PATH)
    lineage_stage_rows = {row["stage"]: row for row in lineage["stages"]}

    for stage, manifest_relative in CANONICAL_STAGES.items():
        manifest = _load_json(branch, manifest_relative)
        rows = _manifest_rows(manifest)
        observed = aggregate_rows(rows)
        expected = lineage_aggregates[stage]
        report.add(
            f"aggregate_{stage}",
            observed == expected,
            f"{stage} manifest rows reaggregate to the sealed value",
            observed,
            expected,
        )
        lineage_row = lineage_stage_rows.get(stage)
        if lineage_row is not None:
            report.add(
                f"lineage_path_{stage}",
                lineage_row["manifest_relative_path"] == manifest_relative,
                f"{stage} lineage manifest path matches the canonical stage map",
                lineage_row["manifest_relative_path"],
                manifest_relative,
            )
        if stage == HISTORICAL_FAILURE_STAGE:
            declared = frozenset(lineage_row["superseded_shared_paths"]) if lineage_row else frozenset()
            report.add(
                "historical_failure_superseded_paths",
                declared == SUPERSEDED_SHARED_PATHS,
                "the archived FS2 failure declares exactly the two superseded shared paths",
                ",".join(sorted(declared)),
                ",".join(sorted(SUPERSEDED_SHARED_PATHS)),
            )
            archive = _load_json(branch, SUPERSESSION_EVIDENCE)
            for item in archive["archived_files"]:
                relative_path = str(item["relative_path"])
                path = branch.path(relative_path)
                if not path.is_file() or path.stat().st_size != int(item["size_bytes"]):
                    report.file_mismatches.append(f"{stage}:archive_size_or_missing:{relative_path}")
                elif sha256_file(path) != item["sha256"]:
                    report.file_mismatches.append(f"{stage}:archive_sha256:{relative_path}")

        if not verify_every_file:
            continue
        for row in rows:
            relative_path = str(row["relative_path"])
            if stage == HISTORICAL_FAILURE_STAGE and relative_path in SUPERSEDED_SHARED_PATHS:
                # Verified through the archive manifest above; the live file is the
                # documented post-remediation version by design.
                continue
            path = branch.path(relative_path)
            if not path.is_file():
                report.file_mismatches.append(f"{stage}:missing:{relative_path}")
                continue
            if path.stat().st_size != int(row["size_bytes"]):
                report.file_mismatches.append(f"{stage}:size:{relative_path}")
                continue
            if sha256_file(path) != row["sha256"]:
                report.file_mismatches.append(f"{stage}:sha256:{relative_path}")

    for stage, gate_relative in PREDICTION_GATES.items():
        gate = _load_json(branch, gate_relative)
        observed = aggregate_rows(gate["stage_a_files"])
        expected = lineage_aggregates[f"{stage}_PREDICTIONS"]
        report.add(
            f"prediction_aggregate_{stage}",
            observed == expected and gate["prediction_aggregate_sha256"] == expected,
            f"{stage} Stage-A prediction aggregate reaggregates to the sealed value",
            observed,
            expected,
        )

    final_manifest = _load_json(branch, FINAL_ARTIFACT_HASHES)
    final_rows = _manifest_rows(final_manifest)
    observed = aggregate_rows(final_rows)
    report.add(
        "final_artifact_aggregate",
        observed == final_manifest["aggregate_fsa_sha256"] == payload["final_artifact_aggregate_sha256"],
        "final audit artifact aggregate recomputes and matches the seal",
        observed,
        str(payload["final_artifact_aggregate_sha256"]),
    )
    if verify_every_file:
        for row in final_rows:
            relative_path = str(row["relative_path"])
            path = branch.path(relative_path)
            if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
                report.file_mismatches.append(f"FSA:size_or_missing:{relative_path}")
                continue
            if sha256_file(path) != row["sha256"]:
                report.file_mismatches.append(f"FSA:sha256:{relative_path}")

    for relative_path in sorted(SEALED_RELATIVE_PATHS):
        report.add(
            f"sealed_file_present_{Path(relative_path).name}",
            branch.path(relative_path).is_file(),
            "the sealed query-label file is present and was reduced to a digest only",
        )

    status = _load_json(branch, STATUS_RELATIVE_PATH)
    report.add(
        "status_classification",
        status["classification"] == payload["classification"],
        "final status classification matches the seal payload",
        status["classification"],
        payload["classification"],
    )
    report.add(
        "project_permanently_closed",
        bool(status["project_permanently_closed"]) and bool(payload["project_permanently_closed"]),
        "the authoritative branch is recorded as permanently closed",
        str(status["project_permanently_closed"]),
        "True",
    )
    return report


def is_bytecode_cache(relative_path: str) -> bool:
    """Return whether a path is a Python bytecode cache rather than scientific content."""

    parts = relative_path.split("/")
    return "__pycache__" in parts and parts[-1].endswith(".pyc")


def verify_preaudit_baseline(
    root: str | Path | None = None,
) -> tuple[int, list[str], list[str]]:
    """Recheck every pre-audit baseline row by size and SHA-256.

    Modification time is deliberately not compared: a migrated copy has new
    filesystem timestamps while its bytes are unchanged.

    Returns ``(row_count, mismatches, absent_bytecode_caches)``. Bytecode caches
    are reported separately rather than silently ignored: they are deterministic
    interpreter-version-specific derivatives of preserved sources, carry no
    scientific content and were already absent from the authoritative source at
    migration time.
    """

    branch = FrozenBranch.resolve(root)
    baseline = _load_json(branch, PREAUDIT_BASELINE)
    mismatches: list[str] = []
    absent_bytecode: list[str] = []
    for row in baseline["files"]:
        relative_path = str(row["relative_path"])
        path = branch.path(relative_path)
        if not path.is_file():
            if is_bytecode_cache(relative_path):
                absent_bytecode.append(relative_path)
            else:
                mismatches.append(f"missing:{relative_path}")
            continue
        if path.stat().st_size != int(row["size_bytes"]):
            mismatches.append(f"size:{relative_path}")
            continue
        if sha256_file(path) != row["sha256"]:
            mismatches.append(f"sha256:{relative_path}")
    return len(baseline["files"]), mismatches, absent_bytecode
