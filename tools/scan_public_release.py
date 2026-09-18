#!/usr/bin/env python3
"""Deterministic secret, private-path, artifact, and large-file release scan."""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs" / "public_release_scan"
GENERATED_REPORTS = {
    "outputs/public_release_scan/secret_scan.json",
    "outputs/public_release_scan/private_data_scan.json",
    "outputs/public_release_scan/large_file_report.csv",
    "outputs/public_release_scan/github_size_report.json",
}
BANNED_SUFFIXES = {
    ".7z", ".arrow", ".bundle", ".ckpt", ".feather", ".h5", ".hdf5",
    ".joblib", ".key", ".log", ".npy", ".npz", ".onnx", ".p12", ".parquet",
    ".pem", ".pfx", ".pickle", ".pkl", ".pt", ".pth", ".pyc", ".tar", ".tgz", ".zip",
}
BANNED_DIRS = {
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__",
    "checkpoints",
}
SCIENTIFIC_BINARY_SUFFIXES = {".npy", ".npz", ".pt", ".pth"}
ALLOWED_SCIENTIFIC_BINARY_PREFIXES = (
    "results/canonical_metrics/",
    "06_failure_mechanism/composition_warp/",
)
TEXT_LIMIT = 10 * 1024 * 1024
LARGE_REPORT_THRESHOLD = 1024 * 1024
GITHUB_WARNING_THRESHOLD = 50 * 1024 * 1024
GITHUB_BLOCK_THRESHOLD = 100 * 1024 * 1024


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def candidate_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.relative_to(ROOT).parts:
            continue
        if rel(path) in GENERATED_REPORTS:
            continue
        files.append(path)
    return sorted(files, key=lambda item: rel(item).casefold())


def readable_text(path: Path) -> str | None:
    if path.stat().st_size > TEXT_LIMIT:
        return None
    data = path.read_bytes()
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = {char: value.count(char) for char in set(value)}
    return -sum((count / len(value)) * math.log2(count / len(value)) for count in counts.values())


def main() -> int:
    files = candidate_files()
    secret_findings: list[dict[str, object]] = []
    path_findings: list[dict[str, object]] = []
    email_findings: list[dict[str, object]] = []
    artifact_findings: list[dict[str, object]] = []
    large_rows: list[dict[str, object]] = []

    high_confidence = {
        "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
        "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
        "credential_url": re.compile(r"[a-z][a-z0-9+.-]*://[^\s/:]+:[^\s/@]+@", re.I),
    }
    private_key_markers = tuple(
        "-----BEGIN " + kind + "-----"
        for kind in ("PRIVATE KEY", "RSA PRIVATE KEY", "EC PRIVATE KEY", "OPENSSH PRIVATE KEY")
    )
    assignment = re.compile(
        r"(?i)(?:api[_-]?key|password|passwd|client[_-]?secret|access[_-]?token|auth[_-]?token)"
        r"\s*[:=]\s*[\"']([^\"']{12,})[\"']"
    )
    windows_path = re.compile(r"(?<![A-Za-z])\b[A-Za-z]:[\\/]")
    posix_home = re.compile("/(?:" + "home|Users" + r")/[^/\s\"']+")
    unc_path = re.compile(r"(?:^|[\"'(:=\s])\\\\[A-Za-z0-9._-]+\\[A-Za-z0-9$._-]+")
    email = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")

    total_bytes = 0
    for path in files:
        relative = rel(path)
        size = path.stat().st_size
        total_bytes += size
        suffix = path.suffix.lower()
        parts_lower = {part.lower() for part in path.relative_to(ROOT).parts}
        approved_scientific_binary = (
            suffix in SCIENTIFIC_BINARY_SUFFIXES
            and any(relative.startswith(prefix) for prefix in ALLOWED_SCIENTIFIC_BINARY_PREFIXES)
        )
        banned_directory_hit = bool(parts_lower & BANNED_DIRS)
        if approved_scientific_binary and "checkpoints" in parts_lower:
            banned_directory_hit = bool((parts_lower & BANNED_DIRS) - {"checkpoints"})
        if (suffix in BANNED_SUFFIXES and not approved_scientific_binary) or banned_directory_hit:
            artifact_findings.append({"path": relative, "reason": "prohibited artifact type or directory"})
        if path.name.lower() in {".env", "credentials.json", "secrets.json"}:
            artifact_findings.append({"path": relative, "reason": "credential-bearing filename"})
        if size >= LARGE_REPORT_THRESHOLD:
            large_rows.append(
                {
                    "path": relative,
                    "size_bytes": size,
                    "over_50_mib": str(size >= GITHUB_WARNING_THRESHOLD).upper(),
                    "over_100_mib": str(size >= GITHUB_BLOCK_THRESHOLD).upper(),
                }
            )

        text = readable_text(path)
        if text is None:
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            for label, pattern in high_confidence.items():
                if pattern.search(line):
                    secret_findings.append({"path": relative, "line": line_no, "kind": label})
            if any(marker in line for marker in private_key_markers):
                secret_findings.append({"path": relative, "line": line_no, "kind": "private_key"})
            for match in assignment.finditer(line):
                value = match.group(1)
                if not value.startswith("<") and entropy(value) >= 3.5:
                    secret_findings.append(
                        {"path": relative, "line": line_no, "kind": "high_entropy_credential_assignment"}
                    )
            if windows_path.search(line):
                path_findings.append({"path": relative, "line": line_no, "kind": "windows_absolute_path"})
            if posix_home.search(line):
                path_findings.append({"path": relative, "line": line_no, "kind": "posix_user_home"})
            if unc_path.search(line):
                path_findings.append({"path": relative, "line": line_no, "kind": "unc_path"})
            if email.search(line):
                email_findings.append({"path": relative, "line": line_no, "kind": "email_address"})

    REPORTS.mkdir(parents=True, exist_ok=True)
    secret_report = {
        "schema_version": 1,
        "scanner": "built-in high-confidence credential-pattern and entropy scan",
        "files_scanned": len(files),
        "status": "PASS" if not secret_findings else "FAIL",
        "finding_count": len(secret_findings),
        "findings": secret_findings,
        "limitation": "This lightweight scanner is not a substitute for a history-aware enterprise secret scanner.",
    }
    private_findings = path_findings + email_findings + artifact_findings
    private_report = {
        "schema_version": 1,
        "scanner": "built-in path, identifier, extension, cache, and size policy scan",
        "files_scanned": len(files),
        "status": "PASS" if not private_findings else "FAIL",
        "finding_count": len(private_findings),
        "absolute_path_findings": path_findings,
        "email_findings": email_findings,
        "artifact_findings": artifact_findings,
        "policy": (
            "No unapproved raw/processed arrays, model artifacts, archives, caches, credentials, "
            "personal paths, or email addresses. Selected scientific binaries are allowed only "
            "under the declared canonical-result prefixes."
        ),
    }
    size_report = {
        "schema_version": 1,
        "file_count_excluding_git_and_generated_scan_reports": len(files),
        "total_bytes_excluding_git_and_generated_scan_reports": total_bytes,
        "files_at_or_above_1_mib": len(large_rows),
        "files_at_or_above_50_mib": sum(row["over_50_mib"] == "TRUE" for row in large_rows),
        "files_at_or_above_100_mib": sum(row["over_100_mib"] == "TRUE" for row in large_rows),
        "github_policy_result": "PASS" if not any(row["over_100_mib"] == "TRUE" for row in large_rows) else "FAIL",
    }
    (REPORTS / "secret_scan.json").write_bytes(
        (json.dumps(secret_report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    (REPORTS / "private_data_scan.json").write_bytes(
        (json.dumps(private_report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    (REPORTS / "github_size_report.json").write_bytes(
        (json.dumps(size_report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    with (REPORTS / "large_file_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "size_bytes", "over_50_mib", "over_100_mib"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(sorted(large_rows, key=lambda row: (-int(row["size_bytes"]), str(row["path"]))))

    if secret_findings or private_findings:
        print(
            f"PUBLIC RELEASE SCAN FAILED: {len(secret_findings)} secret and "
            f"{len(private_findings)} privacy/artifact findings.",
            file=sys.stderr,
        )
        return 1
    print(
        f"PUBLIC RELEASE SCAN PASSED: {len(files)} files, {total_bytes} bytes, "
        f"{len(large_rows)} files at or above 1 MiB."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
