"""Deterministic, reviewable release-manifest generation and verification.

This module exists because the historical Strict-DG frozen-release manifests
could not be shown to have been produced by any committed code. Everything here
is explicit, versioned and re-runnable: a manifest can always be regenerated
from a declared root and verified without being rewritten.

The module performs no scientific computation and reads no data at import time.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..exceptions import ProtocolViolation


MANIFEST_SCHEMA_VERSION = 1
GENERATOR_NAME = "crfid.governance.strict_release_manifest"
GENERATOR_VERSION = "1.0.0"
MANIFEST_FIELDNAMES = ("relative_path", "size_bytes", "sha256")
MANIFEST_KINDS = ("code", "config", "artifact", "dependency")


class ManifestError(ProtocolViolation):
    """Raised when a manifest cannot be generated or fails verification."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_relative(relative_path: str) -> str:
    """Reject absolute, drive-qualified, traversing or backslash paths."""

    if not relative_path:
        raise ManifestError("Manifest entries require a non-empty relative path")
    if relative_path != relative_path.strip():
        raise ManifestError(f"Manifest path has surrounding whitespace: {relative_path!r}")
    if "\\" in relative_path:
        raise ManifestError(f"Manifest paths must use forward slashes: {relative_path}")
    candidate = Path(relative_path)
    if candidate.is_absolute() or candidate.drive or relative_path.startswith("/"):
        raise ManifestError(f"Manifest paths must be repository-relative: {relative_path}")
    if ".." in candidate.parts:
        raise ManifestError(f"Manifest paths must not traverse upwards: {relative_path}")
    return candidate.as_posix()


def require_relative_output(root: Path, output_path: Path) -> Path:
    """Reject an output path that escapes the declared root."""

    resolved_root = Path(root).resolve()
    resolved = Path(output_path)
    if resolved.is_absolute():
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise ManifestError(
                f"Manifest output must stay inside the declared root: {output_path}"
            ) from exc
        return resolved
    return (resolved_root / resolved).resolve()


@dataclass(frozen=True)
class ManifestSpec:
    """A named group of repository-relative paths to describe."""

    kind: str
    relative_paths: tuple[str, ...]
    description: str = ""
    managed_directories: tuple[str, ...] = field(default=())
    #: Sub-trees inside a managed directory that the manifest deliberately does
    #: not describe, such as the release directory holding the manifests
    #: themselves. Excluding them is declared, never implicit.
    excluded_prefixes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.kind not in MANIFEST_KINDS:
            raise ManifestError(f"Unknown manifest kind: {self.kind}")


def build_rows(root: Path, relative_paths: Iterable[str]) -> list[dict[str, Any]]:
    """Hash every declared path. A missing path is an error, never a skip."""

    resolved_root = Path(root).resolve()
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    seen: set[str] = set()
    for raw in relative_paths:
        relative = require_relative(raw)
        if relative in seen:
            raise ManifestError(f"Duplicate manifest entry: {relative}")
        seen.add(relative)
        target = resolved_root / relative
        if not target.is_file():
            missing.append(relative)
            continue
        rows.append(
            {
                "relative_path": relative,
                "size_bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    if missing:
        raise ManifestError(f"Manifest declares files that do not exist: {sorted(missing)}")
    rows.sort(key=lambda row: row["relative_path"])
    return rows


def write_manifest_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    """Write a deterministic UTF-8 CSV: no BOM, LF endings, minimal quoting."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(MANIFEST_FIELDNAMES), lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in MANIFEST_FIELDNAMES})
    return sha256_file(target)


def read_manifest_csv(path: Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        raise ManifestError(f"Manifest is missing: {target}")
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ManifestError(f"Manifest is empty: {target}")
    for row in rows:
        if set(MANIFEST_FIELDNAMES) - set(row):
            raise ManifestError(f"Manifest is missing required columns: {target}")
        require_relative(row["relative_path"])
    return rows


def unexpected_files(
    root: Path,
    managed_directories: Iterable[str],
    declared: Iterable[str],
    excluded_prefixes: Iterable[str] = (),
) -> list[str]:
    """Return files inside managed directories that no manifest declares."""

    resolved_root = Path(root).resolve()
    known = {require_relative(item) for item in declared}
    excluded = tuple(require_relative(item).rstrip("/") + "/" for item in excluded_prefixes)
    found: list[str] = []
    for directory in managed_directories:
        base = resolved_root / require_relative(directory)
        if not base.is_dir():
            continue
        for candidate in sorted(base.rglob("*")):
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(resolved_root).as_posix()
            if relative in known or relative.startswith(excluded):
                continue
            found.append(relative)
    return sorted(found)


def verify_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify a manifest without rewriting it."""

    resolved_root = Path(root).resolve()
    rows = read_manifest_csv(manifest_path)
    matched: list[str] = []
    mismatched: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in rows:
        relative = row["relative_path"]
        target = resolved_root / relative
        if not target.is_file():
            missing.append(relative)
            continue
        observed = sha256_file(target)
        observed_size = target.stat().st_size
        if observed == row["sha256"] and str(observed_size) == str(row["size_bytes"]):
            matched.append(relative)
        else:
            mismatched.append(
                {
                    "relative_path": relative,
                    "declared_sha256": row["sha256"],
                    "observed_sha256": observed,
                    "declared_size_bytes": int(row["size_bytes"]),
                    "observed_size_bytes": observed_size,
                }
            )
    return {
        "manifest": Path(manifest_path).name,
        "manifest_sha256": sha256_file(manifest_path),
        "entries": len(rows),
        "matched": len(matched),
        "mismatched": mismatched,
        "missing": missing,
        "passed": not mismatched and not missing,
    }


def generate_release_manifests(
    *,
    root: Path,
    release_directory: Path,
    specs: Mapping[str, ManifestSpec],
    release_id: str,
    allow_overwrite: bool = False,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate every declared manifest into a new versioned release directory.

    Refuses to overwrite an existing release unless explicitly allowed, which is
    the write-protection behaviour required for canonical frozen releases.
    """

    resolved_root = Path(root).resolve()
    release = require_relative_output(resolved_root, release_directory)
    existing_index = release / "manifest_verification.json"
    if existing_index.exists() and not allow_overwrite:
        raise ManifestError(
            f"Release already exists and would be overwritten: {release}. "
            "Use a new release identifier, or pass an explicit overwrite request "
            "together with a new release identifier."
        )
    release.mkdir(parents=True, exist_ok=True)

    generated: dict[str, Any] = {}
    all_declared: list[str] = []
    managed: list[str] = []
    excluded: list[str] = []
    for name, spec in specs.items():
        excluded.extend(spec.excluded_prefixes)
        rows = build_rows(resolved_root, spec.relative_paths)
        target = release / f"{name}.csv"
        digest = write_manifest_csv(target, rows)
        generated[name] = {
            "kind": spec.kind,
            "description": spec.description,
            "file": target.relative_to(resolved_root).as_posix(),
            "entries": len(rows),
            "manifest_sha256": digest,
        }
        all_declared.extend(row["relative_path"] for row in rows)
        managed.extend(spec.managed_directories)

    unexpected = unexpected_files(
        resolved_root, sorted(set(managed)), all_declared, sorted(set(excluded))
    )

    index = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "generator_name": GENERATOR_NAME,
        "generator_version": GENERATOR_VERSION,
        "release_id": release_id,
        "manifest_generated_at_utc": utc_now(),
        "scientific_execution_time": "NOT_APPLICABLE_MANIFEST_GENERATION_ONLY",
        "note": (
            "Manifest generation time is recorded separately from scientific execution "
            "time. Generating this manifest performs no training, no inference and no "
            "modification of any scientific artifact."
        ),
        "root_relative_release_directory": release.relative_to(resolved_root).as_posix(),
        "manifests": generated,
        "declared_entry_count": len(all_declared),
        "managed_directories": sorted(set(managed)),
        "declared_excluded_prefixes": sorted(set(excluded)),
        "unexpected_files_in_managed_directories": unexpected,
        "unexpected_file_count": len(unexpected),
        "environment": dict(environment or {}),
    }
    index["index_semantic_sha256"] = canonical_json_sha256(
        {k: v for k, v in index.items() if k != "manifest_generated_at_utc"}
    )
    (release / "manifest_verification.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return index


def verify_release_manifests(*, root: Path, release_directory: Path) -> dict[str, Any]:
    """Re-verify a generated release without rewriting anything."""

    resolved_root = Path(root).resolve()
    release = require_relative_output(resolved_root, release_directory)
    index_path = release / "manifest_verification.json"
    if not index_path.is_file():
        raise ManifestError(f"No release index found at {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))

    results = {}
    declared: list[str] = []
    for name, record in index["manifests"].items():
        manifest_path = resolved_root / record["file"]
        observed_manifest_digest = sha256_file(manifest_path)
        result = verify_manifest(resolved_root, manifest_path)
        result["manifest_file_unchanged"] = observed_manifest_digest == record["manifest_sha256"]
        result["passed"] = result["passed"] and result["manifest_file_unchanged"]
        results[name] = result
        declared.extend(row["relative_path"] for row in read_manifest_csv(manifest_path))

    unexpected = unexpected_files(
        resolved_root,
        index.get("managed_directories", []),
        declared,
        index.get("declared_excluded_prefixes", []),
    )
    report = {
        "manifest_schema_version": index["manifest_schema_version"],
        "release_id": index["release_id"],
        "verified_at_utc": utc_now(),
        "results": results,
        "unexpected_files_in_managed_directories": unexpected,
        "all_manifests_passed": all(item["passed"] for item in results.values()),
        "no_unexpected_files": not unexpected,
    }
    report["verification_passed"] = report["all_manifests_passed"] and report["no_unexpected_files"]
    return report


def environment_identity() -> dict[str, Any]:
    """Best-effort, import-safe record of the numerical dependency identity."""

    import platform
    import sys

    identity: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "architecture_bits": 64 if sys.maxsize > 2**32 else 32,
        "platform": platform.system(),
        "interpreter_basename": os.path.basename(sys.executable),
    }
    for name in ("torch", "numpy", "scipy", "sklearn", "pandas", "yaml"):
        try:
            module = __import__(name)
        except Exception:  # noqa: BLE001 - identity recording must not fail the run
            identity[f"{name}_version"] = "NOT_IMPORTABLE"
            continue
        identity[f"{name}_version"] = getattr(module, "__version__", "UNKNOWN")
    try:
        import torch

        identity["torch_cuda_version"] = torch.version.cuda
        identity["torch_cuda_available"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        identity["torch_cuda_version"] = "NOT_IMPORTABLE"
        identity["torch_cuda_available"] = "NOT_IMPORTABLE"
    return identity
