"""Hashing, canonical JSON and run-record helpers for the branch."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .paths import PROJECT_ROOT, ensure_branch_output


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def array_sha256(values: np.ndarray) -> str:
    """Hash an array by dtype, shape and C-ordered bytes."""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> Path:
    """Write canonical, sorted JSON inside the branch output tree."""

    target = ensure_branch_output(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return target


def write_csv(path: str | Path, header: list[str], rows: list[list[Any]]) -> Path:
    import csv

    target = ensure_branch_output(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    return target


def write_text(path: str | Path, text: str) -> Path:
    target = ensure_branch_output(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def environment_identity() -> dict[str, Any]:
    """Record the executing interpreter and the scientific dependency versions."""

    versions: dict[str, str] = {}
    for name in ("numpy", "torch", "scipy", "sklearn", "pandas"):
        try:
            module = __import__(name)
        except ModuleNotFoundError:
            versions[name] = "ABSENT"
        else:
            versions[name] = str(getattr(module, "__version__", "UNKNOWN"))
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
    }


def code_identity() -> dict[str, Any]:
    """Hash every module of this branch so a run is bound to its own code."""

    package = Path(__file__).resolve().parent
    entries = {}
    for module in sorted(package.glob("*.py")):
        entries[module.name] = sha256_file(module)
    return {
        "module_count": len(entries),
        "module_sha256": entries,
        "package_sha256": canonical_json_sha256(entries),
    }


def relative_to_project(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


@dataclass
class RunRecord:
    """Structured provenance for one executed unit of work."""

    run_id: str
    stage: str
    configuration: dict[str, Any]
    started_at_utc: str = field(default_factory=utc_now)
    completed_at_utc: str | None = None
    status: str = "STARTED"
    failure_reason: str | None = None
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)

    def finish(self, status: str, *, failure_reason: str | None = None) -> "RunRecord":
        self.completed_at_utc = utc_now()
        self.status = status
        self.failure_reason = failure_reason
        return self

    def as_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "configuration": self.configuration,
            "configuration_sha256": canonical_json_sha256(self.configuration),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metrics": self.metrics,
            "resources": self.resources,
            "environment": environment_identity(),
            "code": code_identity(),
        }


def git_is_untouched(paths: list[str]) -> dict[str, Any]:
    """Report whether protected paths differ from HEAD, without running mutations."""

    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--"] + paths,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:  # pragma: no cover - environment dependent
        return {"available": False, "reason": str(exc)}
    return {
        "available": completed.returncode == 0,
        "dirty_entries": [line for line in completed.stdout.splitlines() if line.strip()],
    }
