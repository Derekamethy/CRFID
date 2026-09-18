from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import site
import struct
import sys
from pathlib import Path

DEFAULT_CONFIG = Path(os.environ.get("CRFID_ENVIRONMENT_CONFIG", Path.home() / ".crfid" / "environment.json"))
DEFAULT_PROJECT_LOCK = Path(__file__).resolve().parents[1] / "environment" / "CRFID_ENVIRONMENT_LOCK.json"


def fail(message: str) -> None:
    print(f"FAIL_CRFID_LOCKED_ENVIRONMENT: {message}", file=sys.stderr)
    raise SystemExit(1)


def normalized(path: str | Path) -> str:
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def inside(path: str | Path, root: str | Path) -> bool:
    try:
        return os.path.commonpath([normalized(path), normalized(root)]) == normalized(root)
    except ValueError:
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        fail(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"invalid {label} {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} is not a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--project-lock", type=Path, default=DEFAULT_PROJECT_LOCK)
    args = parser.parse_args()

    project_lock = load_json(args.project_lock, "project environment lock")
    config = load_json(args.config, "global CRFID config")
    canonical_python = Path(config.get("python_executable", ""))
    canonical_root = Path(config.get("environment_root", ""))

    if normalized(sys.executable) != normalized(canonical_python):
        fail(f"wrong interpreter: {sys.executable}; required governed interpreter")
    if normalized(sys.prefix) != normalized(canonical_root):
        fail(f"wrong sys.prefix: {sys.prefix}")

    activated = os.environ.get("VIRTUAL_ENV")
    if activated and normalized(activated) != normalized(canonical_root):
        fail(f"another virtual environment is activated: {activated}")
    if os.environ.get("PYTHONPATH"):
        fail(f"PYTHONPATH contamination: {os.environ['PYTHONPATH']}")

    expected_id = "CRFID_STRICT_DG_WINDOWS_CPU_V1"
    if project_lock.get("environment_id") != expected_id or config.get("environment_id") != expected_id:
        fail("environment ID mismatch")
    if project_lock.get("status") != "LOCKED" or config.get("status") != "LOCKED":
        fail("environment is not marked LOCKED")
    if project_lock.get("fallback_allowed") is not False or config.get("fallback_allowed") is not False:
        fail("fallback must be disabled")

    if sha256(canonical_python) != project_lock.get("python_executable_sha256"):
        fail("canonical interpreter hash mismatch")

    if args.config == DEFAULT_CONFIG and sha256(args.config) != project_lock.get("global_config_sha256"):
        fail("global CRFID environment config hash mismatch")
    if args.config != DEFAULT_CONFIG and sha256(args.config) != project_lock.get("global_config_sha256"):
        fail("supplied CRFID environment config hash mismatch")

    for key in ("lock_file", "freeze_file"):
        locked_path = Path(project_lock.get(key, ""))
        if not locked_path.is_absolute():
            locked_path = args.project_lock.resolve().parents[1] / locked_path
        if not locked_path.is_file():
            fail(f"missing {key}: {locked_path}")
        if sha256(locked_path) != project_lock.get(f"{key}_sha256"):
            fail(f"{key} hash mismatch")
    if sha256(Path(config.get("lock_file", ""))) != project_lock.get("lock_file_sha256"):
        fail("global config lock-file identity mismatch")

    if platform.python_implementation() != project_lock.get("python_implementation"):
        fail("Python implementation mismatch")
    if platform.python_version() != project_lock.get("python_version"):
        fail(f"Python version mismatch: {platform.python_version()}")
    if struct.calcsize("P") * 8 != project_lock.get("architecture_bits"):
        fail("Python architecture mismatch")
    if normalized(sys.base_prefix) != normalized(config.get("base_python_root", "")):
        fail(f"unexpected sys.base_prefix: {sys.base_prefix}")
    base_python = Path(config.get("base_python_executable", ""))
    if not base_python.is_file():
        fail(f"permanent base Python is missing: {base_python}")
    if sha256(base_python) != project_lock.get("base_python_sha256"):
        fail("permanent base Python hash mismatch")
    if config.get("base_python_sha256") != project_lock.get("base_python_sha256"):
        fail("global config permanent base hash mismatch")

    user_site = site.getusersitepackages()
    user_sites = [user_site] if isinstance(user_site, str) else list(user_site)
    for entry in sys.path:
        if entry and any(normalized(entry) == normalized(candidate) for candidate in user_sites):
            fail(f"user site-packages is present on sys.path: {entry}")

    versions = project_lock.get("versions", {})
    imported = {}
    for name in project_lock.get("required_imports", []):
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            fail(f"required import failed for {name}: {type(exc).__name__}: {exc}")
        module_version = getattr(module, "__version__", None)
        if module_version != versions.get(name):
            fail(f"{name} version mismatch: {module_version} != {versions.get(name)}")
        module_file = getattr(module, "__file__", None)
        if not module_file or not inside(module_file, canonical_root):
            fail(f"{name} imported outside canonical environment: {module_file}")
        imported[name] = module

    torch = imported["torch"]
    if project_lock.get("compute_platform") != "cpu" or config.get("compute_platform") != "cpu":
        fail("compute platform lock mismatch")
    if torch.version.cuda is not None or torch.cuda.is_available():
        fail("CPU-only torch lock violated")
    probe = torch.tensor([1.0, 2.0], device="cpu") * 2.0
    if probe.tolist() != [2.0, 4.0]:
        fail("CPU tensor microcheck failed")

    print("PASS_CRFID_LOCKED_ENVIRONMENT")


if __name__ == "__main__":
    main()
