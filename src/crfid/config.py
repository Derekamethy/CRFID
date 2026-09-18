"""Configuration loading, environment substitution and validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from .exceptions import ConfigurationError


_ENV_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)(?::-([^}]*))?\}")
_WORKFLOWS = {
    "prepare_data",
    "baseline",
    "strict_dg",
    "target_assisted",
    "failure_analysis",
    "few_shot",
    "external_data1",
    "openems_redesign",
}


def _substitute_env(text: str, environment: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in environment:
            return environment[name]
        if default is not None:
            return default
        raise ConfigurationError(f"Required environment variable is unset: {name}")

    return _ENV_PATTERN.sub(replace, text)


def _substitute_env_values(value: Any, environment: Mapping[str, str]) -> Any:
    """Substitute placeholders after parsing so Windows paths remain literal."""

    if isinstance(value, dict):
        return {
            key: _substitute_env_values(child, environment)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_substitute_env_values(child, environment) for child in value]
    if isinstance(value, str):
        return _substitute_env(value, environment)
    return value


def _parse_yaml_or_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as json_error:
        try:
            import yaml  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ConfigurationError(
                "Configuration is not JSON-compatible YAML and PyYAML is unavailable"
            ) from json_error
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:  # type: ignore[attr-defined]
            raise ConfigurationError(f"Invalid YAML: {exc}") from exc


def load_config(
    path: str | Path, *, environment: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Load a configuration only when explicitly called."""

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    parsed = _parse_yaml_or_json(config_path.read_text(encoding="utf-8"))
    payload = _substitute_env_values(
        parsed, environment if environment is not None else os.environ
    )
    if not isinstance(payload, dict):
        raise ConfigurationError("Top-level configuration must be a mapping")
    validate_config(payload)
    return payload


def _validate_paths(value: Any, key: str = "") -> None:
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            _validate_paths(child_value, str(child_key))
    elif isinstance(value, list):
        for child in value:
            _validate_paths(child, key)
    elif isinstance(value, str) and (key.endswith("_path") or key.endswith("_root")):
        if not value or "\x00" in value:
            raise ConfigurationError(f"Configured path is empty or invalid: {key}")


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate common and workflow-specific configuration contracts."""

    required = {"schema_version", "workflow", "status", "random_seeds", "output_path"}
    missing = sorted(required.difference(config))
    if missing:
        raise ConfigurationError(f"Missing configuration fields: {missing}")
    if config["schema_version"] != 1:
        raise ConfigurationError("Unsupported configuration schema version")
    workflow = str(config["workflow"])
    if workflow not in _WORKFLOWS:
        raise ConfigurationError(f"Unknown workflow: {workflow}")
    seeds = config["random_seeds"]
    if not isinstance(seeds, list) or not seeds or any(
        not isinstance(seed, int) or seed < 0 for seed in seeds
    ):
        raise ConfigurationError("random_seeds must be a non-empty list of integers")
    _validate_paths(dict(config))
    if workflow == "strict_dg":
        access = config.get("domain_access", {})
        if access.get("development_domains") != ["P1", "P2", "P3"]:
            raise ConfigurationError("Strict DG development domains must be P1, P2 and P3")
        if access.get("target_domain") != "P4" or access.get("target_before_freeze") is not False:
            raise ConfigurationError("Strict DG must forbid P4 before recipe freeze")
    if workflow == "few_shot" and config.get("shot_counts") != [1, 3, 5]:
        raise ConfigurationError("Few-shot configuration must declare 1, 3 and 5 shots")
    if workflow == "external_data1" and config.get("class_order") != [0, 1, 2, 3]:
        raise ConfigurationError("Data1 class order must be [0, 1, 2, 3]")
    if workflow == "openems_redesign" and config.get("execute_solver_by_default") is not False:
        raise ConfigurationError("OpenEMS execution must be disabled by default")


def config_digest(config: Mapping[str, Any]) -> str:
    """Return the SHA-256 of a canonical configuration representation."""

    encoded = json.dumps(dict(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
