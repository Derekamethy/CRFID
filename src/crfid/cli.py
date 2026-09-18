"""Command-line interface for validation, planning and explicit execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from .config import load_config
from .workflows import baseline, external_data1, failure_analysis, few_shot, openems_redesign, prepare_data, strict_dg, target_assisted


RUNNERS: dict[str, Callable[..., Any]] = {
    "prepare_data": prepare_data.run,
    "baseline": baseline.run,
    "strict_dg": strict_dg.run,
    "target_assisted": target_assisted.run,
    "failure_analysis": failure_analysis.run,
    "few_shot": few_shot.run,
    "external_data1": external_data1.run,
    "openems_redesign": openems_redesign.run,
}


def run_config(path: str | Path, *, execute: bool = False) -> Any:
    config = load_config(path)
    result = RUNNERS[str(config["workflow"])](config, execute=execute)
    return result.as_dict() if hasattr(result, "as_dict") else result


def workflow_entrypoint(expected_workflow: str) -> int:
    parser = argparse.ArgumentParser(description=f"CRFID {expected_workflow} workflow")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--execute", action="store_true", help="Perform the configured scientific operation")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if config["workflow"] != expected_workflow:
        parser.error(f"Expected workflow {expected_workflow}, found {config['workflow']}")
    result = RUNNERS[expected_workflow](config, execute=arguments.execute)
    payload = result.as_dict() if hasattr(result, "as_dict") else result
    print(json.dumps(payload, indent=2, default=lambda value: value.tolist()))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crfid")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate-config")
    validate_parser.add_argument("paths", nargs="+", type=Path)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.command == "validate-config":
        for path in arguments.paths:
            load_config(path)
        return 0
    payload = run_config(arguments.config, execute=arguments.execute)
    print(json.dumps(payload, indent=2, default=lambda value: value.tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
