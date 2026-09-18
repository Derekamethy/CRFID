from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from crfid.config import load_config
from crfid.workflows import target_assisted as workflow


MODES = (
    "full",
    "validate_source_release",
    "validate_target_inputs",
    "build_target_partitions",
    "prepare_target_assisted_method",
    "execute_target_adaptation",
    "generate_target_predictions",
    "calculate_target_metrics",
    "aggregate_target_results",
    "reproduce_historical_selection",
    "compare_with_authoritative_outputs",
    "generate_target_assisted_report",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="P4 Target-Assisted Adaptation")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=MODES, required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    mode = arguments.mode
    if mode == "full":
        result = workflow.run(config, execute=True)
    elif mode in {"validate_source_release", "validate_target_inputs", "build_target_partitions", "prepare_target_assisted_method", "reproduce_historical_selection"}:
        result = getattr(workflow, mode)(config)
    else:
        execution = workflow.execute_target_adaptation(config)
        if mode == "execute_target_adaptation":
            result = {key: list(value.shape) for key, value in execution.items()}
        elif mode == "generate_target_predictions":
            result = workflow.generate_target_predictions(config, execution)
        else:
            metrics = workflow.calculate_target_metrics(config, execution["predictions"])
            if mode == "calculate_target_metrics":
                result = metrics
            elif mode == "aggregate_target_results":
                result = workflow.aggregate_target_results(metrics)
            else:
                identities = workflow.generate_target_predictions(config, execution)
                comparison = workflow.compare_with_authoritative_outputs(config, metrics, identities)
                result = comparison if mode == "compare_with_authoritative_outputs" else workflow.generate_target_assisted_report(config, metrics, comparison)
    print(json.dumps(result, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
