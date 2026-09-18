"""Data validation and portable prepared-artifact construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..data.validation import validate_supervised_arrays
from ..preprocessing.resampling import resample_signal, uniform_grid
from .common import WorkflowPlan, build_plan, load_npz


def run(config: dict[str, Any], *, execute: bool = False) -> WorkflowPlan | dict[str, Any]:
    plan = build_plan(config, "DATA_PREPARATION_AND_VALIDATION", execute)
    if not execute:
        return plan
    required = tuple(config.get("required_arrays", ["signals", "labels"]))
    payload = load_npz(str(config["input_path"]), required)
    signals, labels = validate_supervised_arrays(
        payload["signals"], payload["labels"], signal_length=int(config["signal_length"]), class_order=config["class_order"]
    )
    output: dict[str, np.ndarray] = {"signals": signals, "labels": labels}
    if "domains" in payload:
        domains = payload["domains"].astype(str)
        if domains.shape != labels.shape or set(domains.tolist()).difference(config["included_domains"]):
            raise ValueError("Domain metadata is not aligned or contains a disallowed domain")
        output["domains"] = domains
    if "conditions" in payload:
        conditions = payload["conditions"].astype(str)
        if conditions.shape != labels.shape or any(not item for item in conditions.tolist()):
            raise ValueError("Condition metadata is not aligned or contains an empty identifier")
        output["conditions"] = conditions
    resampling = config.get("resampling")
    if resampling is not None:
        source_axis = payload.get("frequency_axis")
        if source_axis is None:
            raise ValueError("Physical resampling requires a frequency_axis array")
        target_axis = uniform_grid(
            float(resampling["target_start_ghz"]),
            float(resampling["target_stop_ghz"]),
            int(resampling["target_length"]),
        )
        output["signals"] = np.stack(
            [resample_signal(signal, source_axis, target_axis) for signal in signals]
        )
        output["frequency_axis"] = target_axis
    destination = Path(str(config["output_path"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **output)
    return {
        "status": "PREPARED",
        "sample_count": int(len(labels)),
        "signal_length": int(output["signals"].shape[1]),
        "output_path": str(destination),
    }
