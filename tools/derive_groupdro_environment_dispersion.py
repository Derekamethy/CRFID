"""Derive an environment-level dispersion diagnostic from frozen GroupDRO results.

Classification: ``POST_HOC_SOURCE_ONLY_DERIVATION_FROM_FROZEN_RESULTS``.

This utility separates two dispersion notions that the historical selector
pooled into one statistic:

* **pooled held-unit dispersion** -- population SD across the 15
  held-position x seed Macro-F1 units. This is the statistic the frozen
  historical eta selector actually consumed at step 4.
* **environment-level dispersion** -- population SD across the three
  held-position mean Macro-F1 values, isolating between-position behaviour.

It reads only already-persisted source-only result files. It never loads a
model, a checkpoint, an embedding, a logit, a prediction array, or any P4
artifact, and it never writes into the canonical result directory. Its output
is a diagnostic and **does not replace the historical selector**: the frozen
selection receipt in
``results/canonical_metrics/groupdro_worst_source/15_SOURCE_ONLY_ETA_SELECTION.md``
remains the sole authority for the historical ``eta = 0.05``.

It is not wired into any workflow, test, or entry point, and nothing runs it
automatically. Manual invocation::

    python tools/derive_groupdro_environment_dispersion.py <repo_root> <output_dir>
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path

POSITIONS = ("P1", "P2", "P3")


def population_sd(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def build_rows(results: Path) -> tuple[list[dict], list[dict]]:
    selection_text = (results / "15_SOURCE_ONLY_ETA_SELECTION.md").read_text(encoding="utf-8")
    match = re.search(r"```json\n(.*)\n```", selection_text, re.S)
    if match is None:
        raise SystemExit("Frozen eta-selection receipt does not contain a JSON block")
    payload = json.loads(match.group(1))

    rows: list[dict] = []
    for summary in payload["all_eta_summaries"]:
        means = [summary["held_position_means"][position] for position in POSITIONS]
        rows.append(
            {
                "scope": "development_eta_grid",
                "method": "groupdro",
                "eta": summary["eta"],
                "pooled_held_unit_population_sd_15_units": summary[
                    "population_sd_held_position_seed_macro_f1"
                ],
                "environment_level_population_sd_3_position_means": population_sd(means),
                "environment_level_max_minus_min_position_mean": max(means) - min(means),
                "held_position_mean_P1": means[0],
                "held_position_mean_P2": means[1],
                "held_position_mean_P3": means[2],
                "source": "15_SOURCE_ONLY_ETA_SELECTION.md",
            }
        )

    with (results / "13_SOURCE_MEAN_AND_VARIANCE_RESULTS.csv").open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            means_map = json.loads(record["held_position_means"])
            means = [means_map[position] for position in POSITIONS]
            rows.append(
                {
                    "scope": "final_source_endpoint",
                    "method": record["method"],
                    "eta": record["eta"],
                    "pooled_held_unit_population_sd_15_units": float(
                        record["held_position_seed_population_sd"]
                    ),
                    "environment_level_population_sd_3_position_means": population_sd(means),
                    "environment_level_max_minus_min_position_mean": max(means) - min(means),
                    "held_position_mean_P1": means[0],
                    "held_position_mean_P2": means[1],
                    "held_position_mean_P3": means[2],
                    "source": "13_SOURCE_MEAN_AND_VARIANCE_RESULTS.csv",
                }
            )

    units: dict[tuple[str, str], list[float]] = {}
    with (results / "10_SOURCE_HELD_POSITION_METRICS.csv").open(encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            units.setdefault((record["method"], record["held_position"]), []).append(
                float(record["macro_f1"])
            )
    within = [
        {
            "method": method,
            "held_position": position,
            "seed_count": len(values),
            "within_position_seed_population_sd": population_sd(values),
            "held_position_mean_macro_f1": statistics.fmean(values),
        }
        for (method, position), values in sorted(units.items())
    ]
    return rows, within


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit(f"usage: {argv[0]} <repo_root> <output_dir>")
    results = Path(argv[1]) / "results" / "canonical_metrics" / "groupdro_worst_source"
    output = Path(argv[2])
    rows, within = build_rows(results)
    write_csv(output / "A2_ENVIRONMENT_LEVEL_DISPERSION_DERIVATION.csv", rows)
    write_csv(output / "A2_WITHIN_POSITION_SEED_DISPERSION_DERIVATION.csv", within)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
