"""Section 7 -- additive composition of resonance-peak shifts, and the
predicted-collision comparison against the existing class-collapse conclusion.

Peak extraction is reused by import from ``crfid.analysis.reference_peaks``.
The four verified resonator windows and the conditional 281-point 5-8 GHz axis
come from the reference-peak branch.

No raw P4 data is read. The only P4-derived information used is the *conclusion
text* of the existing class-collapse finding -- which TagIDs collapse -- which
is the single comparison-only read permitted for this sub-experiment.
"""

from __future__ import annotations

import itertools
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import (  # noqa: E402
    ENCODING_STATES,
    FREQUENCY_AXIS,
    POSITIONS,
    PROJECT_ROOT,
    REFERENCE_WINDOWS,
    SURFACES,
    TAG_IDS,
    Canvas,
    canonical_json_sha256,
    condition_means,
    load_frequency_axis,
    load_source,
    sha256_file,
    write_json,
)

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crfid.analysis.reference_peaks import strongest_window_peaks  # noqa: E402

RESULTS = HERE / "results"
PLOTS = HERE / "plots"

#: Fixed in preregistration.md section 3. Not revised after seeing any result.
COLLISION_THRESHOLD_MULTIPLIER = 1.5

PEAK_NAMES = tuple(REFERENCE_WINDOWS)
PEAK_MODE = "min"

#: Comparison-only reads of existing P4-derived conclusions. Conclusion text
#: only; no raw P4 data, spectra or embeddings are read.
OBSERVED_COLLAPSE_SETS = {
    "few_shot_tracked_collapse_triple": {
        "tag_ids": [5, 6, 7],
        "source": "historical few-shot P4 collapse summary retained in this diagnostic",
        "statement": (
            "The few-shot branch tracks Tags 5, 6 and 7 as the collapse set under P4 "
            "adaptation. This is the set named by the task specification."
        ),
        "role": "primary",
    },
    "strict_dg_zero_recall_set": {
        "tag_ids": [3, 5, 6, 7],
        "source": "historical Strict-DG zero-recall summary retained in this diagnostic",
        "statement": (
            "Tags 3, 5, 6, and 7 exhibit at least one zero-recall seed; Tag 6 is the most "
            "severe pooled failure (recall 0.00178, 4 correct of 2,250)."
        ),
        "role": "secondary",
    },
}


def extract_peaks(data, axis: np.ndarray) -> dict:
    """Peak frequency per (tag, er, surface, position, peak name)."""

    means, _ = condition_means(data)
    peaks: dict[tuple[int, int, str, str], dict] = {}
    interior_flags = []
    for (tag, er, surface, position), spectrum in means.items():
        found = strongest_window_peaks(spectrum, axis, REFERENCE_WINDOWS, mode=PEAK_MODE)
        peaks[(tag, er, surface, position)] = found
        interior_flags.extend(bool(found[name]["interior_local_extremum"]) for name in PEAK_NAMES)
    return {
        "peaks": peaks,
        "interior_local_extremum_fraction": float(np.mean(interior_flags)),
        "extraction_count": len(interior_flags),
    }


def per_tag_statistics(peaks: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-tag mean and within-tag SD of each peak frequency, per position.

    Shapes are (tag, position, peak); the within-tag spread is taken across the
    nine (er, surface) conditions of that tag.
    """

    mean = np.empty((len(TAG_IDS), len(POSITIONS), len(PEAK_NAMES)), dtype=np.float64)
    spread = np.empty_like(mean)
    for tag_index, tag in enumerate(TAG_IDS):
        for position_index, position in enumerate(POSITIONS):
            for peak_index, peak in enumerate(PEAK_NAMES):
                values = [
                    peaks[(tag, er, surface, position)][peak]["frequency"]
                    for er in ENCODING_STATES
                    for surface in SURFACES
                ]
                mean[tag_index, position_index, peak_index] = float(np.mean(values))
                spread[tag_index, position_index, peak_index] = float(np.std(values, ddof=1))
    return mean, spread


def minimum_reliable_spacing(mean: np.ndarray, spread: np.ndarray) -> dict:
    """Pre-registered in-domain minimum reliably-separated peak spacing."""

    entries = []
    for position_index, position in enumerate(POSITIONS):
        for peak_index, peak in enumerate(PEAK_NAMES):
            for left, right in itertools.combinations(range(len(TAG_IDS)), 2):
                separation = abs(
                    mean[left, position_index, peak_index] - mean[right, position_index, peak_index]
                )
                pooled = float(
                    np.sqrt(
                        (
                            spread[left, position_index, peak_index] ** 2
                            + spread[right, position_index, peak_index] ** 2
                        )
                        / 2.0
                    )
                )
                entries.append(
                    {
                        "position": position,
                        "peak": peak,
                        "tag_a": TAG_IDS[left],
                        "tag_b": TAG_IDS[right],
                        "separation_ghz": separation,
                        "pooled_within_tag_sd_ghz": pooled,
                        "reliably_separated": bool(separation > pooled),
                    }
                )
    reliable = [entry for entry in entries if entry["reliably_separated"]]
    if not reliable:
        raise RuntimeError(
            "No tag pair is reliably separated in-domain on any peak; the pre-registered "
            "collision threshold cannot be defined from this data."
        )
    winner = min(reliable, key=lambda entry: entry["separation_ghz"])
    return {
        "delta_min_ghz": winner["separation_ghz"],
        "attained_at": {
            "position": winner["position"],
            "peak": winner["peak"],
            "tag_pair": [winner["tag_a"], winner["tag_b"]],
            "pooled_within_tag_sd_ghz": winner["pooled_within_tag_sd_ghz"],
        },
        "candidate_pair_peak_position_count": len(entries),
        "reliably_separated_count": len(reliable),
        "reliably_separated_fraction": len(reliable) / len(entries),
    }


def predict_p4(peaks: dict) -> tuple[np.ndarray, list[dict]]:
    """Additive composition of the distance and angle peak shifts."""

    predicted = np.empty((len(TAG_IDS), len(PEAK_NAMES)), dtype=np.float64)
    rows: list[dict] = []
    for tag_index, tag in enumerate(TAG_IDS):
        for peak_index, peak in enumerate(PEAK_NAMES):
            values = []
            for er in ENCODING_STATES:
                for surface in SURFACES:
                    p1 = peaks[(tag, er, surface, "P1")][peak]["frequency"]
                    p2 = peaks[(tag, er, surface, "P2")][peak]["frequency"]
                    p3 = peaks[(tag, er, surface, "P3")][peak]["frequency"]
                    distance_shift = p3 - p1
                    angle_shift = p2 - p1
                    estimate = p1 + distance_shift + angle_shift
                    values.append(estimate)
                    rows.append(
                        {
                            "tag_id": tag,
                            "er": er,
                            "surface": surface,
                            "peak": peak,
                            "peak_p1_ghz": p1,
                            "peak_p2_ghz": p2,
                            "peak_p3_ghz": p3,
                            "delta_f_distance_ghz": distance_shift,
                            "delta_f_angle_ghz": angle_shift,
                            "predicted_p4_ghz": estimate,
                        }
                    )
            predicted[tag_index, peak_index] = float(np.mean(values))
    return predicted, rows


def collision_analysis(predicted: np.ndarray, threshold: float) -> list[dict]:
    pairs = []
    for left, right in itertools.combinations(range(len(TAG_IDS)), 2):
        per_peak = {}
        flagged_peaks = []
        for peak_index, peak in enumerate(PEAK_NAMES):
            separation = abs(predicted[left, peak_index] - predicted[right, peak_index])
            collides = bool(separation < threshold)
            per_peak[peak] = {
                "predicted_separation_ghz": separation,
                "predicted_collision": collides,
            }
            if collides:
                flagged_peaks.append(peak)
        separations = [per_peak[peak]["predicted_separation_ghz"] for peak in PEAK_NAMES]
        pairs.append(
            {
                "tag_a": TAG_IDS[left],
                "tag_b": TAG_IDS[right],
                "per_peak": per_peak,
                "minimum_predicted_separation_ghz": float(min(separations)),
                "mean_predicted_separation_ghz": float(np.mean(separations)),
                "collided_peaks": flagged_peaks,
                "predicted_collision": bool(flagged_peaks),
            }
        )
    return pairs


def compare_with_collapse(pairs: list[dict]) -> dict:
    comparisons = {}
    for name, definition in OBSERVED_COLLAPSE_SETS.items():
        members = set(definition["tag_ids"])
        rows = []
        counts = {"hit": 0, "false_alarm": 0, "miss": 0, "correct_rejection": 0}
        for pair in pairs:
            observed = pair["tag_a"] in members and pair["tag_b"] in members
            predicted = pair["predicted_collision"]
            if predicted and observed:
                outcome = "hit"
            elif predicted and not observed:
                outcome = "false_alarm"
            elif observed and not predicted:
                outcome = "miss"
            else:
                outcome = "correct_rejection"
            counts[outcome] += 1
            rows.append(
                {
                    "tag_a": pair["tag_a"],
                    "tag_b": pair["tag_b"],
                    "predicted_collision": predicted,
                    "in_observed_collapse_set": observed,
                    "outcome": outcome,
                    "minimum_predicted_separation_ghz": pair["minimum_predicted_separation_ghz"],
                }
            )
        observed_pairs = counts["hit"] + counts["miss"]
        predicted_pairs = counts["hit"] + counts["false_alarm"]
        comparisons[name] = {
            **{key: value for key, value in definition.items()},
            "observed_collapse_pair_count": observed_pairs,
            "predicted_collision_pair_count": predicted_pairs,
            "counts": counts,
            "recall_of_observed_collapse_pairs": (
                counts["hit"] / observed_pairs if observed_pairs else None
            ),
            "precision_of_predicted_collisions": (
                counts["hit"] / predicted_pairs if predicted_pairs else None
            ),
            "per_pair": rows,
        }
    return comparisons


def render_plot(observed: np.ndarray, predicted: np.ndarray, threshold: float, path: Path) -> None:
    width, height = 1060, 620
    left, right, top, bottom = 96, 1030, 78, 470
    canvas = Canvas(width, height)
    canvas.rect(left, top, right, bottom, (250, 250, 252))

    low = min(float(observed.min()), float(predicted.min())) - 0.08
    high = max(float(observed.max()), float(predicted.max())) + 0.08

    def to_x(frequency: float) -> float:
        return left + (frequency - low) / (high - low) * (right - left)

    row_height = (bottom - top) / len(TAG_IDS)

    def to_y(tag_index: int) -> float:
        return top + row_height * (tag_index + 0.5)

    for name, (window_low, window_high) in REFERENCE_WINDOWS.items():
        if window_high < low or window_low > high:
            continue
        x0, x1 = to_x(max(window_low, low)), to_x(min(window_high, high))
        canvas.rect(x0, top, x1, top - 4, (196, 202, 210))
        canvas.text_center(int((x0 + x1) / 2), top - 18, name, (80, 86, 94))
        for edge in (x0, x1):
            for start in range(top, bottom, 10):
                canvas.line(edge, start, edge, min(start + 5, bottom), (214, 219, 226), 1)

    for tag_index, tag in enumerate(TAG_IDS):
        y = to_y(tag_index)
        canvas.line(left, y, right, y, (235, 238, 242), 1)
        canvas.text_right(left - 10, int(y) - 3, f"Tag {tag}", (70, 76, 84))

    styles = (
        ("P1 observed", (120, 128, 138), "o", 2),
        ("P2 observed", (46, 132, 92), "s", 2),
        ("P3 observed", (200, 148, 40), "^", 3),
    )
    for peak_index in range(len(PEAK_NAMES)):
        for tag_index in range(len(TAG_IDS)):
            y = to_y(tag_index)
            for position_index, (_, color, shape, size) in enumerate(styles):
                canvas.marker(
                    to_x(observed[tag_index, position_index, peak_index]), y, color, size, shape
                )
            x = to_x(predicted[tag_index, peak_index])
            half = max(1.0, (threshold / 2.0) / (high - low) * (right - left))
            canvas.rect(x - half, y - 9, x + half, y + 9, (238, 206, 206))
            canvas.marker(x, y, (197, 58, 50), 4, "x")

    canvas.line(left, bottom, right, bottom, (60, 64, 70), 2)
    canvas.line(left, top, left, bottom, (60, 64, 70), 2)
    for tick in np.arange(np.ceil(low * 10) / 10, high, 0.2):
        x = to_x(float(tick))
        if left <= x <= right:
            canvas.line(x, bottom, x, bottom + 4, (60, 64, 70), 1)
            canvas.text_center(int(x), bottom + 10, f"{float(tick):.1f}", (90, 96, 104))

    canvas.text(left, 18, "Predicted P4 peaks vs observed source peaks, per TagID", (24, 28, 34), 2)
    canvas.text(
        left,
        42,
        "predicted P4 = f(P1) + [f(P3) - f(P1)] + [f(P2) - f(P1)]   pink band = pre-registered collision width",
        (96, 102, 110),
        1,
    )
    canvas.text_center(
        (left + right) // 2,
        bottom + 30,
        "Conditional frequency (GHz) -- axis provenance unresolved, not a physical assertion",
        (60, 64, 70),
    )

    legend_y = bottom + 54
    cursor = left
    for label, color, shape, size in styles:
        canvas.marker(cursor + 6, legend_y + 3, color, size, shape)
        canvas.text(cursor + 18, legend_y, label, (40, 44, 50))
        cursor += 150
    canvas.marker(cursor + 6, legend_y + 3, (197, 58, 50), 4, "x")
    canvas.text(cursor + 18, legend_y, "P4 predicted", (40, 44, 50))
    canvas.text(
        left,
        legend_y + 22,
        f"Collision threshold = 1.5 x Delta_min = {threshold:.6f} GHz. Diagnostic only; not a Strict-DG result.",
        (110, 116, 124),
    )
    canvas.save_png(path)


def main() -> int:
    data = load_source()
    axis = load_frequency_axis()
    extraction = extract_peaks(data, axis)
    peaks = extraction["peaks"]

    observed_mean, observed_spread = per_tag_statistics(peaks)
    spacing = minimum_reliable_spacing(observed_mean, observed_spread)
    threshold = COLLISION_THRESHOLD_MULTIPLIER * spacing["delta_min_ghz"]

    predicted, condition_rows = predict_p4(peaks)
    pairs = collision_analysis(predicted, threshold)
    comparisons = compare_with_collapse(pairs)

    render_plot(observed_mean, predicted, threshold, PLOTS / "predicted_vs_actual_peaks.png")

    peak_payload = {
        "schema_version": 1,
        "sub_experiment": "03_peak_warp_prediction",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "p4_raw_data_used": False,
        "domains_used": list(POSITIONS),
        "peak_extraction": {
            "function": "crfid.analysis.reference_peaks.strongest_window_peaks",
            "mode": PEAK_MODE,
            "windows_ghz": {name: list(bounds) for name, bounds in REFERENCE_WINDOWS.items()},
            "window_source": "results/canonical_metrics/reference_peak/FREQUENCY_VALUE_MAP_REPORT.md",
            "grain": "mean spectrum of each (tag_id, er, surface, position) block of 50 repeats",
            "extraction_count": extraction["extraction_count"],
            "interior_local_extremum_fraction": extraction["interior_local_extremum_fraction"],
            "axis_status": "PHYSICAL_FREQUENCY_AXIS_PROVENANCE_UNRESOLVED",
            "axis_step_ghz": float(np.diff(axis).mean()),
        },
        "composition_rule": {
            "delta_f_distance": "peak(P3) - peak(P1)",
            "delta_f_angle": "peak(P2) - peak(P1)",
            "prediction": "peak_hat(P4) = peak(P1) + delta_f_distance + delta_f_angle",
            "per_tag_aggregation": "mean over the nine (er, surface) conditions of that tag",
        },
        "observed_per_tag_peak_ghz": {
            str(tag): {
                position: {
                    peak: observed_mean[tag_index, position_index, peak_index]
                    for peak_index, peak in enumerate(PEAK_NAMES)
                }
                for position_index, position in enumerate(POSITIONS)
            }
            for tag_index, tag in enumerate(TAG_IDS)
        },
        "observed_within_tag_sd_ghz": {
            str(tag): {
                position: {
                    peak: observed_spread[tag_index, position_index, peak_index]
                    for peak_index, peak in enumerate(PEAK_NAMES)
                }
                for position_index, position in enumerate(POSITIONS)
            }
            for tag_index, tag in enumerate(TAG_IDS)
        },
        "predicted_p4_peak_ghz": {
            str(tag): {
                peak: predicted[tag_index, peak_index]
                for peak_index, peak in enumerate(PEAK_NAMES)
            }
            for tag_index, tag in enumerate(TAG_IDS)
        },
        "per_condition_predictions": condition_rows,
        "provenance": {
            "source_signals_sha256": data.signals_sha256,
            "source_registry_sha256": data.registry_sha256,
            "conditional_frequency_axis_sha256": sha256_file(FREQUENCY_AXIS),
        },
    }
    peak_payload["predicted_peaks_semantic_sha256"] = canonical_json_sha256(peak_payload)
    write_json(RESULTS / "predicted_p4_peaks.json", peak_payload)

    collision_payload = {
        "schema_version": 1,
        "sub_experiment": "03_peak_warp_prediction",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "p4_raw_data_used": False,
        "p4_derived_information_used": "existing class-collapse conclusion text only",
        "preregistered_threshold": {
            "name": "predicted_separation_below_1.5x_minimum_reliably_separated_in_domain_spacing",
            "multiplier": COLLISION_THRESHOLD_MULTIPLIER,
            "delta_min_ghz": spacing["delta_min_ghz"],
            "collision_threshold_ghz": threshold,
            "definition": (
                "Delta_min is the smallest per-tag mean peak separation, over all source "
                "positions and peaks, among tag pairs whose separation exceeds their pooled "
                "within-tag standard deviation across the nine (er, surface) conditions."
            ),
            "attained_at": spacing["attained_at"],
            "candidate_pair_peak_position_count": spacing["candidate_pair_peak_position_count"],
            "reliably_separated_count": spacing["reliably_separated_count"],
            "reliably_separated_fraction": spacing["reliably_separated_fraction"],
            "fixed_before_execution": True,
        },
        "pairwise_predictions": pairs,
        "predicted_collision_pairs": [
            [pair["tag_a"], pair["tag_b"]] for pair in pairs if pair["predicted_collision"]
        ],
        "comparison_against_existing_class_collapse_conclusion": comparisons,
        "test_discriminative_power": {
            "pairs_flagged": len([pair for pair in pairs if pair["predicted_collision"]]),
            "pairs_total": len(pairs),
            "discriminative": 0 < len([p for p in pairs if p["predicted_collision"]]) < len(pairs),
            "note": (
                "If every pair is flagged, the hit counts below are attained trivially with a "
                "100% false-alarm rate on the complement and are not evidence that the additive "
                "peak-warp prediction identifies the collapsing tags."
            ),
        },
        "verdict": {
            name: {
                "hits": entry["counts"]["hit"],
                "misses": entry["counts"]["miss"],
                "false_alarms": entry["counts"]["false_alarm"],
                "correct_rejections": entry["counts"]["correct_rejection"],
                "observed_collapse_pairs_recovered": (
                    f"{entry['counts']['hit']}/{entry['observed_collapse_pair_count']}"
                ),
                "evidential_weight": (
                    "none_test_not_discriminative"
                    if not (0 < len([p for p in pairs if p["predicted_collision"]]) < len(pairs))
                    else "informative"
                ),
            }
            for name, entry in comparisons.items()
        },
    }
    collision_payload["collision_semantic_sha256"] = canonical_json_sha256(collision_payload)
    write_json(RESULTS / "collision_prediction_vs_observed.json", collision_payload)

    write_report(peak_payload, collision_payload, observed_mean, predicted)
    primary = comparisons["few_shot_tracked_collapse_triple"]
    print(
        f"delta_min={spacing['delta_min_ghz']:.6f} GHz, threshold={threshold:.6f} GHz, "
        f"predicted collisions={len(collision_payload['predicted_collision_pairs'])}/21"
    )
    print(
        f"primary comparison vs TagID 5/6/7: hits={primary['counts']['hit']}/"
        f"{primary['observed_collapse_pair_count']}, "
        f"false_alarms={primary['counts']['false_alarm']}, misses={primary['counts']['miss']}"
    )
    return 0


def write_report(peaks: dict, collision: dict, observed: np.ndarray, predicted: np.ndarray) -> None:
    threshold = collision["preregistered_threshold"]
    lines = [
        "# Section 7 — Peak-warp prediction and predicted collisions",
        "",
        "Sub-module: `06_failure_mechanism/composition_warp/03_peak_warp_prediction`.",
        "**No raw P4 data was read.** The only P4-derived information used is the",
        "conclusion text of the existing class-collapse finding — which TagIDs collapse —",
        "which is the single comparison-only read permitted for this sub-experiment.",
        "",
        "## Question",
        "",
        "If the additive composition hypothesis holds for resonance geometry, the P4",
        "peak positions are predictable from the source domains alone. Tag identity in",
        "this system is carried by relative resonance geometry, so tags whose predicted",
        "P4 peaks land on top of each other are the tags a classifier should be unable",
        "to tell apart. That gives a source-only prediction of *which classes collapse*,",
        "testable against a conclusion that was reached from P4.",
        "",
        "## Method",
        "",
        "Peak extraction is reused by import from",
        f"`{peaks['peak_extraction']['function']}` in `mode = \"{peaks['peak_extraction']['mode']}\"`,",
        "applied to the mean spectrum of each `(tag_id, er, surface, position)` block of",
        "50 repetitions, on the four verified resonator windows:",
        "",
        "| Window | Conditional range (GHz) |",
        "|---|---|",
    ]
    for name, bounds in peaks["peak_extraction"]["windows_ghz"].items():
        lines.append(f"| `{name}` | {bounds[0]:.3f} – {bounds[1]:.3f} |")
    lines += [
        "",
        f"{peaks['peak_extraction']['extraction_count']} extractions were performed; "
        f"{peaks['peak_extraction']['interior_local_extremum_fraction']:.4f} of them landed on an",
        "interior local minimum rather than a window edge.",
        "",
        "> **Window naming.** The task specification labels these `Lc≈5.30, L1≈5.84,",
        "> L2≈6.46, L3≈7.17`. The repository's own reference-peak report assigns the two",
        "> outer regions the other way round — `L3` is the 5.800–6.350 band and `L1` is",
        "> the 7.050–7.700 band. The four *regions* are the same; only the `L1`/`L3`",
        "> labels differ. The repository's naming is used here, because §7 is required to",
        "> reuse the reference-peak branch rather than re-derive it.",
        "",
        "Then, per TagID and per `(er, surface)` condition:",
        "",
        "```",
        f"delta_f_distance = {peaks['composition_rule']['delta_f_distance']}",
        f"delta_f_angle    = {peaks['composition_rule']['delta_f_angle']}",
        f"{peaks['composition_rule']['prediction']}",
        "```",
        "",
        f"Per-TagID predicted positions are the {peaks['composition_rule']['per_tag_aggregation']}.",
        "",
        "## Pre-registered collision threshold",
        "",
        threshold["definition"],
        "",
        f"`Delta_min = {threshold['delta_min_ghz']:.6f} GHz`, attained at position "
        f"`{threshold['attained_at']['position']}`, peak `{threshold['attained_at']['peak']}`, tag pair "
        f"{threshold['attained_at']['tag_pair']}. "
        f"{threshold['reliably_separated_count']} of {threshold['candidate_pair_peak_position_count']} "
        f"pair/peak/position combinations were reliably separated in-domain "
        f"({threshold['reliably_separated_fraction']:.4f}).",
        "",
        f"**Collision threshold = {threshold['multiplier']} × Delta_min = "
        f"{threshold['collision_threshold_ghz']:.6f} GHz.** Fixed in `preregistration.md` before this",
        "script was run and not revised afterwards.",
        "",
        "## Predicted P4 peak positions (GHz, conditional axis)",
        "",
        "| TagID | " + " | ".join(f"`{name}`" for name in PEAK_NAMES) + " |",
        "|---:|" + "---:|" * len(PEAK_NAMES),
    ]
    for tag_index, tag in enumerate(TAG_IDS):
        cells = " | ".join(f"{predicted[tag_index, index]:.4f}" for index in range(len(PEAK_NAMES)))
        lines.append(f"| {tag} | {cells} |")

    lines += [
        "",
        "## Predicted collisions",
        "",
        f"**{len(collision['predicted_collision_pairs'])} of 21 tag pairs are flagged as predicted collisions**: "
        + (
            ", ".join(f"({a},{b})" for a, b in collision["predicted_collision_pairs"])
            if collision["predicted_collision_pairs"]
            else "none"
        )
        + ".",
        "",
        "| Pair | min predicted separation (GHz) | collided peaks | predicted collision |",
        "|---|---:|---|:---:|",
    ]
    for pair in collision["pairwise_predictions"]:
        collided = ", ".join(f"`{name}`" for name in pair["collided_peaks"]) or "—"
        lines.append(
            f"| ({pair['tag_a']},{pair['tag_b']}) | {pair['minimum_predicted_separation_ghz']:.6f} | "
            f"{collided} | {'**yes**' if pair['predicted_collision'] else 'no'} |"
        )

    flagged = len(collision["predicted_collision_pairs"])
    total_pairs = len(collision["pairwise_predictions"])
    lines += ["", "### Discriminative power of this test", ""]
    if flagged == total_pairs:
        lines += [
            f"**The test flags all {total_pairs} of {total_pairs} pairs, so it is not discriminative.** Any",
            "hit rate it achieves against any observed collapse set is attained trivially,",
            "with a false-alarm rate of 100 % on the complement. The hit counts below are",
            "reported because the specification asks for them, but they are **not evidence**",
            "that the additive peak-warp prediction identifies the collapsing tags.",
            "",
            "The reason is visible in the threshold derivation, and it is the substantive",
            f"finding of this sub-experiment: only {threshold['reliably_separated_count']} of "
            f"{threshold['candidate_pair_peak_position_count']} pair/peak/position combinations",
            f"({threshold['reliably_separated_fraction']:.2%}) are reliably separated in-domain at all. Across",
            "the nine `(er, surface)` conditions of a tag, the within-tag scatter of a peak",
            "position is larger than the between-tag separation almost everywhere. The",
            f"resulting `Delta_min` of {threshold['delta_min_ghz']:.4f} GHz is therefore large — about "
            f"{threshold['delta_min_ghz'] / peaks['peak_extraction']['axis_step_ghz']:.0f} axis steps —",
            "while the predicted P4 separations are an order of magnitude smaller.",
            "",
            "Read plainly: **the four reference peaks do not separate the seven TagIDs in the",
            "source domains in the first place**, so composing their shifts cannot single out",
            "TagIDs 5/6/7. That is a statement about the reference-peak representation, not a",
            "refutation of the collapse conclusion.",
            "",
        ]
    elif flagged == 0:
        lines += [
            f"**The test flags none of the {total_pairs} pairs.** It therefore cannot recover any",
            "observed collapse pair, and the comparison below is a null result rather than a",
            "disagreement.",
            "",
        ]
    else:
        lines += [
            f"The test flags {flagged} of {total_pairs} pairs, so it does discriminate between pairs and",
            "the comparison below carries evidential weight.",
            "",
        ]

    lines += [
        "## Comparison against the existing class-collapse conclusion",
        "",
        "Comparison-only read of existing P4-derived *conclusions*. No raw P4 data,",
        "spectra or embeddings were read.",
        "",
    ]
    for name, entry in collision["comparison_against_existing_class_collapse_conclusion"].items():
        counts = entry["counts"]
        lines += [
            f"### `{name}` ({entry['role']})",
            "",
            f"TagIDs: {entry['tag_ids']}. Source: `{entry['source']}`.",
            "",
            f"> {entry['statement']}",
            "",
            f"Hits **{counts['hit']}/{entry['observed_collapse_pair_count']}** of the observed collapse pairs; "
            f"{counts['false_alarm']} false alarms, {counts['miss']} misses, "
            f"{counts['correct_rejection']} correct rejections out of 21 pairs.",
            "",
            "| Pair | predicted collision | in observed collapse set | outcome |",
            "|---|:---:|:---:|---|",
        ]
        for row in entry["per_pair"]:
            if not (row["predicted_collision"] or row["in_observed_collapse_set"]):
                continue
            lines.append(
                f"| ({row['tag_a']},{row['tag_b']}) | {'yes' if row['predicted_collision'] else 'no'} | "
                f"{'yes' if row['in_observed_collapse_set'] else 'no'} | `{row['outcome']}` |"
            )
        lines += [
            "",
            "Pairs that are neither predicted nor observed are omitted from the table and",
            f"counted as correct rejections ({counts['correct_rejection']}).",
            "",
        ]

    lines += [
        "## Outputs",
        "",
        "* `results/predicted_p4_peaks.json` — extraction settings, observed per-TagID peak means and",
        "  within-tag spreads per source position, per-condition distance/angle shifts, and the predicted",
        "  P4 positions.",
        "* `results/collision_prediction_vs_observed.json` — the pre-registered threshold with its",
        "  derivation, all 21 pairwise predictions, and the hit/miss comparison against both recorded",
        "  class-collapse conclusions.",
        "* `plots/predicted_vs_actual_peaks.png` — observed P1/P2/P3 peaks and the predicted P4 peak per",
        "  TagID, with the collision width drawn around each prediction.",
        "",
        "## Scope",
        "",
        "Every frequency in this report is on the **conditional** 281-point 5–8 GHz axis.",
        "The reference-peak branch release status is `BLOCKED` and the Strict-DG",
        "configuration records `PHYSICAL_FREQUENCY_AXIS_PROVENANCE_UNRESOLVED`. No",
        "physical frequency axis is asserted. Diagnostic and mechanistic only; not",
        "inserted into any existing comparison table.",
    ]
    (HERE / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
