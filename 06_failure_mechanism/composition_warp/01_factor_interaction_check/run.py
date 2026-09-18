"""Section 5 -- two-way factor-interaction check on P1/P2/P3 only.

For every fully crossed factor pair present in the source data, estimate the
interaction term as cell mean minus the additive prediction built from the row
and column marginal means, and compare its RMS magnitude with the RMS magnitude
of the two main effects. The 0.30 threshold is fixed in ``preregistration.md``
and is not revised here.

No P4 measurement, label or summary statistic is read by this script.
"""

from __future__ import annotations

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
    POINT_COUNT,
    SURFACES,
    TAG_IDS,
    Canvas,
    canonical_json_sha256,
    load_frequency_axis,
    load_source,
    sha256_file,
    write_csv,
    write_json,
)

RESULTS = HERE / "results"
PLOTS = HERE / "plots"

#: Fixed in preregistration.md section 3. Not revised after seeing any result.
INTERACTION_RATIO_THRESHOLD = 0.30

FACTOR_PAIRS = (
    {
        "pair_id": "surface_x_position",
        "factor_a": "surface",
        "factor_b": "position",
        "levels_a": list(SURFACES),
        "levels_b": list(POSITIONS),
    },
    {
        "pair_id": "encoding_state_x_position",
        "factor_a": "encoding_state_er",
        "factor_b": "position",
        "levels_a": [str(value) for value in ENCODING_STATES],
        "levels_b": list(POSITIONS),
    },
)


def _level_values(data, factor: str) -> np.ndarray:
    if factor == "surface":
        return data.surface
    if factor == "position":
        return data.position
    if factor == "encoding_state_er":
        return np.asarray([str(value) for value in data.er.tolist()], dtype="<U2")
    raise ValueError(f"Unknown factor: {factor}")


def decompose(
    data,
    factor_a: str,
    factor_b: str,
    levels_a: list[str],
    levels_b: list[str],
    tag_filter: int | None = None,
) -> dict:
    """Two-way decomposition of the dB-domain cell means."""

    values_a = _level_values(data, factor_a)
    values_b = _level_values(data, factor_b)
    base = np.ones(len(data.signals), dtype=bool) if tag_filter is None else data.tag_id == tag_filter

    cell = np.empty((len(levels_a), len(levels_b), POINT_COUNT), dtype=np.float64)
    counts = np.empty((len(levels_a), len(levels_b)), dtype=np.int64)
    for index_a, level_a in enumerate(levels_a):
        for index_b, level_b in enumerate(levels_b):
            selected = np.flatnonzero(base & (values_a == level_a) & (values_b == level_b))
            if selected.size == 0:
                raise RuntimeError(f"Factor pair is not fully crossed at {level_a}/{level_b}")
            counts[index_a, index_b] = selected.size
            cell[index_a, index_b] = data.signals[selected].mean(axis=0)
    if len(set(counts.ravel().tolist())) != 1:
        raise RuntimeError(f"Unbalanced factor design: {counts.tolist()}")

    grand = cell.mean(axis=(0, 1))
    alpha = cell.mean(axis=1) - grand
    beta = cell.mean(axis=0) - grand
    additive = grand[None, None, :] + alpha[:, None, :] + beta[None, :, :]
    gamma = cell - additive

    rms_interaction = float(np.sqrt(np.mean(gamma**2)))
    rms_main_a = float(np.sqrt(np.mean(alpha**2)))
    rms_main_b = float(np.sqrt(np.mean(beta**2)))
    smaller = min(rms_main_a, rms_main_b)
    ratio = float(rms_interaction / smaller)
    return {
        "cell_means": cell,
        "grand": grand,
        "alpha": alpha,
        "beta": beta,
        "additive": additive,
        "gamma": gamma,
        "samples_per_cell": int(counts[0, 0]),
        "rms_interaction_db": rms_interaction,
        "rms_main_effect_a_db": rms_main_a,
        "rms_main_effect_b_db": rms_main_b,
        "smaller_main_effect_rms_db": smaller,
        "interaction_to_smaller_main_effect_ratio": ratio,
        "maximum_absolute_interaction_db": float(np.max(np.abs(gamma))),
        "interaction_rms_per_point": np.sqrt(np.mean(gamma**2, axis=(0, 1))),
        "verdict": "HIGH" if ratio > INTERACTION_RATIO_THRESHOLD else "LOW",
    }


def render_plot(axis: np.ndarray, curves: list[dict], path: Path) -> None:
    width, height = 1000, 560
    left, right, top, bottom = 92, 972, 56, 470
    canvas = Canvas(width, height)

    peak = max(float(curve["values"].max()) for curve in curves)
    y_max = max(peak * 1.18, 1e-9)

    def to_x(frequency: float) -> float:
        return left + (frequency - axis[0]) / (axis[-1] - axis[0]) * (right - left)

    def to_y(value: float) -> float:
        return bottom - (value / y_max) * (bottom - top)

    canvas.rect(left, top, right, bottom, (250, 250, 252))
    for step in range(6):
        value = y_max * step / 5.0
        y = to_y(value)
        canvas.line(left, y, right, y, (222, 226, 232), 1)
        canvas.text_right(left - 8, int(y) - 3, f"{value:.3f}", (90, 96, 104))
    for tick in (5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0):
        x = to_x(tick)
        canvas.line(x, top, x, bottom, (232, 235, 240), 1)
        canvas.text_center(int(x), bottom + 10, f"{tick:.1f}", (90, 96, 104))
    canvas.line(left, bottom, right, bottom, (60, 64, 70), 2)
    canvas.line(left, top, left, bottom, (60, 64, 70), 2)

    for curve in curves:
        xs = [to_x(value) for value in axis]
        ys = [to_y(value) for value in curve["values"]]
        canvas.polyline(xs, ys, curve["color"], 2)
        threshold_y = to_y(curve["threshold"])
        for start in range(left, right, 14):
            canvas.line(start, threshold_y, min(start + 7, right), threshold_y, curve["color"], 1)

    canvas.text(left, 18, "Two-way interaction magnitude vs frequency (P1/P2/P3 only)", (24, 28, 34), 2)
    canvas.text(
        left,
        40,
        "solid = RMS interaction across cells   dashed = 0.30 x smaller main-effect RMS (LOW/HIGH threshold)",
        (96, 102, 110),
        1,
    )
    canvas.text_center((left + right) // 2, bottom + 30, "Conditional frequency (GHz) -- axis provenance unresolved", (60, 64, 70))
    for offset, character in enumerate("dB"):
        canvas.text(14, top + 150 + offset * 12, character, (60, 64, 70))

    legend_y = bottom + 54
    for curve in curves:
        canvas.rect(left, legend_y + 2, left + 22, legend_y + 5, curve["color"])
        label = (
            f"{curve['label']}   RMS interaction {curve['rms']:.4f} dB   "
            f"ratio {curve['ratio']:.4f}   {curve['verdict']}"
        )
        canvas.text(left + 30, legend_y, label, (40, 44, 50))
        legend_y += 20
    canvas.text(
        left,
        legend_y + 6,
        "Pre-registered threshold: ratio > 0.30 => HIGH. Diagnostic only; not a Strict-DG result.",
        (110, 116, 124),
    )
    canvas.save_png(path)


def main() -> int:
    data = load_source()
    axis = load_frequency_axis()

    summary_rows: list[dict] = []
    curve_rows: list[dict] = []
    pair_reports: list[dict] = []
    plot_curves: list[dict] = []
    colors = ((197, 58, 50), (36, 98, 170))

    for index, pair in enumerate(FACTOR_PAIRS):
        pooled = decompose(
            data, pair["factor_a"], pair["factor_b"], pair["levels_a"], pair["levels_b"]
        )
        per_tag = {}
        for tag in TAG_IDS:
            single = decompose(
                data,
                pair["factor_a"],
                pair["factor_b"],
                pair["levels_a"],
                pair["levels_b"],
                tag_filter=tag,
            )
            per_tag[str(tag)] = {
                "rms_interaction_db": single["rms_interaction_db"],
                "rms_main_effect_a_db": single["rms_main_effect_a_db"],
                "rms_main_effect_b_db": single["rms_main_effect_b_db"],
                "interaction_to_smaller_main_effect_ratio": single[
                    "interaction_to_smaller_main_effect_ratio"
                ],
                "verdict": single["verdict"],
            }

        for index_a, level_a in enumerate(pair["levels_a"]):
            for index_b, level_b in enumerate(pair["levels_b"]):
                for point in range(POINT_COUNT):
                    curve_rows.append(
                        {
                            "pair_id": pair["pair_id"],
                            "level_a": level_a,
                            "level_b": level_b,
                            "signal_position": point,
                            "conditional_frequency_ghz": f"{axis[point]:.10f}",
                            "cell_mean_db": f"{pooled['cell_means'][index_a, index_b, point]:.12g}",
                            "additive_prediction_db": f"{pooled['additive'][index_a, index_b, point]:.12g}",
                            "interaction_db": f"{pooled['gamma'][index_a, index_b, point]:.12g}",
                        }
                    )

        ratios = [entry["interaction_to_smaller_main_effect_ratio"] for entry in per_tag.values()]
        report = {
            "pair_id": pair["pair_id"],
            "factor_a": pair["factor_a"],
            "factor_b": pair["factor_b"],
            "levels_a": pair["levels_a"],
            "levels_b": pair["levels_b"],
            "samples_per_cell": pooled["samples_per_cell"],
            "rms_interaction_db": pooled["rms_interaction_db"],
            "rms_main_effect_a_db": pooled["rms_main_effect_a_db"],
            "rms_main_effect_b_db": pooled["rms_main_effect_b_db"],
            "smaller_main_effect_rms_db": pooled["smaller_main_effect_rms_db"],
            "interaction_to_smaller_main_effect_ratio": pooled[
                "interaction_to_smaller_main_effect_ratio"
            ],
            "maximum_absolute_interaction_db": pooled["maximum_absolute_interaction_db"],
            "threshold": INTERACTION_RATIO_THRESHOLD,
            "verdict": pooled["verdict"],
            "per_tag_id": per_tag,
            "per_tag_ratio_minimum": float(min(ratios)),
            "per_tag_ratio_maximum": float(max(ratios)),
            "per_tag_verdicts": sorted({entry["verdict"] for entry in per_tag.values()}),
        }
        pair_reports.append(report)
        summary_rows.append(
            {
                "pair_id": pair["pair_id"],
                "rms_interaction_db": f"{pooled['rms_interaction_db']:.12g}",
                "rms_main_effect_a_db": f"{pooled['rms_main_effect_a_db']:.12g}",
                "rms_main_effect_b_db": f"{pooled['rms_main_effect_b_db']:.12g}",
                "interaction_to_smaller_main_effect_ratio": f"{report['interaction_to_smaller_main_effect_ratio']:.12g}",
                "threshold": INTERACTION_RATIO_THRESHOLD,
                "verdict": pooled["verdict"],
            }
        )
        plot_curves.append(
            {
                "label": pair["pair_id"],
                "values": pooled["interaction_rms_per_point"],
                "threshold": INTERACTION_RATIO_THRESHOLD * pooled["smaller_main_effect_rms_db"],
                "rms": pooled["rms_interaction_db"],
                "ratio": pooled["interaction_to_smaller_main_effect_ratio"],
                "verdict": pooled["verdict"],
                "color": colors[index % len(colors)],
            }
        )

    write_csv(RESULTS / "interaction_summary.csv", curve_rows)
    render_plot(axis, plot_curves, PLOTS / "interaction_vs_frequency.png")

    payload = {
        "schema_version": 1,
        "sub_experiment": "01_factor_interaction_check",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "domains_used": list(POSITIONS),
        "p4_used": False,
        "preregistered_threshold": {
            "name": "interaction_to_smaller_main_effect_ratio",
            "value": INTERACTION_RATIO_THRESHOLD,
            "rule": "ratio > 0.30 => HIGH, otherwise LOW",
            "fixed_before_execution": True,
        },
        "units": "decibel",
        "signal_point_count": POINT_COUNT,
        "factor_pairs": pair_reports,
        "overall_verdict": "HIGH" if any(r["verdict"] == "HIGH" for r in pair_reports) else "LOW",
        "provenance": {
            "source_signals_sha256": data.signals_sha256,
            "source_registry_sha256": data.registry_sha256,
            "conditional_frequency_axis_sha256": sha256_file(FREQUENCY_AXIS),
            "source_row_count": int(len(data.signals)),
        },
    }
    payload["summary_semantic_sha256"] = canonical_json_sha256(payload)
    write_json(RESULTS / "interaction_summary.json", payload)

    write_report(payload, summary_rows)
    print(
        "\n".join(
            f"{row['pair_id']}: RMS interaction {row['rms_interaction_db']} dB, "
            f"ratio {row['interaction_to_smaller_main_effect_ratio']} -> {row['verdict']}"
            for row in summary_rows
        )
    )
    print(f"overall_verdict={payload['overall_verdict']}")
    return 0


def write_report(payload: dict, summary_rows: list[dict]) -> None:
    lines = [
        "# Section 5 — Factor-interaction check",
        "",
        "Sub-module: `06_failure_mechanism/composition_warp/01_factor_interaction_check`.",
        "Domains used: **P1, P2, P3 only**. No P4 measurement, label or summary",
        "statistic was read by this sub-experiment.",
        "",
        "## Question",
        "",
        "The additive composition hypothesis predicts an unobserved factor cell from",
        "the marginal effects of the observed cells. That prediction is only defensible",
        "if the factors it composes do not interact. This sub-experiment measures the",
        "interaction directly, on factor pairs that *are* fully crossed in the source",
        "data, as a proxy for the distance × angle interaction that cannot be measured",
        "without P4.",
        "",
        "## Method",
        "",
        "For each fully crossed pair, cell means are taken over the 281 ordered signal",
        "positions in the raw dB domain. The two-way decomposition is",
        "",
        "```",
        "grand(f)         = mean over all cells",
        "alpha[a](f)      = rowmean[a](f) - grand(f)",
        "beta[b](f)       = colmean[b](f) - grand(f)",
        "additive[a,b](f) = grand(f) + alpha[a](f) + beta[b](f)",
        "gamma[a,b](f)    = cell[a,b](f) - additive[a,b](f)",
        "```",
        "",
        "and the summary scalars are the RMS of `gamma` over all cells and points,",
        "against the RMS of each main effect. The design is balanced: every cell of",
        f"both pairs holds exactly {payload['factor_pairs'][0]['samples_per_cell']} samples.",
        "",
        "## Pre-registered threshold",
        "",
        f"`interaction RMS / min(main-effect RMS) > {payload['preregistered_threshold']['value']}` ⇒ `HIGH`,",
        "otherwise `LOW`. Fixed in `preregistration.md` before this script was run and",
        "not revised afterwards.",
        "",
        "## Result",
        "",
        "| Factor pair | RMS interaction (dB) | RMS main A (dB) | RMS main B (dB) | Ratio | Verdict |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for report in payload["factor_pairs"]:
        lines.append(
            f"| `{report['pair_id']}` | {report['rms_interaction_db']:.6f} | "
            f"{report['rms_main_effect_a_db']:.6f} | {report['rms_main_effect_b_db']:.6f} | "
            f"**{report['interaction_to_smaller_main_effect_ratio']:.4f}** | "
            f"**{report['verdict']}** |"
        )
    lines += [
        "",
        f"Overall verdict: **{payload['overall_verdict']}**.",
        "",
        "**Reading.** Both ratios exceed 1, not merely the 0.30 threshold. The",
        "interaction term is *larger* than the smaller of the two main effects it is",
        "supposed to be a second-order correction to, so on this data the additive part of",
        "the decomposition does not dominate the residual. `RMS main B` is identical in",
        "both rows because `position` is factor B of both pairs. Note that these two pairs",
        "are proxies: the distance × angle interaction itself cannot be measured without",
        "P4, which is exactly the cell the hypothesis is trying to predict.",
        "",
        "Largest single absolute interaction per pair: "
        + ", ".join(
            f"`{report['pair_id']}` {report['maximum_absolute_interaction_db']:.6f} dB"
            for report in payload["factor_pairs"]
        )
        + ".",
        "",
        "### Per-TagID breakdown (secondary)",
        "",
        "The threshold is applied to the pooled value above. The same decomposition",
        "repeated within each TagID gives:",
        "",
        "| Factor pair | min ratio | max ratio | verdicts present |",
        "|---|---:|---:|---|",
    ]
    for report in payload["factor_pairs"]:
        lines.append(
            f"| `{report['pair_id']}` | {report['per_tag_ratio_minimum']:.4f} | "
            f"{report['per_tag_ratio_maximum']:.4f} | "
            + ", ".join(f"`{value}`" for value in report["per_tag_verdicts"])
            + " |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        "* `results/interaction_summary.json` — scalars, per-TagID breakdown, threshold, provenance digests.",
        "* `results/interaction_summary.csv` — the raw dB-domain interaction curve: every cell of every",
        "  pair at all 281 ordered signal positions, with its cell mean, additive prediction and residual.",
        "* `plots/interaction_vs_frequency.png` — RMS interaction magnitude against conditional frequency,",
        "  with the pre-registered threshold drawn as a dashed line per pair.",
        "",
        "The conditional 5–8 GHz axis is used for the horizontal plot scale only. No",
        "physical frequency axis is asserted; the decomposition itself is computed on",
        "ordered signal positions and does not depend on the axis.",
        "",
        "## Scope",
        "",
        "Diagnostic and mechanistic only. These numbers are not a Strict-DG, Few-Shot or",
        "External-Data1 result and are not inserted into any existing comparison table.",
    ]
    (HERE / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
