"""Dependency-free IEEE-style SVG figures.

The locked environment (`environment/CRFID_ENVIRONMENT_LOCK.json`) contains no plotting
library, and this branch must not modify the environment lock. Figures are therefore
emitted as plain SVG: vector, deterministic, byte-reproducible, and directly usable in a
manuscript. Styling follows IEEE conventions -- serif type, single-column widths, thin
rules, no decoration, no colour used as the sole carrier of meaning.
"""

from __future__ import annotations

from typing import Any, Sequence

SERIF = "Times New Roman, Nimbus Roman, Liberation Serif, serif"
COLUMN_WIDTH = 360.0  # ~3.5in at 96 dpi
GREYS = ("#000000", "#4d4d4d", "#7f7f7f", "#a6a6a6", "#cccccc", "#e6e6e6")
HATCHES = ("", "url(#h1)", "url(#h2)", "url(#h3)", "url(#h4)", "url(#h5)")


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _header(width: float, height: float, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.1f}" height="{height:.1f}" '
        f'viewBox="0 0 {width:.1f} {height:.1f}" font-family="{SERIF}">',
        f"<title>{_escape(title)}</title>",
        "<defs>",
        '<pattern id="h1" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<rect width="4" height="4" fill="#ffffff"/><line x1="0" y1="0" x2="0" y2="4" stroke="#000" stroke-width="1.1"/></pattern>',
        '<pattern id="h2" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(-45)">'
        '<rect width="4" height="4" fill="#ffffff"/><line x1="0" y1="0" x2="0" y2="4" stroke="#000" stroke-width="1.1"/></pattern>',
        '<pattern id="h3" width="5" height="5" patternUnits="userSpaceOnUse">'
        '<rect width="5" height="5" fill="#ffffff"/><circle cx="2.5" cy="2.5" r="1" fill="#000"/></pattern>',
        '<pattern id="h4" width="4" height="4" patternUnits="userSpaceOnUse">'
        '<rect width="4" height="4" fill="#ffffff"/><line x1="0" y1="2" x2="4" y2="2" stroke="#000" stroke-width="1"/></pattern>',
        '<pattern id="h5" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<rect width="6" height="6" fill="#ffffff"/><line x1="0" y1="0" x2="0" y2="6" stroke="#000" stroke-width="2"/></pattern>',
        "</defs>",
        f'<rect width="{width:.1f}" height="{height:.1f}" fill="#ffffff"/>',
    ]


def _axes(
    parts: list[str],
    left: float,
    top: float,
    plot_width: float,
    plot_height: float,
    low: float,
    high: float,
    *,
    ticks: int = 5,
    y_label: str = "",
    zero_rule: bool = False,
) -> None:
    parts.append(
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{plot_width:.1f}" height="{plot_height:.1f}" '
        'fill="none" stroke="#000000" stroke-width="0.8"/>'
    )
    span = high - low or 1.0
    for index in range(ticks + 1):
        value = low + span * index / ticks
        y = top + plot_height - (value - low) / span * plot_height
        parts.append(
            f'<line x1="{left - 3:.1f}" y1="{y:.1f}" x2="{left:.1f}" y2="{y:.1f}" stroke="#000" stroke-width="0.8"/>'
        )
        parts.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + plot_width:.1f}" y2="{y:.1f}" '
            'stroke="#e0e0e0" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{left - 5:.1f}" y="{y + 3:.1f}" font-size="7" text-anchor="end">{value:.2f}</text>'
        )
    if zero_rule and low < 0.0 < high:
        y = top + plot_height - (0.0 - low) / span * plot_height
        parts.append(
            f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + plot_width:.1f}" y2="{y:.1f}" '
            'stroke="#000" stroke-width="0.9" stroke-dasharray="3 2"/>'
        )
    if y_label:
        cy = top + plot_height / 2
        parts.append(
            f'<text x="{left - 30:.1f}" y="{cy:.1f}" font-size="8" text-anchor="middle" '
            f'transform="rotate(-90 {left - 30:.1f} {cy:.1f})">{_escape(y_label)}</text>'
        )


def grouped_bar_chart(
    *,
    title: str,
    categories: Sequence[str],
    series: Sequence[tuple[str, Sequence[float]]],
    y_label: str,
    errors: Sequence[Sequence[float]] | None = None,
    width: float = COLUMN_WIDTH,
    height: float = 210.0,
    zero_rule: bool = False,
) -> str:
    left, right, top, bottom = 46.0, 10.0, 24.0, 58.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [value for _, group in series for value in group]
    low = min(0.0, min(values)) if values else 0.0
    high = max(values) if values else 1.0
    if errors:
        high = max(high, max(v + e for (_, g), es in zip(series, errors) for v, e in zip(g, es)))
    pad = (high - low) * 0.12 or 0.05
    low, high = low - (pad if low < 0 else 0.0), high + pad

    parts = _header(width, height, title)
    parts.append(f'<text x="{width/2:.1f}" y="13" font-size="9" text-anchor="middle">{_escape(title)}</text>')
    _axes(parts, left, top, plot_width, plot_height, low, high, y_label=y_label, zero_rule=zero_rule)

    slot = plot_width / max(len(categories), 1)
    bar_width = slot * 0.78 / max(len(series), 1)
    span = high - low or 1.0
    for category_index, category in enumerate(categories):
        base = left + slot * category_index + slot * 0.11
        for series_index, (_, group) in enumerate(series):
            value = group[category_index]
            x = base + bar_width * series_index
            y_value = top + plot_height - (value - low) / span * plot_height
            y_zero = top + plot_height - (max(low, 0.0) - low) / span * plot_height
            y = min(y_value, y_zero)
            bar_height = abs(y_value - y_zero)
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
                f'fill="{HATCHES[series_index % len(HATCHES)] or GREYS[series_index % len(GREYS)]}" '
                'stroke="#000" stroke-width="0.7"/>'
            )
            if errors:
                error = errors[series_index][category_index]
                if error > 0:
                    cx = x + bar_width / 2
                    y_hi = top + plot_height - (value + error - low) / span * plot_height
                    y_lo = top + plot_height - (value - error - low) / span * plot_height
                    parts.append(
                        f'<line x1="{cx:.1f}" y1="{y_hi:.1f}" x2="{cx:.1f}" y2="{y_lo:.1f}" stroke="#000" stroke-width="0.7"/>'
                        f'<line x1="{cx-2:.1f}" y1="{y_hi:.1f}" x2="{cx+2:.1f}" y2="{y_hi:.1f}" stroke="#000" stroke-width="0.7"/>'
                        f'<line x1="{cx-2:.1f}" y1="{y_lo:.1f}" x2="{cx+2:.1f}" y2="{y_lo:.1f}" stroke="#000" stroke-width="0.7"/>'
                    )
        cx = left + slot * category_index + slot / 2
        parts.append(
            f'<text x="{cx:.1f}" y="{top + plot_height + 10:.1f}" font-size="6.5" text-anchor="end" '
            f'transform="rotate(-35 {cx:.1f} {top + plot_height + 10:.1f})">{_escape(category)}</text>'
        )

    legend_y = height - 12
    x = left
    for series_index, (name, _) in enumerate(series):
        parts.append(
            f'<rect x="{x:.1f}" y="{legend_y - 6:.1f}" width="8" height="7" '
            f'fill="{HATCHES[series_index % len(HATCHES)] or GREYS[series_index % len(GREYS)]}" stroke="#000" stroke-width="0.6"/>'
        )
        parts.append(f'<text x="{x + 11:.1f}" y="{legend_y:.1f}" font-size="7">{_escape(name)}</text>')
        x += 12 + 5.2 * len(name)
    parts.append("</svg>")
    return "\n".join(parts)


def paired_seed_chart(
    *,
    title: str,
    comparisons: Sequence[tuple[str, Sequence[float]]],
    seeds: Sequence[int],
    y_label: str = "paired Macro-F1 difference",
    width: float = COLUMN_WIDTH,
    height: float = 210.0,
) -> str:
    return grouped_bar_chart(
        title=title,
        categories=[str(seed) for seed in seeds],
        series=list(comparisons),
        y_label=y_label,
        width=width,
        height=height,
        zero_rule=True,
    )


def confusion_matrix_figure(
    *, title: str, matrix: Sequence[Sequence[int]], width: float = COLUMN_WIDTH
) -> str:
    size = len(matrix)
    cell = 26.0
    left, top = 44.0, 30.0
    height = top + cell * size + 34
    parts = _header(width, height, title)
    parts.append(f'<text x="{width/2:.1f}" y="14" font-size="9" text-anchor="middle">{_escape(title)}</text>')
    peak = max((max(row) for row in matrix), default=1) or 1
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            x = left + cell * column_index
            y = top + cell * row_index
            shade = 255 - int(205 * (value / peak))
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell:.1f}" height="{cell:.1f}" '
                f'fill="rgb({shade},{shade},{shade})" stroke="#000" stroke-width="0.4"/>'
            )
            parts.append(
                f'<text x="{x + cell/2:.1f}" y="{y + cell/2 + 3:.1f}" font-size="7" text-anchor="middle" '
                f'fill="{"#ffffff" if shade < 110 else "#000000"}">{value}</text>'
            )
        parts.append(
            f'<text x="{left - 5:.1f}" y="{top + cell*row_index + cell/2 + 3:.1f}" font-size="7" text-anchor="end">{row_index}</text>'
        )
    for column_index in range(size):
        parts.append(
            f'<text x="{left + cell*column_index + cell/2:.1f}" y="{top - 5:.1f}" font-size="7" text-anchor="middle">{column_index}</text>'
        )
    parts.append(f'<text x="{left + cell*size/2:.1f}" y="{height - 16:.1f}" font-size="7.5" text-anchor="middle">predicted class</text>')
    parts.append(
        f'<text x="{left - 30:.1f}" y="{top + cell*size/2:.1f}" font-size="7.5" text-anchor="middle" '
        f'transform="rotate(-90 {left - 30:.1f} {top + cell*size/2:.1f})">true class</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def scatter_figure(
    *,
    title: str,
    groups: Sequence[tuple[str, Sequence[tuple[float, float]]]],
    x_label: str,
    y_label: str,
    caveat: str = "",
    width: float = COLUMN_WIDTH,
    height: float = 250.0,
) -> str:
    left, right, top, bottom = 46.0, 10.0, 24.0, 62.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    points = [point for _, group in groups for point in group]
    if not points:
        points = [(0.0, 0.0), (1.0, 1.0)]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_low, x_high = min(xs), max(xs)
    y_low, y_high = min(ys), max(ys)
    x_pad = (x_high - x_low) * 0.08 or 0.5
    y_pad = (y_high - y_low) * 0.08 or 0.5
    x_low, x_high = x_low - x_pad, x_high + x_pad
    y_low, y_high = y_low - y_pad, y_high + y_pad

    parts = _header(width, height, title)
    parts.append(f'<text x="{width/2:.1f}" y="13" font-size="9" text-anchor="middle">{_escape(title)}</text>')
    _axes(parts, left, top, plot_width, plot_height, y_low, y_high, y_label=y_label)
    markers = ("circle", "square", "triangle")
    for group_index, (name, group) in enumerate(groups):
        fill = GREYS[group_index % len(GREYS)]
        for x_value, y_value in group:
            x = left + (x_value - x_low) / (x_high - x_low) * plot_width
            y = top + plot_height - (y_value - y_low) / (y_high - y_low) * plot_height
            shape = markers[group_index % len(markers)]
            if shape == "circle":
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.7" fill="{fill}" stroke="none" opacity="0.75"/>')
            elif shape == "square":
                parts.append(f'<rect x="{x-1.6:.1f}" y="{y-1.6:.1f}" width="3.2" height="3.2" fill="{fill}" opacity="0.75"/>')
            else:
                parts.append(
                    f'<polygon points="{x:.1f},{y-2:.1f} {x-1.9:.1f},{y+1.6:.1f} {x+1.9:.1f},{y+1.6:.1f}" fill="{fill}" opacity="0.75"/>'
                )
    parts.append(
        f'<text x="{left + plot_width/2:.1f}" y="{top + plot_height + 16:.1f}" font-size="8" text-anchor="middle">{_escape(x_label)}</text>'
    )
    legend_y = height - (24 if caveat else 12)
    x = left
    for group_index, (name, _) in enumerate(groups):
        parts.append(
            f'<circle cx="{x + 3:.1f}" cy="{legend_y - 3:.1f}" r="2.2" fill="{GREYS[group_index % len(GREYS)]}"/>'
        )
        parts.append(f'<text x="{x + 9:.1f}" y="{legend_y:.1f}" font-size="7">{_escape(name)}</text>')
        x += 14 + 5.0 * len(name)
    if caveat:
        parts.append(
            f'<text x="{left:.1f}" y="{height - 6:.1f}" font-size="6.4" font-style="italic">{_escape(caveat)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def write_figure(path: Any, content: str) -> Any:
    from .integrity import write_text

    return write_text(path, content + "\n")
