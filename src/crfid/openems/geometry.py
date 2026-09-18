"""Pure geometry helpers for the assumption-bounded resonator model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np


RESONATORS = ("Lc", "L3", "L2", "L1")
BIT_RESONATORS = ("L3", "L2", "L1")


def validate_code(code: str) -> None:
    if len(code) != 3 or any(bit not in "01" for bit in code):
        raise ValueError("Code must contain exactly three binary digits")


def present_resonators(code: str) -> set[str]:
    validate_code(code)
    return {"Lc", *(name for bit, name in zip(code, BIT_RESONATORS) if bit == "1")}


def candidate_geometry(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    lengths = {name: float(base["resonator_lengths_mm"][name]) for name in RESONATORS}
    widths = {name: float(base["nominal_strip_width_mm"]) for name in RESONATORS}
    for name, scale in overrides.get("length_scale", {}).items():
        lengths[name] *= float(scale)
    for name, scale in overrides.get("width_scale", {}).items():
        widths[name] *= float(scale)
    geometry = {
        "lengths_mm": lengths,
        "widths_mm": widths,
        "center_spacing_mm": float(overrides.get("center_spacing_mm", base["nominal_center_spacing_mm"])),
        "strip_angle_deg": float(base["strip_angle_deg"]),
        "substrate_diameter_mm": float(base["substrate_diameter_mm"]),
        "substrate_thickness_mm": float(base["substrate_thickness_mm"]),
        "relative_permittivity": float(base["relative_permittivity"]),
        "loss_tangent": float(base["loss_tangent"]),
    }
    validate_geometry(geometry)
    return geometry


def validate_geometry(geometry: Mapping[str, Any]) -> None:
    radius = float(geometry["substrate_diameter_mm"]) / 2.0
    spacing = float(geometry["center_spacing_mm"])
    offsets = np.linspace(-1.5 * spacing, 1.5 * spacing, 4)
    for name, offset in zip(RESONATORS, offsets):
        farthest = math.hypot(
            float(geometry["lengths_mm"][name]) / 2.0,
            abs(float(offset)) + float(geometry["widths_mm"][name]) / 2.0,
        )
        if farthest >= radius - 0.25:
            raise ValueError(f"Assumed {name} strip does not fit the substrate")
    if min(float(value) for value in geometry["widths_mm"].values()) <= 0:
        raise ValueError("Strip widths must be positive")


def strip_polygon(length: float, width: float, angle_degrees: float, offset: float) -> np.ndarray:
    angle = math.radians(angle_degrees)
    along = np.asarray([math.cos(angle), math.sin(angle)])
    across = np.asarray([-math.sin(angle), math.cos(angle)])
    center = offset * across
    return np.asarray(
        [
            center - 0.5 * length * along - 0.5 * width * across,
            center + 0.5 * length * along - 0.5 * width * across,
            center + 0.5 * length * along + 0.5 * width * across,
            center - 0.5 * length * along + 0.5 * width * across,
        ]
    )
