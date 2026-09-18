"""Conservative redesign confirmation gates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfirmationEvidence:
    baseline_global_minimum: float
    candidate_global_minimum: float
    baseline_worst_pair: str
    candidate_worst_pair: str
    convergence_rms_db: float
    maximum_peak_shift_mhz: float
    resonance_gate_passed: bool


def assess_confirmation(
    evidence: ConfirmationEvidence,
    *,
    minimum_improvement_fraction: float,
    convergence_rms_db_max: float,
    peak_shift_mhz_max: float,
) -> dict[str, object]:
    if evidence.baseline_global_minimum <= 0:
        raise ValueError("Baseline global minimum must be positive")
    improvement = (
        evidence.candidate_global_minimum - evidence.baseline_global_minimum
    ) / evidence.baseline_global_minimum
    gates = {
        "global_minimum_improved": improvement >= minimum_improvement_fraction,
        "convergence_passed": evidence.convergence_rms_db <= convergence_rms_db_max,
        "peak_shift_passed": evidence.maximum_peak_shift_mhz <= peak_shift_mhz_max,
        "resonance_gate_passed": evidence.resonance_gate_passed,
    }
    status = "CONFIRMED" if all(gates.values()) else "NOT_CONFIRMED"
    return {
        "status": status,
        "global_minimum_improvement_fraction": improvement,
        "bottleneck_moved": evidence.baseline_worst_pair != evidence.candidate_worst_pair,
        "gates": gates,
    }
