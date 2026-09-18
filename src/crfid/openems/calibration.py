"""Real-measurement calibration boundary."""

from __future__ import annotations

from pathlib import Path


def calibration_status(measured_csv: str | Path | None) -> dict[str, object]:
    if measured_csv is None:
        return {"status": "WAITING_FOR_REAL_VNA_CSV", "calibration_applied": False}
    path = Path(measured_csv)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"status": "REAL_DATA_SUPPLIED_VALIDATION_REQUIRED", "calibration_applied": False}
