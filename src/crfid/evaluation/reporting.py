"""Explicit machine-readable report writing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json_report(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
