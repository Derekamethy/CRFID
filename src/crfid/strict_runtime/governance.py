"""Small, write-scoped helpers for the strict source-only runtime."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .paths import PROJECT_ROOT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise RuntimeError(f"Write outside the public repository blocked: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
