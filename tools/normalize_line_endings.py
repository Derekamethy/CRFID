#!/usr/bin/env python3
"""Normalize UTF-8 text to the LF policy declared by .gitattributes."""

from __future__ import annotations

import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    required_markers = (ROOT / "pyproject.toml", ROOT / "README.md", ROOT / "src" / "crfid")
    if not all(path.exists() for path in required_markers):
        raise SystemExit(f"refusing unexpected repository root: {ROOT}")
    changed = 0
    for path in sorted(ROOT.rglob("*"), key=lambda item: str(item).casefold()):
        if not path.is_file() or ".git" in path.relative_to(ROOT).parts:
            continue
        data = path.read_bytes()
        if b"\x00" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        encoded = normalized.encode("utf-8")
        if encoded != data:
            if not path.stat().st_mode & stat.S_IWRITE:
                path.chmod(path.stat().st_mode | stat.S_IWRITE)
            path.write_bytes(encoded)
            changed += 1
    print(f"LINE ENDING NORMALIZATION COMPLETE: {changed} UTF-8 text files changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
