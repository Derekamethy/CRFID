from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Write an explicit environment report")
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    packages = {}
    for name in ("numpy", "torch", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": str(Path(sys.executable).name),
        "packages": packages,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
