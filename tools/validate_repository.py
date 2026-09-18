from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".toml", ".cff", ".txt", ".json"}
REQUIRED = {
    "README.md",
    "PROJECT_STATUS.md",
    "EXPERIMENT_REGISTRY.md",
    "SCIENTIFIC_SPINE.md",
    "REPRODUCIBILITY.md",
    "DATA_LINEAGE.md",
    "pyproject.toml",
    "src/crfid",
    "configs",
    "workflows",
    "tests",
    "results/canonical_metrics",
}


def prohibited_terms() -> tuple[str, ...]:
    return (
        "whole" + " code",
        "last" + " code",
        "Freezing" + " Five",
        "Final " + "whole" + " code",
        "new" + "code",
        "Dataset" + "-2-Compare",
        "Co" + "dex",
        "Clau" + "de",
        "One" + "Drive",
        "Stage" + " 17",
        "Round" + " 6",
        "Round" + " 7",
        "R" + "6",
        "R" + "7",
    )


def main() -> int:
    failures = []
    for required in sorted(REQUIRED):
        if not (ROOT / required).exists():
            failures.append(f"missing:{required}")
    windows_root = r"[A-Za-z]:[\\/](?:" + "Users|Documents and Settings" + r")[\\/]"
    unix_root = "/" + "home/" + r"[^/]+/"
    absolute_pattern = re.compile(windows_root + "|" + unix_root)
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            except SyntaxError as exc:
                failures.append(f"syntax:{relative}:{exc.lineno}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if absolute_pattern.search(text):
                failures.append(f"absolute_path:{relative}")
            for term in prohibited_terms():
                if term in text or term in relative:
                    failures.append(f"prohibited_name:{relative}")
                    break
    result = {"verdict": "PASS" if not failures else "FAIL", "failure_count": len(failures), "failures": failures}
    print(json.dumps(result, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
