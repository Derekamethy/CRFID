"""Verify an extracted Few-Shot full-evidence bundle without reading scientific values."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

EXPECTED_FILES = 5807
EXPECTED_BYTES = 268302053
EXPECTED_AGGREGATE = "ab17eeda5eebeac518a4cf2eb0998fb77686acbb162c1ed736b71bfef9627d23"
EXPECTED_MANIFEST_SHA256 = "d5645c8840504a60d3126ecea585ceeff9f3bd1909fd297fd899c4f51e68d249"
EXPECTED_LINE_SEAL = "887afb78fec01e4f3b374447907165ddbf006ad9471e938568be217bf8692386"
SEAL_PATH = "06_final_comparative_audit_and_archive/FINAL_FEW_SHOT_LINE_SEAL.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    frozen = bundle / "frozen_branch"
    manifest = bundle / "frozen_payload_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    declared = {row["relative_path"]: row for row in rows}
    actual = {
        path.relative_to(frozen).as_posix(): path
        for path in frozen.rglob("*")
        if path.is_file()
    }
    aggregate = hashlib.sha256()
    mismatches: list[str] = []
    for row in rows:
        relative = row["relative_path"]
        aggregate.update(
            f"{relative}|{row['size_bytes']}|{row['sha256']}\n".encode("utf-8")
        )
        path = actual.get(relative)
        if path is None:
            mismatches.append(f"missing:{relative}")
        elif path.stat().st_size != int(row["size_bytes"]):
            mismatches.append(f"size:{relative}")
        elif sha256_file(path) != row["sha256"]:
            mismatches.append(f"sha256:{relative}")
    seal = json.loads((frozen / SEAL_PATH).read_text(encoding="utf-8"))
    checks = {
        "manifest_sha256": sha256_file(manifest) == EXPECTED_MANIFEST_SHA256,
        "file_count": len(rows) == EXPECTED_FILES,
        "total_bytes": sum(int(row["size_bytes"]) for row in rows) == EXPECTED_BYTES,
        "inventory_aggregate": aggregate.hexdigest() == EXPECTED_AGGREGATE,
        "exact_file_set": set(actual) == set(declared),
        "all_file_hashes": not mismatches,
        "authoritative_line_seal": seal["seal_sha256"] == EXPECTED_LINE_SEAL,
    }
    result = {
        "status": "PASS_EXTERNAL_FEW_SHOT_EVIDENCE" if all(checks.values()) else "FAIL_EXTERNAL_FEW_SHOT_EVIDENCE",
        "checks": checks,
        "mismatches": mismatches[:50],
        "sealed_query_label_values_read": False,
        "model_or_inference_executed": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
