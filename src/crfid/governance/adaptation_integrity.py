"""Hash custody for strict source import and frozen adaptation specifications."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..exceptions import AdaptationSpecificationError, SourceReleaseIntegrityError


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_source_release(release: str | Path, expected_recipe_sha256: str) -> dict[str, Any]:
    root = Path(release).resolve()
    manifest_path = root / "FROZEN_RELEASE_MANIFEST.json"
    if not manifest_path.is_file():
        raise SourceReleaseIntegrityError("Frozen source release manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("recipe_sha256") != expected_recipe_sha256:
        raise SourceReleaseIntegrityError("Strict source recipe identity changed")
    entries = [manifest["recipe"], manifest["environment"], *manifest["checkpoints"], *manifest["preprocessing"]]
    verified: list[dict[str, Any]] = []
    for entry in entries:
        relative = Path(str(entry["relative_path"]))
        marker = Path("outputs") / "strict_dg" / "frozen_release"
        try:
            local = relative.relative_to(marker)
        except ValueError as error:
            raise SourceReleaseIntegrityError("Manifest entry escapes the frozen source release") from error
        artifact = root / local
        actual = sha256_file(artifact)
        if actual != entry["sha256"]:
            raise SourceReleaseIntegrityError(f"Frozen source artifact changed: {local.as_posix()}")
        verified.append({"relative_path": local.as_posix(), "sha256": actual})
    return {
        "strict_recipe_sha256": expected_recipe_sha256,
        "manifest_semantic_sha256": manifest["manifest_semantic_sha256"],
        "checkpoint_identities": [item for item in verified if item["relative_path"].startswith("checkpoints/")],
        "preprocessing_identities": [item for item in verified if item["relative_path"].startswith("preprocessing/")],
        "imported_seeds": [int(item["seed"]) for item in manifest["checkpoints"]],
        "source_model_architecture": json.loads((root / "recipe.json").read_text(encoding="utf-8"))["architecture"],
        "input_representation": json.loads((root / "recipe.json").read_text(encoding="utf-8"))["representation"],
        "class_order": json.loads((root / "recipe.json").read_text(encoding="utf-8"))["class_order"],
        "environment_identity": verified[1],
        "verification_status": "PASS",
        "source_artifacts_modified": False,
    }


def verify_specification(specification: dict[str, Any], expected_sha256: str) -> None:
    if canonical_json_sha256(specification) != expected_sha256:
        raise AdaptationSpecificationError("Adaptation specification identity changed")
