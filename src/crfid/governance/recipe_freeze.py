"""Immutable recipe seals."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _canonical_bytes(recipe: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(recipe), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class RecipeSeal:
    algorithm: str
    digest: str


def create_recipe_seal(recipe: Mapping[str, Any]) -> RecipeSeal:
    return RecipeSeal("sha256", hashlib.sha256(_canonical_bytes(recipe)).hexdigest())


def verify_recipe_seal(recipe: Mapping[str, Any], seal: RecipeSeal) -> bool:
    return seal.algorithm == "sha256" and create_recipe_seal(recipe).digest == seal.digest
