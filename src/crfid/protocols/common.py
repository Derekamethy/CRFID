"""Common protocol request types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Resource(str, Enum):
    FEATURES = "features"
    LABELS = "labels"
    OUTCOMES = "outcomes"


class Purpose(str, Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    SELECTION = "selection"
    ADAPTATION = "adaptation"
    EVALUATION = "evaluation"


@dataclass(frozen=True)
class AccessRequest:
    domain: str
    resource: Resource
    purpose: Purpose
    partition: str | None = None


@dataclass(frozen=True)
class ProtocolDeclaration:
    name: str
    scientific_label: str
    allowed_domains: tuple[str, ...]
    target_assisted: bool
