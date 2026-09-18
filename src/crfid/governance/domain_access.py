"""Reusable allow-list contract for data-resource requests."""

from __future__ import annotations

from dataclasses import dataclass

from ..exceptions import DomainAccessViolation
from ..protocols.common import AccessRequest, Purpose, Resource


@dataclass(frozen=True)
class AccessRule:
    domain: str
    resource: Resource
    purpose: Purpose
    partition: str | None = None

    def matches(self, request: AccessRequest) -> bool:
        return (
            self.domain == request.domain
            and self.resource is request.resource
            and self.purpose is request.purpose
            and (self.partition is None or self.partition == request.partition)
        )


class AccessContract:
    def __init__(self, name: str, rules: tuple[AccessRule, ...]) -> None:
        if not name or not rules:
            raise ValueError("Access contract requires a name and rules")
        self.name = name
        self.rules = rules

    def authorize(self, request: AccessRequest) -> None:
        if not any(rule.matches(request) for rule in self.rules):
            raise DomainAccessViolation(f"Access denied by {self.name}: {request}")
