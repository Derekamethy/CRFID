"""Explicit scientific protocol classes."""

__all__ = ["StrictDGProtocol", "TargetAssistedProtocol", "FewShotProtocol", "ExternalPortabilityProtocol"]


def __getattr__(name: str):
    if name == "StrictDGProtocol":
        from .strict_dg import StrictDGProtocol

        return StrictDGProtocol
    if name == "TargetAssistedProtocol":
        from .target_assisted import TargetAssistedProtocol

        return TargetAssistedProtocol
    if name == "FewShotProtocol":
        from .few_shot import FewShotProtocol

        return FewShotProtocol
    if name == "ExternalPortabilityProtocol":
        from .external_portability import ExternalPortabilityProtocol

        return ExternalPortabilityProtocol
    raise AttributeError(name)
