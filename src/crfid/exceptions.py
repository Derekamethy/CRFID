"""Dedicated exceptions for invalid scientific and execution states."""


class CRFIDError(Exception):
    """Base exception for the package."""


class ConfigurationError(CRFIDError):
    """Raised when a configuration is incomplete or unsafe."""


class DataValidationError(CRFIDError):
    """Raised when data violate a declared schema."""


class ProtocolViolation(CRFIDError):
    """Raised when a workflow violates its scientific contract."""


class DomainAccessViolation(ProtocolViolation):
    """Raised when a protocol requests forbidden domain information."""


class TargetAccessViolation(DomainAccessViolation):
    """Raised when P4 access exceeds the target-assisted contract."""


class SourceReleaseIntegrityError(ProtocolViolation):
    """Raised when an imported frozen source artifact fails verification."""


class AdaptationSpecificationError(ProtocolViolation):
    """Raised when a frozen target-assisted specification is altered."""


class SplitIsolationError(ProtocolViolation):
    """Raised when protected groups overlap across partitions."""


class RecipeFreezeError(ProtocolViolation):
    """Raised when a recipe seal is absent, invalid or changed."""


class DependencyUnavailableError(CRFIDError):
    """Raised when an optional scientific dependency is required."""


class SimulationDisabledError(CRFIDError):
    """Raised when simulation was not explicitly enabled."""
