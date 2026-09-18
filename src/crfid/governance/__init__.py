"""Scientific governance and integrity helpers."""

from .domain_access import AccessContract
from .recipe_freeze import RecipeSeal

__all__ = [
    "AccessContract",
    "RecipeSeal",
    "ManifestSpec",
    "generate_release_manifests",
    "verify_release_manifests",
    "TargetAccessToken",
    "authorize_target_access",
    "require_token",
]


def __getattr__(name: str):
    """Expose the Strict-DG governance API without import-time side effects."""

    if name in {"ManifestSpec", "generate_release_manifests", "verify_release_manifests"}:
        from . import strict_release_manifest

        return getattr(strict_release_manifest, name)
    if name in {"TargetAccessToken", "authorize_target_access", "require_token"}:
        from . import strict_target_authorization

        return getattr(strict_target_authorization, name)
    raise AttributeError(name)
