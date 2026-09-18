"""Machine-enforced target authorization for the official Strict-DG workflow.

The external audit established that the Strict-DG target loader could be called
with no recipe freeze and no authorization, and that the tested
``StrictDGProtocol`` was never instantiated by any execution script. This module
supplies the missing enforcement.

Design contract:

* A :class:`TargetAccessToken` cannot be constructed from a boolean flag, a
  ``None`` default, or a hand-written dictionary. It is minted only by
  :func:`authorize_target_access`, which re-derives every identity it asserts
  from the frozen release on disk.
* The token binds the recipe digest, the preprocessing-state digest and every
  final checkpoint digest. If any of them changes, a previously minted token
  stops validating.
* Feature access and label access are separate capabilities. Labels are
  released only after predictions have been serialized, so target labels cannot
  influence the predictions they score.
* Every access is appended to an in-token log.

Python cannot cryptographically prevent a local process from opening a file.
What this module enforces is that the *official* Strict-DG execution path
cannot reach target data without a validated token derived from a verified
frozen release.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..exceptions import DomainAccessViolation, RecipeFreezeError
from ..protocols.common import AccessRequest, Purpose, Resource
from ..protocols.strict_dg import StrictDGProtocol


TOKEN_SCHEMA_VERSION = 1
_MINT_GUARD = object()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


class TargetAuthorizationError(DomainAccessViolation):
    """Raised when target access is requested without valid authorization."""


@dataclass
class TargetAccessToken:
    """Proof that the frozen release was verified before any target read."""

    schema_version: int
    recipe_sha256: str
    preprocessing_state_sha256: str
    checkpoint_sha256_by_seed: dict[int, str]
    authorization_file_sha256: str
    frozen_release_directory: str
    issued_at_utc: str
    features_released: bool = False
    predictions_serialized: bool = False
    access_log: list[dict[str, Any]] = field(default_factory=list)
    _mint_guard: Any = None

    def __post_init__(self) -> None:
        if self._mint_guard is not _MINT_GUARD:
            raise TargetAuthorizationError(
                "TargetAccessToken must be minted by authorize_target_access(); "
                "direct construction is not a valid authorization"
            )
        self._mint_guard = None

    @property
    def token_sha256(self) -> str:
        return canonical_json_sha256(
            {
                "schema_version": self.schema_version,
                "recipe_sha256": self.recipe_sha256,
                "preprocessing_state_sha256": self.preprocessing_state_sha256,
                "checkpoint_sha256_by_seed": {
                    str(seed): value for seed, value in sorted(self.checkpoint_sha256_by_seed.items())
                },
                "authorization_file_sha256": self.authorization_file_sha256,
            }
        )

    def revalidate(self) -> None:
        """Re-derive every bound identity from disk; raise if anything moved."""

        release = Path(self.frozen_release_directory)
        recipe = release / "recipe.json"
        if not recipe.is_file() or sha256_file(recipe) != self.recipe_sha256:
            raise TargetAuthorizationError("Frozen recipe changed after authorization")
        state = release / "preprocessing" / "state.json"
        if not state.is_file():
            raise TargetAuthorizationError("Frozen preprocessing state is missing")
        declared = json.loads(state.read_text(encoding="utf-8")).get("state_sha256")
        if declared != self.preprocessing_state_sha256:
            raise TargetAuthorizationError("Frozen preprocessing state changed after authorization")
        for seed, expected in self.checkpoint_sha256_by_seed.items():
            path = release / "checkpoints" / f"seed_{seed}.pt"
            if not path.is_file() or sha256_file(path) != expected:
                raise TargetAuthorizationError(f"Frozen checkpoint changed after authorization: seed {seed}")
        authorization = release / "target_access_authorization.json"
        if not authorization.is_file() or sha256_file(authorization) != self.authorization_file_sha256:
            raise TargetAuthorizationError("Target-access authorization file changed after authorization")

    def release_features(self, *, purpose: str) -> None:
        self.revalidate()
        self.features_released = True
        self.access_log.append(
            {"at_utc": utc_now(), "resource": "features", "purpose": purpose, "granted": True}
        )

    def mark_predictions_serialized(self) -> None:
        if not self.features_released:
            raise TargetAuthorizationError("Predictions cannot be serialized before features are released")
        self.predictions_serialized = True
        self.access_log.append(
            {"at_utc": utc_now(), "resource": "predictions", "purpose": "serialization", "granted": True}
        )

    def release_labels(self, *, purpose: str) -> None:
        """Labels are a separate capability, unlocked only for final scoring."""

        if purpose != "final_scoring":
            self.access_log.append(
                {"at_utc": utc_now(), "resource": "labels", "purpose": purpose, "granted": False}
            )
            raise TargetAuthorizationError(
                f"Target labels may only be released for final_scoring, not {purpose!r}"
            )
        if not self.predictions_serialized:
            self.access_log.append(
                {"at_utc": utc_now(), "resource": "labels", "purpose": purpose, "granted": False}
            )
            raise TargetAuthorizationError(
                "Target labels cannot be released before predictions are serialized"
            )
        self.revalidate()
        self.access_log.append(
            {"at_utc": utc_now(), "resource": "labels", "purpose": purpose, "granted": True}
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "token_sha256": self.token_sha256,
            "recipe_sha256": self.recipe_sha256,
            "preprocessing_state_sha256": self.preprocessing_state_sha256,
            "checkpoint_sha256_by_seed": {
                str(seed): value for seed, value in sorted(self.checkpoint_sha256_by_seed.items())
            },
            "authorization_file_sha256": self.authorization_file_sha256,
            "issued_at_utc": self.issued_at_utc,
            "features_released": self.features_released,
            "predictions_serialized": self.predictions_serialized,
            "access_log": list(self.access_log),
        }


def authorize_target_access(
    *,
    frozen_release_directory: Path,
    protocol: StrictDGProtocol,
    expected_seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
) -> TargetAccessToken:
    """Verify the frozen release, drive the protocol gate, and mint a token.

    Every identity asserted by the returned token is re-derived here from disk.
    The protocol object is genuinely exercised: the recipe is frozen from the
    on-disk recipe, the seal is verified against it, and the resulting
    authorization is checked with a real :class:`AccessRequest`.
    """

    release = Path(frozen_release_directory)
    if not release.is_dir():
        raise TargetAuthorizationError(f"Frozen release directory is missing: {release}")

    recipe_path = release / "recipe.json"
    sidecar = release / "recipe.sha256"
    authorization_path = release / "target_access_authorization.json"
    manifest_path = release / "FROZEN_RELEASE_MANIFEST.json"
    state_path = release / "preprocessing" / "state.json"
    for required in (recipe_path, sidecar, authorization_path, manifest_path, state_path):
        if not required.is_file():
            raise TargetAuthorizationError(f"Frozen release is incomplete: {required.name} is missing")

    recipe_digest = sha256_file(recipe_path)
    declared_digest = sidecar.read_text(encoding="ascii").split()[0]
    if recipe_digest != declared_digest:
        raise TargetAuthorizationError("Frozen recipe does not match its recorded digest")

    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("authorized") is not True:
        raise TargetAuthorizationError("Target access is not authorized")
    if authorization.get("recipe_sha256") != recipe_digest:
        raise TargetAuthorizationError("Authorization is bound to a different recipe")

    # The historical authorization records its own canonical-JSON digest. Verify
    # it rather than trusting it, which the historical evaluation script did not.
    declared_authorization_digest = authorization.get("authorization_sha256")
    recomputed = canonical_json_sha256(
        {k: v for k, v in authorization.items() if k != "authorization_sha256"}
    )
    if declared_authorization_digest != recomputed:
        raise TargetAuthorizationError(
            "Target-access authorization payload does not match its own declared digest"
        )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    preprocessing_digest = state.get("state_sha256")
    recomputed_state = canonical_json_sha256({k: v for k, v in state.items() if k != "state_sha256"})
    if preprocessing_digest != recomputed_state:
        raise TargetAuthorizationError("Frozen preprocessing state does not match its own declared digest")
    if authorization.get("preprocessing_state_sha256") != preprocessing_digest:
        raise TargetAuthorizationError("Authorization is bound to a different preprocessing state")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_AND_TARGET_AUTHORIZED":
        raise TargetAuthorizationError("Frozen release is not marked frozen and authorized")
    declared_checkpoints = {int(row["seed"]): row for row in manifest.get("checkpoints", [])}
    if tuple(sorted(declared_checkpoints)) != tuple(sorted(expected_seeds)):
        raise TargetAuthorizationError("Frozen release does not declare the expected seed set")
    if authorization.get("checkpoint_count") != len(expected_seeds):
        raise TargetAuthorizationError("Authorization checkpoint count does not match the seed set")

    checkpoint_digests: dict[int, str] = {}
    for seed, row in sorted(declared_checkpoints.items()):
        path = release / "checkpoints" / f"seed_{seed}.pt"
        if not path.is_file():
            raise TargetAuthorizationError(f"Frozen checkpoint is missing: seed {seed}")
        observed = sha256_file(path)
        if observed != row["sha256"]:
            raise TargetAuthorizationError(f"Frozen checkpoint does not match the manifest: seed {seed}")
        checkpoint_digests[seed] = observed

    # Drive the real protocol gate. Before freeze it must refuse; after a
    # verified freeze it must permit evaluation only.
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    try:
        protocol.authorize(AccessRequest("P4", Resource.FEATURES, Purpose.EVALUATION))
    except DomainAccessViolation:
        pass
    else:
        raise TargetAuthorizationError(
            "Protocol object already permitted P4 before this authorization was issued"
        )
    seal = protocol.freeze_recipe(recipe)
    try:
        protocol.authorize_final_evaluation(recipe, seal)
    except RecipeFreezeError as exc:  # pragma: no cover - defensive
        raise TargetAuthorizationError(f"Recipe seal verification failed: {exc}") from exc
    protocol.authorize(AccessRequest("P4", Resource.FEATURES, Purpose.EVALUATION))
    for forbidden in (Purpose.TRAINING, Purpose.ADAPTATION, Purpose.SELECTION):
        try:
            protocol.authorize(AccessRequest("P4", Resource.FEATURES, forbidden))
        except DomainAccessViolation:
            continue
        raise TargetAuthorizationError(f"Protocol permitted a forbidden P4 purpose: {forbidden}")

    return TargetAccessToken(
        schema_version=TOKEN_SCHEMA_VERSION,
        recipe_sha256=recipe_digest,
        preprocessing_state_sha256=preprocessing_digest,
        checkpoint_sha256_by_seed=checkpoint_digests,
        authorization_file_sha256=sha256_file(authorization_path),
        frozen_release_directory=str(release.resolve()),
        issued_at_utc=utc_now(),
        _mint_guard=_MINT_GUARD,
    )


def require_token(token: TargetAccessToken | None) -> TargetAccessToken:
    """Reject a missing, wrongly typed or stale authorization."""

    if token is None:
        raise TargetAuthorizationError(
            "Strict-DG target access requires a TargetAccessToken; there is no default authorization"
        )
    if not isinstance(token, TargetAccessToken):
        raise TargetAuthorizationError(
            f"Strict-DG target access requires a TargetAccessToken, received {type(token).__name__}"
        )
    token.revalidate()
    return token
