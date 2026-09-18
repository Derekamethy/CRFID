"""Machine-enforced Gate-E access to the sealed Paper4 P4 target domain.

Contract, mirroring the canonical Strict-DG target gate but bound to this branch's own
preregistration:

* a :class:`PreregistrationToken` cannot be hand-constructed; it is minted only by
  :func:`authorize_target_access`, which re-derives the preregistration digest and every
  frozen checkpoint digest from disk;
* features and labels are separate capabilities;
* labels are released only for ``final_scoring`` and only after every prediction array
  has been serialised and hashed, so target labels cannot influence the predictions they
  score;
* once labels have been released the token refuses to release features again, so no
  prediction can be regenerated after label access;
* every access attempt, granted or refused, is appended to an access ledger.

Python cannot stop a local process from opening a file. What is enforced here is that
this branch's *official* evaluation path cannot reach P4 without a validated token
derived from a frozen preregistration.
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import paths
from .integrity import canonical_json_sha256, sha256_file, utc_now

_MINT_GUARD = object()
TOKEN_SCHEMA_VERSION = 1
SIGNAL_POINT_COUNT = 281
SURFACE_COLUMNS = ("A3", "A2", "A1")
POSITION_COLUMNS = ("P4", "P3", "P2", "P1")
METADATA_COLUMNS = SURFACE_COLUMNS + POSITION_COLUMNS + ("ER", "TagID")
EXPECTED_HEADER = METADATA_COLUMNS + tuple(str(index) for index in range(SIGNAL_POINT_COUNT))


class TargetAuthorizationError(RuntimeError):
    """Raised when target access is requested without valid authorization."""


@dataclass
class PreregistrationToken:
    schema_version: int
    preregistration_sha256: str
    checkpoint_sha256_by_unit: dict[str, str]
    registered_treatments: tuple[str, ...]
    preregistration_path: str
    checkpoint_directory: str
    issued_at_utc: str
    features_released: bool = False
    predictions_closed: bool = False
    labels_released: bool = False
    access_log: list[dict[str, Any]] = field(default_factory=list)
    _mint_guard: Any = None

    def __post_init__(self) -> None:
        if self._mint_guard is not _MINT_GUARD:
            raise TargetAuthorizationError(
                "PreregistrationToken must be minted by authorize_target_access(); "
                "direct construction is not a valid authorization"
            )
        self._mint_guard = None

    @property
    def token_sha256(self) -> str:
        return canonical_json_sha256(
            {
                "schema_version": self.schema_version,
                "preregistration_sha256": self.preregistration_sha256,
                "checkpoint_sha256_by_unit": dict(sorted(self.checkpoint_sha256_by_unit.items())),
                "registered_treatments": sorted(self.registered_treatments),
            }
        )

    def _log(self, resource: str, purpose: str, granted: bool, detail: str = "") -> None:
        self.access_log.append(
            {
                "at_utc": utc_now(),
                "resource": resource,
                "purpose": purpose,
                "granted": granted,
                "detail": detail,
            }
        )

    def revalidate(self) -> None:
        """Re-derive every bound identity from disk; raise if anything moved."""

        path = Path(self.preregistration_path)
        if not path.is_file() or sha256_file(path) != self.preregistration_sha256:
            raise TargetAuthorizationError("Preregistration changed after authorization")
        directory = Path(self.checkpoint_directory)
        for unit, expected in self.checkpoint_sha256_by_unit.items():
            checkpoint = directory / f"{unit}.pt"
            if not checkpoint.is_file() or sha256_file(checkpoint) != expected:
                raise TargetAuthorizationError(f"Frozen checkpoint changed after authorization: {unit}")

    def require_registered(self, treatment_id: str, unit: str) -> None:
        if treatment_id not in self.registered_treatments:
            self._log("checkpoint", "prediction", False, treatment_id)
            raise TargetAuthorizationError(
                f"{treatment_id} is not a preregistered treatment; the evaluator refuses it"
            )
        if unit not in self.checkpoint_sha256_by_unit:
            self._log("checkpoint", "prediction", False, unit)
            raise TargetAuthorizationError(f"{unit} is not a preregistered evaluation unit")

    def release_features(self, *, purpose: str) -> None:
        if self.labels_released:
            self._log("features", purpose, False, "after_label_release")
            raise TargetAuthorizationError(
                "Target features cannot be released after labels; regeneration is forbidden"
            )
        self.revalidate()
        self.features_released = True
        self._log("features", purpose, True)

    def close_predictions(self, prediction_digests: dict[str, str]) -> None:
        if not self.features_released:
            self._log("predictions", "closure", False, "features_not_released")
            raise TargetAuthorizationError("Predictions cannot be closed before features are released")
        missing = sorted(set(self.checkpoint_sha256_by_unit) - set(prediction_digests))
        if missing:
            self._log("predictions", "closure", False, f"missing:{missing[:3]}")
            raise TargetAuthorizationError(f"Prediction set is incomplete: {missing[:5]}")
        extra = sorted(set(prediction_digests) - set(self.checkpoint_sha256_by_unit))
        if extra:
            self._log("predictions", "closure", False, f"unregistered:{extra[:3]}")
            raise TargetAuthorizationError(f"Prediction set contains unregistered units: {extra[:5]}")
        self.predictions_closed = True
        self._log("predictions", "closure", True, f"units:{len(prediction_digests)}")

    def release_labels(self, *, purpose: str) -> None:
        if purpose != "final_scoring":
            self._log("labels", purpose, False)
            raise TargetAuthorizationError(
                f"Target labels may only be released for final_scoring, not {purpose!r}"
            )
        if not self.predictions_closed:
            self._log("labels", purpose, False, "predictions_not_closed")
            raise TargetAuthorizationError(
                "Target labels cannot be released before predictions are serialised and hashed"
            )
        self.revalidate()
        self.labels_released = True
        self._log("labels", purpose, True)

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "token_sha256": self.token_sha256,
            "preregistration_sha256": self.preregistration_sha256,
            "registered_treatments": sorted(self.registered_treatments),
            "evaluation_unit_count": len(self.checkpoint_sha256_by_unit),
            "issued_at_utc": self.issued_at_utc,
            "features_released": self.features_released,
            "predictions_closed": self.predictions_closed,
            "labels_released": self.labels_released,
            "access_log": list(self.access_log),
        }


def authorize_target_access(
    *, preregistration_path: Path, checkpoint_directory: Path
) -> PreregistrationToken:
    """Verify the frozen preregistration and checkpoints, then mint a token."""

    import json

    path = Path(preregistration_path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file():
        raise TargetAuthorizationError(f"Preregistration is missing: {path}")
    if not sidecar.is_file():
        raise TargetAuthorizationError(f"Preregistration digest sidecar is missing: {sidecar}")
    digest = sha256_file(path)
    declared = sidecar.read_text(encoding="ascii").split()[0]
    if digest != declared:
        raise TargetAuthorizationError("Preregistration does not match its recorded digest")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("gate_d_status") != "PASS_FINAL_PREREGISTRATION_FROZEN":
        raise TargetAuthorizationError("Preregistration is not marked frozen")
    if payload.get("target_accessed_before_preregistration") is not False:
        raise TargetAuthorizationError("Preregistration does not assert a sealed target")

    directory = Path(checkpoint_directory)
    units: dict[str, str] = {}
    for entry in payload["evaluation_units"]:
        unit = entry["unit_id"]
        checkpoint = directory / f"{unit}.pt"
        if not checkpoint.is_file():
            raise TargetAuthorizationError(f"Frozen checkpoint is missing: {unit}")
        observed = sha256_file(checkpoint)
        if observed != entry["checkpoint_sha256"]:
            raise TargetAuthorizationError(f"Frozen checkpoint does not match the preregistration: {unit}")
        units[unit] = observed
    if not units:
        raise TargetAuthorizationError("Preregistration declares no evaluation units")

    return PreregistrationToken(
        schema_version=TOKEN_SCHEMA_VERSION,
        preregistration_sha256=digest,
        checkpoint_sha256_by_unit=units,
        registered_treatments=tuple(sorted(payload["registered_treatments"])),
        preregistration_path=str(path.resolve()),
        checkpoint_directory=str(directory.resolve()),
        issued_at_utc=utc_now(),
        _mint_guard=_MINT_GUARD,
    )


class SealedLabels:
    """Target labels that can only be opened through a token, once predictions close."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = np.ascontiguousarray(values, dtype=np.int64)
        self._values.setflags(write=False)

    def open(self, token: PreregistrationToken, *, purpose: str = "final_scoring") -> np.ndarray:
        token.release_labels(purpose=purpose)
        return self._values

    def __len__(self) -> int:
        return int(self._values.shape[0])


@dataclass(frozen=True)
class TargetDataset:
    signals: np.ndarray
    sealed_labels: SealedLabels
    surfaces: np.ndarray
    condition_ids: np.ndarray
    file_records: tuple[dict[str, Any], ...]
    signals_sha256: str


def _active_one_hot(row: dict[str, str], columns: tuple[str, ...], role: str) -> str:
    values = {column: float(row[column]) for column in columns}
    if any(value not in (0.0, 1.0) for value in values.values()):
        raise TargetAuthorizationError(f"Invalid {role} one-hot values in the target file")
    active = [column for column, value in values.items() if value == 1.0]
    if len(active) != 1:
        raise TargetAuthorizationError(f"Expected exactly one active {role}")
    return active[0]


def load_target(token: PreregistrationToken, *, purpose: str) -> TargetDataset:
    """Read the three sealed P4 files exactly once, with labels kept sealed."""

    token.release_features(purpose=purpose)
    root = paths.sealed_target_root()
    signals: list[np.ndarray] = []
    labels: list[int] = []
    surfaces: list[str] = []
    conditions: list[str] = []
    records: list[dict[str, Any]] = []
    for name in paths.SEALED_P4_FILE_NAMES:
        path = root / name
        expected_surface = name.split("_")[0]
        raw_bytes = path.read_bytes()
        decoded = raw_bytes.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(decoded, newline=""))
        if reader.fieldnames is None or tuple(reader.fieldnames) != EXPECTED_HEADER:
            raise TargetAuthorizationError(f"Unexpected target schema: {name}")
        rows = 0
        for row in reader:
            surface = _active_one_hot(row, SURFACE_COLUMNS, "surface")
            position = _active_one_hot(row, POSITION_COLUMNS, "position")
            if surface != expected_surface or position != "P4":
                raise TargetAuthorizationError(f"Target file/domain mismatch: {name}")
            tag_id = int(float(row["TagID"]))
            er = int(float(row["ER"]))
            if tag_id not in range(1, 8) or er not in {0, 1, 2}:
                raise TargetAuthorizationError("Unexpected target tag/permittivity metadata")
            values = np.fromiter(
                (float(row[str(index)]) for index in range(SIGNAL_POINT_COUNT)),
                dtype=np.float64,
                count=SIGNAL_POINT_COUNT,
            )
            if not np.isfinite(values).all():
                raise TargetAuthorizationError("Non-finite target signal")
            signals.append(values)
            labels.append(tag_id - 1)
            surfaces.append(surface)
            conditions.append(f"tag={tag_id}|er={er}|surface={surface}|position=P4")
            rows += 1
        records.append(
            {
                "file_name": name,
                "file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "row_count": rows,
                "size_bytes": len(raw_bytes),
            }
        )
    stacked = np.ascontiguousarray(np.stack(signals, axis=0))
    if stacked.shape != (3150, SIGNAL_POINT_COUNT):
        raise TargetAuthorizationError(f"Unexpected target population shape: {stacked.shape}")
    digest = hashlib.sha256()
    digest.update(str(stacked.dtype).encode("ascii"))
    digest.update(str(stacked.shape).encode("ascii"))
    digest.update(stacked.tobytes(order="C"))
    return TargetDataset(
        signals=stacked,
        sealed_labels=SealedLabels(np.asarray(labels, dtype=np.int64)),
        surfaces=np.asarray(surfaces, dtype="<U2"),
        condition_ids=np.asarray(conditions, dtype=object),
        file_records=tuple(records),
        signals_sha256=digest.hexdigest(),
    )
