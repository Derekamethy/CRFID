"""Machine-readable claim boundary for External Data1."""

from __future__ import annotations

from ..exceptions import ProtocolViolation
from ..external_data1.schema import SCIENTIFIC_CLAIM_TYPE


PRIMARY_CLAIM = (
    "The canonical preprocessing, classical baseline and CNN evaluation pipeline "
    "was successfully applied to an independently acquired four-class Data1 dataset "
    "under a frozen within-Data1 train/test protocol."
)
ALLOWED_CONCLUSIONS = {
    "DATA1_LEARNABILITY_CLEAR",
    "CNN_PIPELINE_PORTABILITY_SUPPORTED",
    "FIRST_DIFFERENCE_PORTABILITY_NOT_SUPPORTED",
}
FORBIDDEN_CLAIM_FRAGMENTS = (
    "seven-class model generalized",
    "cross-dataset zero-shot",
    "validates paper4 class",
    "improves strict dg",
    "target-assisted",
    "ssl pretraining was executed",
)


def validate_external_claim(
    claim_type: str, primary_claim: str, conclusions: set[str]
) -> None:
    if claim_type != SCIENTIFIC_CLAIM_TYPE:
        raise ProtocolViolation("External Data1 scientific claim type changed")
    if primary_claim != PRIMARY_CLAIM:
        raise ProtocolViolation("External Data1 primary claim exceeds the frozen boundary")
    if not conclusions.issubset(ALLOWED_CONCLUSIONS):
        raise ProtocolViolation("Unsupported External Data1 conclusion")
    normalized = primary_claim.casefold()
    if any(fragment in normalized for fragment in FORBIDDEN_CLAIM_FRAGMENTS):
        raise ProtocolViolation("Forbidden cross-dataset claim detected")


def reject_direct_external_transfer_claim(claim: str) -> None:
    normalized = claim.casefold()
    if any(fragment in normalized for fragment in FORBIDDEN_CLAIM_FRAGMENTS):
        raise ProtocolViolation("Direct external-transfer claims are unsupported")

