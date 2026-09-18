"""External unlabelled Data1 assistance under a strict source-only Paper4 P1-P3 -> P4 protocol.

This branch is scientifically independent of:

* the existing ``crfid.external_data1`` portability branch (within-Data1 pipeline
  validation), which it never imports, reads for decisions, or modifies;
* the historical retrospective P4-informed SSL/NCM/flooding experiments, which are
  reference-only and whose selections are treated as target-tainted;
* ``crfid.few_shot``, which uses P4 support samples and is a different protocol.

Every module here is newly implemented. The canonical Strict-DG protocol is
reconstructed from current v2 evidence and reproduced as a *matched* control so
that any measured gain can be attributed to self-supervision or to external
Data1 rather than to an architecture or implementation change.
"""

from __future__ import annotations

BRANCH_ID = "external_data1_ssl_strict_dg"
BRANCH_CLAIM_TYPE = "EXTERNAL_UNLABELLED_DATA_ASSISTED_STRICT_SOURCE_ONLY_DG"
SCHEMA_VERSION = 1

__all__ = ["BRANCH_ID", "BRANCH_CLAIM_TYPE", "SCHEMA_VERSION"]
