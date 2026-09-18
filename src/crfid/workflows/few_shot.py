"""Condition-disjoint target prototype few-shot workflow.

Scope note: this runner is a structural planner over a single support/query
episode. It is **not** the authoritative Few-Shot implementation and does not
reproduce any reported result. The authoritative line was executed once, audited
and permanently sealed; its executed code and artifacts are preserved under
``outputs/few_shot/frozen_branch``. To verify the reported numbers use
``workflows/05_few_shot_p4_adaptation/verify_frozen_few_shot_branch.py``, and
read the frozen artifacts through :mod:`crfid.few_shot`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..adaptation.prototypes import build_target_prototypes, squared_euclidean_predict
from ..adaptation.support_query import SupportQuerySplit
from .common import WorkflowPlan, build_plan, load_npz


def run(config: dict[str, Any], *, execute: bool = False) -> WorkflowPlan | dict[str, Any]:
    plan = build_plan(config, "CONTROLLED_TARGET_FEW_SHOT_ADAPTATION", execute)
    if not execute:
        return plan
    payload = load_npz(
        str(config["input_path"]),
        ("support_embeddings", "support_labels", "support_sample_ids", "support_conditions", "query_embeddings", "query_conditions"),
    )
    split = SupportQuerySplit(
        support_indices=np.arange(len(payload["support_labels"])),
        query_indices=np.arange(len(payload["query_embeddings"])) + len(payload["support_labels"]),
        support_conditions=tuple(payload["support_conditions"].astype(str).tolist()),
        query_conditions=tuple(payload["query_conditions"].astype(str).tolist()),
    )
    split.validate(len(payload["support_labels"]) + len(payload["query_embeddings"]))
    shot = int(config["shot_count"])
    prototypes = build_target_prototypes(
        payload["support_embeddings"].astype("float32"),
        payload["support_labels"],
        payload["support_sample_ids"].astype(str).tolist(),
        shot_count=shot,
    )
    prediction, distances = squared_euclidean_predict(payload["query_embeddings"].astype("float32"), prototypes)
    return {"status": "QUERY_PREDICTIONS_CREATED_LABELS_SEALED", "prediction": prediction, "distances": distances}
