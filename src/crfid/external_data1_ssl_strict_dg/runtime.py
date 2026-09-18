"""Deterministic runtime configuration shared by every executable in this branch.

The intra-op thread count is pinned because float reduction order in convolution and
matrix kernels depends on it: two runs of the same seed with different thread counts can
diverge. Pinning it to a constant makes every unit reproducible regardless of how many
branch processes run concurrently.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

INTRA_OP_THREADS = 4
INTER_OP_THREADS = 1


def configure(seed: int | None = None) -> dict[str, Any]:
    """Pin threads and, when a seed is given, seed every generator this branch uses."""

    os.environ.setdefault("PYTHONHASHSEED", "0")
    record: dict[str, Any] = {
        "intra_op_threads": INTRA_OP_THREADS,
        "inter_op_threads": INTER_OP_THREADS,
    }
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover
        record["torch"] = "ABSENT"
        return record
    torch.set_num_threads(INTRA_OP_THREADS)
    torch.set_num_interop_threads_available = True
    try:
        torch.set_num_interop_threads(INTER_OP_THREADS)
    except RuntimeError:
        # Already initialised in this process; the pinned value from the first call
        # stands and is recorded as such.
        record["inter_op_threads"] = int(torch.get_num_interop_threads())
    record["observed_intra_op_threads"] = int(torch.get_num_threads())
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        record["seed"] = int(seed)
    return record
