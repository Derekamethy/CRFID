"""Paper4 P1-P3 source-only loading.

Signals and labels come from the read-only canonical Strict-DG source inputs, which
contain exactly the three source domains and no target row. The fold partitions come
from the canonical ``SOURCE_ONLY_LOPO_SPLITS.csv`` so that this branch's source-side
protocol is *matched* to the canonical anchor rather than re-invented.

Nothing in this module can reach P4: the arrays it opens do not contain P4, and every
loader asserts that the domain set is exactly ``{P1, P2, P3}``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from . import paths
from .integrity import array_sha256, sha256_file

SOURCE_DOMAINS = ("P1", "P2", "P3")
TARGET_DOMAIN = "P4"
CLASS_ORDER = tuple(range(7))
RAW_LENGTH = 281
FOLD_IDS = ("S1", "S2", "S3")
FOLD_OUTER_DOMAIN = {"S1": "P1", "S2": "P2", "S3": "P3"}
PARTITIONS = ("inner_train", "inner_validation", "outer_held")


class SourceDataViolation(RuntimeError):
    """Raised when source loading would admit a target row."""


@dataclass(frozen=True)
class SourceCorpus:
    signals: np.ndarray  # [9450, 281] float64
    labels: np.ndarray  # [9450] int64
    domains: np.ndarray  # [9450] '<U2'
    signals_sha256: str
    labels_sha256: str

    def subset(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = np.asarray(indices, dtype=np.int64)
        return self.signals[rows], self.labels[rows]


@lru_cache(maxsize=1)
def load_source_corpus() -> SourceCorpus:
    """Load the canonical P1-P3 corpus and prove the target is absent."""

    signals = np.load(paths.STRICT_DG_SOURCE_SIGNALS)
    labels = np.load(paths.STRICT_DG_SOURCE_LABELS)
    if signals.ndim != 2 or signals.shape[1] != RAW_LENGTH or signals.dtype != np.float64:
        raise SourceDataViolation("Canonical source signals have an unexpected shape or dtype")
    if labels.shape != (signals.shape[0],) or labels.dtype != np.int64:
        raise SourceDataViolation("Canonical source labels are not aligned")
    if not np.isfinite(signals).all():
        raise SourceDataViolation("Canonical source signals contain non-finite values")
    if set(np.unique(labels).tolist()) != set(CLASS_ORDER):
        raise SourceDataViolation("Canonical source labels do not cover the seven classes")

    domains = _registry_domains(signals.shape[0])
    observed = set(np.unique(domains).tolist())
    if observed != set(SOURCE_DOMAINS):
        raise SourceDataViolation(f"Source corpus domain set is not P1-P3: {sorted(observed)}")
    if TARGET_DOMAIN in observed:  # pragma: no cover - unreachable given the check above
        raise SourceDataViolation("Target domain present in the source corpus")

    return SourceCorpus(
        signals=signals,
        labels=labels,
        domains=domains,
        signals_sha256=array_sha256(signals),
        labels_sha256=array_sha256(labels),
    )


@lru_cache(maxsize=1)
def _registry_domains(expected_rows: int) -> np.ndarray:
    rows: list[str] = []
    with paths.STRICT_DG_SOURCE_REGISTRY.open(newline="", encoding="utf-8") as handle:
        for index, record in enumerate(csv.DictReader(handle)):
            if int(record["registry_row"]) != index:
                raise SourceDataViolation("Canonical source registry rows are not contiguous")
            rows.append(record["position"])
    if len(rows) != expected_rows:
        raise SourceDataViolation("Canonical source registry length does not match the arrays")
    return np.asarray(rows, dtype="<U2")


@lru_cache(maxsize=1)
def load_fold_partitions() -> dict[str, dict[str, np.ndarray]]:
    """Return ``{fold_id: {partition: row indices}}`` from the canonical LOPO manifest."""

    collected: dict[str, dict[str, list[int]]] = {
        fold: {partition: [] for partition in PARTITIONS} for fold in FOLD_IDS
    }
    with paths.STRICT_DG_LOPO_SPLITS.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            fold = record["fold_id"]
            partition = record["partition"]
            position = record["position"]
            if fold not in collected or partition not in PARTITIONS:
                raise SourceDataViolation(f"Unexpected canonical split row: {fold}/{partition}")
            if position == TARGET_DOMAIN:
                raise SourceDataViolation("Canonical LOPO manifest referenced the target domain")
            if partition == "outer_held" and position != FOLD_OUTER_DOMAIN[fold]:
                raise SourceDataViolation(f"Fold {fold} outer-held domain is not {FOLD_OUTER_DOMAIN[fold]}")
            if partition != "outer_held" and position == FOLD_OUTER_DOMAIN[fold]:
                raise SourceDataViolation(f"Fold {fold} development partition leaked its held domain")
            collected[fold][partition].append(int(record["registry_row"]))

    partitions: dict[str, dict[str, np.ndarray]] = {}
    for fold, groups in collected.items():
        indexed = {name: np.asarray(sorted(values), dtype=np.int64) for name, values in groups.items()}
        union = np.concatenate(list(indexed.values()))
        if len(np.unique(union)) != len(union):
            raise SourceDataViolation(f"Fold {fold} partitions overlap")
        partitions[fold] = indexed
    return partitions


def all_source_indices() -> np.ndarray:
    return np.arange(load_source_corpus().signals.shape[0], dtype=np.int64)


def source_input_identity() -> dict[str, Any]:
    corpus = load_source_corpus()
    return {
        "signal_file": paths.STRICT_DG_SOURCE_SIGNALS.name,
        "signal_file_sha256": sha256_file(paths.STRICT_DG_SOURCE_SIGNALS),
        "label_file": paths.STRICT_DG_SOURCE_LABELS.name,
        "label_file_sha256": sha256_file(paths.STRICT_DG_SOURCE_LABELS),
        "split_file": paths.STRICT_DG_LOPO_SPLITS.name,
        "split_file_sha256": sha256_file(paths.STRICT_DG_LOPO_SPLITS),
        "registry_file": paths.STRICT_DG_SOURCE_REGISTRY.name,
        "registry_file_sha256": sha256_file(paths.STRICT_DG_SOURCE_REGISTRY),
        "signal_array_sha256": corpus.signals_sha256,
        "label_array_sha256": corpus.labels_sha256,
        "sample_count": int(corpus.signals.shape[0]),
        "raw_length": RAW_LENGTH,
        "domains_present": sorted(set(corpus.domains.tolist())),
        "target_domain_present": False,
    }
