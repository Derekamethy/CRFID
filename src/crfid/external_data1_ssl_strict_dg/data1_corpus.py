"""Label-free external Data1 signal corpus.

Data1 is an *unlabelled* external corpus for this branch. The label column exists in
the source CSVs and is stripped at the file-reading boundary. It is read exactly once
per file, only to confirm provenance against the frozen Data1 authority (integral
values inside the declared local class order), and is then discarded. No function in
this module returns, stores, caches or serialises a Data1 label, and the cached
corpus artefact contains signals only.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from . import paths
from .integrity import array_sha256, sha256_file

SET_NAMES = tuple(f"set_{index}.csv" for index in range(1, 10))
SIGNAL_LENGTH = 1601
DECLARED_LOCAL_CLASS_ORDER = (0, 1, 2, 3)
DECLARED_TOTAL_SAMPLES = 12250
DECLARED_FILE_SHA256 = {
    "set_1.csv": "43a09751248820cfd4fc664b56acc25317e99930321c6ced63596a272ec56396",
    "set_2.csv": "8c5c47549e3a3ce3a47cdd7acf8e690a5dbe4db61c7a0795acb6c94aad3bfbbe",
    "set_3.csv": "0652992f722976066b770cb61fce25ffda55f4318f8c7dd8800b32ada59217b8",
    "set_4.csv": "e70ab5a278cbde9129a91f9c4099a71537e550d9f60cc29323e8f54f9424d118",
    "set_5.csv": "1e3db4767630ec73cc9f9f159fc7dcf6391dfdf29f8e25205cf8fdd837c0aaf9",
    "set_6.csv": "0b2157bb6ef78ae0745cfffea76d0a718e4d43c77e7998c770584cf8a7a3b570",
    "set_7.csv": "883c20dacc508498b555e585a3b658b702ce3937a49fa281bd43bf17eea32964",
    "set_8.csv": "791c11b1b23d54ce055e9b557395595d450f864b2f0a2dab74ea9c0c588f477d",
    "set_9.csv": "87db5dd2f509130be67f540ceea024d1381f0cb99a3010ae4e32b9d703e4315a",
}

CACHE_RELATIVE = ("01_manifests", "data1_unlabelled_signals_float64.npy")


class Data1LabelLeak(RuntimeError):
    """Raised if a Data1 label would escape the reading boundary."""


@dataclass(frozen=True)
class Data1FileRecord:
    name: str
    sha256: str
    row_count: int
    signal_length: int
    signal_sha256: str
    label_column_conforms: bool
    signal_minimum: float
    signal_maximum: float
    signal_mean: float


def _read_one_file(path: Path) -> tuple[np.ndarray, Data1FileRecord]:
    """Read one Data1 CSV, strip the label column, and return signals only."""

    import pandas as pd

    frame = pd.read_csv(path, header=0, dtype="float64")
    table = frame.to_numpy(dtype=np.float64, copy=True)
    del frame
    if table.ndim != 2 or table.shape[1] != SIGNAL_LENGTH + 1:
        raise Data1LabelLeak(f"Data1 file has an unexpected column count: {path.name}")
    if not np.isfinite(table).all():
        raise Data1LabelLeak(f"Data1 file contains non-finite values: {path.name}")

    # Provenance-only inspection of the label column. Nothing derived from it leaves
    # this block except a boolean conformance flag.
    label_column = table[:, -1]
    integral = np.array_equal(label_column, np.round(label_column))
    within_order = bool(
        set(np.unique(label_column).astype(np.int64).tolist()).issubset(DECLARED_LOCAL_CLASS_ORDER)
    )
    conforms = bool(integral and within_order)
    del label_column

    signals = np.ascontiguousarray(table[:, :SIGNAL_LENGTH])
    del table
    if not conforms:
        raise Data1LabelLeak(
            f"Data1 label column does not conform to the frozen authority: {path.name}"
        )
    record = Data1FileRecord(
        name=path.name,
        sha256=sha256_file(path),
        row_count=int(signals.shape[0]),
        signal_length=int(signals.shape[1]),
        signal_sha256=array_sha256(signals),
        label_column_conforms=conforms,
        signal_minimum=float(signals.min()),
        signal_maximum=float(signals.max()),
        signal_mean=float(signals.mean()),
    )
    return signals, record


def build_corpus(*, cache: bool = True) -> tuple[np.ndarray, list[Data1FileRecord]]:
    """Read all nine Data1 sets, strip labels, and return the signal corpus."""

    root = paths.data1_root()
    blocks: list[np.ndarray] = []
    records: list[Data1FileRecord] = []
    for name in SET_NAMES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Data1 set is missing: {path}")
        signals, record = _read_one_file(path)
        expected = DECLARED_FILE_SHA256.get(name)
        if expected is not None and record.sha256 != expected:
            raise Data1LabelLeak(
                f"Data1 file digest differs from the frozen authority: {name}"
            )
        blocks.append(signals)
        records.append(record)
    corpus = np.ascontiguousarray(np.concatenate(blocks, axis=0))
    if corpus.shape != (DECLARED_TOTAL_SAMPLES, SIGNAL_LENGTH):
        raise Data1LabelLeak(
            f"Data1 corpus shape {corpus.shape} contradicts the frozen authority"
        )
    if cache:
        target = paths.branch_output(*CACHE_RELATIVE)
        np.save(target, corpus)
    return corpus, records


@lru_cache(maxsize=1)
def load_corpus() -> np.ndarray:
    """Return the cached label-free Data1 signal corpus, building it if needed."""

    cached = paths.BRANCH_OUTPUT_ROOT.joinpath(*CACHE_RELATIVE)
    if cached.is_file():
        corpus = np.load(cached)
        if corpus.shape != (DECLARED_TOTAL_SAMPLES, SIGNAL_LENGTH):
            raise Data1LabelLeak("Cached Data1 corpus has an unexpected shape")
        return corpus
    corpus, _ = build_corpus(cache=True)
    return corpus


def corpus_identity() -> dict[str, Any]:
    corpus = load_corpus()
    return {
        "sample_count": int(corpus.shape[0]),
        "signal_length": int(corpus.shape[1]),
        "corpus_sha256": array_sha256(corpus),
        "labels_present_in_corpus": False,
        "label_use": "NONE_LABELS_STRIPPED_AT_READ_BOUNDARY",
    }
