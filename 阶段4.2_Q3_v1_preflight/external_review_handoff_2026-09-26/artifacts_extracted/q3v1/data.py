"""Fail-closed aligned data contract; only train/valid are exposed in preflight."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import pickle
from pathlib import Path

import numpy as np

EXPECTED_SHA256 = "66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd"
DIMS = {"text": (50, 768), "audio": (50, 74), "vision": (50, 35), "text_bert": (3, 50)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_content_indices(text_bert: np.ndarray) -> np.ndarray:
    """Provisional BERT content positions, preserving official 0..49 indices.

    This validates token structure, not the provenance of text/audio/vision rows.
    """
    arr = np.asarray(text_bert)
    if arr.shape != (3, 50) or not np.isfinite(arr).all() or not np.equal(arr, np.floor(arr)).all():
        raise ValueError("text_bert must be finite integer-valued [3,50]")
    ids, attention, segments = arr.astype(np.int64)
    if not np.isin(attention, (0, 1)).all() or not np.isin(segments, (0, 1)).all():
        raise ValueError("invalid attention/segment values")
    active = np.flatnonzero(attention == 1)
    if len(active) < 3 or not np.array_equal(active, np.arange(len(active))):
        raise ValueError("attention must be a nonempty contiguous prefix")
    if ids[0] != 101 or ids[active[-1]] != 102 or np.any(ids[1:active[-1]] == 102):
        raise ValueError("CLS/SEP structure not verified")
    if np.any(ids[active[-1] + 1:] != 0):
        raise ValueError("padding token IDs not zero")
    return active[1:-1].astype(np.int64)


@dataclass(frozen=True)
class AlignedSample:
    sample_id: str
    raw_text: str
    text: np.ndarray
    audio: np.ndarray
    vision: np.ndarray
    content_indices: np.ndarray
    classification_label: int
    regression_label: float

    @property
    def validity_mask(self) -> np.ndarray:
        mask = np.zeros(50, dtype=bool)
        mask[self.content_indices] = True
        return mask


def _validate_split(split: dict, name: str) -> list[AlignedSample]:
    required = set(DIMS) | {"id", "raw_text", "classification_labels", "regression_labels"}
    if not required.issubset(split):
        raise ValueError(f"{name}: missing fields {sorted(required - set(split))}")
    n = len(split["id"])
    if any(len(split[key]) != n for key in required):
        raise ValueError(f"{name}: first dimension mismatch")
    if any(np.shape(split[key])[1:] != shape for key, shape in DIMS.items()):
        raise ValueError(f"{name}: tensor shape mismatch")
    ids = [str(value) for value in split["id"]]
    if len(set(ids)) != n:
        raise ValueError(f"{name}: duplicate sample id")
    result = []
    for i, sample_id in enumerate(ids):
        indices = validate_content_indices(split["text_bert"][i])
        features = [np.asarray(split[key][i], dtype=np.float32) for key in ("text", "audio", "vision")]
        if any(not np.isfinite(feature).all() for feature in features):
            raise ValueError(f"{name}/{sample_id}: nonfinite feature")
        raw_cls = float(split["classification_labels"][i])
        regression = float(split["regression_labels"][i])
        if raw_cls not in (0.0, 1.0, 2.0) or not np.isfinite(regression) or not -3 <= regression <= 3:
            raise ValueError(f"{name}/{sample_id}: invalid label")
        result.append(AlignedSample(sample_id, str(split["raw_text"][i]), *features, indices, int(raw_cls), regression))
    return result


def load_split(path: str | Path, split: str, verify_hash: bool = True) -> list[AlignedSample]:
    if split not in ("train", "valid"):
        raise PermissionError("preflight only exposes train or valid; test is selection-locked")
    source = Path(path)
    if verify_hash and sha256_file(source) != EXPECTED_SHA256:
        raise ValueError("aligned PKL SHA-256 mismatch")
    # The official pickle container necessarily deserializes all three splits.
    # Only the requested train/valid mapping is accessed; test is never returned.
    with source.open("rb") as stream:
        container = pickle.load(stream)
    return _validate_split(container[split], split)


class TrainScaler:
    """Train-only position-wise statistics over the validated content mask."""

    def __init__(self) -> None:
        self.mean: dict[str, np.ndarray] = {}
        self.std: dict[str, np.ndarray] = {}
        self.small_std: dict[str, int] = {}
        self.source_split: str | None = None

    def fit(self, samples: list[AlignedSample], *, split: str) -> "TrainScaler":
        if split != "train" or not samples:
            raise PermissionError("scaler fit requires nonempty train split")
        for key in ("text", "audio", "vision"):
            count = 0
            total = np.zeros(getattr(samples[0], key).shape[1], dtype=np.float64)
            squares = np.zeros_like(total)
            for sample in samples:
                values = getattr(sample, key)[sample.content_indices].astype(np.float64)
                count += len(values)
                total += values.sum(axis=0)
                squares += np.square(values).sum(axis=0)
            mean = total / count
            raw_std = np.sqrt(np.maximum(squares / count - mean * mean, 0.0))
            small = raw_std < 1e-6
            self.mean[key] = mean
            self.std[key] = np.where(small, 1.0, raw_std)
            self.small_std[key] = int(small.sum())
        self.source_split = split
        return self

    def transform(self, sample: AlignedSample) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.source_split != "train":
            raise RuntimeError("train scaler not fitted")
        output = []
        mask = sample.validity_mask
        for key in ("text", "audio", "vision"):
            values = ((getattr(sample, key).astype(np.float64) - self.mean[key]) / self.std[key]).astype(np.float32)
            values[~mask] = 0
            output.append(values)
        return *output, mask
