"""Sequence builder for Model B.

1. :func:`build_feature_cache` runs Model A + the frozen CNN over every frame of every indexed
   sequence (plus synthetic gradual-layer-shift variants) and stores one ``.npz`` per sequence:
   ``features (T, D)``, ``labels (T, C)`` with ``-1`` = unknown, and Model-A baseline scores.
2. :class:`printguard.training.datasets.WindowDataset` cuts cached sequences into sliding windows.

Caching makes the temporal model cheap to train (seconds on CPU) and keeps Model-A inference,
which dominates cost, to a single pass.
"""

from __future__ import annotations

import json
import logging
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from printguard.data.augment import apply_layer_shift, sample_params
from printguard.taxonomy import TEMPORAL_CLASSES

LOGGER = logging.getLogger(__name__)


@dataclass
class SequenceRecord:
    """One recording sequence from ``data/processed/sequences/index.json``."""

    id: str
    kind: str
    split: str
    frames: list[str]
    stringing: list[int]


def load_index(path: str | Path) -> list[SequenceRecord]:
    """Read the sequence index written by :mod:`printguard.data.prepare`."""
    return [SequenceRecord(**r) for r in json.loads(Path(path).read_text())]


def load_frames(paths: Sequence[str], max_side: int = 640) -> list[np.ndarray]:
    """Read BGR frames, downscaling so the longest side is at most ``max_side``."""
    import cv2

    out = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(p)
        h, w = img.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        out.append(img)
    return out


def cache_name(seq_id: str, variant: str) -> str:
    """File-system safe cache file name for a sequence variant."""
    return f"{seq_id.replace('/', '__')}{'#' + variant if variant else ''}.npz"


def variants_for(rec: SequenceRecord, aug: dict[str, Any]) -> list[str]:
    """Which cached variants to build for a sequence: original + synthetic layer shifts.

    Synthetic shifts are applied to defect-free sequences (all splits, so val/test also contain
    shift onsets to measure delay) and, for training diversity, optionally to stringing ones.
    """
    out = [""]
    if not aug.get("enabled", True):
        return out
    n = int(aug.get("variants_clean", 2)) if rec.kind == "without_stringing" else 0
    if rec.kind == "with_stringing" and rec.split == "train":
        n = int(aug.get("variants_stringing_train", 1))
    out += [f"shift{k}" for k in range(n)]
    return out


def build_feature_cache(
    records: Sequence[SequenceRecord],
    frame_fn: Any,
    cache_dir: str | Path,
    aug: dict[str, Any],
    seed: int = 42,
    max_side: int = 640,
    budget_s: float | None = None,
    frame_stride: int = 1,
) -> list[Path]:
    """Compute and cache per-frame features for every sequence variant (resumable).

    Args:
        records: sequences to process.
        frame_fn: callable ``frames -> (features (T, D), baseline_scores (T, C))``; see
            :func:`printguard.training.train_b.make_frame_fn`.
        cache_dir: output directory for ``.npz`` files.
        aug: ``augmentation`` section of the Model-B config.
        seed: base RNG seed (each variant derives its own, so caches are reproducible).
        max_side: frame downscale before Model A.
        budget_s: stop after this many seconds (work resumes on the next call).
        frame_stride: keep every n-th frame (CPU smoke runs); labels are subsampled alike.
    """
    import time

    t0 = time.time()
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for rec in records:
        for variant in variants_for(rec, aug):
            out = cache / cache_name(rec.id, variant)
            if out.exists():
                written.append(out)
                continue
            if budget_s is not None and time.time() - t0 > budget_s:
                LOGGER.info("time budget reached; rerun to continue")
                return written
            frames = load_frames(rec.frames[::frame_stride], max_side)
            stringing = np.asarray(rec.stringing[::frame_stride], dtype=np.int64)
            shift = np.zeros(len(frames), dtype=np.int64)
            meta: dict[str, Any] = {"id": rec.id, "variant": variant, "split": rec.split, "kind": rec.kind}
            if variant:
                rng = np.random.default_rng([seed, zlib.crc32(rec.id.encode()), int(variant[5:])])
                params = sample_params(
                    frames,
                    rng,
                    onset_range=tuple(aug.get("onset_range", (0.3, 0.6))),
                    ramp_range=tuple(aug.get("ramp_range", (8, 25))),
                    shift_range=tuple(aug.get("shift_range", (0.03, 0.08))),
                )
                frames, shift = apply_layer_shift(frames, params)
                meta["layer_shift_params"] = params.__dict__
            feats, base = frame_fn(frames)
            labels = np.stack([stringing, shift], axis=1)
            np.savez_compressed(out, features=feats.astype(np.float32), labels=labels, baseline=base.astype(np.float32),
                                meta=json.dumps(meta))
            written.append(out)
            LOGGER.info("cached %s (%d frames)", out.name, len(frames))
    return written


@dataclass
class CachedSequence:
    """A cached sequence variant loaded in memory."""

    name: str
    split: str
    features: np.ndarray
    labels: np.ndarray
    baseline: np.ndarray
    meta: dict[str, Any]


def load_cache(cache_dir: str | Path, split: str | None = None) -> list[CachedSequence]:
    """Load cached sequences (optionally one split), sorted by name for determinism."""
    out = []
    for p in sorted(Path(cache_dir).glob("*.npz")):
        z = np.load(p, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if split and meta["split"] != split:
            continue
        out.append(CachedSequence(p.stem, meta["split"], z["features"], z["labels"], z["baseline"], meta))
    return out


class FeatureNormalizer:
    """Per-dimension standardisation fitted on training frames."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean, self.std = mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)

    @classmethod
    def fit(cls, seqs: Sequence[CachedSequence]) -> FeatureNormalizer:
        """Fit mean/std on the concatenated frames of ``seqs``."""
        x = np.concatenate([s.features for s in seqs])
        return cls(x.mean(0), x.std(0))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def state(self) -> dict[str, list[float]]:
        """Serialisable state (stored in the checkpoint)."""
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


NUM_TEMPORAL = len(TEMPORAL_CLASSES)
