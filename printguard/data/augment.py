"""Synthetic *gradual* layer-shift augmentation for frame sequences.

Real layer shifts are step-like (a skipped stepper step), but a slowly slipping belt or a
loose pulley produces a shift that grows over many layers. No public dataset records that
with an onset timestamp, so we synthesise it on defect-free sequences:

1. pick an onset frame ``t0`` and a ramp length ``R``;
2. estimate the object's top edge at ``t0`` from the difference to the first frame
   (everything printed after ``t0`` lies above that row);
3. for ``t >= t0`` translate the image region above the cut row horizontally by
   ``dx(t) = max_shift * min(1, (t - t0 + 1) / R)``, filling the exposed strip by edge
   replication.

Labels are 1 from ``t0`` on (the defect exists from its onset even while still subtle), so
detection delay is measured from the true onset. This is a *simulation*: it is used to show
that a temporal model can react to a slowly growing displacement, not as a substitute for
real recordings (see ``docs/data_card.md``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LayerShiftParams:
    """Parameters of one synthetic gradual layer shift."""

    onset: int
    ramp: int
    max_shift_frac: float
    direction: int  # +1 right, -1 left
    cut_row: int

    def shift_px(self, t: int, width: int) -> int:
        """Horizontal displacement (pixels) applied at frame ``t``."""
        if t < self.onset:
            return 0
        frac = min(1.0, (t - self.onset + 1) / max(1, self.ramp))
        return int(round(self.direction * frac * self.max_shift_frac * width))


def estimate_cut_row(first: np.ndarray, at_onset: np.ndarray, thresh: int = 30, min_pixels: int = 25) -> int:
    """Topmost image row where ``at_onset`` differs from ``first`` (the printed part's top edge).

    Falls back to the vertical centre when nothing changed noticeably.
    """
    diff = np.abs(first.astype(np.int16) - at_onset.astype(np.int16)).max(axis=2)
    rows = np.where((diff > thresh).sum(axis=1) >= min_pixels)[0]
    return int(rows.min()) if rows.size else first.shape[0] // 2


def shift_region(frame: np.ndarray, cut_row: int, dx: int) -> np.ndarray:
    """Translate rows ``[0, cut_row)`` horizontally by ``dx`` pixels with edge replication."""
    if dx == 0 or cut_row <= 0:
        return frame
    out = frame.copy()
    top = frame[:cut_row]
    w = frame.shape[1]
    dx = int(np.clip(dx, -(w - 1), w - 1))
    if dx > 0:
        out[:cut_row, dx:] = top[:, : w - dx]
        out[:cut_row, :dx] = top[:, :1]
    else:
        out[:cut_row, : w + dx] = top[:, -dx:]
        out[:cut_row, w + dx :] = top[:, -1:]
    return out


def sample_params(
    frames: Sequence[np.ndarray],
    rng: np.random.Generator,
    onset_range: tuple[float, float] = (0.3, 0.6),
    ramp_range: tuple[int, int] = (8, 25),
    shift_range: tuple[float, float] = (0.03, 0.08),
) -> LayerShiftParams:
    """Draw random shift parameters for a sequence (onset as a fraction of its length)."""
    n = len(frames)
    onset = int(rng.uniform(*onset_range) * n)
    onset = int(np.clip(onset, 1, max(1, n - 2)))
    cut = estimate_cut_row(frames[0], frames[onset])
    return LayerShiftParams(
        onset=onset,
        ramp=int(rng.integers(ramp_range[0], ramp_range[1] + 1)),
        max_shift_frac=float(rng.uniform(*shift_range)),
        direction=int(rng.choice([-1, 1])),
        cut_row=cut,
    )


def apply_layer_shift(frames: Sequence[np.ndarray], params: LayerShiftParams) -> tuple[list[np.ndarray], np.ndarray]:
    """Apply a gradual layer shift; returns augmented frames and per-frame 0/1 labels."""
    out, labels = [], np.zeros(len(frames), dtype=np.int64)
    for t, f in enumerate(frames):
        dx = params.shift_px(t, f.shape[1])
        out.append(shift_region(f, params.cut_row, dx))
        labels[t] = int(t >= params.onset)
    return out, labels
