"""Alert policy: turns noisy per-frame scores into operator alerts.

An alert opens when ``k`` of the last ``n`` scores for a label exceed ``threshold`` (the same
rule used by the offline metrics), and the label is then muted for ``cooldown_frames`` so an
operator is not flooded with duplicates of one ongoing defect.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class AlertPolicy:
    """k-of-n persistence with per-label cooldown."""

    threshold: float = 0.5
    k: int = 3
    n: int = 5
    cooldown_frames: int = 50
    thresholds: dict[str, float] = field(default_factory=dict)
    _hist: dict[str, deque[bool]] = field(default_factory=lambda: defaultdict(deque), init=False, repr=False)
    _muted_until: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def update(self, frame_idx: int, scores: dict[str, float]) -> list[str]:
        """Feed one frame's scores; returns the labels for which a new alert opens now."""
        fired = []
        for label, score in scores.items():
            h = self._hist[label]
            h.append(score >= self.thresholds.get(label, self.threshold))
            while len(h) > self.n:
                h.popleft()
            if sum(h) >= self.k and frame_idx >= self._muted_until.get(label, -1):
                fired.append(label)
                self._muted_until[label] = frame_idx + self.cooldown_frames
        return fired

    def reset(self) -> None:
        """Forget all history (new print / new source)."""
        self._hist.clear()
        self._muted_until.clear()


def trend(values: list[float]) -> float:
    """Least-squares slope of the recent scores (per frame); > 0 means the defect is growing."""
    m = len(values)
    if m < 2:
        return 0.0
    mx = (m - 1) / 2
    my = sum(values) / m
    num = sum((i - mx) * (v - my) for i, v in enumerate(values))
    den = sum((i - mx) ** 2 for i in range(m))
    return num / den if den else 0.0
