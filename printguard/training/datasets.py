"""PyTorch dataset of sliding windows over cached Model-B feature sequences."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from printguard.data.sequences import CachedSequence, FeatureNormalizer


class WindowDataset(Dataset):
    """Sliding windows ``(x (T, D), y (T, C), mask (T, C))`` over cached sequences.

    Sequences shorter than ``window`` are left-padded with their first frame and the padded
    steps are masked out.
    """

    def __init__(self, seqs: Sequence[CachedSequence], norm: FeatureNormalizer, window: int, stride: int) -> None:
        self.items: list[tuple[np.ndarray, np.ndarray]] = []
        for s in seqs:
            x, y = norm(s.features), s.labels.astype(np.float32)
            for start, stop in _windows(len(x), window, stride):
                xs, ys = x[start:stop], y[start:stop]
                pad = window - len(xs)
                if pad:
                    xs = np.concatenate([np.repeat(xs[:1], pad, 0), xs])
                    ys = np.concatenate([np.full((pad, y.shape[1]), -1.0, np.float32), ys])
                self.items.append((xs, ys))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = self.items[i]
        mask = (y >= 0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(np.clip(y, 0, 1)), torch.from_numpy(mask)


def _windows(n: int, window: int, stride: int) -> Iterator[tuple[int, int]]:
    if n <= window:
        yield 0, n
        return
    starts = list(range(0, n - window + 1, stride))
    if starts[-1] != n - window:
        starts.append(n - window)
    for s in starts:
        yield s, s + window
