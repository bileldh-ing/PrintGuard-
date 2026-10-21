"""Model B: temporal models over sliding windows of per-frame features.

Both models are *causal*: the output at step ``t`` depends only on frames ``<= t``, so the
same network runs online in the inference worker (one new frame in, one decision out) and
offline on whole windows during training. Outputs are per-timestep multi-label logits over
:data:`printguard.taxonomy.TEMPORAL_CLASSES`.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import torch
from torch import Tensor, nn


class _CausalConv1d(nn.Conv1d):
    """1-D convolution padded on the left only, so no future frame leaks into step ``t``."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilation: int) -> None:
        super().__init__(c_in, c_out, kernel_size, dilation=dilation)
        self._left = (kernel_size - 1) * dilation

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(nn.functional.pad(x, (self._left, 0)))


class _ChannelNorm(nn.Module):
    """LayerNorm over channels at each time step.

    ``GroupNorm``/``BatchNorm`` over ``(C, T)`` would mix statistics across time and leak future
    frames into past outputs, breaking causality (caught by ``tests/test_temporal.py``).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv1d(channels, channels, kernel_size, dilation),
            _ChannelNorm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            _CausalConv1d(channels, channels, kernel_size, dilation),
            _ChannelNorm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class TCN(nn.Module):
    """Causal temporal convolutional network (Bai et al., 2018) with residual dilated blocks.

    Args:
        in_dim: per-frame feature size.
        num_classes: number of temporal outputs.
        channels: hidden width.
        levels: number of residual blocks; dilation doubles per level.
        kernel_size: convolution kernel size.
        dropout: dropout probability.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        channels: int = 64,
        levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, channels))
        self.blocks = nn.Sequential(*[_TemporalBlock(channels, kernel_size, 2**i, dropout) for i in range(levels)])
        self.head = nn.Linear(channels, num_classes)
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2**levels - 1)

    def forward(self, x: Tensor) -> Tensor:
        """``x``: ``(batch, time, in_dim)`` -> logits ``(batch, time, num_classes)``."""
        h = self.inp(x).transpose(1, 2)
        h = self.blocks(h).transpose(1, 2)
        return self.head(h)


class TemporalTransformer(nn.Module):
    """Small causal Transformer encoder (pre-norm) with sinusoidal positions."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        d_model: int = 64,
        heads: int = 4,
        layers: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, heads, d_model * ff_mult, dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.register_buffer("pos", _sinusoid(max_len, d_model), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """``x``: ``(batch, time, in_dim)`` -> logits ``(batch, time, num_classes)``."""
        t = x.shape[1]
        h = self.inp(x) + self.pos[:t]
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.encoder(h, mask=mask)
        return self.head(self.norm(h))


def _sinusoid(n: int, d: int) -> Tensor:
    pos = torch.arange(n).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


def build_temporal_model(cfg: dict[str, Any], in_dim: int, num_classes: int) -> nn.Module:
    """Instantiate a temporal model from the ``model`` section of a Model-B config."""
    kind = cfg.get("type", "tcn").lower()
    classes = {"tcn": TCN, "transformer": TemporalTransformer}
    if kind not in classes:
        raise ValueError(f"Unknown temporal model type {kind!r} (expected 'tcn' or 'transformer')")
    cls = classes[kind]
    # configs inherit from each other, so ignore keys that belong to the other architecture
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self", "in_dim", "num_classes"}
    return cls(in_dim, num_classes, **{k: v for k, v in cfg.items() if k in accepted})
