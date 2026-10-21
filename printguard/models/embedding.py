"""Frozen MobileNetV3-Small embedding trunk (PyTorch). ONNX twin: ``OnnxEmbedder``."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from printguard.inference.onnx_backend import IMAGENET_MEAN, IMAGENET_STD


class EmbeddingCNN(nn.Module):
    """Global-average-pooled MobileNetV3-Small features with an optional fixed random projection."""

    def __init__(self, out_dim: int | None = 128, pretrained: bool = True, seed: int = 0) -> None:
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        net = None
        if pretrained:
            try:
                net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            except Exception:  # offline: fall back to a random (but seeded) trunk
                net = None
        self.pretrained = net is not None
        if net is None:
            torch.manual_seed(seed)
            net = mobilenet_v3_small(weights=None)
        self.trunk = net.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        native = 576
        if out_dim and out_dim < native:
            g = torch.Generator().manual_seed(seed)
            self.register_buffer("proj", torch.randn(native, out_dim, generator=g) / np.sqrt(out_dim))
            self.out_dim = out_dim
        else:
            self.proj = None
            self.out_dim = native
        self.eval().requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, 3, H, W)`` ImageNet-normalised RGB -> ``(B, out_dim)``."""
        z = self.pool(self.trunk(x)).flatten(1)
        return z @ self.proj if self.proj is not None else z


class TorchEmbedder:
    """Callable wrapper with the same interface as ``OnnxEmbedder``."""

    def __init__(self, out_dim: int | None = 128, imgsz: int = 160, pretrained: bool = True) -> None:
        self.cnn, self.imgsz = EmbeddingCNN(out_dim, pretrained), imgsz
        self.out_dim = self.cnn.out_dim

    @torch.inference_mode()
    def __call__(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        import cv2

        batch = []
        for f in frames_bgr:
            rgb = cv2.cvtColor(cv2.resize(f, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            batch.append(((rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))
        return self.cnn(torch.from_numpy(np.stack(batch))).numpy().astype(np.float32)
