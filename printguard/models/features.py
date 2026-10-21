"""Per-frame feature vectors for Model B (torch-free at import time).

``feature = [CNN embedding | detection summary | classifier probabilities]``

* **CNN embedding** - pooled MobileNetV3-Small features (ImageNet weights) reduced by a fixed
  random projection; served by ONNX Runtime (``embedder: *.onnx``) or PyTorch.
* **Detection summary** - per detector class: max confidence, box count, summed box area.
* **Classifier probabilities** - the Trier frame classifier's softmax output.

The extractor is frozen; only the temporal model is trained.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from printguard.models.detector import FrameResult

Embedder = Callable[[Sequence[np.ndarray]], np.ndarray]


def detection_summary(result: FrameResult | None, det_classes: Sequence[str], cls_classes: Sequence[str]) -> np.ndarray:
    """Fixed-length vector summarising Model-A output for one frame (zeros if ``None``)."""
    vec = np.zeros(3 * len(det_classes) + len(cls_classes), dtype=np.float32)
    if result is None:
        return vec
    for i, name in enumerate(det_classes):
        dets = [d for d in result.detections if d.label == name]
        if dets:
            vec[3 * i] = max(d.confidence for d in dets)
            vec[3 * i + 1] = min(len(dets), 10) / 10.0
            vec[3 * i + 2] = min(1.0, sum(d.area_fraction(result.width, result.height) for d in dets))
    off = 3 * len(det_classes)
    for j, name in enumerate(cls_classes):
        vec[off + j] = result.class_probs.get(name, 0.0)
    return vec


def make_embedder(
    embedder: str | None, embed_dim: int = 128, imgsz: int = 160, pretrained: bool = True
) -> Embedder | None:
    """``*.onnx`` path -> ONNX Runtime embedder; ``"torch"`` -> PyTorch; ``None``/``"none"`` -> no embedding."""
    if not embedder or embedder == "none":
        return None
    if str(embedder).endswith(".onnx"):
        from printguard.inference.onnx_backend import OnnxEmbedder

        if not Path(embedder).exists():
            raise FileNotFoundError(f"{embedder} not found - run `python -m printguard.export embedder`")
        return OnnxEmbedder(embedder)
    from printguard.models.embedding import TorchEmbedder

    return TorchEmbedder(embed_dim, imgsz, pretrained)


class FrameFeatureExtractor:
    """Turns BGR frames (+ optional Model-A results) into Model-B feature vectors."""

    def __init__(self, det_classes: Sequence[str], cls_classes: Sequence[str], embedder: Embedder | None) -> None:
        self.det_classes, self.cls_classes = tuple(det_classes), tuple(cls_classes)
        self.embedder = embedder

    @property
    def dim(self) -> int:
        """Length of one feature vector."""
        emb = int(getattr(self.embedder, "out_dim", 0)) if self.embedder is not None else 0
        return emb + 3 * len(self.det_classes) + len(self.cls_classes)

    def __call__(
        self, frames_bgr: Sequence[np.ndarray], results: Sequence[FrameResult | None] | None = None
    ) -> np.ndarray:
        """Return an ``(N, dim)`` float32 array for ``N`` frames."""
        results = results if results is not None else [None] * len(frames_bgr)
        summ = np.stack([detection_summary(r, self.det_classes, self.cls_classes) for r in results])
        if self.embedder is None:
            return summ
        return np.concatenate([self.embedder(frames_bgr), summ], axis=1)
