"""Online inference: one frame in, one :class:`FrameEvent` out.

``frame -> Model A (boxes + frame classes) -> features -> ring buffer -> Model B -> alerts``

Every stage is optional so the same pipeline runs with only a detector, only the temporal
model, or neither (video passthrough for UI development).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from printguard.inference.alerts import AlertPolicy, trend
from printguard.models.detector import Detection, FrameAnalyzer
from printguard.taxonomy import DETECT_CLASSES, FRAME_CLS_CLASSES, NO_DEFECT, TEMPORAL_CLASSES


@dataclass
class FrameEvent:
    """Everything the dashboard needs about one processed frame."""

    frame_idx: int
    timestamp: float
    width: int
    height: int
    detections: list[Detection] = field(default_factory=list)
    class_probs: dict[str, float] = field(default_factory=dict)
    temporal_scores: dict[str, float] = field(default_factory=dict)
    trends: dict[str, float] = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return asdict(self)


class OnlinePipeline:
    """Stateful per-stream pipeline (one instance per camera)."""

    def __init__(
        self,
        analyzer: FrameAnalyzer | None = None,
        temporal_ckpt: str | Path | None = None,
        policy: AlertPolicy | None = None,
        window: int = 32,
        trend_len: int = 10,
        embedder: str | None = None,
    ) -> None:
        self.analyzer = analyzer
        self.policy = policy or AlertPolicy()
        self.window, self.trend_len = window, trend_len
        self.temporal = self.norm = self.extractor = None
        self.temporal_backend: str | None = None
        if temporal_ckpt and Path(temporal_ckpt).exists():
            from printguard.models.features import FrameFeatureExtractor, make_embedder

            if str(temporal_ckpt).endswith(".onnx"):  # torch-free path (Docker / Raspberry Pi)
                from printguard.inference.onnx_backend import OnnxTemporal

                self.temporal = OnnxTemporal(temporal_ckpt)
                f = dict(self.temporal.features)
                self.temporal_backend = "onnx"
            else:
                from printguard.training.train_b import load_temporal

                self.temporal, self.norm, ck = load_temporal(temporal_ckpt)
                f = dict(ck["features"])
                self.temporal_backend = "torch"
            if embedder:  # deployment may point at a different copy of the same embedder
                f["embedder"] = embedder
            emb = make_embedder(
                f.get("embedder"), f.get("embed_dim", 128), f.get("imgsz", 160), f.get("pretrained", True)
            )
            self.extractor = FrameFeatureExtractor(DETECT_CLASSES, FRAME_CLS_CLASSES, emb)
        self._feats: deque[np.ndarray] = deque(maxlen=window)
        self._scores: dict[str, deque[float]] = {}
        self._idx = 0

    @property
    def has_models(self) -> bool:
        """Whether any model is loaded."""
        return self.analyzer is not None or self.temporal is not None

    def reset(self) -> None:
        """Clear temporal state (call when the source changes)."""
        self._feats.clear()
        self._scores.clear()
        self.policy.reset()
        self._idx = 0

    def frame_scores(self, ev: FrameEvent) -> dict[str, float]:
        """Scores fed to the alert policy: temporal scores when available, else Model-A scores."""
        if ev.temporal_scores:
            return dict(ev.temporal_scores)
        scores: dict[str, float] = {}
        for label in DETECT_CLASSES:
            scores[label] = max((d.confidence for d in ev.detections if d.label == label), default=0.0)
        for label, p in ev.class_probs.items():
            if label != NO_DEFECT:
                scores[label] = max(scores.get(label, 0.0), p)
        return scores

    def temporal_scores(self, feats: np.ndarray) -> dict[str, float]:
        """Model-B scores of the newest frame given the ``(time, dim)`` feature buffer."""
        if self.temporal_backend == "onnx":
            return self.temporal(feats)
        import torch

        x = torch.from_numpy(self.norm(feats))[None]
        with torch.inference_mode():
            probs = torch.sigmoid(self.temporal(x))[0, -1].numpy()
        return {c: float(p) for c, p in zip(TEMPORAL_CLASSES, probs, strict=True)}

    def __call__(self, frame_bgr: np.ndarray, timestamp: float | None = None) -> FrameEvent:
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        ev = FrameEvent(self._idx, timestamp if timestamp is not None else time.time(), w, h)
        result = self.analyzer(frame_bgr) if self.analyzer else None
        if result is not None:
            ev.detections, ev.class_probs = result.detections, result.class_probs
        if self.temporal is not None and self.extractor is not None:
            self._feats.append(self.extractor([frame_bgr], [result])[0])
            ev.temporal_scores = self.temporal_scores(np.stack(self._feats))
        scores = self.frame_scores(ev)
        for label, s in scores.items():
            hist = self._scores.setdefault(label, deque(maxlen=self.trend_len))
            hist.append(s)
            ev.trends[label] = trend(list(hist))
        ev.alerts = self.policy.update(self._idx, scores)
        ev.latency_ms = (time.perf_counter() - t0) * 1000
        self._idx += 1
        return ev

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> OnlinePipeline:
        """Build from the ``pipeline`` section of ``configs/server.yaml``."""
        analyzer = None
        a = cfg.get("model_a", {})
        try:
            analyzer = FrameAnalyzer.from_config(a)
        except ValueError:
            analyzer = None  # no Model-A weights available
        pol = cfg.get("alerts", {})
        return cls(
            analyzer=analyzer,
            temporal_ckpt=cfg.get("model_b", {}).get("weights"),
            policy=AlertPolicy(
                threshold=pol.get("threshold", 0.5),
                k=pol.get("k", 3),
                n=pol.get("n", 5),
                cooldown_frames=pol.get("cooldown_frames", 50),
                thresholds=pol.get("thresholds", {}),
            ),
            window=cfg.get("model_b", {}).get("window", 32),
            embedder=cfg.get("model_b", {}).get("embedder"),
        )
