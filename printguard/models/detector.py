"""Model A: frame-level defect models built on Ultralytics YOLO.

Two heads are combined because the public data supports two different label types:

* :class:`BoxDetector` - YOLO detection trained on YOLO-Strings (real stringing boxes);
* :class:`FrameClassifier` - YOLO classification trained on the Trier FFF dataset, which only
  has image-level labels (extrusion error / layer shift / no error). No boxes are invented
  for it; the dashboard shows its output as a frame-level badge, not a box.

:class:`FrameAnalyzer` runs both and returns one :class:`FrameResult` per frame.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One bounding box in pixel coordinates of the analysed frame."""

    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    def area_fraction(self, width: int, height: int) -> float:
        """Box area as a fraction of the frame area."""
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(max(1, width * height))


@dataclass
class FrameResult:
    """Model-A output for a single frame."""

    width: int
    height: int
    detections: list[Detection] = field(default_factory=list)
    class_probs: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return asdict(self)


class BoxDetector:
    """Thin wrapper over an Ultralytics detection model (``.pt`` or exported ``.onnx``)."""

    def __init__(
        self, weights: str | Path, conf: float = 0.25, iou: float = 0.5, imgsz: int = 640, device: str = "cpu"
    ):
        from ultralytics import YOLO

        self.model = YOLO(str(weights), task="detect")
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device

    @property
    def names(self) -> dict[int, str]:
        """Class-id to label mapping stored in the weights."""
        return dict(self.model.names)

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        res = self.model.predict(
            frame_bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device, verbose=False
        )[0]
        out: list[Detection] = []
        if res.boxes is None:
            return out
        for xyxy, conf, cls in zip(
            res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy(), strict=True
        ):
            out.append(Detection(self.names[int(cls)], float(conf), tuple(float(v) for v in xyxy)))  # type: ignore[arg-type]
        return out


class FrameClassifier:
    """Thin wrapper over an Ultralytics classification model."""

    def __init__(self, weights: str | Path, imgsz: int = 224, device: str = "cpu"):
        from ultralytics import YOLO

        self.model = YOLO(str(weights), task="classify")
        self.imgsz, self.device = imgsz, device

    @property
    def names(self) -> dict[int, str]:
        """Class-id to label mapping stored in the weights."""
        return dict(self.model.names)

    def __call__(self, frame_bgr: np.ndarray) -> dict[str, float]:
        res = self.model.predict(frame_bgr, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        probs = res.probs.data.cpu().numpy()
        return {self.names[i]: float(p) for i, p in enumerate(probs)}


DetectorFn = Callable[[np.ndarray], list[Detection]]
ClassifierFn = Callable[[np.ndarray], dict[str, float]]


class FrameAnalyzer:
    """Runs whichever Model-A heads are configured and merges their output.

    Heads are plain callables, so Ultralytics (``.pt``) and ONNX Runtime (``.onnx``) backends
    are interchangeable.
    """

    def __init__(self, detector: DetectorFn | None = None, classifier: ClassifierFn | None = None) -> None:
        if detector is None and classifier is None:
            raise ValueError("FrameAnalyzer needs at least one of detector/classifier")
        self.detector, self.classifier = detector, classifier

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> FrameAnalyzer:
        """Build from a ``model_a`` config section; missing weight files are skipped.

        ``*.onnx`` weights use the torch-free ONNX Runtime backend, anything else Ultralytics.
        """
        det: DetectorFn | None = None
        clf: ClassifierFn | None = None
        dcfg, ccfg = cfg.get("detector") or {}, cfg.get("classifier") or {}
        if dcfg.get("weights") and Path(dcfg["weights"]).exists():
            if str(dcfg["weights"]).endswith(".onnx"):
                from printguard.inference.onnx_backend import OnnxDetector

                det = OnnxDetector(dcfg["weights"], conf=dcfg.get("conf", 0.25), iou=dcfg.get("iou", 0.5))
            else:
                det = BoxDetector(dcfg["weights"], conf=dcfg.get("conf", 0.25), imgsz=dcfg.get("imgsz", 640))
        if ccfg.get("weights") and Path(ccfg["weights"]).exists():
            if str(ccfg["weights"]).endswith(".onnx"):
                from printguard.inference.onnx_backend import OnnxClassifier

                clf = OnnxClassifier(ccfg["weights"])
            else:
                clf = FrameClassifier(ccfg["weights"], imgsz=ccfg.get("imgsz", 224))
        return cls(det, clf)

    def __call__(self, frame_bgr: np.ndarray) -> FrameResult:
        h, w = frame_bgr.shape[:2]
        return FrameResult(
            width=w,
            height=h,
            detections=self.detector(frame_bgr) if self.detector else [],
            class_probs=self.classifier(frame_bgr) if self.classifier else {},
        )
