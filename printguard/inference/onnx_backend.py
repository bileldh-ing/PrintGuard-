"""Torch-free ONNX Runtime backends for Model A and the feature embedder.

Used for Raspberry Pi / low-memory deployment and for feature extraction: the process only
needs ``numpy``, ``opencv`` and ``onnxruntime`` (~150 MB RSS instead of ~500 MB with PyTorch).
Pre/post-processing mirrors Ultralytics (letterbox + NMS for detection; shortest-side resize +
centre crop for classification) and is checked against it in ``tests/test_onnx_backend.py``.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from printguard.models.detector import Detection

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _session(path: str | Path, threads: int | None = None):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _names(sess) -> dict[int, str]:
    meta = sess.get_modelmeta().custom_metadata_map
    return {int(k): v for k, v in ast.literal_eval(meta["names"]).items()} if "names" in meta else {}


def letterbox(img: np.ndarray, size: int, color: int = 114) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to ``size x size``; returns (image, scale, (pad_x, pad_y))."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
    px, py = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), color, dtype=np.uint8)
    out[py : py + nh, px : px + nw] = resized
    return out, r, (px, py)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    """Greedy non-maximum suppression on ``xyxy`` boxes; returns kept indices (score order)."""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        ovr = inter / np.maximum(area_i + area_r - inter, 1e-9)
        order = rest[ovr <= iou]
    return keep


class OnnxDetector:
    """YOLO (v8/11) detection model exported to ONNX (output ``(1, 4 + nc, N)``)."""

    def __init__(self, path: str | Path, conf: float = 0.25, iou: float = 0.5, threads: int | None = None) -> None:
        self.sess = _session(path, threads)
        self.names = _names(self.sess)
        self.imgsz = int(self.sess.get_inputs()[0].shape[2])
        self.conf, self.iou = conf, iou

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        img, r, (px, py) = letterbox(frame_bgr, self.imgsz)
        x = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        pred = self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0][0].T  # (N, 4 + nc)
        cls_scores = pred[:, 4:]
        cls = cls_scores.argmax(1)
        score = cls_scores[np.arange(len(cls)), cls]
        m = score >= self.conf
        if not m.any():
            return []
        xywh, score, cls = pred[m, :4], score[m], cls[m]
        boxes = np.stack(
            [
                xywh[:, 0] - xywh[:, 2] / 2,
                xywh[:, 1] - xywh[:, 3] / 2,
                xywh[:, 0] + xywh[:, 2] / 2,
                xywh[:, 1] + xywh[:, 3] / 2,
            ],
            1,
        )
        h, w = frame_bgr.shape[:2]
        out: list[Detection] = []
        for c in np.unique(cls):
            idx = np.where(cls == c)[0]
            for k in nms(boxes[idx], score[idx], self.iou):
                b = boxes[idx[k]].copy()
                b[[0, 2]] = np.clip((b[[0, 2]] - px) / r, 0, w)
                b[[1, 3]] = np.clip((b[[1, 3]] - py) / r, 0, h)
                out.append(Detection(self.names.get(int(c), str(c)), float(score[idx[k]]), tuple(float(v) for v in b)))
        return sorted(out, key=lambda d: -d.confidence)


class OnnxClassifier:
    """YOLO classification model exported to ONNX (softmax output ``(1, nc)``)."""

    def __init__(self, path: str | Path, threads: int | None = None) -> None:
        self.sess = _session(path, threads)
        self.names = _names(self.sess)
        self.imgsz = int(self.sess.get_inputs()[0].shape[2])

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Shortest side -> imgsz, centre crop, RGB, [0, 1] (Ultralytics classify transforms)."""
        h, w = frame_bgr.shape[:2]
        s = self.imgsz / min(h, w)
        img = cv2.resize(
            frame_bgr, (max(self.imgsz, round(w * s)), max(self.imgsz, round(h * s))), interpolation=cv2.INTER_LINEAR
        )
        hh, ww = img.shape[:2]
        y0, x0 = (hh - self.imgsz) // 2, (ww - self.imgsz) // 2
        img = img[y0 : y0 + self.imgsz, x0 : x0 + self.imgsz, ::-1]
        return img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    def __call__(self, frame_bgr: np.ndarray) -> dict[str, float]:
        probs = self.sess.run(None, {self.sess.get_inputs()[0].name: self.preprocess(frame_bgr)})[0][0]
        return {self.names.get(i, str(i)): float(p) for i, p in enumerate(probs)}


class OnnxEmbedder:
    """Frozen embedding CNN exported by ``python -m printguard.export embedder``."""

    def __init__(self, path: str | Path, threads: int | None = None) -> None:
        self.sess = _session(path, threads)
        self.imgsz = int(self.sess.get_inputs()[0].shape[2])
        self.out_dim = int(self.sess.get_outputs()[0].shape[1])

    def __call__(self, frames_bgr: Sequence[np.ndarray]) -> np.ndarray:
        batch = []
        for f in frames_bgr:
            rgb = cv2.cvtColor(cv2.resize(f, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            batch.append(((rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))
        return self.sess.run(None, {self.sess.get_inputs()[0].name: np.stack(batch)})[0].astype(np.float32)


class OnnxTemporal:
    """Model B (TCN / Transformer) exported by ``python -m printguard.export temporal``.

    The ONNX file is self-contained: class names, feature normaliser and feature-extractor
    settings live in its metadata. The graph has a fixed window; because Model B is causal, a
    shorter buffer is right-padded and read at its last real frame (identical to running the
    PyTorch checkpoint on the short buffer), and a longer one keeps its newest ``window`` frames.
    """

    def __init__(self, path: str | Path, threads: int | None = None) -> None:
        import json

        self.sess = _session(path, threads)
        meta = self.sess.get_modelmeta().custom_metadata_map
        if meta.get("printguard_model") != "temporal":
            raise ValueError(f"{path} is not a PrintGuard temporal export (re-run `printguard.export temporal`)")
        self.classes: list[str] = json.loads(meta["classes"])
        norm = json.loads(meta["normalizer"])
        self.mean = np.asarray(norm["mean"], dtype=np.float32)
        self.std = np.maximum(np.asarray(norm["std"], dtype=np.float32), 1e-6)
        self.features: dict = json.loads(meta.get("features", "{}"))
        self.model_type = meta.get("model_type", "tcn")
        self.in_dim = int(meta["in_dim"])
        self.window = int(meta.get("window", 32))
        self._input = self.sess.get_inputs()[0].name

    def logits(self, feats: np.ndarray) -> np.ndarray:
        """Per-step logits ``(t, classes)`` for a ``(t, in_dim)`` buffer of *unnormalised* features.

        ``t`` is ``min(len(feats), window)``: longer buffers keep their newest ``window`` frames.
        """
        if feats.ndim != 2 or feats.shape[1] != self.in_dim or len(feats) == 0:
            raise ValueError(f"expected (time, {self.in_dim}) features, got {feats.shape}")
        x = ((feats[-self.window :] - self.mean) / self.std).astype(np.float32)
        t = len(x)
        if t < self.window:  # causal model: padding after the last real frame cannot change it
            x = np.concatenate([x, np.zeros((self.window - t, self.in_dim), dtype=np.float32)])
        return self.sess.run(None, {self._input: x[None]})[0][0, :t]

    def __call__(self, feats: np.ndarray) -> dict[str, float]:
        """Sigmoid scores of the last time step (what the online pipeline alerts on)."""
        z = self.logits(feats)[-1].astype(np.float64)
        return {c: float(1.0 / (1.0 + np.exp(-v))) for c, v in zip(self.classes, z, strict=True)}
