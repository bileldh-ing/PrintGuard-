from pathlib import Path

import numpy as np
import pytest

from printguard.inference.onnx_backend import letterbox, nms

WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "smoke"


def test_letterbox_geometry() -> None:
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out, r, (px, py) = letterbox(img, 64)
    assert out.shape == (64, 64, 3) and r == pytest.approx(0.32)
    assert (px, py) == (0, 16)


def test_nms_suppresses_overlaps() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7])
    assert nms(boxes, scores, 0.5) == [0, 2]


@pytest.mark.skipif(not (WEIGHTS / "model_a_classify.onnx").exists(), reason="smoke weights not present")
def test_onnx_classifier_outputs_distribution() -> None:
    from printguard.inference.onnx_backend import OnnxClassifier

    clf = OnnxClassifier(WEIGHTS / "model_a_classify.onnx")
    probs = clf(np.full((240, 320, 3), 127, dtype=np.uint8))
    assert set(probs) == {"extrusion_error", "layer_shift", "no_error"}
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.skipif(not (WEIGHTS / "model_a_detect.onnx").exists(), reason="smoke weights not present")
def test_onnx_detector_boxes_inside_frame() -> None:
    from printguard.inference.onnx_backend import OnnxDetector

    det = OnnxDetector(WEIGHTS / "model_a_detect.onnx", conf=0.0001)
    out = det(np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8))
    assert all(0 <= d.xyxy[0] <= d.xyxy[2] <= 320 and 0 <= d.xyxy[1] <= d.xyxy[3] <= 240 for d in out)
