"""Model B ONNX export: parity with PyTorch and torch-free use in the online pipeline."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from printguard.export import export_temporal  # noqa: E402
from printguard.inference.onnx_backend import OnnxTemporal  # noqa: E402
from printguard.inference.pipeline import OnlinePipeline  # noqa: E402
from printguard.models.temporal import build_temporal_model  # noqa: E402
from printguard.taxonomy import DETECT_CLASSES, FRAME_CLS_CLASSES, TEMPORAL_CLASSES  # noqa: E402
from printguard.training.train_b import load_temporal  # noqa: E402

# no embedder: features = detection summary + classifier probabilities only
IN_DIM = 3 * len(DETECT_CLASSES) + len(FRAME_CLS_CLASSES)
CFGS = {
    "tcn": {"type": "tcn", "channels": 16, "levels": 3, "kernel_size": 3, "dropout": 0.0},
    "transformer": {"type": "transformer", "d_model": 16, "heads": 2, "layers": 1, "dropout": 0.0},
}


def _checkpoint(tmp_path: Path, kind: str) -> Path:
    torch.manual_seed(0)
    model = build_temporal_model(CFGS[kind], IN_DIM, len(TEMPORAL_CLASSES))
    rng = np.random.default_rng(0)
    path = tmp_path / f"model_b_{kind}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_cfg": CFGS[kind],
            "in_dim": IN_DIM,
            "classes": list(TEMPORAL_CLASSES),
            "normalizer": {"mean": rng.normal(size=IN_DIM).tolist(), "std": rng.uniform(0.5, 2, IN_DIM).tolist()},
            "features": {"embedder": "none"},
            "n_params": sum(p.numel() for p in model.parameters()),
        },
        path,
    )
    return path


@pytest.mark.parametrize("kind", ["tcn", "transformer"])
def test_onnx_matches_torch_for_every_buffer_length(tmp_path: Path, kind: str) -> None:
    ckpt = _checkpoint(tmp_path, kind)
    onnx_path = export_temporal(ckpt, window=8)
    model, norm, _ = load_temporal(ckpt)
    rt = OnnxTemporal(onnx_path)
    assert rt.classes == list(TEMPORAL_CLASSES) and rt.window == 8 and rt.model_type == kind
    feats = np.random.default_rng(1).normal(size=(12, IN_DIM)).astype(np.float32)
    for t in (1, 3, 8):  # a ring buffer that is still filling up must match torch exactly
        with torch.inference_mode():
            ref = model(torch.from_numpy(norm(feats[:t]))[None])[0].numpy()
        np.testing.assert_allclose(rt.logits(feats[:t]), ref, rtol=1e-4, atol=1e-4)
    with torch.inference_mode():  # longer buffers keep the newest `window` frames
        ref = model(torch.from_numpy(norm(feats[-8:]))[None])[0].numpy()
    np.testing.assert_allclose(rt.logits(feats), ref, rtol=1e-4, atol=1e-4)


def test_onnx_rejects_wrong_feature_dim(tmp_path: Path) -> None:
    rt = OnnxTemporal(export_temporal(_checkpoint(tmp_path, "tcn"), window=4))
    with pytest.raises(ValueError):
        rt.logits(np.zeros((4, IN_DIM + 1), dtype=np.float32))


def test_pipeline_runs_onnx_model_b_like_torch(tmp_path: Path) -> None:
    ckpt = _checkpoint(tmp_path, "tcn")
    onnx_path = export_temporal(ckpt, window=4)
    p_onnx = OnlinePipeline(temporal_ckpt=onnx_path, window=4)
    p_torch = OnlinePipeline(temporal_ckpt=ckpt, window=4)
    assert p_onnx.temporal_backend == "onnx" and p_torch.temporal_backend == "torch"
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    for _ in range(6):  # buffer fills up, then slides
        a, b = p_onnx(frame, 0.0), p_torch(frame, 0.0)
        assert set(a.temporal_scores) == set(TEMPORAL_CLASSES)
        for c in TEMPORAL_CLASSES:
            assert a.temporal_scores[c] == pytest.approx(b.temporal_scores[c], abs=1e-4)
