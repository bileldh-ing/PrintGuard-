"""Export models to ONNX for torch-free CPU / Raspberry Pi inference.

Usage::

    python -m printguard.export yolo --weights weights/model_a_detect.pt --imgsz 640
    python -m printguard.export yolo --weights weights/model_a_classify.pt --imgsz 224
    python -m printguard.export embedder --out weights/embedder.onnx --embed-dim 128 --imgsz 160
    python -m printguard.export temporal --weights weights/model_b_tcn.pt --window 32

YOLO exports go through Ultralytics (``model.export(format="onnx")``) and keep the class names
in the ONNX metadata. The embedder and temporal models use ``torch.onnx.export``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

LOGGER = logging.getLogger("printguard.export")


def export_yolo(weights: str | Path, imgsz: int, opset: int = 17) -> Path:
    """Export an Ultralytics detection/classification checkpoint; returns the ``.onnx`` path."""
    from ultralytics import YOLO

    out = YOLO(str(weights)).export(format="onnx", imgsz=imgsz, opset=opset, dynamic=False, simplify=False)
    return Path(out)


def export_embedder(out: str | Path, embed_dim: int = 128, imgsz: int = 160, pretrained: bool = True) -> Path:
    """Export the frozen MobileNetV3-Small embedding CNN (incl. its fixed projection)."""
    import torch

    from printguard.models.embedding import EmbeddingCNN

    model = EmbeddingCNN(embed_dim, pretrained).eval()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, imgsz, imgsz)

    class _Plain(torch.nn.Module):  # strip inference_mode decorator for tracing
        def __init__(self, m: EmbeddingCNN) -> None:
            super().__init__()
            self.m = m

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.m.pool(self.m.trunk(x)).flatten(1)
            return z @ self.m.proj if self.m.proj is not None else z

    torch.onnx.export(
        _Plain(model),
        dummy,
        str(out),
        input_names=["images"],
        output_names=["embedding"],
        dynamic_axes={"images": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    LOGGER.info("embedder (pretrained=%s) -> %s", model.pretrained, out)
    return out


def export_temporal(weights: str | Path, window: int = 32, out: str | Path | None = None) -> Path:
    """Export a Model-B checkpoint to a self-contained ONNX file.

    The graph has a fixed ``(1, window, in_dim)`` input (static shapes trace cleanly for both the
    TCN and ``nn.MultiheadAttention`` and run faster on small CPUs). Both architectures are
    causal, so :class:`~printguard.inference.onnx_backend.OnnxTemporal` right-pads a ring buffer
    that is still filling up and reads the output at the last real frame, which gives exactly
    the PyTorch scores. Everything the torch-free runtime needs is stored in the ONNX metadata:
    class names, feature normaliser, feature-extractor settings, model type and window length.
    """
    import json

    import onnx
    import torch

    from printguard.training.train_b import load_temporal

    model, _, ck = load_temporal(weights)
    out = Path(out or Path(weights).with_suffix(".onnx"))
    dummy = torch.zeros(1, window, ck["in_dim"])
    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["features"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )
    proto = onnx.load(str(out))
    meta = {
        "printguard_model": "temporal",
        "model_type": str(ck["model_cfg"].get("type", "tcn")),
        "classes": json.dumps(list(ck["classes"])),
        "normalizer": json.dumps({k: list(map(float, v)) for k, v in ck["normalizer"].items()}),
        "features": json.dumps(ck.get("features", {})),
        "in_dim": str(ck["in_dim"]),
        "window": str(window),
    }
    del proto.metadata_props[:]
    for k, v in meta.items():
        proto.metadata_props.add(key=k, value=v)
    onnx.save(proto, str(out))
    LOGGER.info("temporal model (%s, window %d) -> %s", meta["model_type"], window, out)
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["yolo", "embedder", "temporal"])
    ap.add_argument("--weights")
    ap.add_argument("--out")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--window", type=int, default=32)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.what == "yolo":
        print(export_yolo(args.weights, args.imgsz))
    elif args.what == "embedder":
        print(export_embedder(args.out or "weights/embedder.onnx", args.embed_dim, args.imgsz))
    else:
        print(export_temporal(args.weights, args.window, args.out))


if __name__ == "__main__":
    main()
