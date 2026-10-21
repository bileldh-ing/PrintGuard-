"""CPU inference benchmark (FPS / latency) for every model in the pipeline.

Usage::

    python -m printguard.evaluation.benchmark --detector weights/model_a_detect.onnx \
        --classifier weights/model_a_classify.onnx --embedder weights/embedder.onnx \
        --temporal weights/model_b_tcn.pt --out results/benchmark.json

Reports median / p90 latency and throughput per component and for the full per-frame
pipeline, together with the host CPU so numbers from a laptop and a Raspberry Pi can be compared.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def time_fn(fn: Callable[[], Any], warmup: int = 3, runs: int = 30) -> dict[str, float]:
    """Latency statistics (ms) of ``fn`` after ``warmup`` calls."""
    for _ in range(warmup):
        fn()
    lat = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    med = statistics.median(lat)
    return {
        "median_ms": round(med, 2),
        "p90_ms": round(lat[int(0.9 * (len(lat) - 1))], 2),
        "fps": round(1000 / med, 1) if med else 0.0,
        "runs": runs,
    }


def host_info() -> dict[str, Any]:
    """CPU / OS description for the report."""
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("model name", "hardware", "cpu model")):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "machine": platform.machine(),
        "cpu": cpu,
        "cores": os.cpu_count(),
        "python": platform.python_version(),
        "system": platform.system(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Benchmark the configured components on a synthetic 960x540 frame."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)
    res: dict[str, Any] = {"host": host_info(), "frame": "960x540 synthetic", "components": {}}
    fns: list[Callable[[], Any]] = []
    if args.detector:
        from printguard.models.detector import FrameAnalyzer

        det = FrameAnalyzer.from_config({"detector": {"weights": args.detector, "imgsz": args.det_imgsz}}).detector
        res["components"][f"detector ({Path(args.detector).name})"] = time_fn(lambda: det(frame), runs=args.runs)
        fns.append(lambda: det(frame))
    if args.classifier:
        from printguard.models.detector import FrameAnalyzer

        clf = FrameAnalyzer.from_config(
            {"classifier": {"weights": args.classifier, "imgsz": args.cls_imgsz}}
        ).classifier
        res["components"][f"classifier ({Path(args.classifier).name})"] = time_fn(lambda: clf(frame), runs=args.runs)
        fns.append(lambda: clf(frame))
    if args.embedder:
        from printguard.models.features import make_embedder

        emb = make_embedder(args.embedder)
        res["components"][f"embedder ({Path(args.embedder).name})"] = time_fn(lambda: emb([frame]), runs=args.runs)
        fns.append(lambda: emb([frame]))
    if args.temporal:
        if args.temporal.endswith(".onnx"):
            from printguard.inference.onnx_backend import _session

            sess = _session(args.temporal)
            shape = [d if isinstance(d, int) else 1 for d in sess.get_inputs()[0].shape]
            x = np.zeros(shape, dtype=np.float32)
            tfn = lambda: sess.run(None, {sess.get_inputs()[0].name: x})  # noqa: E731
        else:
            import torch

            from printguard.training.train_b import load_temporal

            model, _, ck = load_temporal(args.temporal)
            x = torch.zeros(1, args.window, ck["in_dim"])

            def tfn() -> Any:
                with torch.inference_mode():
                    return model(x)

        res["components"][f"temporal ({Path(args.temporal).name}, window {args.window})"] = time_fn(tfn, runs=args.runs)
        fns.append(tfn)
    if fns:
        res["pipeline_per_frame"] = time_fn(lambda: [f() for f in fns], runs=args.runs)
    return res


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector")
    ap.add_argument("--classifier")
    ap.add_argument("--embedder")
    ap.add_argument("--temporal")
    ap.add_argument("--det-imgsz", type=int, default=640)
    ap.add_argument("--cls-imgsz", type=int, default=224)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out", default="results/benchmark.json")
    args = ap.parse_args()
    res = run(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
