"""End-to-end Model-B training/evaluation on a synthetic feature cache (seconds on CPU)."""

import json
from pathlib import Path

import numpy as np

from printguard.training import train_b


def _write_cache(cache: Path, rng: np.random.Generator) -> None:
    cache.mkdir(parents=True)
    for split, n in (("train", 8), ("val", 3), ("test", 3)):
        for i in range(n):
            t = 40
            onset = int(rng.integers(10, 30)) if i % 2 == 0 else t
            labels = np.zeros((t, 2), dtype=np.int64)
            labels[onset:, 1] = 1
            feats = rng.normal(size=(t, 6)).astype(np.float32)
            feats[:, 0] += 3 * labels[:, 1]  # an informative feature
            base = np.clip(feats[:, :2] / 6 + 0.5, 0, 1).astype(np.float32)
            meta = json.dumps({"id": f"{split}{i}", "variant": "", "split": split, "kind": "without_stringing"})
            np.savez_compressed(cache / f"{split}{i}.npz", features=feats, labels=labels, baseline=base, meta=meta)


def test_train_and_evaluate(tmp_path: Path) -> None:
    _write_cache(tmp_path / "cache", np.random.default_rng(0))
    cfg = {
        "seed": 0,
        "data": {"cache_dir": str(tmp_path / "cache"), "frame_stride": 1},
        "features": {"embedder": None},
        "model": {"type": "tcn", "channels": 16, "levels": 2},
        "train": {
            "window": 16,
            "stride": 4,
            "epochs": 15,
            "batch_size": 8,
            "lr": 0.01,
            "weight_decay": 0.0,
            "output": str(tmp_path / "b.pt"),
        },
        "evaluation": {
            "threshold": 0.5,
            "persistence_k": 2,
            "persistence_n": 3,
            "seconds_per_frame": 30,
            "results_dir": str(tmp_path / "res"),
        },
    }
    ckpt = train_b.cmd_train(cfg)
    assert ckpt.exists()
    rdir = train_b.cmd_evaluate(cfg, "test")
    res = json.loads((rdir / "metrics_test.json").read_text())
    assert res["model_b"]["layer_shift"]["frame"]["average_precision"] > 0.8  # trivially learnable signal
    assert "| layer_shift | B (temporal)" in (rdir / "metrics_test.md").read_text()
