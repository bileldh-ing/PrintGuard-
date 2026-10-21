"""Model B pipeline: feature extraction -> temporal training -> A-vs-B evaluation.

Usage::

    python -m printguard.training.train_b features --config configs/model_b_tcn.yaml [--override ...]
    python -m printguard.training.train_b train    --config configs/model_b_tcn.yaml [--override ...]
    python -m printguard.training.train_b evaluate --config configs/model_b_tcn.yaml [--override ...]

``features`` is resumable (``--budget`` seconds per call) because Model-A inference over every
frame dominates the cost; ``train`` and ``evaluate`` only read the cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from printguard.config import apply_profile, load_config
from printguard.data.sequences import (
    CachedSequence,
    FeatureNormalizer,
    build_feature_cache,
    load_cache,
    load_index,
)
from printguard.evaluation import metrics as M
from printguard.taxonomy import DETECT_CLASSES, FRAME_CLS_CLASSES, TEMPORAL_CLASSES

if TYPE_CHECKING:  # torch is imported lazily so the `features` step stays torch-free
    import torch
    from torch import nn

LOGGER = logging.getLogger("printguard.train_b")
FrameFn = Callable[[Sequence[np.ndarray]], tuple[np.ndarray, np.ndarray]]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_frame_fn(cfg: dict[str, Any]) -> FrameFn:
    """Build ``frames -> (features, baseline_scores)`` from the config's Model-A + feature settings.

    ``baseline_scores[:, c]`` is Model A's own per-frame score for temporal class ``c`` (max
    stringing box confidence; classifier probability of ``layer_shift``), used to compare A vs B.
    """
    from printguard.models.detector import FrameAnalyzer
    from printguard.models.features import FrameFeatureExtractor, make_embedder

    analyzer = FrameAnalyzer.from_config(cfg["model_a"])
    fcfg = cfg["features"]
    embedder = make_embedder(
        fcfg.get("embedder"), fcfg.get("embed_dim", 128), fcfg.get("imgsz", 160), fcfg.get("pretrained", True)
    )
    extractor = FrameFeatureExtractor(DETECT_CLASSES, FRAME_CLS_CLASSES, embedder)

    def fn(frames: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        results = [analyzer(f) for f in frames]
        feats = np.concatenate([extractor(frames[i : i + 16], results[i : i + 16]) for i in range(0, len(frames), 16)])
        base = np.zeros((len(frames), len(TEMPORAL_CLASSES)), dtype=np.float32)
        for t, r in enumerate(results):
            base[t, 0] = max((d.confidence for d in r.detections if d.label == "stringing"), default=0.0)
            base[t, 1] = r.class_probs.get("layer_shift", 0.0)
        return feats, base

    return fn


def cmd_features(cfg: dict[str, Any], budget: float | None) -> int:
    """Populate the feature cache; returns the number of cached sequence variants."""
    records = load_index(cfg["data"]["index"])
    fn = make_frame_fn(cfg)
    done = build_feature_cache(
        records,
        fn,
        cfg["data"]["cache_dir"],
        cfg["augmentation"],
        cfg.get("seed", 42),
        cfg["data"].get("max_side", 640),
        budget_s=budget,
        frame_stride=int(cfg["data"].get("frame_stride", 1)),
    )
    total = sum(1 + _n_variants(r, cfg["augmentation"]) for r in records)
    LOGGER.info("feature cache: %d/%d sequence variants", len(done), total)
    return len(done)


def _n_variants(rec: Any, aug: dict[str, Any]) -> int:
    from printguard.data.sequences import variants_for

    return len(variants_for(rec, aug)) - 1


def masked_bce(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor
) -> torch.Tensor:
    """Binary cross-entropy averaged over observed (mask=1) entries only."""
    from torch import nn

    loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def predict(model: nn.Module, seqs: Sequence[CachedSequence], norm: FeatureNormalizer) -> list[np.ndarray]:
    """Per-frame sigmoid scores ``(T, C)`` for whole sequences (causal => identical to online use)."""
    import torch

    model.eval()
    with torch.inference_mode():
        return [torch.sigmoid(model(torch.from_numpy(norm(s.features))[None]))[0].numpy() for s in seqs]


def evaluate_scores(
    seqs: Sequence[CachedSequence], scores: Sequence[np.ndarray], ecfg: dict[str, Any]
) -> dict[str, Any]:
    """Frame metrics, detection delay and false alarms per temporal class."""
    thr, k, n, spf = ecfg["threshold"], ecfg["persistence_k"], ecfg["persistence_n"], ecfg["seconds_per_frame"]
    out: dict[str, Any] = {}
    for c, name in enumerate(TEMPORAL_CLASSES):
        y_all = np.concatenate([s.labels[:, c] for s in seqs])
        p_all = np.concatenate([sc[:, c] for sc in scores])
        states = [M.persistent(sc[:, c], thr, k, n) for sc in scores]
        delays = []
        for s, st in zip(seqs, states, strict=True):
            pos = np.where(s.labels[:, c] == 1)[0]
            if pos.size:
                delays.append(M.detection_delay(st, int(pos[0])))
        out[name] = {
            "frame": M.binary_pr(p_all, y_all, thr) | {"average_precision": M.average_precision(p_all, y_all)},
            "delay": M.summarize_delays(delays, spf),
            "false_alarms": M.false_alarms(states, [s.labels[:, c] for s in seqs], spf),
            "positive_frames": int((y_all == 1).sum()),
            "negative_frames": int((y_all == 0).sum()),
        }
    return out


def cmd_train(cfg: dict[str, Any]) -> Path:
    """Train the temporal model on the cached features; keeps the best epoch by val mean AP."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from printguard.models.temporal import build_temporal_model
    from printguard.training.datasets import WindowDataset

    seed_everything(int(cfg.get("seed", 42)))
    cache = cfg["data"]["cache_dir"]
    train, val = load_cache(cache, "train"), load_cache(cache, "val")
    if not train or not val:
        raise RuntimeError(f"Empty feature cache in {cache}; run the 'features' step first")
    norm = FeatureNormalizer.fit(train)
    tcfg = cfg["train"]
    ds = WindowDataset(train, norm, tcfg["window"], tcfg["stride"])
    loader = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(0))
    in_dim = train[0].features.shape[1]
    model = build_temporal_model(cfg["model"], in_dim, len(TEMPORAL_CLASSES))
    labels = np.concatenate([s.labels for s in train])
    pos = np.clip((labels == 1).sum(0), 1, None)
    neg = np.clip((labels == 0).sum(0), 1, None)
    pos_weight = torch.tensor(np.clip(neg / pos, 0.2, 10.0), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg["epochs"])
    history, best, best_state = [], -1.0, None
    t0 = time.time()
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        total = 0.0
        for x, y, m in loader:
            opt.zero_grad()
            loss = masked_bce(model(x), y, m, pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(x)
        sched.step()
        val_scores = predict(model, val, norm)
        ev = evaluate_scores(val, val_scores, cfg["evaluation"])
        aps = [ev[c]["frame"]["average_precision"] for c in TEMPORAL_CLASSES if ev[c]["positive_frames"]]
        mean_ap = float(np.mean(aps)) if aps else 0.0
        history.append({"epoch": epoch, "train_loss": total / len(ds), "val_mean_ap": mean_ap})
        LOGGER.info("epoch %d loss %.4f val mAP %.3f", epoch, total / len(ds), mean_ap)
        if mean_ap > best:
            best, best_state = mean_ap, {k: v.detach().clone() for k, v in model.state_dict().items()}
    out = Path(tcfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "model_cfg": cfg["model"],
            "in_dim": in_dim,
            "classes": list(TEMPORAL_CLASSES),
            "normalizer": norm.state(),
            "features": cfg["features"],
            "history": history,
            "best_val_mean_ap": best,
            "train_seconds": time.time() - t0,
            "n_params": sum(p.numel() for p in model.parameters()),
        },
        out,
    )
    LOGGER.info("saved %s (best val mAP %.3f)", out, best)
    return out


def load_temporal(path: str | Path) -> tuple[nn.Module, FeatureNormalizer, dict[str, Any]]:
    """Load a Model-B checkpoint written by :func:`cmd_train`."""
    import torch

    from printguard.models.temporal import build_temporal_model

    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_temporal_model(ck["model_cfg"], ck["in_dim"], len(ck["classes"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    norm = FeatureNormalizer(np.asarray(ck["normalizer"]["mean"]), np.asarray(ck["normalizer"]["std"]))
    return model, norm, ck


def cmd_evaluate(cfg: dict[str, Any], split: str = "test") -> Path:
    """Evaluate Model B and the Model-A baseline on ``split``; writes JSON + Markdown."""
    seqs = load_cache(cfg["data"]["cache_dir"], split)
    model, norm, ck = load_temporal(cfg["train"]["output"])
    stride = int(cfg["data"].get("frame_stride", 1))
    ecfg = dict(cfg["evaluation"])
    ecfg["frame_stride"] = stride
    ecfg["seconds_per_frame"] = float(ecfg["seconds_per_frame"]) * stride  # one cached frame = `stride` layers
    scores_b = predict(model, seqs, norm)
    scores_a = [s.baseline for s in seqs]
    res = {
        "split": split,
        "sequences": len(seqs),
        "frames": int(sum(len(s.features) for s in seqs)),
        "model": cfg["model"]["type"],
        "n_params": ck["n_params"],
        "evaluation_settings": ecfg,
        "model_b": evaluate_scores(seqs, scores_b, ecfg),
        "model_a_baseline": evaluate_scores(seqs, scores_a, ecfg),
    }
    rdir = Path(cfg["evaluation"]["results_dir"])
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / f"metrics_{split}.json").write_text(json.dumps(res, indent=2))
    (rdir / f"metrics_{split}.md").write_text(render_markdown(res))
    LOGGER.info("wrote %s", rdir)
    return rdir


def render_markdown(res: dict[str, Any]) -> str:
    """Markdown table comparing Model A (frame-wise) and Model B (temporal)."""
    spf = res["evaluation_settings"]["seconds_per_frame"]
    stride = res["evaluation_settings"].get("frame_stride", 1)
    lines = [
        f"### Model B ({res['model']}) vs Model A baseline - split `{res['split']}`",
        f"{res['sequences']} sequence variants, {res['frames']} frames. Alarm rule: "
        f"{res['evaluation_settings']['persistence_k']}-of-{res['evaluation_settings']['persistence_n']} frames "
        f"above {res['evaluation_settings']['threshold']}. One evaluated frame = {stride} printed layer(s); "
        f"per-hour figures assume {spf:g} s per evaluated frame.",
        "",
        "| Class | Model | Frame AP | P | R | Detected seq. | Median delay (frames) "
        "| False alarms /100 frames | /print-hour* |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in TEMPORAL_CLASSES:
        for label, key in (("A (frame)", "model_a_baseline"), ("B (temporal)", "model_b")):
            m = res[key][c]
            d = m["delay"]
            lines.append(
                f"| {c} | {label} | {m['frame']['average_precision']:.3f} | {m['frame']['precision']:.3f} | "
                f"{m['frame']['recall']:.3f} | {d['detected']}/{d['sequences']} | "
                f"{d.get('median_delay_frames', float('nan')):.1f} | {m['false_alarms']['per_100_frames']:.2f} | "
                f"{m['false_alarms']['per_print_hour']:.2f} |"
            )
    lines.append("")
    lines.append(f"\\* assumes {spf:g} s per evaluated frame; the dataset does not publish layer times.")
    return "\n".join(lines) + "\n"


def resolve(config: str, override: str | None) -> dict[str, Any]:
    """Load a Model-B config with an optional profile override."""
    return apply_profile(load_config(config), override)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["features", "train", "evaluate", "all"])
    ap.add_argument("--config", default="configs/model_b_tcn.yaml")
    ap.add_argument("--override", default=None)
    ap.add_argument("--budget", type=float, default=None, help="seconds for the features step (resumable)")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = resolve(args.config, args.override)
    if args.step in ("features", "all"):
        cmd_features(cfg, args.budget)
    if args.step in ("train", "all"):
        cmd_train(cfg)
    if args.step in ("evaluate", "all"):
        cmd_evaluate(cfg, args.split)


if __name__ == "__main__":
    main()
