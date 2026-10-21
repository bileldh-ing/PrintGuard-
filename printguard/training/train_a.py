"""Train Model A (YOLO detection or classification head) from a YAML config.

Usage::

    python -m printguard.training.train_a --config configs/model_a_detect.yaml
    python -m printguard.training.train_a --config configs/model_a_classify.yaml \
        --override configs/smoke/model_a_classify.yaml --device cpu
    python -m printguard.training.train_a --config ... --resume   # continue an interrupted run
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from printguard.config import deep_update, load_config

LOGGER = logging.getLogger("printguard.train_a")


def resolve_config(config: str, override: str | None = None) -> dict[str, Any]:
    """Load a Model-A config and apply an optional override file (e.g. a smoke profile)."""
    cfg = load_config(config)
    if override:
        cfg = deep_update(cfg, load_config(override))
    if cfg.get("task") not in {"detect", "classify"}:
        raise ValueError("Model-A config needs task: detect | classify")
    return cfg


def run_dir(cfg: dict[str, Any]) -> Path:
    """Directory Ultralytics writes the run to."""
    t = cfg["train"]
    return Path(t.get("project", "runs/model_a")).resolve() / t.get("name", cfg["task"])


def train(cfg: dict[str, Any], device: str = "cpu", resume: bool = False) -> Path:
    """Train (or resume) and copy ``best.pt`` to ``cfg['output_weights']``. Returns that path."""
    from ultralytics import YOLO

    rdir = run_dir(cfg)
    last = rdir / "weights" / "last.pt"
    if resume and last.exists():
        LOGGER.info("resuming from %s", last)
        YOLO(str(last)).train(resume=True)
    else:
        params = dict(cfg["train"])
        params["project"] = str(Path(params.get("project", "runs/model_a")).resolve())
        YOLO(cfg["model"]).train(data=str(Path(cfg["data"]).resolve()), device=device, **params)
    best = rdir / "weights" / "best.pt"
    if not best.exists():
        best = last
    out = Path(cfg["output_weights"])
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    (out.with_suffix(".json")).write_text(json.dumps({"config": cfg, "run_dir": str(rdir)}, indent=2, default=str))
    LOGGER.info("saved %s", out)
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", default=None, help="YAML merged on top of --config (e.g. a smoke profile)")
    ap.add_argument("--device", default="cpu", help="'cpu', '0', '0,1', 'mps'")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(train(resolve_config(args.config, args.override), args.device, args.resume))


if __name__ == "__main__":
    main()
