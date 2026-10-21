"""YAML configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file. Relative paths are resolved against the repository root."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / p
    with p.open() as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{p} must contain a mapping at top level")
    if "base" in cfg:  # simple inheritance: `base: other.yaml`, then override keys
        base = load_config(cfg.pop("base"))
        cfg = deep_update(base, cfg)
    return cfg


def apply_profile(cfg: dict[str, Any], override: str | Path | None) -> dict[str, Any]:
    """Merge a profile override (e.g. ``configs/smoke/model_b.yaml``) into ``cfg``.

    When the override sets ``profile: <name>``, output artefacts are redirected to
    ``weights/<name>/`` and ``results/<name>/`` so smoke runs never overwrite real ones.
    """
    if not override:
        return cfg
    merged = deep_update(cfg, load_config(override))
    prof = merged.get("profile")
    if prof:
        if "train" in merged and "output" in merged["train"]:
            merged["train"]["output"] = str(Path("weights") / prof / Path(merged["train"]["output"]).name)
        if "evaluation" in merged and "results_dir" in merged["evaluation"]:
            merged["evaluation"]["results_dir"] = str(
                Path("results") / prof / Path(merged["evaluation"]["results_dir"]).name
            )
    return merged


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], val)
        else:
            out[key] = val
    return out
