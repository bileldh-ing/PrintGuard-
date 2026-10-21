"""Build replay videos for the dashboard demo from held-out *test* sequences.

* ``data/demo/stringing_<seq>.mp4`` - a real YOLO-Strings recording with stringing;
* ``data/demo/layer_shift_<seq>.mp4`` - a defect-free test recording with a *synthetic* gradual
  layer shift (see ``printguard/data/augment.py``), clearly labelled as synthetic in the overlay.

Usage: ``python scripts/make_demo_video.py [--fps 4]``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from printguard.data.augment import apply_layer_shift, sample_params
from printguard.data.sequences import load_frames, load_index


def write(frames: list[np.ndarray], out: Path, fps: float, caption: str) -> None:
    """Write frames to an MP4 with a small caption."""
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i, f in enumerate(frames):
        f = f.copy()
        cv2.putText(
            f,
            f"{caption} | layer {i + 1}",
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        vw.write(f)
    vw.release()
    print(out)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default="data/processed/sequences/index.json")
    ap.add_argument("--out", default="data/demo")
    ap.add_argument("--fps", type=float, default=4)
    args = ap.parse_args()
    test = [r for r in load_index(args.index) if r.split == "test"]
    s = next(r for r in test if r.kind == "with_stringing")
    write(
        load_frames(s.frames, 640),
        Path(args.out) / f"stringing_{s.id.split('/')[-1]}.mp4",
        args.fps,
        "YOLO-Strings test sequence (real stringing)",
    )
    c = max((r for r in test if r.kind == "without_stringing"), key=lambda r: len(r.frames))
    frames = load_frames(c.frames, 640)
    shifted, _ = apply_layer_shift(frames, sample_params(frames, np.random.default_rng(7)))
    write(
        shifted,
        Path(args.out) / f"layer_shift_{c.id.split('/')[-1]}.mp4",
        args.fps,
        "SYNTHETIC gradual layer shift on a clean test print",
    )


if __name__ == "__main__":
    main()
