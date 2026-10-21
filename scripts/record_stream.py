"""Record the live WebSocket stream of a running PrintGuard server into a GIF + summary.

Each GIF frame is exactly what the server sent (JPEG + detections + scores), rendered the way
the dashboard draws it - useful for the README without a browser.

Usage: ``python scripts/record_stream.py --url ws://127.0.0.1:8000/ws --frames 40 --out docs/assets/stream.gif``
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path

import cv2
import numpy as np

COLORS = {"stringing": (11, 158, 245), "layer_shift": (250, 139, 167), "extrusion_error": (68, 68, 239)}


def render(msg: dict) -> np.ndarray:
    """Draw boxes, frame-classifier badge and temporal scores onto the streamed JPEG."""
    img = cv2.imdecode(np.frombuffer(base64.b64decode(msg["jpeg"]), np.uint8), cv2.IMREAD_COLOR)
    ev = msg["event"]
    sx, sy = img.shape[1] / ev["width"], img.shape[0] / ev["height"]
    for d in ev["detections"]:
        x1, y1, x2, y2 = (int(v * s) for v, s in zip(d["xyxy"], (sx, sy, sx, sy), strict=True))
        c = COLORS.get(d["label"], (166, 184, 20))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        cv2.putText(
            img,
            f"{d['label']} {d['confidence']:.0%}",
            (x1, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            c,
            1,
            cv2.LINE_AA,
        )
    panel = np.full((70, img.shape[1], 3), (22, 16, 11), np.uint8)
    y = 18
    scores = ev["temporal_scores"] or {k: v for k, v in ev["class_probs"].items() if k != "no_error"}
    src = "temporal (B)" if ev["temporal_scores"] else "frame (A)"
    cv2.putText(
        panel,
        f"PrintGuard AI  frame {ev['frame_idx']}  scores: {src}",
        (8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (230, 237, 243),
        1,
        cv2.LINE_AA,
    )
    for i, (k, v) in enumerate(sorted(scores.items())):
        yy = y + 18 * (i + 1)
        cv2.putText(panel, f"{k:<14}", (8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (176, 155, 139), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (130, yy - 9), (130 + int(200 * v), yy - 1), COLORS.get(k, (166, 184, 20)), -1)
        cv2.putText(panel, f"{v:.0%}", (340, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 237, 243), 1, cv2.LINE_AA)
    if msg.get("alerts"):
        cv2.putText(
            panel,
            "ALERT: " + ", ".join(a["label"] for a in msg["alerts"]),
            (img.shape[1] - 230, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (68, 68, 239),
            2,
            cv2.LINE_AA,
        )
    return np.vstack([img, panel])


async def record(url: str, n: int) -> list[dict]:
    """Collect ``n`` frame messages from the server WebSocket."""
    import websockets

    out: list[dict] = []
    async with websockets.connect(url, max_size=2**24, proxy=None) as ws:  # local server: bypass proxies
        while len(out) < n:
            msg = json.loads(await ws.recv())
            if msg.get("type") == "frame":
                out.append(msg)
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--out", default="docs/assets/stream.gif")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--every", type=int, default=1, help="keep every n-th recorded frame in the GIF")
    args = ap.parse_args()
    msgs = asyncio.run(record(args.url, args.frames))
    from PIL import Image

    frames = []
    for m in msgs[:: args.every]:
        img = render(m)
        h = round(img.shape[0] * args.width / img.shape[1])
        frames.append(Image.fromarray(cv2.cvtColor(cv2.resize(img, (args.width, h)), cv2.COLOR_BGR2RGB)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = [f.quantize(colors=96, method=Image.Quantize.MEDIANCUT) for f in frames]  # keep the GIF small
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=250, loop=0, optimize=True)
    cv2.imwrite(str(out.with_suffix(".jpg")), render(msgs[-1]))
    summary = {
        "frames": len(msgs),
        "alerts": [a for m in msgs for a in m["alerts"]],
        "mean_latency_ms": float(np.mean([m["event"]["latency_ms"] for m in msgs])),
        "status": msgs[-1]["status"],
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
