"""Evaluate Model A on its held-out test splits.

* **classifier** (Trier, image-level): per-class precision / recall / F1, accuracy and a
  confusion matrix (JSON + PNG). Runs on ``.onnx`` (torch-free) or ``.pt`` weights.
* **detector** (YOLO-Strings, boxes): Ultralytics ``val`` on the ``test`` split -> per-class
  precision, recall, mAP@0.5 and mAP@0.5:0.95.

Usage::

    python -m printguard.evaluation.eval_a classifier --weights weights/model_a_classify.onnx
    python -m printguard.evaluation.eval_a detector --weights weights/model_a_detect.pt --imgsz 640
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from printguard.evaluation.metrics import classification_report

LOGGER = logging.getLogger("printguard.eval_a")
IMG_EXT = {".jpg", ".jpeg", ".png"}


def eval_classifier(weights: str | Path, data_dir: str | Path, split: str = "test", imgsz: int = 224) -> dict[str, Any]:
    """Classification report of the frame classifier on ``<data_dir>/<split>/<class>/*.jpg``."""
    import cv2

    if str(weights).endswith(".onnx"):
        from printguard.inference.onnx_backend import OnnxClassifier

        clf = OnnxClassifier(weights)
    else:
        from printguard.models.detector import FrameClassifier

        clf = FrameClassifier(weights, imgsz=imgsz)
    root = Path(data_dir) / split
    class_names = sorted(p.name for p in root.iterdir() if p.is_dir())
    y_true, y_pred = [], []
    for ci, cname in enumerate(class_names):
        for img_path in sorted((root / cname).iterdir()):
            if img_path.suffix.lower() not in IMG_EXT:
                continue
            probs = clf(cv2.imread(str(img_path)))
            y_true.append(ci)
            y_pred.append(class_names.index(max(probs, key=probs.get)))
    rep = classification_report(y_true, y_pred, class_names)
    rep |= {"split": split, "weights": str(weights)}
    return rep


def plot_confusion(rep: dict[str, Any], out: Path, title: str) -> None:
    """Save a confusion-matrix heat map."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(rep["confusion_matrix"])
    names = rep["class_names"]
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black"
            )
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def eval_detector(weights: str | Path, data_yaml: str | Path, split: str = "test", imgsz: int = 640) -> dict[str, Any]:
    """Ultralytics validation of the box detector on ``split``."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    m = model.val(
        data=str(data_yaml), split=split, imgsz=imgsz, batch=4, device="cpu", plots=False, verbose=False, workers=0
    )
    names = model.names
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[names[int(c)]] = {
            "precision": float(p),
            "recall": float(r),
            "mAP50": float(ap50),
            "mAP50_95": float(ap),
        }
    return {
        "split": split,
        "weights": str(weights),
        "imgsz": imgsz,
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "per_class": per_class,
        "speed_ms_per_image": {k: float(v) for k, v in m.speed.items()},
        "images": _count_images(data_yaml, split),
    }


def _count_images(data_yaml: str | Path, split: str) -> int:
    import yaml

    cfg = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(cfg.get("path", Path(data_yaml).parent))
    folder = root / cfg.get(split, f"images/{split}")
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMG_EXT) if folder.exists() else 0


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("head", choices=["classifier", "detector"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=None, help="trier_cls dir or YOLO dataset.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="results/model_a")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.head == "classifier":
        rep = eval_classifier(args.weights, args.data or "data/processed/trier_cls", args.split, args.imgsz)
        (out / f"classifier_{args.split}.json").write_text(json.dumps(rep, indent=2))
        plot_confusion(rep, out / f"classifier_{args.split}_confusion.png", f"Frame classifier - {args.split}")
    else:
        rep = eval_detector(
            args.weights, args.data or "data/processed/yolo_strings/dataset.yaml", args.split, args.imgsz
        )
        (out / f"detector_{args.split}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "confusion_matrix"}, indent=2))


if __name__ == "__main__":
    main()
