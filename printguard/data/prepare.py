"""Turn downloaded raw files into training-ready datasets with leakage-safe splits.

Outputs under ``data/processed/``::

    trier_cls/{train,val,test}/<class>/<camera>__<file>.jpg   image-level classification (Model A, cls head)
    yolo_strings/images/{train,val,test}/*.jpg + labels/...    YOLO detection (Model A, det head)
    yolo_strings/dataset.yaml
    sequences/index.json                                       ordered frames + frame labels (Model B)
    splits.csv                                                 every file, its group and split

Split rules (see ``docs/data_card.md``):

* Trier: the official ``test`` archive is the test set. ``base`` frames are grouped into
  contiguous blocks (a proxy for a print, since the archive has no print IDs) and whole blocks
  go to train or val, stratified per class.
* YOLO-Strings: whole recording sequences go to one split (``split.yolo_strings_*_sequences``),
  so a print's FLIR and microscope views and all Roboflow copies stay together.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from printguard.config import load_config
from printguard.data.selection import frame_index, roboflow_group, sequence_id

LOGGER = logging.getLogger("printguard.prepare")
IMG_EXT = (".jpg", ".jpeg", ".png")


#: Longest image side written by :func:`_link`; 0 keeps originals (hard links). Set from the profile.
MAX_SIDE = 0


def _link(src: Path, dst: Path) -> None:
    """Hard-link ``src`` to ``dst`` or, when :data:`MAX_SIDE` is set, write a downscaled copy.

    Downscaling (aspect preserved) keeps CPU smoke runs within a small memory budget; YOLO
    labels are normalised, so they stay valid.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if MAX_SIDE:
        import cv2

        img = cv2.imread(str(src))
        if img is not None:
            h, w = img.shape[:2]
            scale = MAX_SIDE / max(h, w)
            if scale < 1:
                img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def split_groups(groups_by_class: dict[str, list[str]], val_fraction: float, seed: int) -> dict[str, str]:
    """Assign whole groups to ``train``/``val``, stratified per class (at least one val group each)."""
    rng = random.Random(seed)
    out: dict[str, str] = {}
    for cls in sorted(groups_by_class):
        gs = sorted(groups_by_class[cls])
        rng.shuffle(gs)
        n_val = max(1, round(len(gs) * val_fraction)) if len(gs) > 1 else 0
        for i, g in enumerate(gs):
            out[g] = "val" if i < n_val else "train"
    return out


def yolo_strings_split(name: str, val_seqs: set[str], test_seqs: set[str]) -> str:
    """Split of a YOLO-Strings file by its recording-sequence number (no id -> train)."""
    sid = sequence_id(name)
    if sid in test_seqs:
        return "test"
    if sid in val_seqs:
        return "val"
    return "train"


def prepare_trier(sel: dict[str, Any], cfg: dict[str, Any], raw: Path, out: Path, rows: list[dict[str, str]]) -> None:
    """Build the image-classification tree from the Trier selection."""
    class_map = cfg["sources"]["trier_fff"]["class_map"]
    groups: dict[str, list[str]] = {}
    for archive_groups in sel.values():
        groups.update(archive_groups)
    base = {g: m for g, m in groups.items() if g.startswith("base/")}
    by_class: dict[str, list[str]] = defaultdict(list)
    for g in base:
        by_class[g.split("/")[1]].append(g)
    assign = split_groups(by_class, cfg["split"]["val_fraction"], int(cfg.get("seed", 42)))
    for g, members in groups.items():
        split = "test" if g.startswith("test/") else assign.get(g, "train")
        _, folder, cam, _ = g.split("/")
        cls = class_map[folder]
        for m in members:
            src = raw / m
            if not src.exists():
                continue
            dst = out / "trier_cls" / split / cls / f"{cam}__{src.name}"
            _link(src, dst)
            rows.append({"dataset": "trier_cls", "file": str(dst), "group": g, "split": split, "label": cls})


def prepare_yolo_strings(
    sel: dict[str, list[str]], cfg: dict[str, Any], raw: Path, out: Path, rows: list[dict[str, str]]
) -> None:
    """Build the YOLO detection tree (images without a label file become background images)."""
    val_seqs = set(cfg["split"]["yolo_strings_val_sequences"])
    test_seqs = set(cfg["split"]["yolo_strings_test_sequences"])
    root = out / "yolo_strings"
    for g, members in sel.items():
        if not g.startswith("yolo/"):
            continue
        imgs = [m for m in members if m.lower().endswith(IMG_EXT)]
        txts = {roboflow_group(m) + Path(m).stem: m for m in members if m.endswith(".txt")}
        for img in imgs:
            src = raw / img
            if not src.exists():
                continue
            split = yolo_strings_split(src.name, val_seqs, test_seqs)
            stem = Path(img).stem.replace(" ", "")
            _link(src, root / "images" / split / f"{stem}{src.suffix.lower()}")
            lbl = root / "labels" / split / f"{stem}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            txt = txts.get(roboflow_group(img) + Path(img).stem)
            content = (raw / txt).read_text() if txt and (raw / txt).exists() else ""
            if not lbl.exists() or lbl.read_text() != content:
                lbl.write_text(content)
            rows.append({"dataset": "yolo_strings", "file": str(lbl), "group": str(sequence_id(src.name)),
                         "split": split, "label": "boxes" if lbl.stat().st_size else "background"})
    (root / "dataset.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: stringing\n"
    )


def prepare_sequences(
    sel: dict[str, list[str]], cfg: dict[str, Any], raw: Path, out: Path, rows: list[dict[str, str]]
) -> Path:
    """Index recording sequences with per-frame stringing labels.

    Frames of ``with_stringing`` sequences are labelled from the YOLO-Strings box annotations of
    the same frame (box present -> stringing). Frames without an annotation file are marked
    ``-1`` (unknown) and masked out of the loss and the metrics. ``without_stringing`` frames are 0.
    """
    val_seqs = set(cfg["split"]["yolo_strings_val_sequences"])
    test_seqs = set(cfg["split"]["yolo_strings_test_sequences"])
    labels: dict[str, int] = {}
    for m in sel.get("seqlabels", []):
        p = raw / m
        if p.exists():
            key = roboflow_group(m).replace(" ", "")
            labels[key] = max(labels.get(key, 0), int(bool(p.read_text().strip())))
    index = []
    for g, frames in sorted(sel.items()):
        if not g.startswith("seq/"):
            continue
        _, kind, seq = g.split("/")
        frames = [f for f in frames if (raw / f).exists()]
        frames.sort(key=lambda f: frame_index(f))
        if kind == "with_stringing":
            y = [labels.get(Path(f).stem.replace(" ", ""), -1) for f in frames]
        else:
            y = [0] * len(frames)
        num = seq.replace("sequence_", "")
        split = "test" if num in test_seqs else "val" if num in val_seqs else "train"
        index.append({"id": f"{kind}/{seq}", "kind": kind, "split": split,
                      "frames": [str(raw / f) for f in frames], "stringing": y})
        rows.append({"dataset": "sequences", "file": f"{kind}/{seq}", "group": num, "split": split, "label": kind})
    path = out / "sequences" / "index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=1))
    return path


def run(config: str, profile: str) -> Path:
    """Prepare every dataset for ``profile``; returns ``data/processed``."""
    global MAX_SIDE
    cfg = load_config(config)
    MAX_SIDE = int(cfg["profiles"][profile].get("resize_max_side", 0))
    root = Path(cfg.get("root", "data"))
    raw, out = root / "raw", root / "processed"
    sel = json.loads((root / "manifests" / f"selection_{profile}.json").read_text())
    rows: list[dict[str, str]] = []
    trier_id, ys_id = cfg["sources"]["trier_fff"]["record_id"], cfg["sources"]["yolo_strings"]["record_id"]
    prepare_trier(sel[trier_id], cfg, raw / trier_id, out, rows)
    ys_sel = sel[ys_id][cfg["sources"]["yolo_strings"]["archive"]]
    prepare_yolo_strings(ys_sel, cfg, raw / ys_id, out, rows)
    prepare_sequences(ys_sel, cfg, raw / ys_id, out, rows)
    with (out / "splits.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "file", "group", "split", "label"])
        w.writeheader()
        w.writerows(rows)
    _check_no_leakage(rows)
    LOGGER.info("prepared %d files into %s", len(rows), out)
    return out


def _check_no_leakage(rows: list[dict[str, str]]) -> None:
    """Fail loudly if any group appears in more than one split."""
    seen: dict[tuple[str, str], str] = {}
    for r in rows:
        if r["dataset"] == "yolo_strings" and r["group"] == "None":
            continue
        key = (r["dataset"], r["group"])
        if seen.setdefault(key, r["split"]) != r["split"]:
            raise RuntimeError(f"Leakage: group {key} in {seen[key]} and {r['split']}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--profile", default="smoke", choices=["smoke", "full"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(run(args.config, args.profile))


if __name__ == "__main__":
    main()
