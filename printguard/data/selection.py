"""Pure functions that pick which archive members to fetch.

Kept free of I/O so the selection logic (and the leakage guarantees that depend on it)
can be unit-tested against a synthetic member list.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence

_INDEX_RE = re.compile(r"(\d+)\.(?:jpe?g|png)$", re.IGNORECASE)
_ROBOFLOW_RE = re.compile(r"_(?:jpg|jpeg|png)\.rf\.[0-9a-f]+\.(?:jpg|jpeg|png|txt)$", re.IGNORECASE)
_SEQ_PREFIX_RE = re.compile(r"^([A-Za-z]*\d*?)_?\d+$")
_SEQ_ID_RE = re.compile(r"(?:data|good)\d*_(?:flir|micro)_(\d+)", re.IGNORECASE)


def frame_index(name: str) -> int:
    """Trailing integer of an image file name (``camera1_017.jpg`` -> 17)."""
    m = _INDEX_RE.search(name)
    if not m:
        raise ValueError(f"No frame index in {name!r}")
    return int(m.group(1))


def evenly_spaced_blocks(indices: Sequence[int], n_blocks: int, block_len: int) -> list[list[int]]:
    """Choose ``n_blocks`` runs of ``block_len`` consecutive positions, spread over ``indices``.

    ``indices`` must be sorted. Returns lists of *values* from ``indices``. ``n_blocks == 0``
    returns every index chunked into blocks (used by the ``full`` profile).
    """
    if not indices:
        return []
    if n_blocks <= 0:
        return [list(indices[i : i + block_len]) for i in range(0, len(indices), block_len)]
    n = len(indices)
    block_len = min(block_len, n)
    n_blocks = min(n_blocks, max(1, n // block_len))
    if n_blocks == 1:
        starts = [0]
    else:
        starts = [round(i * (n - block_len) / (n_blocks - 1)) for i in range(n_blocks)]
    return [list(indices[s : s + block_len]) for s in starts]


def select_trier_members(
    members: Iterable[str],
    split_root: str,
    variant: str,
    classes: Sequence[str],
    cameras: Sequence[str],
    n_blocks: int,
    block_len: int,
) -> dict[str, list[str]]:
    """Select contiguous frame blocks per (class, camera) from the Trier archive listing.

    Returns ``{group_id: [member, ...]}`` where ``group_id`` is
    ``"<split_root>/<class>/<camera>/b<k>"``. Groups are the unit for train/val splitting:
    consecutive frames are near-duplicates, so they must never straddle a split.
    """
    buckets: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for m in members:
        parts = m.split("/")
        if len(parts) != 5 or parts[0] != split_root or parts[1] != variant:
            continue
        _, _, cls, cam, fname = parts
        if cls in classes and cam in cameras and fname:
            try:
                buckets[(cls, cam)].append((frame_index(fname), m))
            except ValueError:
                continue
    groups: dict[str, list[str]] = {}
    for (cls, cam), items in sorted(buckets.items()):
        items.sort()
        by_idx = dict(items)
        blocks = evenly_spaced_blocks([i for i, _ in items], n_blocks, block_len)
        for k, block in enumerate(blocks):
            groups[f"{split_root}/{cls}/{cam}/b{k:04d}"] = [by_idx[i] for i in block]
    return groups


def roboflow_group(name: str) -> str:
    """Source-image key of a Roboflow export file (strips the ``_jpg.rf.<hash>`` suffix).

    Roboflow writes augmented copies of one source image with different hashes; grouping by
    this key keeps all copies on the same side of a split.
    """
    base = name.rsplit("/", 1)[-1]
    return _ROBOFLOW_RE.sub("", base)


def sequence_prefix(source_key: str) -> str:
    """Capture-session prefix of a YOLO-Strings source key (``mic6_72`` -> ``mic6``).

    Frames sharing a prefix come from the same recording session and are grouped for splitting.
    Keys without an alphabetic prefix (``_10``) fall into the ``"_"`` group.
    """
    key = source_key.lstrip("_")
    if "_" in key:
        return key.rsplit("_", 1)[0]
    m = _SEQ_PREFIX_RE.match(key)
    return (m.group(1) if m and m.group(1) else "_") or "_"


def sequence_id(name: str) -> str | None:
    """Recording-sequence number in a YOLO-Strings file name (``data1_micro_18_37`` -> ``"18"``).

    The FLIR and microscope views and the ``data``/``good`` portions of one print share the
    number, so it is the leakage-safe split unit. Returns ``None`` when the name carries no id.
    """
    m = _SEQ_ID_RE.search(name)
    return str(int(m.group(1))) if m else None


def select_yolo_strings_members(members: Iterable[str], max_groups: int, seed: int) -> dict[str, list[str]]:
    """Pick image+label pairs from the YOLO-Strings ``yolo/`` tree, grouped by source image.

    Returns ``{source_key: [img, label, ...]}``. ``max_groups == 0`` keeps everything.
    Sampling is stratified by capture-session prefix so every session is represented.
    """
    import random

    pairs: dict[str, list[str]] = defaultdict(list)
    for m in members:
        if "/yolo/" not in m or not m.lower().endswith((".jpg", ".jpeg", ".png", ".txt")):
            continue
        if m.endswith(("_darknet.labels", "desktop.ini")):
            continue
        pairs[roboflow_group(m)].append(m)
    keys = sorted(k for k, v in pairs.items() if any(x.lower().endswith((".jpg", ".jpeg", ".png")) for x in v))
    if max_groups and len(keys) > max_groups:
        rng = random.Random(seed)
        by_prefix: dict[str, list[str]] = defaultdict(list)
        for k in keys:
            by_prefix[sequence_prefix(k)].append(k)
        chosen: list[str] = []
        prefixes = sorted(by_prefix)
        for p in prefixes:
            rng.shuffle(by_prefix[p])
        while len(chosen) < max_groups:
            progressed = False
            for p in prefixes:
                if by_prefix[p] and len(chosen) < max_groups:
                    chosen.append(by_prefix[p].pop())
                    progressed = True
            if not progressed:
                break
        keys = sorted(chosen)
    return {k: sorted(pairs[k]) for k in keys}


def select_sequence_members(
    members: Iterable[str],
    camera: str,
    with_stringing: Sequence[str] | str,
    without_stringing: Sequence[str] | str,
    stride: int,
    max_frames: int,
) -> dict[str, list[str]]:
    """Pick frames of YOLO-Strings recording sequences.

    Returns ``{"<label>/<sequence>": [frames in temporal order]}``.
    """
    seqs: dict[str, list[tuple[int, str]]] = defaultdict(list)
    wanted = {"with_stringing": with_stringing, "without_stringing": without_stringing}
    for m in members:
        parts = m.split("/")
        if len(parts) != 6 or parts[1] != "sequences" or parts[4] != camera or not parts[5]:
            continue
        label, seq = parts[2], parts[3]
        sel = wanted.get(label)
        if sel is None or (sel != "all" and seq not in sel):
            continue
        try:
            seqs[f"{label}/{seq}"].append((frame_index(parts[5]), m))
        except ValueError:
            continue
    out: dict[str, list[str]] = {}
    for key, items in sorted(seqs.items()):
        items.sort()
        frames = [m for _, m in items][:: max(1, stride)]
        out[key] = frames[:max_frames] if max_frames else frames
    return out


def select_sequence_label_members(members: Iterable[str], camera: str, sequences: Sequence[str] | str) -> list[str]:
    """YOLO label files that annotate frames of the given ``with_stringing`` sequences.

    YOLO-Strings annotates (almost) every frame of its stringing sequences in the ``yolo/`` tree
    (``data1_<camera>_<seq>_<frame>_jpg.rf.<hash>.txt``). These give *frame-level* stringing
    labels, i.e. a real defect-onset time for temporal evaluation.
    """
    wanted = None if sequences == "all" else {s.replace("sequence_", "") for s in sequences}
    pat = re.compile(rf"^data\d*_{re.escape(camera)}_(\d+)_\s*\d+$")
    out = []
    for m in members:
        if "/yolo/" not in m or not m.endswith(".txt"):
            continue
        g = pat.match(roboflow_group(m))
        if g and (wanted is None or g.group(1) in wanted):
            out.append(m)
    return sorted(out)
