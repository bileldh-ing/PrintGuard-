"""Download PrintGuard training data from Zenodo.

Usage::

    python -m printguard.data.download --profile smoke   # ~0.5 GB, CPU/CI friendly
    python -m printguard.data.download --profile full    # selected members of every archive
    python -m printguard.data.download --profile full --whole-archives  # ~50 GB, MD5-verified

Outputs (under ``data/``)::

    manifests/<record>.json                 record metadata + published MD5 checksums
    manifests/<record>__<archive>.members.json   cached ZIP central-directory listing
    manifests/selection_<profile>.json      exactly which members were chosen, with groups
    raw/<record>/<member path>              extracted files (CRC-32 verified)
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from printguard.config import load_config
from printguard.data import selection
from printguard.data.zenodo import (
    ZenodoRecord,
    ZipMember,
    download_file,
    extract_remote_members,
    fetch_record,
    list_remote_zip,
    save_manifest,
)

LOGGER = logging.getLogger("printguard.download")


def _listing(record: ZenodoRecord, archive: str, manifests: Path) -> dict[str, ZipMember]:
    cache = manifests / f"{record.record_id}__{archive}.members.json"
    if cache.exists():
        rows = json.loads(cache.read_text())
        if rows and isinstance(rows[0], dict):
            return {r["name"]: ZipMember(**r) for r in rows}
    listing = list_remote_zip(record.file(archive).url)
    cache.write_text(json.dumps([asdict(m) for m in listing]))
    return {m.name: m for m in listing}


def _members(record: ZenodoRecord, archive: str, manifests: Path) -> list[str]:
    return list(_listing(record, archive, manifests))


def plan(cfg: dict[str, Any], profile: str, manifests: Path) -> dict[str, Any]:
    """Resolve a profile into ``{record_id: {archive: {group: [members]}}}`` (no file transfer)."""
    prof = cfg["profiles"][profile]
    seed = int(cfg.get("seed", 42))
    src = cfg["sources"]
    out: dict[str, Any] = {}

    trier = fetch_record(src["trier_fff"]["record_id"])
    save_manifest(trier, manifests / f"{trier.record_id}.json")
    t = prof["trier"]
    tr: dict[str, dict[str, list[str]]] = {}
    for split_root, archive_key, blocks in (("base", "base", t["base_blocks"]), ("test", "test", t["test_blocks"])):
        archive = src["trier_fff"]["archives"][archive_key]
        tr[archive] = selection.select_trier_members(
            _members(trier, archive, manifests), split_root, t["variant"], t["classes"], t["cameras"], blocks, t["block_len"]
        )
    if t.get("include_stringing_archive"):
        archive = src["trier_fff"]["archives"]["stringing"]
        mem = _members(trier, archive, manifests)
        groups: dict[str, list[str]] = {}
        for cls in ("stringing_error", "no_stringing_error"):
            # the stringing archive has one more path level (".../full_res/cameraX/img")
            sub = [m.replace("additional_defect_data/", "additional/", 1) for m in mem]
            sel = selection.select_trier_members(
                [m.replace(f"additional/{cls}/full_res/", f"additional/full_res/{cls}/", 1) for m in sub],
                "additional", "full_res", [cls], t["cameras"], t["base_blocks"], t["block_len"],
            )
            for g, ms in sel.items():
                groups[g] = [
                    m.replace(f"additional/full_res/{cls}/", f"additional_defect_data/{cls}/full_res/", 1) for m in ms
                ]
        tr[archive] = groups
    out[trier.record_id] = tr

    ys = fetch_record(src["yolo_strings"]["record_id"])
    save_manifest(ys, manifests / f"{ys.record_id}.json")
    archive = src["yolo_strings"]["archive"]
    mem = _members(ys, archive, manifests)
    y = prof["yolo_strings"]
    s = y["sequences"]
    out[ys.record_id] = {
        archive: {
            **{f"yolo/{k}": v for k, v in selection.select_yolo_strings_members(mem, y["max_groups"], seed).items()},
            "seqlabels": selection.select_sequence_label_members(mem, s["camera"], s["with_stringing"]),
            **{
                f"seq/{k}": v
                for k, v in selection.select_sequence_members(
                    mem, s["camera"], s["with_stringing"], s["without_stringing"], s["stride"], s["max_frames"]
                ).items()
            },
        }
    }
    return out


def run(config: str, profile: str, whole_archives: bool, workers: int) -> Path:
    """Plan and execute a download. Returns the path of the selection manifest."""
    cfg = load_config(config)
    root = Path(cfg.get("root", "data"))
    manifests, raw = root / "manifests", root / "raw"
    manifests.mkdir(parents=True, exist_ok=True)

    if whole_archives:
        for key in ("trier_fff", "yolo_strings"):
            rec = fetch_record(cfg["sources"][key]["record_id"])
            save_manifest(rec, manifests / f"{rec.record_id}.json")
            for f in rec.files:
                LOGGER.info("downloading %s (%.1f GB) with MD5 check", f.key, f.size / 1e9)
                download_file(f, root / "archives" / rec.record_id)
        return manifests

    sel = plan(cfg, profile, manifests)
    sel_path = manifests / f"selection_{profile}.json"
    sel_path.write_text(json.dumps(sel, indent=1))
    lock, count = threading.Lock(), [0]

    def _progress(n: int, total: int) -> None:
        with lock:
            count[0] += n
            if count[0] % 50 < n or count[0] == total:
                LOGGER.info("  %d/%d files", count[0], total)

    for record_id, archives in sel.items():
        rec = fetch_record(record_id)
        for archive, groups in archives.items():
            members = sorted({m for ms in groups.values() for m in ms})
            LOGGER.info("%s/%s: %d members in %d groups", record_id, archive, len(members), len(groups))
            count[0] = 0
            index = _listing(rec, archive, manifests)
            extract_remote_members(
                rec.file(archive).url, [index[m] for m in members], raw / record_id, workers=workers, progress=_progress
            )
    return sel_path


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--profile", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--whole-archives", action="store_true", help="download complete ZIPs and verify MD5")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(run(args.config, args.profile, args.whole_archives, args.workers))


if __name__ == "__main__":
    main()
