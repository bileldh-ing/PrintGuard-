"""Zenodo access: record metadata, checksummed full downloads and partial ZIP extraction.

The Trier FFF dataset (record 18628777) ships as three ZIP archives of 12-20 GB each.
Downloading them whole is not practical on a laptop or CI runner, so this module can
also read a ZIP's central directory over HTTP range requests and extract only the
members we need. Integrity is verified at two levels:

* full archives: the MD5 published by Zenodo is checked after download;
* partial extraction: every member is CRC-32 checked by :mod:`zipfile` on read.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import threading
import time
import zipfile
import zlib
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

LOGGER = logging.getLogger(__name__)
ZENODO_API = "https://zenodo.org/api/records/{record_id}"
CHUNK = 1 << 20


@dataclass(frozen=True)
class ZenodoFile:
    """One file attached to a Zenodo record."""

    key: str
    size: int
    checksum: str  # "md5:<hex>"
    url: str

    @property
    def md5(self) -> str:
        algo, _, digest = self.checksum.partition(":")
        if algo != "md5":
            raise ValueError(f"Unsupported checksum algorithm: {algo}")
        return digest


@dataclass(frozen=True)
class ZenodoRecord:
    """Minimal view of a Zenodo record."""

    record_id: str
    title: str
    version: str | None
    license: str | None
    doi: str | None
    files: tuple[ZenodoFile, ...]

    def file(self, key: str) -> ZenodoFile:
        for f in self.files:
            if f.key == key:
                return f
        raise KeyError(f"{key!r} not in record {self.record_id}: {[f.key for f in self.files]}")


def fetch_record(record_id: str | int, session: requests.Session | None = None) -> ZenodoRecord:
    """Fetch record metadata (title, license, file list with MD5 checksums) from the Zenodo API."""
    http = session or requests.Session()
    resp = http.get(ZENODO_API.format(record_id=record_id), timeout=30)
    resp.raise_for_status()
    return parse_record(resp.json())


def parse_record(payload: dict) -> ZenodoRecord:
    """Parse a Zenodo ``/api/records/<id>`` JSON payload."""
    meta = payload.get("metadata", {})
    files = tuple(
        ZenodoFile(key=f["key"], size=int(f["size"]), checksum=f["checksum"], url=f["links"]["self"])
        for f in payload.get("files", [])
    )
    lic = meta.get("license")
    return ZenodoRecord(
        record_id=str(payload.get("id", payload.get("recid", ""))),
        title=meta.get("title", ""),
        version=meta.get("version"),
        license=lic.get("id") if isinstance(lic, dict) else lic,
        doi=payload.get("doi") or meta.get("doi"),
        files=files,
    )


def save_manifest(record: ZenodoRecord, path: Path) -> None:
    """Write the record's file list and checksums to a JSON manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=2))


def md5sum(path: Path) -> str:
    """Return the hex MD5 digest of a file, streamed in 1 MiB chunks."""
    digest = hashlib.md5()  # noqa: S324 - integrity check against Zenodo, not security
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(zf: ZenodoFile, dest_dir: Path, session: requests.Session | None = None) -> Path:
    """Download a whole file with resume support and verify its MD5 checksum.

    Raises:
        IOError: if the checksum does not match after download.
    """
    http = session or requests.Session()
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / zf.key
    if out.exists() and out.stat().st_size == zf.size and md5sum(out) == zf.md5:
        LOGGER.info("%s already present and verified", out)
        return out
    part = out.with_suffix(out.suffix + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    with http.get(zf.url, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        mode = "ab" if resp.status_code == 206 else "wb"
        with part.open(mode) as fh:
            for chunk in resp.iter_content(CHUNK):
                fh.write(chunk)
    if md5sum(part) != zf.md5:
        raise OSError(f"MD5 mismatch for {zf.key}; delete {part} and retry")
    part.rename(out)
    return out


@dataclass(frozen=True)
class ZipMember:
    """Central-directory entry of a remote ZIP, enough to fetch the member with one request."""

    name: str
    size: int
    offset: int
    compressed_size: int
    method: int
    crc: int


def list_remote_zip(url: str) -> list[ZipMember]:
    """Read a remote ZIP's central directory over HTTP range requests (no archive download)."""
    from remotezip import RemoteZip

    with RemoteZip(url) as rz:
        return [
            ZipMember(i.filename, i.file_size, i.header_offset, i.compress_size, i.compress_type, i.CRC)
            for i in rz.infolist()
        ]


def _decode_member(blob: bytes, member: ZipMember) -> bytes:
    """Decode one member from bytes that start at its local file header; verify CRC-32."""
    if blob[:4] != b"PK\x03\x04":
        raise OSError(f"Bad local header for {member.name}")
    name_len, extra_len = struct.unpack("<HH", blob[26:30])
    start = 30 + name_len + extra_len
    data = blob[start : start + member.compressed_size]
    if len(data) < member.compressed_size:
        raise OSError(f"Truncated data for {member.name}")
    if member.method == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(data, -15)
    elif member.method != zipfile.ZIP_STORED:
        raise OSError(f"Unsupported compression {member.method} for {member.name}")
    if zlib.crc32(data) & 0xFFFFFFFF != member.crc:
        raise OSError(f"CRC-32 mismatch for {member.name}")
    return data


def plan_ranges(
    members: Iterable[ZipMember], max_gap: int = 4 << 20, max_span: int = 48 << 20, slack: int = 1024
) -> list[list[ZipMember]]:
    """Group members whose bytes are close in the archive so one range request fetches them all.

    Zenodo rate-limits requests (~130/min), not bytes, so coalescing neighbouring members
    (e.g. consecutive frames of one camera) is far faster than one request per file.
    """
    ordered = sorted(members, key=lambda m: m.offset)
    clusters: list[list[ZipMember]] = []
    for m in ordered:
        if clusters:
            first, last = clusters[-1][0], clusters[-1][-1]
            last_end = last.offset + 30 + len(last.name.encode()) + slack + last.compressed_size
            end = m.offset + 30 + len(m.name.encode()) + slack + m.compressed_size
            if m.offset - last_end <= max_gap and end - first.offset <= max_span:
                clusters[-1].append(m)
                continue
        clusters.append([m])
    return clusters


def _get(session: requests.Session, url: str, lo: int, hi: int, retries: int = 6) -> bytes:
    for attempt in range(retries):
        resp = session.get(url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=120)
        if resp.status_code == 429:  # rate limited: wait for the window to reset
            wait = float(resp.headers.get("Retry-After") or 0) or 10 * (attempt + 1)
            LOGGER.info("rate limited, sleeping %.0fs", wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.content
    raise OSError(f"Gave up on range {lo}-{hi} after {retries} attempts")


def fetch_cluster(url: str, cluster: list[ZipMember], session: requests.Session, slack: int = 1024) -> dict[str, bytes]:
    """Fetch a cluster of members with one range request and return ``{name: data}``."""
    lo = cluster[0].offset
    last = cluster[-1]
    hi = last.offset + 30 + len(last.name.encode()) + slack + last.compressed_size
    blob = _get(session, url, lo, hi)
    out: dict[str, bytes] = {}
    for m in cluster:
        rel = m.offset - lo
        try:
            out[m.name] = _decode_member(blob[rel:], m)
        except OSError:  # extra field bigger than the slack: refetch this member alone
            end = m.offset + 30 + len(m.name.encode()) + 65536 + m.compressed_size
            out[m.name] = _decode_member(_get(session, url, m.offset, end), m)
    return out


def extract_remote_members(
    url: str,
    members: Iterable[ZipMember],
    dest_dir: Path,
    workers: int = 4,
    progress: Callable[[int, int], None] | None = None,
) -> list[Path]:
    """Extract selected members of a remote ZIP without downloading the archive.

    Neighbouring members are fetched together (see :func:`plan_ranges`), every member is
    CRC-32 verified, and existing files are skipped so interrupted runs resume.
    """
    members = list(members)
    todo = [m for m in members if not (dest_dir / m.name).exists()]
    sessions = threading.local()

    def _one(cluster: list[ZipMember]) -> int:
        if not hasattr(sessions, "s"):
            sessions.s = requests.Session()
        for name, data in fetch_cluster(url, cluster, sessions.s).items():
            target = dest_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.rename(target)
        if progress:
            progress(len(cluster), len(todo))
        return len(cluster)

    clusters = plan_ranges(todo)
    LOGGER.info("%d files in %d range requests", len(todo), len(clusters))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(_one, c) for c in clusters]):
            fut.result()
    return [dest_dir / m.name for m in members]
