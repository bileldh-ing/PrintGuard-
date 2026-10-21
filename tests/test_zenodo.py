import io
import zipfile

from printguard.data.zenodo import ZipMember, fetch_cluster, parse_record, plan_ranges


def _zip_bytes() -> tuple[bytes, list[ZipMember]]:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a/one.txt", "hello " * 50, compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("a/two.bin", bytes(range(256)) * 4, compress_type=zipfile.ZIP_STORED)
        zf.writestr("b/three.txt", "printguard", compress_type=zipfile.ZIP_DEFLATED)
    data = buf.getvalue()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = [
            ZipMember(i.filename, i.file_size, i.header_offset, i.compress_size, i.compress_type, i.CRC)
            for i in zf.infolist()
        ]
    return data, members


class _RangeSession:
    """Fake requests.Session serving byte ranges of an in-memory archive."""

    def __init__(self, blob: bytes) -> None:
        self.blob, self.calls = blob, 0

    def get(self, url: str, headers: dict, timeout: float) -> object:
        self.calls += 1
        lo, hi = (int(v) for v in headers["Range"].removeprefix("bytes=").split("-"))
        blob = self.blob

        class R:
            status_code = 206
            content = blob[lo : hi + 1]
            headers: dict = {}

            def raise_for_status(self) -> None: ...

        return R()


def test_parse_record() -> None:
    rec = parse_record(
        {
            "id": 1,
            "doi": "10.5281/zenodo.1",
            "metadata": {"title": "T", "version": "2", "license": {"id": "cc-by-4.0"}},
            "files": [{"key": "a.zip", "size": 3, "checksum": "md5:abc", "links": {"self": "https://x/a.zip/content"}}],
        }
    )
    assert rec.title == "T" and rec.license == "cc-by-4.0"
    assert rec.file("a.zip").md5 == "abc"


def test_plan_ranges_coalesces_neighbours() -> None:
    ms = [ZipMember(f"f{i}", 10, i * 100, 10, 0, 0) for i in range(5)] + [ZipMember("far", 10, 10**9, 10, 0, 0)]
    clusters = plan_ranges(ms, max_gap=1000, max_span=10**6)
    assert [len(c) for c in clusters] == [5, 1]


def test_fetch_cluster_decodes_and_checks_crc() -> None:
    blob, members = _zip_bytes()
    sess = _RangeSession(blob)
    out = fetch_cluster("http://fake", plan_ranges(members)[0], sess)  # type: ignore[arg-type]
    assert sess.calls == 1
    assert out["a/one.txt"] == ("hello " * 50).encode()
    assert out["a/two.bin"] == bytes(range(256)) * 4
    assert out["b/three.txt"] == b"printguard"
