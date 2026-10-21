"""API tests with a fake Model A (no weights needed) and a fake printer."""

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from printguard.inference.alerts import AlertPolicy
from printguard.inference.pipeline import OnlinePipeline
from printguard.models.detector import Detection, FrameAnalyzer
from server.app import create_app


class FakeDetector:
    def __call__(self, frame: np.ndarray) -> list[Detection]:
        return [Detection("stringing", 0.9, (10.0, 10.0, 50.0, 40.0))]


class FakePrinter:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def pause(self) -> tuple[bool, str]:
        self.calls += 1
        return True, "HTTP 204"


def _settings(tmp_path: Path, pause_enabled: bool = False, dry_run: bool = True) -> dict:
    return {
        "server": {"database": str(tmp_path / "t.sqlite3"), "stream_fps": 50, "max_width": 160},
        "source": {"autostart": False},
        "printer": {"pause_enabled": pause_enabled, "dry_run": dry_run, "min_seconds_between_pauses": 0},
    }


def _factory() -> OnlinePipeline:
    return OnlinePipeline(FrameAnalyzer(FakeDetector()), policy=AlertPolicy(k=2, n=3, cooldown_frames=1000))


def _wait(client: TestClient, cond: Callable[[dict], bool], timeout: float = 10.0) -> dict:
    t0, st = time.time(), {}
    while time.time() - t0 < timeout:
        st = client.get("/api/status").json()
        if cond(st):
            return st
        time.sleep(0.05)
    raise AssertionError(f"timeout; last status {st}")


@pytest.fixture()
def printer() -> FakePrinter:
    return FakePrinter()


def test_health_and_dashboard(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path), _factory)) as c:
        assert c.get("/api/health").json() == {"status": "ok"}
        assert "PrintGuard" in c.get("/").text
        assert c.get("/api/status").json()["printer"]["pause_enabled"] is False


def test_replay_file_records_events_and_alerts(tmp_path: Path, video_file: Path) -> None:
    with TestClient(create_app(_settings(tmp_path), _factory)) as c:
        r = c.post("/api/source", json={"kind": "file", "uri": str(video_file), "loop": False, "target_fps": 30})
        assert r.status_code == 200
        _wait(c, lambda s: s["worker"]["frames_processed"] >= 5)
        events = c.get("/api/events").json()
        assert events and events[0]["label"] == "stringing" and events[0]["bbox"] == [10.0, 10.0, 50.0, 40.0]
        alerts = c.get("/api/alerts").json()
        assert len(alerts) == 1 and alerts[0]["status"] == "open"  # cooldown prevents duplicates
        assert c.get("/api/frame.jpg").headers["content-type"] == "image/jpeg"
        with c.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "status"
        c.post("/api/source/stop")


def test_bad_source_rejected(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path), _factory)) as c:
        assert c.post("/api/source", json={"kind": "file", "uri": "missing.mp4"}).status_code == 404
        assert c.post("/api/source", json={"kind": "usb", "uri": "cam"}).status_code == 422
        assert c.post("/api/source", json={"kind": "ftp", "uri": "x"}).status_code == 422


def _open_alert(c: TestClient, video_file: Path) -> int:
    c.post("/api/source", json={"kind": "file", "uri": str(video_file), "loop": False, "target_fps": 30})
    _wait(c, lambda s: s["worker"]["frames_processed"] >= 4)
    c.post("/api/source/stop")
    return c.get("/api/alerts").json()[0]["id"]


def test_pause_is_disabled_by_default(tmp_path: Path, video_file: Path, printer: FakePrinter) -> None:
    with TestClient(create_app(_settings(tmp_path), _factory, printer)) as c:
        aid = _open_alert(c, video_file)
        r = c.post(f"/api/alerts/{aid}/approve-pause", json={"operator": "bilel"})
        assert r.status_code == 403 and printer.calls == 0


def test_pause_requires_approval_and_happens_once(tmp_path: Path, video_file: Path, printer: FakePrinter) -> None:
    with TestClient(create_app(_settings(tmp_path, pause_enabled=True, dry_run=False), _factory, printer)) as c:
        aid = _open_alert(c, video_file)
        assert printer.calls == 0, "no pause may be sent without an operator approval"
        assert c.post(f"/api/alerts/{aid}/approve-pause", json={"operator": ""}).status_code == 422
        r = c.post(f"/api/alerts/{aid}/approve-pause", json={"operator": "bilel"})
        assert r.status_code == 200 and r.json()["sent"] and printer.calls == 1
        assert c.post(f"/api/alerts/{aid}/approve-pause", json={"operator": "bilel"}).status_code == 409
        assert c.get("/api/pause-requests").json()[0]["operator"] == "bilel"


def test_dismiss_and_token(tmp_path: Path, video_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with TestClient(create_app(_settings(tmp_path), _factory)) as c:
        aid = _open_alert(c, video_file)
        monkeypatch.setenv("PRINTGUARD_TOKEN", "s3cret")
        assert c.post(f"/api/alerts/{aid}/dismiss", json={"operator": "bilel"}).status_code == 401
        r = c.post(f"/api/alerts/{aid}/dismiss", json={"operator": "bilel"}, headers={"Authorization": "Bearer s3cret"})
        assert r.status_code == 200 and r.json()["status"] == "dismissed"
