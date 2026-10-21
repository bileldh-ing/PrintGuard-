import time

import pytest

from server.db import Timeline
from server.printer import MoonrakerClient, OctoPrintClient, PauseGuard, PauseRefused


class FakeClient:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def pause(self) -> tuple[bool, str]:
        self.calls += 1
        return True, "HTTP 204"


def _guard(**cfg: object) -> tuple[PauseGuard, Timeline, FakeClient]:
    tl = Timeline(":memory:")
    client = FakeClient()
    base = {"pause_enabled": True, "dry_run": False, "min_seconds_between_pauses": 60}
    return PauseGuard(base | cfg, tl, client), tl, client


def test_disabled_by_default() -> None:
    tl = Timeline(":memory:")
    guard = PauseGuard({}, tl, FakeClient())
    assert guard.enabled is False and guard.dry_run is True
    aid = tl.add_alert(None, time.time(), 1, "stringing", 0.9)
    with pytest.raises(PauseRefused, match="disabled"):
        guard.approve(aid, "bilel")


def test_requires_operator_and_open_alert() -> None:
    guard, tl, client = _guard()
    aid = tl.add_alert(None, time.time(), 1, "stringing", 0.9)
    with pytest.raises(PauseRefused, match="operator"):
        guard.approve(aid, " ")
    with pytest.raises(PauseRefused, match="Unknown"):
        guard.approve(999, "bilel")
    tl.set_alert_status(aid, "dismissed", "bilel")
    with pytest.raises(PauseRefused, match="dismissed"):
        guard.approve(aid, "bilel")
    assert client.calls == 0


def test_dry_run_never_contacts_printer() -> None:
    guard, tl, client = _guard(dry_run=True)
    aid = tl.add_alert(None, time.time(), 1, "layer_shift", 0.8)
    res = guard.approve(aid, "bilel")
    assert res.ok and not res.sent and client.calls == 0
    assert tl.alert(aid)["status"] == "pause_requested"
    assert tl.pause_requests()[0]["dry_run"] == 1


def test_sends_once_and_rate_limits() -> None:
    guard, tl, client = _guard()
    a1 = tl.add_alert(None, time.time(), 1, "stringing", 0.9)
    a2 = tl.add_alert(None, time.time(), 2, "stringing", 0.9)
    assert guard.approve(a1, "bilel").sent
    with pytest.raises(PauseRefused, match="already"):
        guard.approve(a1, "bilel")
    with pytest.raises(PauseRefused, match="Rate limit"):
        guard.approve(a2, "bilel")
    assert client.calls == 1


class _Session:
    def __init__(self, status: int) -> None:
        self.status = status
        self.last: dict = {}

    def post(self, url: str, **kw: object) -> object:
        self.last = {"url": url, **kw}
        status = self.status

        class R:
            status_code = status
            text = ""

        return R()


def test_octoprint_and_moonraker_requests() -> None:
    s = _Session(204)
    ok, _ = OctoPrintClient("http://octopi.local/", "KEY", session=s).pause()  # type: ignore[arg-type]
    assert ok and s.last["url"] == "http://octopi.local/api/job"
    assert s.last["json"] == {"command": "pause", "action": "pause"} and s.last["headers"]["X-Api-Key"] == "KEY"
    s = _Session(200)
    ok, _ = MoonrakerClient("http://klipper:7125", session=s).pause()  # type: ignore[arg-type]
    assert ok and s.last["url"] == "http://klipper:7125/printer/print/pause"
