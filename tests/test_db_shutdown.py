"""Timeline must not crash when written to after close or when the DB turns read-only."""

from __future__ import annotations

import sqlite3

from server.db import Timeline


def test_write_after_close_is_dropped(tmp_path):
    tl = Timeline(tmp_path / "t.sqlite3")
    sid = tl.start_session("file", "demo.mp4")
    assert sid > 0
    tl.close()
    assert tl.closed
    # Worker thread flushing late during SIGTERM: must be a no-op, not an exception.
    assert tl.add_event(sid, 1.0, 0, "stringing", 0.9, "detector") == -1
    tl.end_session(sid)
    assert tl.events() == []
    tl.close()  # idempotent


def test_readonly_database_is_guarded(tmp_path):
    path = tmp_path / "ro.sqlite3"
    Timeline(path).close()
    tl = Timeline(path)
    # Re-open the underlying connection read-only to simulate a read-only volume at teardown.
    tl._conn.close()
    tl._conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    assert tl.add_alert(None, 1.0, 0, "stringing", 0.8) == -1
    tl.close()
