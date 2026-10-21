"""SQLite timeline: sessions, detection events, operator alerts and pause requests.

Uses the standard-library ``sqlite3`` with one connection guarded by a lock, which is plenty
for one printer at a few frames per second and keeps the deployment dependency-free.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    ended_at REAL,
    source_kind TEXT NOT NULL,
    source_uri TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER REFERENCES sessions(id),
    ts REAL NOT NULL,
    frame_idx INTEGER NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL,          -- detector | classifier | temporal
    bbox TEXT,                     -- JSON [x1, y1, x2, y2] in frame pixels, or NULL
    trend REAL
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER REFERENCES sessions(id),
    ts REAL NOT NULL,
    frame_idx INTEGER NOT NULL,
    label TEXT NOT NULL,
    score REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',   -- open | dismissed | pause_requested | pause_failed
    decided_by TEXT,
    decided_at REAL,
    note TEXT
);
CREATE TABLE IF NOT EXISTS pause_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id INTEGER NOT NULL REFERENCES alerts(id),
    ts REAL NOT NULL,
    operator TEXT NOT NULL,
    backend TEXT NOT NULL,
    dry_run INTEGER NOT NULL,
    ok INTEGER NOT NULL,
    detail TEXT
);
"""


class _DroppedWrite:
    """Stand-in cursor for a write skipped during shutdown."""

    lastrowid = -1


class Timeline:
    """Thread-safe accessor for the SQLite timeline."""

    def __init__(self, path: str | Path) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._closed = False
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    @property
    def closed(self) -> bool:
        """True once :meth:`close` ran; later writes are dropped instead of raising."""
        return self._closed

    def _exec(self, sql: str, args: tuple[Any, ...] = ()) -> sqlite3.Cursor | _DroppedWrite:
        """Run one write statement.

        During shutdown (SIGTERM) the worker thread may still flush an event while the lifespan
        handler closes the connection, or the database file may have become read-only (container
        teardown, full disk, revoked volume). Those writes are logged and dropped (``lastrowid``
        is ``-1``): losing the last frame's event is better than crashing the shutdown path.
        """
        with self._lock:
            if self._closed:
                log.debug("timeline closed; dropping write: %s", sql.split("(")[0])
                return _DroppedWrite()
            try:
                cur = self._conn.execute(sql, args)
                self._conn.commit()
                return cur
            except (sqlite3.OperationalError, sqlite3.ProgrammingError) as exc:
                msg = str(exc).lower()
                if any(k in msg for k in ("readonly", "read-only", "closed", "locked")):
                    log.warning("timeline write dropped (%s): %s", exc, sql.split("(")[0])
                    return _DroppedWrite()
                raise

    def _rows(self, sql: str, args: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._lock:
            if self._closed:
                return []
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    # sessions ---------------------------------------------------------------
    def start_session(self, kind: str, uri: str) -> int:
        """Open a session for a new video source; returns its id."""
        return int(
            self._exec(
                "INSERT INTO sessions(started_at, source_kind, source_uri) VALUES (?,?,?)", (time.time(), kind, uri)
            ).lastrowid
        )

    def end_session(self, session_id: int) -> None:
        """Mark a session as finished."""
        self._exec("UPDATE sessions SET ended_at=? WHERE id=?", (time.time(), session_id))

    def sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Most recent sessions first."""
        return self._rows("SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,))

    # events -----------------------------------------------------------------
    def add_event(
        self,
        session_id: int | None,
        ts: float,
        frame_idx: int,
        label: str,
        confidence: float,
        source: str,
        bbox: list[float] | None = None,
        trend: float | None = None,
    ) -> int:
        """Record one detection / classification / temporal score above the logging threshold."""
        return int(
            self._exec(
                "INSERT INTO events(session_id, ts, frame_idx, label, confidence, source, bbox, trend) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (session_id, ts, frame_idx, label, confidence, source, json.dumps(bbox) if bbox else None, trend),
            ).lastrowid
        )

    def events(
        self, since: float | None = None, limit: int = 200, session_id: int | None = None
    ) -> list[dict[str, Any]]:
        """Events newest first, optionally after ``since`` (unix seconds) and for one session."""
        sql, args = "SELECT * FROM events WHERE 1=1", []
        if since is not None:
            sql += " AND ts > ?"
            args.append(since)
        if session_id is not None:
            sql += " AND session_id = ?"
            args.append(session_id)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        rows = self._rows(sql, tuple(args))
        for r in rows:
            r["bbox"] = json.loads(r["bbox"]) if r["bbox"] else None
        return rows

    # alerts -----------------------------------------------------------------
    def add_alert(self, session_id: int | None, ts: float, frame_idx: int, label: str, score: float) -> int:
        """Open a new operator alert."""
        return int(
            self._exec(
                "INSERT INTO alerts(session_id, ts, frame_idx, label, score) VALUES (?,?,?,?,?)",
                (session_id, ts, frame_idx, label, score),
            ).lastrowid
        )

    def alert(self, alert_id: int) -> dict[str, Any] | None:
        """One alert by id."""
        rows = self._rows("SELECT * FROM alerts WHERE id=?", (alert_id,))
        return rows[0] if rows else None

    def alerts(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Alerts newest first, optionally filtered by status."""
        if status:
            return self._rows("SELECT * FROM alerts WHERE status=? ORDER BY id DESC LIMIT ?", (status, limit))
        return self._rows("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))

    def set_alert_status(self, alert_id: int, status: str, operator: str, note: str | None = None) -> None:
        """Record an operator decision on an alert."""
        self._exec(
            "UPDATE alerts SET status=?, decided_by=?, decided_at=?, note=? WHERE id=?",
            (status, operator, time.time(), note, alert_id),
        )

    # pause audit log --------------------------------------------------------
    def add_pause_request(
        self, alert_id: int, operator: str, backend: str, dry_run: bool, ok: bool, detail: str
    ) -> int:
        """Append to the pause-request audit log."""
        return int(
            self._exec(
                "INSERT INTO pause_requests(alert_id, ts, operator, backend, dry_run, ok, detail) "
                "VALUES (?,?,?,?,?,?,?)",
                (alert_id, time.time(), operator, backend, int(dry_run), int(ok), detail),
            ).lastrowid
        )

    def pause_requests(self, limit: int = 50) -> list[dict[str, Any]]:
        """Pause audit log, newest first."""
        return self._rows("SELECT * FROM pause_requests ORDER BY id DESC LIMIT ?", (limit,))

    def close(self) -> None:
        """Close the connection (idempotent; safe to call from a signal-triggered shutdown)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._conn.commit()
            except sqlite3.Error as exc:  # e.g. "attempt to write a readonly database" at teardown
                log.warning("timeline final commit failed: %s", exc)
            finally:
                self._conn.close()
