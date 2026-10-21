"""Background inference worker and WebSocket broadcaster.

The worker owns the video source and the :class:`~printguard.inference.pipeline.OnlinePipeline`
and runs in a daemon thread, so slow CPU inference never blocks the FastAPI event loop. Results
are pushed to WebSocket clients through per-client asyncio queues that drop stale frames.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import cv2

from printguard.inference.pipeline import FrameEvent, OnlinePipeline
from printguard.taxonomy import NO_DEFECT
from server.db import Timeline
from server.ingest import SourceSpec, frames

LOGGER = logging.getLogger("printguard.worker")


class Broadcaster:
    """Fan-out of JSON messages from the worker thread to WebSocket clients."""

    def __init__(self) -> None:
        self._clients: set[asyncio.Queue[dict[str, Any]]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        """Remember the server event loop (called at startup)."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Register a client queue."""
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=4)
        with self._lock:
            self._clients.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove a client queue."""
        with self._lock:
            self._clients.discard(q)

    @property
    def n_clients(self) -> int:
        """Connected WebSocket clients."""
        return len(self._clients)

    def publish(self, msg: dict[str, Any]) -> None:
        """Thread-safe publish; a full client queue drops its oldest message."""
        if self._loop is None:
            return
        with self._lock:
            clients = list(self._clients)
        for q in clients:
            self._loop.call_soon_threadsafe(_put_latest, q, msg)


def _put_latest(q: asyncio.Queue[dict[str, Any]], msg: dict[str, Any]) -> None:
    if q.full():
        with contextlib.suppress(asyncio.QueueEmpty):
            q.get_nowait()
    q.put_nowait(msg)


class InferenceWorker:
    """Runs one video source through the pipeline and records results."""

    def __init__(
        self,
        pipeline_factory: Callable[[], OnlinePipeline],
        timeline: Timeline,
        broadcaster: Broadcaster,
        settings: dict[str, Any],
    ) -> None:
        self._factory = pipeline_factory
        self._pipeline: OnlinePipeline | None = None
        self.timeline, self.broadcaster = timeline, broadcaster
        srv = settings.get("server", {})
        self.max_width = int(srv.get("max_width", 960))
        self.stream_period = 1.0 / float(srv.get("stream_fps", 8))
        self.jpeg_quality = int(srv.get("jpeg_quality", 70))
        self.log_min_conf = float(srv.get("event_min_confidence", 0.3))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.spec: SourceSpec | None = None
        self.session_id: int | None = None
        self.frames_processed = 0
        self.last_error: str | None = None
        self.latest_jpeg: bytes | None = None
        self._times: deque[float] = deque(maxlen=30)
        self._latency: deque[float] = deque(maxlen=30)

    @property
    def pipeline(self) -> OnlinePipeline:
        """Pipeline, built on first use (model loading is slow)."""
        if self._pipeline is None:
            self._pipeline = self._factory()
        return self._pipeline

    @property
    def running(self) -> bool:
        """Whether a source is being processed."""
        return self._thread is not None and self._thread.is_alive()

    def start(self, spec: SourceSpec) -> int:
        """Stop any current source and start ``spec``; returns the new session id."""
        spec.validate()
        self.stop()
        self.pipeline.reset()
        self.spec, self.frames_processed, self.last_error = spec, 0, None
        self.session_id = self.timeline.start_session(spec.kind, spec.uri)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="printguard-worker", daemon=True)
        self._thread.start()
        return self.session_id

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the current source (no-op when idle)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self._thread = None
        if self.session_id is not None:
            self.timeline.end_session(self.session_id)

    def status(self) -> dict[str, Any]:
        """Snapshot for ``/api/status``."""
        fps = 0.0
        if len(self._times) > 1:
            fps = (len(self._times) - 1) / max(1e-6, self._times[-1] - self._times[0])
        p = self._pipeline
        return {
            "running": self.running,
            "source": self.spec.__dict__ if self.spec else None,
            "session_id": self.session_id,
            "frames_processed": self.frames_processed,
            "fps": round(fps, 2),
            "latency_ms": round(sum(self._latency) / len(self._latency), 1) if self._latency else None,
            "models": {
                "detector": bool(p and p.analyzer and p.analyzer.detector),
                "classifier": bool(p and p.analyzer and p.analyzer.classifier),
                "temporal": bool(p and p.temporal is not None),
            },
            "clients": self.broadcaster.n_clients,
            "last_error": self.last_error,
        }

    # ------------------------------------------------------------------
    def _run(self) -> None:
        assert self.spec is not None
        last_push = 0.0
        try:
            for frame in frames(self.spec, self._stop.is_set, self.max_width):
                ev = self.pipeline(frame)
                self.frames_processed += 1
                self._times.append(time.monotonic())
                self._latency.append(ev.latency_ms)
                alerts = self._record(ev)
                now = time.monotonic()
                if alerts or now - last_push >= self.stream_period:
                    last_push = now
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                    if ok:
                        self.latest_jpeg = buf.tobytes()
                        self.broadcaster.publish(
                            {
                                "type": "frame",
                                "jpeg": base64.b64encode(self.latest_jpeg).decode(),
                                "event": ev.to_dict(),
                                "alerts": alerts,
                                "status": self.status(),
                            }
                        )
        except Exception as exc:  # surface to the dashboard instead of dying silently
            LOGGER.exception("worker crashed")
            self.last_error = str(exc)
            self.broadcaster.publish({"type": "error", "message": str(exc)})
        finally:
            self.broadcaster.publish({"type": "status", "status": self.status()})

    def _record(self, ev: FrameEvent) -> list[dict[str, Any]]:
        sid = self.session_id
        for d in ev.detections:
            if d.confidence >= self.log_min_conf:
                self.timeline.add_event(
                    sid,
                    ev.timestamp,
                    ev.frame_idx,
                    d.label,
                    d.confidence,
                    "detector",
                    list(d.xyxy),
                    ev.trends.get(d.label),
                )
        for label, p in ev.class_probs.items():
            if label != NO_DEFECT and p >= 0.5:
                self.timeline.add_event(
                    sid, ev.timestamp, ev.frame_idx, label, p, "classifier", None, ev.trends.get(label)
                )
        for label, p in ev.temporal_scores.items():
            if p >= 0.5:
                self.timeline.add_event(
                    sid, ev.timestamp, ev.frame_idx, label, p, "temporal", None, ev.trends.get(label)
                )
        out = []
        scores = self.pipeline.frame_scores(ev)
        for label in ev.alerts:
            aid = self.timeline.add_alert(sid, ev.timestamp, ev.frame_idx, label, scores.get(label, 0.0))
            out.append(
                {
                    "id": aid,
                    "label": label,
                    "score": scores.get(label, 0.0),
                    "frame_idx": ev.frame_idx,
                    "ts": ev.timestamp,
                    "status": "open",
                }
            )
        return out
