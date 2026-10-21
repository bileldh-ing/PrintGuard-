"""FastAPI application: REST API, WebSocket live feed and the dashboard.

Run with ``python -m server`` (or ``make serve``). Endpoints:

=========================================  ===================================================
``GET  /``                                 dashboard (static single page)
``GET  /api/health``                       liveness probe
``GET  /api/status``                       worker, models, fps, printer-safety status
``POST /api/source``                       start a file / usb / rtsp source
``POST /api/source/stop``                  stop the current source
``GET  /api/events``                       detection timeline (SQLite)
``GET  /api/alerts``                       operator alerts
``POST /api/alerts/{id}/dismiss``          operator dismisses an alert
``POST /api/alerts/{id}/approve-pause``    operator approves a pause request (safety-gated)
``GET  /api/pause-requests``               pause audit log
``GET  /api/frame.jpg``                    latest processed frame
``WS   /ws``                               live frames + events + alerts
=========================================  ===================================================

Mutating endpoints require ``Authorization: Bearer <PRINTGUARD_TOKEN>`` when that environment
variable is set (recommended whenever the server is reachable beyond localhost).
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from printguard.inference.pipeline import OnlinePipeline
from server.db import Timeline
from server.ingest import SourceSpec
from server.printer import PauseGuard, PauseRefused, PrinterClient
from server.settings import load_settings
from server.worker import Broadcaster, InferenceWorker

DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "dashboard"


class SourceRequest(BaseModel):
    """Body of ``POST /api/source``."""

    kind: str = Field(pattern="^(file|usb|rtsp)$")
    uri: str = Field(min_length=1, max_length=2048)
    loop: bool = True
    target_fps: float = Field(default=4.0, gt=0, le=30)


class OperatorDecision(BaseModel):
    """Body of the alert decision endpoints."""

    operator: str = Field(min_length=1, max_length=64)
    note: str | None = Field(default=None, max_length=500)


def require_token(authorization: str | None = Header(default=None)) -> None:
    """Bearer-token check for mutating endpoints (enabled only when PRINTGUARD_TOKEN is set)."""
    token = os.environ.get("PRINTGUARD_TOKEN")
    if token and authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")


def create_app(
    settings: dict[str, Any] | None = None,
    pipeline_factory: Callable[[], OnlinePipeline] | None = None,
    printer_client: PrinterClient | None = None,
) -> FastAPI:
    """Application factory (tests inject settings, a fake pipeline and a fake printer)."""
    cfg = settings or load_settings()
    srv = cfg.get("server", {})
    timeline = Timeline(srv.get("database", "data/printguard.sqlite3"))
    broadcaster = Broadcaster()
    factory = pipeline_factory or (lambda: OnlinePipeline.from_config(cfg.get("pipeline", {})))
    worker = InferenceWorker(factory, timeline, broadcaster, cfg)
    guard = PauseGuard(cfg.get("printer", {}), timeline, printer_client)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        broadcaster.bind(asyncio.get_running_loop())
        src = cfg.get("source", {})
        if src.get("autostart") and src.get("uri"):
            worker.start(SourceSpec(src["kind"], str(src["uri"]), src.get("loop", True), src.get("target_fps", 4)))
        try:
            yield
        finally:
            # Stop the worker first so it stops writing, then close the DB. Both are guarded so a
            # SIGTERM during teardown (or a read-only volume) never turns into a crash.
            try:
                worker.stop()
            except Exception:
                logging.getLogger(__name__).exception("worker stop failed during shutdown")
            timeline.close()

    app = FastAPI(title="PrintGuard AI", version="0.1.0", lifespan=lifespan)
    app.state.worker, app.state.timeline, app.state.guard = worker, timeline, guard
    if DASHBOARD_DIR.exists():
        app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(DASHBOARD_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return {"worker": worker.status(), "printer": guard.status()}

    @app.post("/api/source", dependencies=[Depends(require_token)])
    def start_source(req: SourceRequest) -> dict[str, Any]:
        spec = SourceSpec(req.kind, req.uri, req.loop, req.target_fps)
        if spec.kind == "file" and not Path(spec.uri).exists():
            raise HTTPException(404, f"File not found: {spec.uri}")
        try:
            sid = worker.start(spec)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"session_id": sid, "status": worker.status()}

    @app.post("/api/source/stop", dependencies=[Depends(require_token)])
    def stop_source() -> dict[str, Any]:
        worker.stop()
        return {"status": worker.status()}

    @app.get("/api/events")
    def events(
        since: float | None = None, limit: int = Query(200, ge=1, le=5000), session_id: int | None = None
    ) -> list[dict[str, Any]]:
        return timeline.events(since, limit, session_id)

    @app.get("/api/alerts")
    def alerts(status: str | None = None, limit: int = Query(100, ge=1, le=1000)) -> list[dict[str, Any]]:
        return timeline.alerts(status, limit)

    @app.get("/api/sessions")
    def sessions(limit: int = Query(20, ge=1, le=500)) -> list[dict[str, Any]]:
        return timeline.sessions(limit)

    @app.post("/api/alerts/{alert_id}/dismiss", dependencies=[Depends(require_token)])
    def dismiss(alert_id: int, body: OperatorDecision) -> dict[str, Any]:
        alert = timeline.alert(alert_id)
        if alert is None:
            raise HTTPException(404, "Unknown alert")
        if alert["status"] != "open":
            raise HTTPException(409, f"Alert already {alert['status']}")
        timeline.set_alert_status(alert_id, "dismissed", body.operator, body.note)
        return timeline.alert(alert_id) or {}

    @app.post("/api/alerts/{alert_id}/approve-pause", dependencies=[Depends(require_token)])
    def approve_pause(alert_id: int, body: OperatorDecision) -> dict[str, Any]:
        if not guard.enabled:
            raise HTTPException(403, "Pause requests are disabled in configs/server.yaml (printer.pause_enabled)")
        try:
            res = guard.approve(alert_id, body.operator)
        except PauseRefused as exc:
            raise HTTPException(409, str(exc)) from exc
        return res.__dict__ | {"alert": timeline.alert(alert_id)}

    @app.get("/api/pause-requests")
    def pause_requests(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
        return timeline.pause_requests(limit)

    @app.get("/api/frame.jpg")
    def frame() -> Response:
        if worker.latest_jpeg is None:
            raise HTTPException(404, "No frame yet")
        return Response(worker.latest_jpeg, media_type="image/jpeg")

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        q = broadcaster.subscribe()
        try:
            await socket.send_json({"type": "status", "status": worker.status(), "printer": guard.status()})
            while True:
                await socket.send_json(await q.get())
        except WebSocketDisconnect:
            pass
        finally:
            broadcaster.unsubscribe(q)

    return app
