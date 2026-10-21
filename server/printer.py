"""Operator-approved pause requests for OctoPrint and Moonraker (Klipper).

Safety design
-------------
* There is **no code path that pauses a printer automatically**. :class:`PauseGuard` is only
  called from the approve endpoint, which requires an operator name and an alert id.
* Pausing is disabled by default (``printer.pause_enabled: false``) and runs in ``dry_run``
  mode by default, which records the request without contacting the printer.
* An alert can be approved once; dismissed or already-handled alerts are rejected.
* Pause requests are rate-limited (``min_seconds_between_pauses``) and every attempt -
  successful, failed or dry-run - is written to the audit log.
* API keys are read from an environment variable, never from the config file or the repo.
* Only *pause* is implemented. Resume/cancel stay manual at the printer.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

import requests

from server.db import Timeline

LOGGER = logging.getLogger("printguard.printer")


class PrinterClient(Protocol):
    """Minimal printer interface: send a pause command, return (ok, detail)."""

    name: str

    def pause(self) -> tuple[bool, str]: ...


class OctoPrintClient:
    """OctoPrint REST: ``POST /api/job`` with ``{"command": "pause", "action": "pause"}``."""

    name = "octoprint"

    def __init__(self, url: str, api_key: str, timeout: float = 5.0, session: requests.Session | None = None) -> None:
        self.url, self.api_key, self.timeout = url.rstrip("/"), api_key, timeout
        self.http = session or requests.Session()

    def pause(self) -> tuple[bool, str]:
        resp = self.http.post(
            f"{self.url}/api/job",
            json={"command": "pause", "action": "pause"},
            headers={"X-Api-Key": self.api_key},
            timeout=self.timeout,
        )
        # 204 No Content on success, 409 if no job is printing
        return resp.status_code == 204, f"HTTP {resp.status_code} {resp.text[:200]}"


class MoonrakerClient:
    """Moonraker (Klipper) REST: ``POST /printer/print/pause``."""

    name = "moonraker"

    def __init__(
        self, url: str, api_key: str | None = None, timeout: float = 5.0, session: requests.Session | None = None
    ) -> None:
        self.url, self.api_key, self.timeout = url.rstrip("/"), api_key, timeout
        self.http = session or requests.Session()

    def pause(self) -> tuple[bool, str]:
        headers = {"X-Api-Key": self.api_key} if self.api_key else {}
        resp = self.http.post(f"{self.url}/printer/print/pause", headers=headers, timeout=self.timeout)
        return resp.status_code == 200, f"HTTP {resp.status_code} {resp.text[:200]}"


def make_client(cfg: dict[str, Any]) -> PrinterClient:
    """Instantiate the configured backend; the API key comes from ``cfg['api_key_env']``."""
    key = os.environ.get(cfg.get("api_key_env", ""), "")
    timeout = float(cfg.get("request_timeout_s", 5))
    backend = cfg.get("backend", "octoprint")
    if backend == "octoprint":
        return OctoPrintClient(cfg["url"], key, timeout)
    if backend == "moonraker":
        return MoonrakerClient(cfg["url"], key or None, timeout)
    raise ValueError(f"Unknown printer backend {backend!r}")


class PauseRefused(Exception):
    """Raised when a pause request is rejected by the safety rules."""


@dataclass
class PauseResult:
    """Outcome of an approved pause request."""

    alert_id: int
    sent: bool
    ok: bool
    dry_run: bool
    detail: str


class PauseGuard:
    """Gatekeeper between an operator approval and the printer."""

    def __init__(self, cfg: dict[str, Any], timeline: Timeline, client: PrinterClient | None = None) -> None:
        self.enabled = bool(cfg.get("pause_enabled", False))
        self.dry_run = bool(cfg.get("dry_run", True))
        self.min_gap = float(cfg.get("min_seconds_between_pauses", 60))
        self.backend = cfg.get("backend", "octoprint")
        self.timeline = timeline
        self._client = client
        self._cfg = cfg
        self._last_sent = 0.0

    @property
    def client(self) -> PrinterClient:
        """Lazily-built printer client."""
        if self._client is None:
            self._client = make_client(self._cfg)
        return self._client

    def status(self) -> dict[str, Any]:
        """Public status for the dashboard (no secrets)."""
        return {
            "pause_enabled": self.enabled,
            "dry_run": self.dry_run,
            "backend": self.backend,
            "min_seconds_between_pauses": self.min_gap,
        }

    def approve(self, alert_id: int, operator: str) -> PauseResult:
        """Operator approved pausing for ``alert_id``. Applies every safety rule, then (maybe) sends.

        Raises:
            PauseRefused: pausing disabled, unknown/handled alert, missing operator or rate limit.
        """
        if not self.enabled:
            raise PauseRefused("Pause requests are disabled (printer.pause_enabled=false)")
        if not operator or not operator.strip():
            raise PauseRefused("An operator name is required to approve a pause")
        alert = self.timeline.alert(alert_id)
        if alert is None:
            raise PauseRefused(f"Unknown alert {alert_id}")
        if alert["status"] != "open":
            raise PauseRefused(f"Alert {alert_id} is already {alert['status']}")
        now = time.time()
        if not self.dry_run and now - self._last_sent < self.min_gap:
            raise PauseRefused(f"Rate limit: last pause {now - self._last_sent:.0f}s ago (min {self.min_gap:.0f}s)")
        if self.dry_run:
            ok, detail, sent = True, "dry-run: pause NOT sent to printer", False
        else:
            try:
                ok, detail = self.client.pause()
            except requests.RequestException as exc:
                ok, detail = False, f"request failed: {exc}"
            sent = True
            self._last_sent = now
        self.timeline.add_pause_request(alert_id, operator, self.backend, self.dry_run, ok, detail)
        self.timeline.set_alert_status(alert_id, "pause_requested" if ok else "pause_failed", operator, detail)
        LOGGER.warning("pause for alert %s approved by %s: ok=%s %s", alert_id, operator, ok, detail)
        return PauseResult(alert_id, sent, ok, self.dry_run, detail)
