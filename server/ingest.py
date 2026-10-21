"""Video ingestion with OpenCV: video files (optionally looped), USB cameras and RTSP streams."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

LOGGER = logging.getLogger("printguard.ingest")


@dataclass
class SourceSpec:
    """What to open: ``kind`` is ``file``, ``usb`` or ``rtsp``; ``uri`` a path, index or URL."""

    kind: str
    uri: str
    loop: bool = False
    target_fps: float = 4.0

    def validate(self) -> None:
        """Reject malformed specs early (before a worker thread starts)."""
        if self.kind not in {"file", "usb", "rtsp"}:
            raise ValueError(f"Unknown source kind {self.kind!r}")
        if self.kind == "usb" and not self.uri.isdigit():
            raise ValueError("USB source uri must be a camera index such as '0'")
        if self.kind == "rtsp" and not self.uri.startswith(("rtsp://", "rtsps://", "http://", "https://")):
            raise ValueError("RTSP source uri must start with rtsp:// (or http(s):// for MJPEG)")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")


def open_capture(spec: SourceSpec) -> cv2.VideoCapture:
    """Open an OpenCV capture for the spec."""
    if spec.kind == "usb":
        cap = cv2.VideoCapture(int(spec.uri))
    elif spec.kind == "rtsp":
        cap = cv2.VideoCapture(spec.uri, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always process the freshest frame
    else:
        cap = cv2.VideoCapture(spec.uri)
    if not cap.isOpened():
        raise OSError(f"Could not open {spec.kind} source {spec.uri!r}")
    return cap


def resize_max_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale so the width is at most ``max_width`` (aspect preserved)."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    s = max_width / w
    return cv2.resize(frame, (max_width, round(h * s)), interpolation=cv2.INTER_AREA)


def frames(spec: SourceSpec, stop: callable, max_width: int = 960, reconnect_s: float = 2.0) -> Iterator[np.ndarray]:
    """Yield frames at roughly ``spec.target_fps`` until ``stop()`` returns True.

    * files are paced to the target rate using their own FPS (frames in between are skipped)
      and rewound when ``loop`` is set;
    * live sources always return the latest frame; a dropped RTSP stream is reopened.
    """
    spec.validate()
    period = 1.0 / spec.target_fps
    while not stop():
        try:
            cap = open_capture(spec)
        except OSError as exc:
            if spec.kind == "file":
                raise
            LOGGER.warning("%s; retrying in %.0fs", exc, reconnect_s)
            time.sleep(reconnect_s)
            continue
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        skip = max(1, round(src_fps / spec.target_fps)) if spec.kind == "file" else 1
        idx, last = 0, 0.0
        while not stop():
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if spec.kind == "file" and (idx - 1) % skip:
                continue
            wait = period - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.monotonic()
            yield resize_max_width(frame, max_width)
        cap.release()
        if spec.kind == "file" and not spec.loop:
            return
        if spec.kind != "file":
            LOGGER.warning("stream ended; reconnecting")
            time.sleep(reconnect_s)
