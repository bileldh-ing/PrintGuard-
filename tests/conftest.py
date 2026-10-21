"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def video_file(tmp_path: Path) -> Path:
    """A tiny synthetic MP4 (40 frames, 160x120) with a moving square."""
    import cv2

    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (160, 120))
    for i in range(40):
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(img, (10 + i, 40), (40 + i, 70), (0, 0, 255), -1)
        writer.write(img)
    writer.release()
    return path
