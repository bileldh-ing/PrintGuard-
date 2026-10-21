from pathlib import Path

import numpy as np
import pytest

from server.ingest import SourceSpec, frames, resize_max_width


def test_spec_validation() -> None:
    SourceSpec("usb", "0").validate()
    SourceSpec("rtsp", "rtsp://cam/stream").validate()
    bad = [
        SourceSpec("usb", "cam"),
        SourceSpec("rtsp", "cam"),
        SourceSpec("ftp", "x"),
        SourceSpec("file", "x", target_fps=0),
    ]
    for spec in bad:
        with pytest.raises(ValueError):
            spec.validate()


def test_file_frames_are_resized(video_file: Path) -> None:
    got = list(frames(SourceSpec("file", str(video_file), loop=False, target_fps=200), lambda: False, max_width=80))
    assert 0 < len(got) <= 40
    assert got[0].shape[1] == 80


def test_resize_max_width_keeps_small_frames() -> None:
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    assert resize_max_width(img, 100) is img
