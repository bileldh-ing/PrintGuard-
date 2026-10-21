"""Defect taxonomy shared by every component.

Only classes backed by public labelled data are trained. ``spaghetti`` and ``warping`` are
part of the target taxonomy but neither dataset used here labels them; they stay reserved
until own recordings (or another dataset) provide examples.
"""

from __future__ import annotations

from typing import Final

NO_DEFECT: Final = "no_error"

#: Classes of the frame-level image classifier (Trier FFF dataset, image-level labels).
FRAME_CLS_CLASSES: Final[tuple[str, ...]] = ("extrusion_error", "layer_shift", "no_error")

#: Classes of the box detector (YOLO-Strings, real bounding boxes).
DETECT_CLASSES: Final[tuple[str, ...]] = ("stringing",)

#: Per-timestep outputs of the temporal model (multi-label).
TEMPORAL_CLASSES: Final[tuple[str, ...]] = ("stringing", "layer_shift")

#: Target classes without training data yet.
RESERVED_CLASSES: Final[tuple[str, ...]] = ("spaghetti", "warping")

#: Union shown in the dashboard / event stream.
ALL_DEFECTS: Final[tuple[str, ...]] = ("stringing", "extrusion_error", "layer_shift", *RESERVED_CLASSES)
