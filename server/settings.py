"""Server settings loaded from ``configs/server.yaml`` (path overridable via ``PRINTGUARD_CONFIG``)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from printguard.config import load_config

DEFAULT_CONFIG = "configs/server.yaml"


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    """Load server settings; environment variable ``PRINTGUARD_CONFIG`` wins over the default."""
    return load_config(path or os.environ.get("PRINTGUARD_CONFIG", DEFAULT_CONFIG))
