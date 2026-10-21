"""``python -m server`` - start the PrintGuard API + dashboard with uvicorn."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from server.app import create_app
from server.settings import load_settings


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="PrintGuard AI server")
    ap.add_argument("--config", default=None, help="server YAML (default: configs/server.yaml or $PRINTGUARD_CONFIG)")
    ap.add_argument("--source", default=None, help="video file / camera index / rtsp URL to start immediately")
    ap.add_argument("--kind", default=None, choices=["file", "usb", "rtsp"])
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_settings(args.config)
    if args.source:
        kind = args.kind or ("usb" if args.source.isdigit() else "rtsp" if "://" in args.source else "file")
        cfg.setdefault("source", {}).update({"autostart": True, "uri": args.source, "kind": kind})
    srv = cfg.get("server", {})
    uvicorn.run(create_app(cfg), host=args.host or srv.get("host", "0.0.0.0"), port=args.port or srv.get("port", 8000))


if __name__ == "__main__":
    main()
