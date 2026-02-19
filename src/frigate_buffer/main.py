#!/usr/bin/env python3
"""
Frigate Event Buffer - Entry Point

Listens to Frigate MQTT topics, tracks events through their lifecycle,
sends Ring-style notifications to Home Assistant, and manages rolling retention.

Run with: python -m frigate_buffer.main
"""

import logging
import signal
import sys
from pathlib import Path

from frigate_buffer.config import load_config
from frigate_buffer.logging_utils import setup_logging
from frigate_buffer.orchestrator import StateAwareOrchestrator
from frigate_buffer.services.video import (
    ensure_detection_model_ready,
    log_gpu_status,
)

# Early logging for config loading (reconfigured after config is loaded)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger('frigate-buffer')

orchestrator = None


def _load_version() -> str:
    """Load version from version.txt.

    Looks in two places so it works in both dev and Docker:
    1. Next to this module (package dir) — used when installed via pip; version.txt
       is included as package data and copied into the image at build time.
    2. Project root (three levels up from this file) — used when running from
       source (e.g. pip install -e . or pytest from repo root).
    """
    try:
        pkg_dir = Path(__file__).resolve().parent
        for candidate in (pkg_dir / "version.txt", pkg_dir.parent.parent / "version.txt"):
            if candidate.exists():
                return candidate.read_text().strip()
    except OSError:
        pass
    return "unknown"


def signal_handler(sig, frame):
    """Handle shutdown signals."""
    global orchestrator
    logger.info(f"Received signal {sig}, shutting down...")
    if orchestrator:
        orchestrator.stop()
    sys.exit(0)


def main():
    """Main entry point."""
    # Load configuration
    config = load_config()

    # Setup logging with configured level
    setup_logging(config.get('LOG_LEVEL', 'INFO'))

    VERSION = _load_version()
    logger.info("VERSION = %s", VERSION)

    # GPU diagnostics (NVDEC used for decode)
    log_gpu_status()

    # Ensure multi-cam detection model is available (download if not cached)
    ensure_detection_model_ready(config)

    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Create and start orchestrator
    global orchestrator
    orchestrator = StateAwareOrchestrator(config)
    orchestrator.start()


if __name__ == '__main__':
    main()
