#!/usr/bin/env python3
"""
Frigate Event Buffer - Bootstrap and entry point.

Provides bootstrap() for the WSGI worker (frigate_buffer.wsgi). The server is
started via run_server.py (Gunicorn); do not run Flask's built-in server.

Run the app with: python run_server.py
"""

import logging
import os
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
# Use 12-hour time format for consistency with rest of app (see constants.DISPLAY_*).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("frigate-buffer")


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
        for candidate in (
            pkg_dir / "version.txt",
            pkg_dir.parent.parent / "version.txt",
        ):
            if candidate.exists():
                return candidate.read_text().strip()
    except OSError:
        pass
    return "unknown"


def bootstrap() -> tuple[dict, StateAwareOrchestrator]:
    """Load config, setup logging/GPU/YOLO, create and return (config, orchestrator).

    Used by the WSGI entry point (wsgi.py). Does not start the web server.
    When notifications.mobile_app is enabled, initializes Firebase Admin SDK;
    on missing or invalid credentials, logs a warning and disables the mobile provider.
    """
    config = load_config()
    setup_logging(config.get("LOG_LEVEL", "INFO"))

    storage_path = config.get("STORAGE_PATH", "/app/storage")
    yolo_config_dir = os.path.join(storage_path, "ultralytics")
    os.makedirs(yolo_config_dir, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", yolo_config_dir)
    os.makedirs(os.path.join(storage_path, "yolo_models"), exist_ok=True)

    version = _load_version()
    logger.info("VERSION = %s", version)

    log_gpu_status()
    ensure_detection_model_ready(config)

    # Optional Firebase init for FCM (mobile_app). Use GOOGLE_APPLICATION_CREDENTIALS
    # from config path if set and env not already set. Pass project ID from env
    # (GOOGLE_CLOUD_PROJECT) or config so Cloud Messaging has a project.
    if config.get("NOTIFICATIONS_MOBILE_APP_ENABLED"):
        creds_path = config.get("MOBILE_APP_GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        if creds_path and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        project_id = config.get("MOBILE_APP_FIREBASE_PROJECT_ID", "").strip()
        try:
            import firebase_admin
            from firebase_admin import credentials

            if creds_path and os.path.isfile(creds_path):
                cred = credentials.Certificate(creds_path)
                firebase_admin.initialize_app(cred)
            elif project_id:
                firebase_admin.initialize_app(options={"projectId": project_id})
            else:
                firebase_admin.initialize_app()
        except Exception as e:
            logger.warning(
                "Firebase credentials not found or invalid, "
                "disabling mobile_app provider: %s",
                e,
            )
            config["NOTIFICATIONS_MOBILE_APP_ENABLED"] = False

    orchestrator = StateAwareOrchestrator(config)
    return config, orchestrator


def main():
    """Entry point: direct user to run_server.py (Gunicorn is the only server)."""
    logger.error(
        "Frigate Event Buffer must be started with run_server.py (Gunicorn). "
        "Do not use python -m frigate_buffer.main to run the server."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
