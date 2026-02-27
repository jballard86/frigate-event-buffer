#!/usr/bin/env python3
"""
Frigate Event Buffer - Gunicorn launcher.

Reads FLASK_HOST and FLASK_PORT from config (via config.load_config()) and
execs Gunicorn with a single worker and multiple threads. Uses os.execvp so
Gunicorn replaces this process (PID 1 in Docker), ensuring signals pass through.

Usage: python run_server.py

Do not hardcode bind address in Dockerfile; this script provides dynamic binding.
"""

import os
import sys

# Ensure package is importable when run from repo root (e.g. during development).
if __name__ == "__main__":
    _src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)

from frigate_buffer.config import load_config


def main() -> None:
    config = load_config()
    host = config.get("FLASK_HOST", "0.0.0.0")
    port = config.get("FLASK_PORT", 5055)

    # Single-worker guardrail: wsgi.py requires this env to be set.
    os.environ["FRIGATE_BUFFER_SINGLE_WORKER"] = "1"

    # Gunicorn replaces this process so Docker SIGTERM goes to Gunicorn.
    argv = [
        sys.executable,
        "-m",
        "gunicorn",
        "--bind",
        f"{host}:{port}",
        "-w",
        "1",
        "--threads",
        "4",
        "--capture-output",
        "--enable-stdio-inheritance",
        "frigate_buffer.wsgi:application",
    ]
    os.execvp(sys.executable, argv)


if __name__ == "__main__":
    main()
