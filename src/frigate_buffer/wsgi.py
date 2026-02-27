"""
WSGI entry point for Gunicorn.

Bootstraps config and StateAwareOrchestrator, starts background services
(MQTT, notifier, scheduler), and exposes the Flask app. Must be run with
exactly one Gunicorn worker (enforced via FRIGATE_BUFFER_SINGLE_WORKER set by
run_server.py). Registers graceful shutdown on SIGTERM/SIGINT so orchestrator.stop()
is called when the container stops.
"""

import logging
import os
import signal

from frigate_buffer.main import bootstrap

logger = logging.getLogger("frigate-buffer")

# Module-level orchestrator reference for signal handler (set in create_application).
_orchestrator = None


def _shutdown_handler(signum: int, frame) -> None:
    """Call orchestrator.stop() on SIGTERM/SIGINT so MQTT and threads shut down cleanly."""
    global _orchestrator
    logger.info("Received signal %s, shutting down orchestrator...", signum)
    if _orchestrator:
        _orchestrator.stop()
    raise SystemExit(0)


def create_application():
    """Create the WSGI application: bootstrap, start_services, return Flask app."""
    global _orchestrator

    # Mandatory single-worker guardrail: run_server.py sets this before execvp.
    if os.environ.get("FRIGATE_BUFFER_SINGLE_WORKER") != "1":
        raise RuntimeError(
            "Gunicorn must be started via run_server.py with exactly one worker (-w 1). "
            "Multiple workers would break GPU_LOCK and StateAwareOrchestrator. "
            "Set FRIGATE_BUFFER_SINGLE_WORKER=1 if invoking gunicorn manually with -w 1."
        )

    config, orchestrator = bootstrap()
    _orchestrator = orchestrator

    orchestrator.start_services()

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    return orchestrator.flask_app


application = create_application()
