"""Error buffer and logging setup for the stats dashboard."""

import logging
import time
import threading
from typing import List

logger = logging.getLogger('frigate-buffer')


class ErrorBuffer:
    """Thread-safe rotating buffer of recent ERROR/WARNING log records (max 10)."""

    def __init__(self, max_size: int = 10):
        self._entries: List[dict] = []
        self._max_size = max_size
        self._lock = threading.Lock()

    def append(self, timestamp: str, level: str, message: str) -> None:
        with self._lock:
            self._entries.append({
                "ts": timestamp,
                "level": level,
                "message": message[:500] if message else ""
            })
            if len(self._entries) > self._max_size:
                self._entries.pop(0)

    def get_all(self) -> List[dict]:
        with self._lock:
            return list(reversed(self._entries))


class ErrorBufferHandler(logging.Handler):
    """Logging handler that writes ERROR/WARNING to ErrorBuffer."""

    def __init__(self, buffer: ErrorBuffer):
        super().__init__(level=logging.WARNING)
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.created))
            self._buffer.append(ts, record.levelname, record.getMessage())
        except Exception:
            self.handleError(record)


error_buffer = ErrorBuffer(max_size=10)


def setup_logging(log_level: str):
    """Configure logging with the specified level."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Reconfigure the root logger
    logging.getLogger().setLevel(level)
    logger.setLevel(level)

    # Add error buffer handler for stats dashboard (avoid duplicate if called twice)
    if not any(isinstance(h, ErrorBufferHandler) for h in logger.handlers):
        logger.addHandler(ErrorBufferHandler(error_buffer))

    # Suppress werkzeug per-request logging (floods logs with GET /events)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

    logger.info(f"Log level set to {log_level.upper()}")
