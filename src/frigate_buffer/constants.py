"""
Shared constants for path filtering, HTTP streaming, and display.

Centralizes NON_CAMERA_DIRS and HTTP chunk size so callers do not duplicate
definitions or magic numbers. Also provides a small tensor type-check helper
(is_tensor) used by file manager, video service, and crop_utils.
"""

from typing import Any

# Directories under storage root that are not cameras; exclude from event/camera listing,
# cleanup, and storage stats (e.g. ultralytics, yolo_models, daily_reports, daily_reviews).
NON_CAMERA_DIRS: frozenset[str] = frozenset({
    "ultralytics",
    "yolo_models",
    "daily_reports",
    "daily_reviews",
})

# Chunk size for HTTP streaming (e.g. proxy snapshot/latest.jpg, download iter_content).
# Used so streaming responses do not load entire body into memory.
HTTP_STREAM_CHUNK_SIZE: int = 8192

# Larger chunk size for file downloads (snapshot, clip) where throughput matters more than latency.
HTTP_DOWNLOAD_CHUNK_SIZE: int = 65536

# Frigate proxy request timeouts (seconds). Used by web/frigate_proxy for snapshot and latest.jpg.
FRIGATE_PROXY_SNAPSHOT_TIMEOUT: int = 15
FRIGATE_PROXY_LATEST_TIMEOUT: int = 10

# AI/display: max chars to log for proxy error response body; max frame width for API (preserve aspect).
LOG_MAX_RESPONSE_BODY: int = 2000
FRAME_MAX_WIDTH: int = 1280

# Storage stats cache TTL (seconds). Default when STORAGE_STATS_MAX_AGE_SECONDS not in config.
DEFAULT_STORAGE_STATS_MAX_AGE_SECONDS: int = 30 * 60

# Error buffer for stats dashboard: max number of recent ERROR/WARNING log entries.
ERROR_BUFFER_MAX_SIZE: int = 10


def is_tensor(x: Any) -> bool:
    """True if x is a torch.Tensor; avoids importing torch at module level for tests."""
    return type(x).__name__ == "Tensor"
