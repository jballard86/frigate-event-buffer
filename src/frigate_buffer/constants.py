"""
Shared constants for path filtering, HTTP streaming, and display.

Centralizes NON_CAMERA_DIRS and HTTP chunk size so callers do not duplicate
definitions or magic numbers.
"""

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
