"""
Shared constants for path filtering, HTTP streaming, and display.

Centralizes NON_CAMERA_DIRS and HTTP chunk size so callers do not duplicate
definitions or magic numbers. Also provides a small tensor type-check helper
(is_tensor) used by file manager, video service, and crop_utils.
"""

from typing import Any

# Directories under storage root that are not cameras; exclude from event/camera
# listing, cleanup, and storage stats (e.g. ultralytics, yolo_models, daily_reports,
# daily_reviews). "saved" holds user-kept events; excluded from cleanup and from
# normal camera/event listing.
NON_CAMERA_DIRS: frozenset[str] = frozenset(
    {
        "ultralytics",
        "yolo_models",
        "daily_reports",
        "daily_reviews",
        "saved",
    }
)

# Chunk size for HTTP streaming (e.g. proxy snapshot/latest.jpg, download
# iter_content). Used so streaming responses do not load entire body into memory.
HTTP_STREAM_CHUNK_SIZE: int = 8192

# Larger chunk size for file downloads (snapshot, clip) where throughput matters
# more than latency.
HTTP_DOWNLOAD_CHUNK_SIZE: int = 65536

# Frigate proxy request timeouts (seconds). Used by web/frigate_proxy for
# snapshot and latest.jpg.
FRIGATE_PROXY_SNAPSHOT_TIMEOUT: int = 15
FRIGATE_PROXY_LATEST_TIMEOUT: int = 10

# Gemini proxy request timeouts (seconds). Quick-title = single image; analysis =
# multi-frame or text-only.
GEMINI_PROXY_QUICK_TITLE_TIMEOUT: int = 30
GEMINI_PROXY_ANALYSIS_TIMEOUT: int = 60

# AI/display: max chars to log for proxy error response body; max frame width for
# API (preserve aspect).
LOG_MAX_RESPONSE_BODY: int = 2000
FRAME_MAX_WIDTH: int = 1280

# Storage stats cache TTL (seconds). Default when STORAGE_STATS_MAX_AGE_SECONDS
# not in config.
DEFAULT_STORAGE_STATS_MAX_AGE_SECONDS: int = 30 * 60

# Error buffer for stats dashboard: max number of recent ERROR/WARNING log entries.
ERROR_BUFFER_MAX_SIZE: int = 10

# NVDEC/decoder init failure: log this prefix so crash-loop logs are searchable
# before container restart.
NVDEC_INIT_FAILURE_PREFIX: str = "NVDEC hardware initialization failed"

# Video compilation dynamic zoom: min crop size as fraction of frame (max
# zoom-in); crop window never smaller than this.
ZOOM_MIN_FRAME_FRACTION: float = 0.4
# Video compilation dynamic zoom: padding around bounding box (e.g. 0.10 = bbox
# + 10%).
ZOOM_CONTENT_PADDING: float = 0.10

# Video compilation: default native resolution when sidecar is missing or empty.
COMPILATION_DEFAULT_NATIVE_WIDTH: int = 1920
COMPILATION_DEFAULT_NATIVE_HEIGHT: int = 1080

# When nearest sidecar entry has no detections (e.g. person left frame), search
# for nearest entry with detections within this many seconds; hold that crop.
HOLD_CROP_MAX_DISTANCE_SEC: float = 5.0

# Dynamic slice trimming: pre-roll and post-roll around first/last detection
# across all cameras (used by timeline_ema._trim_slices_to_action_window).
ACTION_PREROLL_SEC: float = 3.0
ACTION_POSTROLL_SEC: float = 3.0

# AI notification mode: mutually exclusive paths (settings.ai_mode / AI_MODE).
# "frigate" = Frigate GenAI only (tracked_object, frigate/reviews, review
# summarize at CE close). "external_api" = Buffer Gemini/Quick Title only (quick
# title, multi-clip CE analysis).
AI_MODE_FRIGATE: str = "frigate"
AI_MODE_EXTERNAL_API: str = "external_api"

# -----------------------------------------------------------------------------
# Time display (user-facing): 12-hour format with AM/PM
# All user-visible timestamps (UI, reports, summaries, logs, API responses)
# use 12-hour format for consistency. Internal/ISO formats (e.g. timeline entry
# "ts") remain ISO 8601 where parsing is required.
# -----------------------------------------------------------------------------
DISPLAY_DATETIME_FORMAT: str = "%Y-%m-%d %I:%M:%S %p"
DISPLAY_TIME_ONLY_FORMAT: str = "%I:%M %p"


def is_tensor(x: Any) -> bool:
    """True if x is a torch.Tensor; avoids importing torch at module level for
    tests."""
    return type(x).__name__ == "Tensor"
