"""Event state models and helper functions."""

import time
import uuid
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ExtractedFrame:
    """Single frame from multi-clip extraction: image, timestamp, camera, and optional metadata (e.g. is_full_frame_resize)."""
    frame: Any  # BGR numpy array (cv2/opencv)
    timestamp_sec: float
    camera: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameMetadata:
    """Per-frame tracked object data from Frigate MQTT (tracked_object_update).
    Box is always [ymin, xmin, ymax, xmax] normalized 0-1 (see state manager normalization).
    """
    frame_time: float
    box: tuple[float, float, float, float]  # ymin, xmin, ymax, xmax normalized 0-1
    area: float
    score: float

# Patterns that indicate "no concerns" from GenAI review summary (skip summarized notification)
NO_CONCERNS_PATTERNS = (
    "no concerns were found during this time period",
    "no concerns were found",
    "no concerns",
)


def _is_no_concerns(summary: str) -> bool:
    """Return True if the summary indicates no concerns (skip summarized notification)."""
    normalized = (summary or "").strip().lower()
    return any(p in normalized for p in NO_CONCERNS_PATTERNS)


class EventPhase(Enum):
    """Tracks the lifecycle phase of a Frigate event."""
    NEW = auto()        # Phase 1: Initial detection from frigate/events type=new
    DESCRIBED = auto()  # Phase 2: AI description received from tracked_object_update
    FINALIZED = auto()  # Phase 3: GenAI metadata received from frigate/reviews
    SUMMARIZED = auto() # Phase 4: Review summary received from Frigate API


@runtime_checkable
class NotificationEvent(Protocol):
    """Protocol for event notification data."""
    event_id: str
    camera: str
    label: str
    phase: EventPhase
    created_at: float
    threat_level: int
    clip_downloaded: bool
    snapshot_downloaded: bool

    # Optional fields
    ai_description: str | None
    genai_title: str | None
    genai_description: str | None
    review_summary: str | None
    folder_path: str | None
    image_url_override: str | None


@dataclass(slots=True)
class EventState:
    """Represents the current state of a tracked Frigate event."""
    event_id: str
    camera: str
    label: str
    phase: EventPhase = EventPhase.NEW
    created_at: float = field(default_factory=time.time)

    # Phase 2 data (from tracked_object_update)
    ai_description: str | None = None

    # Phase 3 data (from frigate/reviews)
    genai_title: str | None = None
    genai_description: str | None = None
    genai_scene: str | None = None  # Longer narrative from Frigate metadata.scene
    severity: str | None = None
    threat_level: int = 0  # 0=normal, 1=suspicious, 2=critical

    # Phase 4 data (from review summary API)
    review_summary: str | None = None

    # File management
    folder_path: str | None = None
    clip_downloaded: bool = False
    snapshot_downloaded: bool = False
    summary_written: bool = False
    review_summary_written: bool = False

    # Event end tracking
    end_time: float | None = None
    has_clip: bool = False
    has_snapshot: bool = False

    # Notification override
    image_url_override: str | None = None


def _generate_consolidated_id(start_ts: float) -> tuple:
    """Generate our internal consolidated event ID. Returns (full_id, folder_name)."""
    ts_int = int(start_ts)
    short_uuid = uuid.uuid4().hex[:8]
    full_id = f"ce_{ts_int}_{short_uuid}"
    folder_name = f"{ts_int}_{short_uuid}"
    return full_id, folder_name


@dataclass(slots=True)
class ConsolidatedEvent:
    """A consolidated event grouping multiple Frigate events (same real-world activity)."""

    consolidated_id: str
    folder_name: str
    folder_path: str
    start_time: float
    last_activity_time: float
    end_time_max: float = 0  # Max end_time across sub-events (updated as events end)
    cameras: list[str] = field(default_factory=list)
    frigate_event_ids: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    # Best-so-far (never regress)
    best_title: str | None = None
    best_description: str | None = None
    best_threat_level: int = 0

    # Primary (first) Frigate event for immediate clip/snapshot
    primary_event_id: str | None = None
    primary_camera: str | None = None
    snapshot_downloaded: bool = False
    clip_downloaded: bool = False

    # Notification tracking (avoid duplicate sends per group)
    clip_ready_sent: bool = False
    finalized_sent: bool = False

    # CE close: when True, no more sub-events will be added
    closing: bool = False
    closed: bool = False

    # Legacy-compat: expose as EventState-like for notifier
    @property
    def event_id(self) -> str:
        return self.consolidated_id

    @property
    def camera(self) -> str:
        return self.primary_camera or (self.cameras[0] if self.cameras else "unknown")

    @property
    def label(self) -> str:
        """Return comma-separated list of unique labels."""
        if not self.labels:
            return "person"  # Fallback
        unique = sorted(list(set(self.labels)))
        return ", ".join(unique)

    @property
    def created_at(self) -> float:
        return self.start_time

    @property
    def phase(self):
        return EventPhase.SUMMARIZED  # Simplified for notification

    @property
    def genai_title(self) -> str | None:
        return self.best_title

    @property
    def genai_description(self) -> str | None:
        return self.best_description

    @property
    def threat_level(self) -> int:
        return self.best_threat_level

    @property
    def severity(self) -> str | None:
        return "detection"

    @property
    def review_summary(self) -> str | None:
        return None  # Set when we get full-event summary

    @property
    def end_time(self) -> float | None:
        return self.last_activity_time

    @property
    def has_clip(self) -> bool:
        return self.clip_downloaded

    @property
    def has_snapshot(self) -> bool:
        return self.snapshot_downloaded

    @property
    def image_url_override(self) -> str | None:
        return None

    @property
    def ai_description(self) -> str | None:
        return None
