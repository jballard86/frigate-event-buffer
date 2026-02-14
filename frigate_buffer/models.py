"""Event state models and helper functions."""

import time
import uuid
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List

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


@dataclass
class EventState:
    """Represents the current state of a tracked Frigate event."""
    event_id: str
    camera: str
    label: str
    phase: EventPhase = EventPhase.NEW
    created_at: float = field(default_factory=time.time)

    # Phase 2 data (from tracked_object_update)
    ai_description: Optional[str] = None

    # Phase 3 data (from frigate/reviews)
    genai_title: Optional[str] = None
    genai_description: Optional[str] = None
    genai_scene: Optional[str] = None  # Longer narrative from Frigate metadata.scene
    severity: Optional[str] = None
    threat_level: int = 0  # 0=normal, 1=suspicious, 2=critical

    # Phase 4 data (from review summary API)
    review_summary: Optional[str] = None

    # File management
    folder_path: Optional[str] = None
    clip_downloaded: bool = False
    snapshot_downloaded: bool = False
    summary_written: bool = False
    review_summary_written: bool = False

    # Event end tracking
    end_time: Optional[float] = None
    has_clip: bool = False
    has_snapshot: bool = False


def _generate_consolidated_id(start_ts: float) -> tuple:
    """Generate our internal consolidated event ID. Returns (full_id, folder_name)."""
    ts_int = int(start_ts)
    short_uuid = uuid.uuid4().hex[:8]
    full_id = f"ce_{ts_int}_{short_uuid}"
    folder_name = f"{ts_int}_{short_uuid}"
    return full_id, folder_name


@dataclass
class ConsolidatedEvent:
    """A consolidated event grouping multiple Frigate events (same real-world activity)."""

    consolidated_id: str
    folder_name: str
    folder_path: str
    start_time: float
    last_activity_time: float
    end_time_max: float = 0  # Max end_time across sub-events (updated as events end)
    cameras: List[str] = field(default_factory=list)
    frigate_event_ids: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)

    # Best-so-far (never regress)
    best_title: Optional[str] = None
    best_description: Optional[str] = None
    best_threat_level: int = 0

    # Primary (first) Frigate event for immediate clip/snapshot
    primary_event_id: Optional[str] = None
    primary_camera: Optional[str] = None
    snapshot_downloaded: bool = False
    clip_downloaded: bool = False

    # Notification tracking (avoid duplicate sends per group)
    clip_ready_sent: bool = False
    finalized_sent: bool = False

    # CE close: when True, no more sub-events will be added
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
    def genai_title(self) -> Optional[str]:
        return self.best_title

    @property
    def genai_description(self) -> Optional[str]:
        return self.best_description

    @property
    def threat_level(self) -> int:
        return self.best_threat_level

    @property
    def severity(self) -> Optional[str]:
        return "detection"

    @property
    def review_summary(self) -> Optional[str]:
        return None  # Set when we get full-event summary

    @property
    def end_time(self) -> Optional[float]:
        return self.last_activity_time

    @property
    def has_clip(self) -> bool:
        return self.clip_downloaded

    @property
    def has_snapshot(self) -> bool:
        return self.snapshot_downloaded
