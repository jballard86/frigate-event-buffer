"""Thread-safe manager for active event states."""

import logging
import threading
import time
from typing import Any

from frigate_buffer.models import EventState, EventPhase, FrameMetadata

logger = logging.getLogger('frigate-buffer')

# Max per-frame metadata entries per event to bound memory
MAX_FRAME_METADATA_PER_EVENT = 500


def _normalize_box(
    box: Any,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> tuple[float, float, float, float] | None:
    """
    Normalize box to [ymin, xmin, ymax, xmax] in 0-1 range.
    Frigate may send [x1, y1, x2, y2] in pixels or normalized.
    Returns None if box is invalid or cannot be normalized.
    """
    if not box or not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        a, b, c, d = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    except (TypeError, ValueError):
        return None
    # Assume [x1, y1, x2, y2] (Frigate convention): xmin=a, ymin=b, xmax=c, ymax=d
    # Convert to [ymin, xmin, ymax, xmax]
    xmin, ymin = min(a, c), min(b, d)
    xmax, ymax = max(a, c), max(b, d)
    if frame_width and frame_height and frame_width > 0 and frame_height > 0:
        # Pixel coordinates: normalize to 0-1
        ymin, xmin = ymin / frame_height, xmin / frame_width
        ymax, xmax = ymax / frame_height, xmax / frame_width
    else:
        # If any value > 1, assume pixels but we have no dimensions: use 1920x1080 as fallback
        if max(xmin, ymin, xmax, ymax) > 1.0:
            ymin, xmin = ymin / 1080.0, xmin / 1920.0
            ymax, xmax = ymax / 1080.0, xmax / 1920.0
    # Clamp to [0, 1]
    ymin = max(0.0, min(1.0, ymin))
    xmin = max(0.0, min(1.0, xmin))
    ymax = max(0.0, min(1.0, ymax))
    xmax = max(0.0, min(1.0, xmax))
    if ymax <= ymin or xmax <= xmin:
        return None
    return (ymin, xmin, ymax, xmax)


class EventStateManager:
    """Thread-safe manager for active event states across multiple cameras."""

    def __init__(self):
        self._events: dict[str, EventState] = {}
        self._frame_metadata: dict[str, list[FrameMetadata]] = {}
        self._lock = threading.RLock()

    def create_event(self, event_id: str, camera: str, label: str,
                     start_time: float) -> EventState:
        """Create a new event in NEW phase."""
        with self._lock:
            if event_id in self._events:
                logger.debug(f"Event {event_id} already exists, returning existing")
                return self._events[event_id]

            event = EventState(
                event_id=event_id,
                camera=camera,
                label=label,
                created_at=start_time or time.time()
            )
            self._events[event_id] = event
            logger.info(f"Created event state: {event_id} ({label} on {camera})")
            return event

    def get_event(self, event_id: str) -> EventState | None:
        """Get event by ID (thread-safe read)."""
        with self._lock:
            return self._events.get(event_id)

    def set_ai_description(self, event_id: str, description: str) -> bool:
        """Set AI description and advance to DESCRIBED phase."""
        with self._lock:
            event = self._events.get(event_id)
            if event and event.phase == EventPhase.NEW:
                event.ai_description = description
                event.phase = EventPhase.DESCRIBED
                logger.info(f"Event {event_id} advanced to DESCRIBED phase")
                logger.debug(f"AI description: {description[:100]}..." if len(description) > 100 else f"AI description: {description}")
                return True
            elif event:
                # Update description even if already past NEW phase
                event.ai_description = description
                logger.debug(f"Updated AI description for {event_id} (already past NEW phase)")
                return True
            logger.debug(f"Cannot set AI description: event {event_id} not found")
            return False

    def set_genai_metadata(self, event_id: str, title: str | None,
                           description: str | None, severity: str,
                           threat_level: int = 0,
                           scene: str | None = None) -> bool:
        """Set GenAI review metadata and advance to FINALIZED phase."""
        with self._lock:
            event = self._events.get(event_id)
            if event:
                event.genai_title = title
                event.genai_description = description
                if scene is not None:
                    event.genai_scene = scene
                event.severity = severity
                event.threat_level = threat_level
                event.phase = EventPhase.FINALIZED
                logger.info(f"Event {event_id} advanced to FINALIZED (threat_level={threat_level})")
                logger.debug(f"GenAI title: {title}, severity: {severity}")
                return True
            logger.debug(f"Cannot set GenAI metadata: event {event_id} not found")
            return False

    def set_review_summary(self, event_id: str, summary: str) -> bool:
        """Set review summary and advance to SUMMARIZED phase."""
        with self._lock:
            event = self._events.get(event_id)
            if event:
                event.review_summary = summary
                event.phase = EventPhase.SUMMARIZED
                logger.info(f"Event {event_id} advanced to SUMMARIZED phase")
                return True
            logger.debug(f"Cannot set review summary: event {event_id} not found")
            return False

    def mark_event_ended(self, event_id: str, end_time: float,
                         has_clip: bool, has_snapshot: bool) -> EventState | None:
        """Mark event as ended, returning the event for processing."""
        with self._lock:
            event = self._events.get(event_id)
            if event:
                event.end_time = end_time
                event.has_clip = has_clip
                event.has_snapshot = has_snapshot
                logger.debug(f"Event {event_id} marked ended (clip={has_clip}, snapshot={has_snapshot})")
            return event

    def add_frame_metadata(
        self,
        event_id: str,
        frame_time: float,
        box: Any,
        area: float,
        score: float,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> bool:
        """Append per-frame metadata for an event. Box is normalized to [ymin, xmin, ymax, xmax] 0-1. Capped at MAX_FRAME_METADATA_PER_EVENT."""
        normalized = _normalize_box(box, frame_width, frame_height)
        if normalized is None and box:
            logger.debug("Skipping frame metadata: box could not be normalized")
        with self._lock:
            if event_id not in self._frame_metadata:
                self._frame_metadata[event_id] = []
            lst = self._frame_metadata[event_id]
            if len(lst) >= MAX_FRAME_METADATA_PER_EVENT:
                lst.pop(0)
            self._frame_metadata[event_id].append(
                FrameMetadata(
                    frame_time=frame_time,
                    box=normalized if normalized else (0.0, 0.0, 1.0, 1.0),
                    area=float(area or 0),
                    score=float(score or 0),
                )
            )
        return True

    def get_frame_metadata(self, event_id: str) -> list[FrameMetadata]:
        """Return a copy of frame metadata list for the event (or empty list)."""
        with self._lock:
            return list(self._frame_metadata.get(event_id, []))

    def clear_frame_metadata(self, event_id: str) -> None:
        """Remove all frame metadata for the event (e.g. after analysis or on event removal)."""
        with self._lock:
            self._frame_metadata.pop(event_id, None)

    def remove_event(self, event_id: str) -> EventState | None:
        """Remove event from active tracking and clear its frame metadata."""
        with self._lock:
            self._frame_metadata.pop(event_id, None)
            removed = self._events.pop(event_id, None)
            if removed:
                logger.info(f"Removed event from tracking: {event_id}")
            return removed

    def get_active_event_ids(self) -> list[str]:
        """Get list of all active event IDs (for cleanup protection)."""
        with self._lock:
            return list(self._events.keys())

    def get_stats(self) -> dict:
        """Get statistics about active events."""
        with self._lock:
            by_phase = {phase.name: 0 for phase in EventPhase}
            by_camera = {}

            for event in self._events.values():
                by_phase[event.phase.name] += 1
                by_camera[event.camera] = by_camera.get(event.camera, 0) + 1

            return {
                "total_active": len(self._events),
                "by_phase": by_phase,
                "by_camera": by_camera
            }
