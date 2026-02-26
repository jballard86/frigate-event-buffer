"""Smart Zone Filtering: per-camera rules for when to start an event
(zones + exceptions)."""

import logging
from typing import Any

logger = logging.getLogger("frigate-buffer")


class SmartZoneFilter:
    """Decides whether to start an event based on CAMERA_EVENT_FILTERS
    (tracked_zones, exceptions)."""

    def __init__(self, config: dict) -> None:
        self._config = config

    @staticmethod
    def normalize_sub_label(sub_label: Any) -> str | None:
        """Extract matchable string from Frigate sub_label (format varies by version).
        Handles: None, string, [name, score], empty values, unexpected types.
        """
        if sub_label is None:
            return None
        if isinstance(sub_label, str):
            return sub_label.strip() if sub_label.strip() else None
        if isinstance(sub_label, (list, tuple)) and len(sub_label) > 0:
            first = sub_label[0]
            if isinstance(first, str):
                return first.strip() if first.strip() else None
            if first is not None:
                return str(first).strip() or None
        return None

    def should_start_event(
        self,
        camera: str,
        label: str,
        sub_label: Any,
        entered_zones: list[str],
        current_zones: list[str] | None = None,
    ) -> bool:
        """Smart Zone Filtering: decide if we should create an event.

        Returns True to start, False to ignore (defer).
        No event_filters for camera -> legacy behavior (always start).
        When tracked_zones is set, we start if the object is in a tracked zone
        per either entered_zones or current_zones, so we react on the first
        message where the object is currently in zone even if entered_zones is
        empty (avoids delayed first notification when Frigate populates zone
        only in later messages).
        """
        filters = self._config.get("CAMERA_EVENT_FILTERS", {}).get(camera)
        if not filters:
            return True

        exceptions = filters.get("exceptions") or []
        tracked_zones = filters.get("tracked_zones") or []

        # 1. Exceptions: create event regardless of zone
        if exceptions:
            exc_set = {e.strip().lower() for e in exceptions if e}
            if label and label.strip().lower() in exc_set:
                return True
            norm = self.normalize_sub_label(sub_label)
            if norm and norm.lower() in exc_set:
                return True

        # 2. Tracked zones: start if object is in a tracked zone per entered OR current
        if not tracked_zones:
            return True
        entered = entered_zones or []
        current = current_zones or []
        tracked_set = {z.strip().lower() for z in tracked_zones if z}
        entered_lower = [z.strip().lower() for z in entered if z]
        current_lower = [z.strip().lower() for z in current if z]
        return any(z in tracked_set for z in entered_lower) or any(
            z in tracked_set for z in current_lower
        )
