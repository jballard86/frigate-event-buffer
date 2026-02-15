"""Manages consolidated events (time-gap grouped)."""

import logging
import threading
import time
from typing import Dict, Optional, List, TYPE_CHECKING

from frigate_buffer.models import ConsolidatedEvent, _generate_consolidated_id

if TYPE_CHECKING:
    from frigate_buffer.managers.file import FileManager

logger = logging.getLogger('frigate-buffer')


class ConsolidatedEventManager:
    """Manages consolidated events (time-gap grouped). Uses events/{ce_id}/{camera}/ storage."""

    def __init__(self, file_manager: 'FileManager', event_gap_seconds: int = 120,
                 on_close_callback=None):
        self._file_manager = file_manager
        self._events: Dict[str, ConsolidatedEvent] = {}
        self._frigate_to_ce: Dict[str, str] = {}  # frigate_event_id -> consolidated_id
        self._active_ce_id: Optional[str] = None  # Currently active (receiving new sub-events)
        self._close_timers: Dict[str, threading.Timer] = {}  # ce_id -> Timer
        self._lock = threading.RLock()
        self.event_gap_seconds = event_gap_seconds
        self._on_close_callback = on_close_callback  # (ce_id: str) -> None

    @property
    def on_close_callback(self):
        return self._on_close_callback

    @on_close_callback.setter
    def on_close_callback(self, callback):
        self._on_close_callback = callback

    def get_or_create(
        self,
        event_id: str,
        camera: str,
        label: str,
        start_time: float,
    ) -> tuple:
        """
        Get existing consolidated event or create new. Returns (ConsolidatedEvent, is_new, camera_folder).
        Creates events/{ce_id}/{camera}/ storage structure.
        event_gap_seconds: time since last event before next starts a NEW group.
        """
        now = time.time()
        with self._lock:
            if self._active_ce_id:
                ce = self._events.get(self._active_ce_id)
                # Add to existing CE if within time gap (cross-camera grouping)
                if (ce and not ce.closing and not ce.closed and
                        (now - ce.last_activity_time) < self.event_gap_seconds):
                    ce.frigate_event_ids.append(event_id)
                    ce.last_activity_time = now
                    if camera not in ce.cameras:
                        ce.cameras.append(camera)
                    if label and label not in ce.labels:
                        ce.labels.append(label)
                    self._frigate_to_ce[event_id] = ce.consolidated_id
                    # Ensure camera subdir exists
                    camera_folder = self._file_manager.ensure_consolidated_camera_folder(
                        ce.folder_path, camera
                    )
                    return ce, False, camera_folder

            # New consolidated event: create events/{folder_name}/ and events/{folder_name}/{camera}/
            full_id, folder_name = _generate_consolidated_id(start_time)
            base_path = self._file_manager.create_consolidated_event_folder(folder_name)
            camera_folder = self._file_manager.ensure_consolidated_camera_folder(base_path, camera)
            ce = ConsolidatedEvent(
                consolidated_id=full_id,
                folder_name=folder_name,
                folder_path=base_path,
                start_time=start_time,
                last_activity_time=now,
                cameras=[camera],
                frigate_event_ids=[event_id],
                labels=[label] if label else [],
                primary_event_id=event_id,
                primary_camera=camera,
            )
            self._events[full_id] = ce
            self._frigate_to_ce[event_id] = full_id
            self._active_ce_id = full_id
            return ce, True, camera_folder

    def get_by_frigate_event(self, event_id: str) -> Optional[ConsolidatedEvent]:
        with self._lock:
            ce_id = self._frigate_to_ce.get(event_id)
            return self._events.get(ce_id) if ce_id else None

    def update_activity(self, event_id: str, activity_time: Optional[float] = None,
                       end_time: Optional[float] = None) -> None:
        with self._lock:
            ce_id = self._frigate_to_ce.get(event_id)
            if ce_id and ce_id in self._events:
                ce = self._events[ce_id]
                ce.last_activity_time = activity_time or time.time()
                if end_time is not None and end_time > ce.end_time_max:
                    ce.end_time_max = end_time

    def schedule_close_timer(self, ce_id: str) -> None:
        """Schedule or reschedule the close timer for this CE. When it fires, CE is closed."""
        with self._lock:
            if ce_id not in self._events:
                return
            ce = self._events[ce_id]
            if ce.closed:
                return
            # Cancel existing timer
            if ce_id in self._close_timers:
                self._close_timers[ce_id].cancel()
                del self._close_timers[ce_id]
            # Schedule new timer
            def _fire():
                self._on_close_timer(ce_id)
            t = threading.Timer(float(self.event_gap_seconds), _fire)
            t.daemon = True
            self._close_timers[ce_id] = t
            t.start()
            logger.debug(f"Scheduled CE close timer for {ce_id} in {self.event_gap_seconds}s")

    def mark_closing(self, ce_id: str) -> bool:
        """
        Mark event as closing to prevent new additions.
        Returns True if successful (state changed), False if already closing/closed.
        """
        with self._lock:
            ce = self._events.get(ce_id)
            if not ce or ce.closing or ce.closed:
                return False
            ce.closing = True
            if self._active_ce_id == ce_id:
                self._active_ce_id = None
            if ce_id in self._close_timers:
                self._close_timers[ce_id].cancel()
                del self._close_timers[ce_id]
            return True

    def _on_close_timer(self, ce_id: str) -> None:
        """Called when close timer fires. Invoke callback to finalize."""
        # Note: We do NOT mark closed here anymore to avoid race conditions.
        # The callback (finalize_consolidated_event) must acquire the closing lock.
        if self._on_close_callback:
            try:
                self._on_close_callback(ce_id)
            except Exception as e:
                logger.exception(f"CE close callback error for {ce_id}: {e}")

    def update_best(self, event_id: str, title: Optional[str] = None,
                    description: Optional[str] = None, threat_level: Optional[int] = None) -> None:
        with self._lock:
            ce = self.get_by_frigate_event(event_id)
            if ce:
                if title and (not ce.best_title or len(str(title)) > len(str(ce.best_title or ""))):
                    ce.best_title = title
                if description and (not ce.best_description or len(str(description)) > len(str(ce.best_description or ""))):
                    ce.best_description = description
                if threat_level is not None and threat_level > ce.best_threat_level:
                    ce.best_threat_level = threat_level

    def mark_inactive(self, consolidated_id: str) -> None:
        with self._lock:
            if self._active_ce_id == consolidated_id:
                self._active_ce_id = None

    def remove(self, consolidated_id: str) -> Optional[ConsolidatedEvent]:
        with self._lock:
            ce = self._events.pop(consolidated_id, None)
            if ce:
                for fid in ce.frigate_event_ids:
                    self._frigate_to_ce.pop(fid, None)
                if self._active_ce_id == consolidated_id:
                    self._active_ce_id = None
            return ce

    def get_active_consolidated_ids(self) -> List[str]:
        with self._lock:
            return list(self._events.keys())

    def get_all(self) -> List[ConsolidatedEvent]:
        with self._lock:
            return list(self._events.values())
