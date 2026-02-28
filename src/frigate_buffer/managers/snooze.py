"""In-memory snooze state per camera for muting notifications or AI processing.

Snoozes expire by time; expired entries are removed on read (get_active_snoozes,
is_notifications_snoozed, is_ai_snoozed) so no background job is required.
"""

import logging
import threading
import time

logger = logging.getLogger("frigate-buffer")


class SnoozeManager:
    """Thread-safe in-memory manager for per-camera snooze state.

    Tracks expiration_time (Unix timestamp), snooze_notifications, and snooze_ai
    per camera. Used by API routes and (when wired) by notification/AI code to
    selectively mute notifications or AI processing until expiration.
    """

    def __init__(self) -> None:
        self._snoozes: dict[str, dict] = {}
        self._lock = threading.RLock()

    def set_snooze(
        self,
        camera: str,
        duration_minutes: int,
        snooze_notifications: bool = True,
        snooze_ai: bool = True,
    ) -> float:
        """Set or overwrite snooze for a camera.

        Args:
            camera: Camera name to snooze.
            duration_minutes: Minutes until snooze expires.
            snooze_notifications: If True, notifications for this camera
                should be muted while snoozed.
            snooze_ai: If True, AI processing for this camera should be
                skipped while snoozed.

        Returns:
            Expiration time as Unix timestamp.
        """
        expiration_time = time.time() + duration_minutes * 60
        with self._lock:
            self._snoozes[camera] = {
                "expiration_time": expiration_time,
                "snooze_notifications": snooze_notifications,
                "snooze_ai": snooze_ai,
            }
        return expiration_time

    def clear_snooze(self, camera: str) -> None:
        """Remove snooze for the given camera. Idempotent if not snoozed."""
        with self._lock:
            self._snoozes.pop(camera, None)

    def get_active_snoozes(self) -> dict[str, dict]:
        """Return all currently active (non-expired) snoozes.

        Expired entries are removed before building the result. Each value
        has keys: expiration_time (float), snooze_notifications (bool),
        snooze_ai (bool).

        Returns:
            Dict mapping camera name to snooze details.
        """
        now = time.time()
        with self._lock:
            expired = [
                c for c, d in self._snoozes.items() if d["expiration_time"] <= now
            ]
            for c in expired:
                del self._snoozes[c]
            return {
                cam: {
                    "expiration_time": data["expiration_time"],
                    "snooze_notifications": data["snooze_notifications"],
                    "snooze_ai": data["snooze_ai"],
                }
                for cam, data in self._snoozes.items()
            }

    def is_notifications_snoozed(self, camera: str) -> bool:
        """True if camera has an active snooze with snooze_notifications True.

        If the snooze is expired, it is removed and False is returned.
        """
        with self._lock:
            data = self._snoozes.get(camera)
            if data is None:
                return False
            if data["expiration_time"] <= time.time():
                del self._snoozes[camera]
                return False
            return bool(data["snooze_notifications"])

    def is_ai_snoozed(self, camera: str) -> bool:
        """True if camera has an active snooze with snooze_ai True.

        If the snooze is expired, it is removed and False is returned.
        """
        with self._lock:
            data = self._snoozes.get(camera)
            if data is None:
                return False
            if data["expiration_time"] <= time.time():
                del self._snoozes[camera]
                return False
            return bool(data["snooze_ai"])
