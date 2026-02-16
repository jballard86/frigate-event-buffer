"""Timeline logging service: writes event timeline entries (HA, MQTT, Frigate API) to event folders."""

import logging
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from frigate_buffer.managers.file import FileManager
    from frigate_buffer.managers.consolidation import ConsolidatedEventManager
    from frigate_buffer.models import EventState

logger = logging.getLogger('frigate-buffer')


class TimelineLogger:
    """Handles all timeline logging: HA notifications, MQTT payloads, Frigate API requests/responses."""

    def __init__(
        self,
        file_manager: 'FileManager',
        consolidated_manager: 'ConsolidatedEventManager',
    ) -> None:
        self._file_manager = file_manager
        self._consolidated_manager = consolidated_manager

    def folder_for_event(self, event: Optional['EventState']) -> Optional[str]:
        """Folder for timeline (CE root if consolidated, else event folder)."""
        if not event:
            return None
        ce = self._consolidated_manager.get_by_frigate_event(event.event_id)
        return ce.folder_path if ce else event.folder_path

    def log_ha(self, event: Optional['EventState'], status: str, payload: dict) -> None:
        """Log HA notification payload to event timeline."""
        folder = self.folder_for_event(event)
        if folder:
            self._file_manager.append_timeline_entry(folder, {
                "source": "ha_notification",
                "direction": "out",
                "label": f"Sent to Home Assistant: {status}",
                "data": payload
            })

    def log_mqtt(self, folder_path: str, topic: str, payload: dict, label: str) -> None:
        """Log MQTT payload from Frigate to event timeline."""
        if folder_path:
            self._file_manager.append_timeline_entry(folder_path, {
                "source": "frigate_mqtt",
                "direction": "in",
                "label": label,
                "data": {"topic": topic, "payload": payload}
            })

    def log_frigate_api(
        self,
        folder_path: str,
        direction: str,
        label: str,
        data: dict,
    ) -> None:
        """Log Frigate API request/response to event timeline. direction: 'in' or 'out'."""
        if folder_path:
            self._file_manager.append_timeline_entry(folder_path, {
                "source": "frigate_api",
                "direction": direction,
                "label": label,
                "data": data
            })
