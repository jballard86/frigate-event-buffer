"""Manager modules for file, state, consolidation, zone filtering, and preferences."""

from frigate_buffer.managers.consolidation import ConsolidatedEventManager
from frigate_buffer.managers.file import FileManager
from frigate_buffer.managers.preferences import PreferencesManager
from frigate_buffer.managers.state import EventStateManager
from frigate_buffer.managers.zone_filter import SmartZoneFilter

__all__ = [
    "FileManager",
    "EventStateManager",
    "ConsolidatedEventManager",
    "SmartZoneFilter",
    "PreferencesManager",
]
