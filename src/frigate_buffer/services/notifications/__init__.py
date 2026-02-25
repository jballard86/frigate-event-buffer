"""Notification service: provider interface, dispatcher, and providers (e.g. Home Assistant MQTT).

BaseNotificationProvider, NotificationDispatcher, and HomeAssistantMqttProvider
are implemented in Phases 2–4. No imports from this package in orchestrator/lifecycle until Phase 5.
"""

from frigate_buffer.services.notifications.base import (
    BaseNotificationProvider,
    NotificationResult,
)
from frigate_buffer.services.notifications.dispatcher import NotificationDispatcher
from frigate_buffer.services.notifications.providers import (
    HomeAssistantMqttProvider,
    PushoverProvider,
)

__all__ = [
    "BaseNotificationProvider",
    "NotificationDispatcher",
    "NotificationResult",
    "HomeAssistantMqttProvider",
    "PushoverProvider",
]
