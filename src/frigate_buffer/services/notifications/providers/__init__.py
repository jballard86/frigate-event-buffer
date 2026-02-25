"""Notification providers (e.g. Home Assistant MQTT)."""

from frigate_buffer.services.notifications.providers.ha_mqtt import HomeAssistantMqttProvider

__all__ = ["HomeAssistantMqttProvider"]
