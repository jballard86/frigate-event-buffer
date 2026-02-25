"""Home Assistant MQTT notification provider.

Publishes to frigate/custom/notifications. Builds HA-specific payload (URLs, tags, clear_tag).
No queue or rate limiting; the Dispatcher owns that. Returns NotificationResult for timeline.
"""

import json
import logging
import os
import threading
from typing import Any

import paho.mqtt.client as mqtt

from frigate_buffer.services.notifications.base import (
    BaseNotificationProvider,
    NotificationResult,
)

logger = logging.getLogger("frigate-buffer")

# Protocol type for event-like objects (NotificationEvent protocol)
NotificationEventLike = Any


class HomeAssistantMqttProvider(BaseNotificationProvider):
    """Sends notifications to Home Assistant via MQTT (frigate/custom/notifications)."""

    TOPIC = "frigate/custom/notifications"

    def __init__(
        self,
        mqtt_client: mqtt.Client,
        buffer_ip: str,
        flask_port: int,
        frigate_url: str = "",
        storage_path: str = "",
    ) -> None:
        self.mqtt_client = mqtt_client
        self.buffer_ip = buffer_ip
        self.flask_port = flask_port
        self.frigate_url = frigate_url.rstrip("/")
        self.storage_path = storage_path
        # Clear-previous-notification state (same tag = add clear_tag to replace in HA)
        self._last_notification_tag: str | None = None
        self._next_send_is_new_event: bool = False
        self._lock = threading.Lock()

    def send(
        self,
        event: NotificationEventLike,
        status: str,
        message: str | None = None,
        tag_override: str | None = None,
    ) -> NotificationResult | None:
        """Build HA payload, publish to MQTT, return NotificationResult for timeline."""
        current_tag = tag_override or f"frigate_{event.event_id}"
        with self._lock:
            last_tag = self._last_notification_tag
            next_is_new = self._next_send_is_new_event
            add_clear_tag = (
                last_tag is not None and last_tag == current_tag and not next_is_new
            )
            clear_flag_after_send = next_is_new

        payload = self._build_payload(event, status, message, current_tag, add_clear_tag, last_tag)

        try:
            result = self.mqtt_client.publish(
                self.TOPIC,
                json.dumps(payload),
                retain=False,
            )
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                with self._lock:
                    self._last_notification_tag = current_tag
                    if clear_flag_after_send:
                        self._next_send_is_new_event = False
                logger.info("Published notification for %s: %s", event.event_id, status)
                logger.debug("Notification payload: %s", json.dumps(payload, indent=2))
                return {
                    "provider": "HA_MQTT",
                    "status": "success",
                    "payload": payload,
                }
            logger.warning("Failed to publish notification: rc=%s", result.rc)
            return {"provider": "HA_MQTT", "status": "failure", "message": f"rc={result.rc}"}
        except Exception as e:
            logger.error("Error publishing notification: %s", e)
            return {"provider": "HA_MQTT", "status": "failure", "message": str(e)}

    def send_overflow(self) -> NotificationResult | None:
        """Publish overflow summary to MQTT; return NotificationResult."""
        import time

        payload = {
            "event_id": "overflow_summary",
            "status": "overflow",
            "phase": "OVERFLOW",
            "camera": "multiple",
            "label": "multiple",
            "title": "Multiple Security Events",
            "message": "Multiple notifications were queued. Click to review all events on your Security Alert Dashboard.",
            "image_url": None,
            "video_url": None,
            "tag": "frigate_overflow",
            "timestamp": time.time(),
        }
        try:
            result = self.mqtt_client.publish(
                self.TOPIC,
                json.dumps(payload),
                retain=False,
            )
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info("Published overflow notification")
                return {"provider": "HA_MQTT", "status": "success", "payload": payload}
            logger.warning("Failed to publish overflow notification: rc=%s", result.rc)
            return {"provider": "HA_MQTT", "status": "failure", "message": f"rc={result.rc}"}
        except Exception as e:
            logger.error("Error publishing overflow notification: %s", e)
            return {"provider": "HA_MQTT", "status": "failure", "message": str(e)}

    def mark_last_event_ended(self) -> None:
        """Next notification will be treated as a new event (no clear_tag)."""
        with self._lock:
            self._next_send_is_new_event = True

    def _build_payload(
        self,
        event: NotificationEventLike,
        status: str,
        message: str | None,
        current_tag: str,
        add_clear_tag: bool,
        last_tag: str | None,
    ) -> dict:
        """Build the HA-shaped payload dict."""
        buffer_base = f"http://{self.buffer_ip}:{self.flask_port}"
        image_url = None
        video_url = None

        if getattr(event, "folder_path", None):
            folder_path = event.folder_path
            if self.storage_path and folder_path.startswith(self.storage_path):
                try:
                    rel = os.path.relpath(folder_path, self.storage_path)
                    rel = rel.replace(os.sep, "/")
                    base_url = f"{buffer_base}/files/{rel}"
                except ValueError:
                    folder_name = os.path.basename(folder_path)
                    camera_dir = os.path.basename(os.path.dirname(folder_path))
                    base_url = f"{buffer_base}/files/{camera_dir}/{folder_name}"
            else:
                folder_name = os.path.basename(folder_path)
                camera_dir = os.path.basename(os.path.dirname(folder_path))
                base_url = f"{buffer_base}/files/{camera_dir}/{folder_name}"
            if getattr(event, "image_url_override", None):
                image_url = getattr(event, "image_url_override", None)
            elif getattr(event, "snapshot_downloaded", False):
                image_url = f"{base_url}/snapshot.jpg"
            else:
                image_url = f"{buffer_base}/api/events/{event.event_id}/snapshot.jpg"
            if getattr(event, "clip_downloaded", False):
                from frigate_buffer.services.query import resolve_clip_in_folder

                full_folder = folder_path
                if self.storage_path and not os.path.isabs(full_folder):
                    full_folder = os.path.join(self.storage_path, full_folder)
                clip_basename = resolve_clip_in_folder(full_folder)
                video_url = f"{base_url}/{clip_basename}" if clip_basename else None
        elif self.frigate_url:
            image_url = f"{buffer_base}/api/events/{event.event_id}/snapshot.jpg"

        title = self._build_title(event)
        if not message:
            message = self._build_message(event, status)

        if getattr(event, "folder_path", None):
            folder_path = event.folder_path
            if self.storage_path and "events" in folder_path.replace(os.sep, "/"):
                parts = folder_path.replace(os.sep, "/").split("/")
                if "events" in parts:
                    idx = parts.index("events")
                    if idx + 1 < len(parts):
                        ce_id = parts[idx + 1]
                        player_url = f"http://{self.buffer_ip}:{self.flask_port}/player?camera=events&subdir={ce_id}"
                    else:
                        player_url = f"http://{self.buffer_ip}:{self.flask_port}/player"
                else:
                    player_url = f"http://{self.buffer_ip}:{self.flask_port}/player"
            else:
                folder_name = os.path.basename(folder_path)
                camera_dir = os.path.basename(os.path.dirname(folder_path))
                player_url = f"http://{self.buffer_ip}:{self.flask_port}/player?camera={camera_dir}&subdir={folder_name}"
        else:
            player_url = f"http://{self.buffer_ip}:{self.flask_port}/player"

        phase = getattr(event, "phase", None)
        phase_name = "NEW" if phase is None else (getattr(phase, "name", None) or str(phase))
        threat_level = getattr(event, "threat_level", 0)
        created_at = getattr(event, "created_at", 0.0)

        payload = {
            "event_id": event.event_id,
            "status": status,
            "phase": phase_name,
            "camera": event.camera,
            "label": event.label,
            "title": title,
            "message": message,
            "image_url": image_url,
            "video_url": video_url,
            "player_url": player_url,
            "tag": current_tag,
            "timestamp": created_at,
            "threat_level": threat_level,
            "critical": threat_level >= 2,
        }
        if add_clear_tag and last_tag is not None:
            payload["clear_tag"] = last_tag
        return payload

    def _get_camera_display_name(self, event: NotificationEventLike) -> str:
        return event.camera.replace("_", " ").title()

    def _get_label_display_name(self, event: NotificationEventLike) -> str:
        return event.label.title()

    def _get_default_title(self, event: NotificationEventLike) -> str:
        camera_display = self._get_camera_display_name(event)
        label_display = self._get_label_display_name(event)
        return f"{label_display} detected at {camera_display}"

    def _build_title(self, event: NotificationEventLike) -> str:
        if getattr(event, "genai_title", None):
            return event.genai_title
        return self._get_default_title(event)

    def _build_message(self, event: NotificationEventLike, status: str) -> str:
        best_desc = getattr(event, "genai_description", None) or getattr(event, "ai_description", None)
        fallback = self._get_default_title(event)
        review_summary = getattr(event, "review_summary", None)

        match status:
            case "summarized" if review_summary:
                lines = [
                    l.strip()
                    for l in review_summary.split("\n")
                    if l.strip() and not l.strip().startswith("#")
                ]
                excerpt = lines[0] if lines else "Review summary available"
                return excerpt[:200] + ("..." if len(excerpt) > 200 else "")
            case "clip_ready":
                desc = best_desc or fallback
                if getattr(event, "clip_downloaded", False):
                    return f"Video available. {desc}"
                return f"Video unavailable. {desc}"
            case "finalized":
                return best_desc or f"Event complete: {fallback}"
            case "described":
                return getattr(event, "ai_description", None) or f"{fallback} (details updating)"
            case _:
                return best_desc or fallback
