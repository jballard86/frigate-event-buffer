"""Pushover notification provider.

Sends notifications via Pushover API with phase-based priority (loud
snapshot_ready, silent clip_ready/finalized) and optional image/GIF
attachments. Uses event properties only for player URL construction
(no folder_path string parsing).
"""

import logging
import os
from typing import Any

import requests

from frigate_buffer.services.notifications.base import (
    BaseNotificationProvider,
    NotificationResult,
)

logger = logging.getLogger("frigate-buffer")

PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"
URL_TITLE = "View in Event Viewer"
OVERFLOW_MESSAGE = "Too many events occurring. Notifications temporarily paused."
OVERFLOW_TITLE = "Frigate Event Buffer"

# Allowed statuses; others are skipped with "Filtered intermediate phase".
ALLOWED_PHASES = ("snapshot_ready", "clip_ready", "finalized")

# Threat level >= this uses high priority (1) for snapshot_ready; else normal (0).
THREAT_HIGH_PRIORITY = 2

# Emergency priority requires retry and expire or API rejects.
EMERGENCY_RETRY = 30
EMERGENCY_EXPIRE = 3600

# Protocol type for event-like objects (NotificationEvent protocol).
NotificationEventLike = Any


def _player_url(player_base_url: str, event: NotificationEventLike) -> str:
    """Build player URL from event properties only (no folder_path parsing)."""
    base = (player_base_url or "").rstrip("/")
    prefix = f"{base}/player" if base else "/player"
    # ConsolidatedEvent has folder_name; use camera=events&subdir=folder_name.
    folder_name = getattr(event, "folder_name", None)
    if folder_name:
        return f"{prefix}?camera=events&subdir={folder_name}"
    camera = getattr(event, "camera", None) or "unknown"
    event_id = getattr(event, "event_id", None)
    if event_id:
        return f"{prefix}?camera={camera}&subdir={event_id}"
    return f"{prefix}?camera={camera}"


def _build_title(event: NotificationEventLike) -> str:
    """Title from genai_title or default from camera/label."""
    if getattr(event, "genai_title", None):
        return event.genai_title
    camera = getattr(event, "camera", "unknown").replace("_", " ").title()
    label = getattr(event, "label", "motion").title()
    return f"{label} detected at {camera}"


def _build_message(
    event: NotificationEventLike, status: str, message_override: str | None
) -> str:
    """Message from override, or from event/status."""
    if message_override:
        return message_override
    best_desc = getattr(event, "genai_description", None) or getattr(
        event, "ai_description", None
    )
    fallback = _build_title(event)
    if status == "clip_ready":
        desc = best_desc or fallback
        if getattr(event, "clip_downloaded", False):
            return f"Video available. {desc}"
        return f"Video unavailable. {desc}"
    if status == "finalized":
        return best_desc or f"Event complete: {fallback}"
    return best_desc or fallback


class PushoverProvider(BaseNotificationProvider):
    """Sends notifications via Pushover API with phase filter and
    priority/attachments."""

    def __init__(self, pushover_config: dict, player_base_url: str) -> None:
        self._config = pushover_config or {}
        self._player_base_url = (player_base_url or "").rstrip("/")
        self._token = (self._config.get("pushover_api_token") or "").strip()
        self._user = (self._config.get("pushover_user_key") or "").strip()
        self._device = (self._config.get("device") or "").strip() or None
        self._sound = (self._config.get("default_sound") or "").strip() or None
        self._html = 1 if self._config.get("html", 1) else 0

    def send(
        self,
        event: NotificationEventLike,
        status: str,
        message: str | None = None,
        tag_override: str | None = None,
    ) -> NotificationResult | None:
        """Send notification for allowed phases; skip others with status 'skipped'."""
        if status not in ALLOWED_PHASES:
            return {
                "provider": "PUSHOVER",
                "status": "skipped",
                "message": "Filtered intermediate phase",
            }

        # Priority and attachment by phase.
        if status == "snapshot_ready":
            threat = getattr(event, "threat_level", 0)
            priority = 1 if threat >= THREAT_HIGH_PRIORITY else 0
            attachment_path = None
            if getattr(event, "folder_path", None):
                path = os.path.join(event.folder_path, "latest.jpg")
                if os.path.isfile(path):
                    attachment_path = (path, "image/jpeg", "latest.jpg")
        else:
            # clip_ready or finalized: silent.
            priority = -1
            attachment_path = None
            if getattr(event, "folder_path", None):
                path = os.path.join(event.folder_path, "notification.gif")
                if os.path.isfile(path):
                    attachment_path = (path, "image/gif", "notification.gif")

        title = _build_title(event)
        body = _build_message(event, status, message)
        url = _player_url(self._player_base_url, event)
        ts = int(getattr(event, "created_at", 0) or 0)

        payload: dict[str, Any] = {
            "token": self._token,
            "user": self._user,
            "title": title,
            "message": body,
            "url": url,
            "url_title": URL_TITLE,
            "html": self._html,
            "priority": priority,
            "timestamp": ts,
        }
        if self._device:
            payload["device"] = self._device
        if self._sound:
            payload["sound"] = self._sound
        if priority == 2:
            payload["retry"] = EMERGENCY_RETRY
            payload["expire"] = EMERGENCY_EXPIRE

        files = None
        if attachment_path:
            path, mime, name = attachment_path
            try:
                with open(path, "rb") as f:
                    files = {"attachment": (name, f.read(), mime)}
            except OSError as e:
                logger.warning("Pushover: could not read attachment %s: %s", path, e)

        try:
            if files:
                # Multipart: send fields as form data; requests will use
                # multipart/form-data.
                resp = requests.post(
                    PUSHOVER_API_URL,
                    data=payload,
                    files=files,
                    timeout=30,
                )
            else:
                resp = requests.post(
                    PUSHOVER_API_URL,
                    data=payload,
                    timeout=30,
                )
        except requests.RequestException as e:
            logger.error("Pushover API request failed: %s", e)
            return {
                "provider": "PUSHOVER",
                "status": "failure",
                "message": str(e),
            }

        try:
            data = resp.json()
        except Exception as e:
            logger.warning("Pushover: invalid JSON response: %s", e)
            return {
                "provider": "PUSHOVER",
                "status": "failure",
                "message": resp.text[:500] if resp.text else str(e),
            }

        if resp.status_code >= 400 or data.get("status") != 1:
            errors = data.get("errors") or [resp.text or f"HTTP {resp.status_code}"]
            err_msg = (
                "; ".join(str(e) for e in errors)
                if isinstance(errors, list)
                else str(errors)
            )
            logger.warning("Pushover API error: %s", err_msg)
            return {
                "provider": "PUSHOVER",
                "status": "failure",
                "message": err_msg[:500],
            }

        logger.info(
            "Pushover notification sent for %s: %s",
            getattr(event, "event_id", "?"),
            status,
        )
        return {"provider": "PUSHOVER", "status": "success"}

    def send_overflow(self) -> NotificationResult | None:
        """Send overflow warning with priority 0; return NotificationResult."""
        payload: dict[str, Any] = {
            "token": self._token,
            "user": self._user,
            "title": OVERFLOW_TITLE,
            "message": OVERFLOW_MESSAGE,
            "priority": 0,
            "html": self._html,
        }
        if self._device:
            payload["device"] = self._device
        if self._sound:
            payload["sound"] = self._sound
        try:
            resp = requests.post(PUSHOVER_API_URL, data=payload, timeout=30)
            data = (
                resp.json()
                if resp.headers.get("content-type", "").startswith("application/json")
                else {}
            )
            if resp.status_code >= 400 or data.get("status") != 1:
                errors = data.get("errors") or [resp.text or f"HTTP {resp.status_code}"]
                err_msg = (
                    "; ".join(str(e) for e in errors)
                    if isinstance(errors, list)
                    else str(errors)
                )
                return {
                    "provider": "PUSHOVER",
                    "status": "failure",
                    "message": err_msg[:500],
                }
            logger.info("Pushover overflow notification sent")
            return {"provider": "PUSHOVER", "status": "success"}
        except requests.RequestException as e:
            logger.error("Pushover overflow request failed: %s", e)
            return {"provider": "PUSHOVER", "status": "failure", "message": str(e)}
