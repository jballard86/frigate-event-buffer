"""FCM (Firebase Cloud Messaging) notification provider for the mobile app.

Sends data-only messages so the Android app can handle them in the background.
Token is read dynamically from PreferencesManager (no notification block).
A background keep-alive thread runs a periodic FCM dry-run (every 10 minutes)
at init to warm and maintain the HTTP connection pool and TLS handshake,
avoiding first-notification and post-idle latency.
"""

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from frigate_buffer.services.notifications.base import (
    BaseNotificationProvider,
    NotificationResult,
)

if TYPE_CHECKING:
    from frigate_buffer.managers.preferences import PreferencesManager

logger = logging.getLogger("frigate-buffer")

# FCM data payload keys and values must be strings.
OVERFLOW_PHASE = "OVERFLOW"
OVERFLOW_MESSAGE = "Too many events queued"

# Statuses that have a cropped snapshot available (snapshot_ready or later).
_STATUSES_WITH_CROPPED_SNAPSHOT = frozenset(
    {"snapshot_ready", "clip_ready", "finalized", "summarized"}
)

NotificationEventLike = Any


def _extract_ce_id(tag_override: str | None, event: NotificationEventLike) -> str:
    """Extract event/CE ID: from tag_override (strip frigate_ prefix) or event."""
    if tag_override and tag_override.startswith("frigate_"):
        return tag_override[len("frigate_") :].strip() or getattr(event, "event_id", "")
    return getattr(event, "event_id", "") or ""


def _build_title(event: NotificationEventLike) -> str:
    """Title from genai_title or generic fallback (FCM requires a string)."""
    title = getattr(event, "genai_title", None)
    if title and str(title).strip():
        return str(title).strip()
    camera = (getattr(event, "camera", None) or "unknown").replace("_", " ").title()
    label = (getattr(event, "label", None) or "motion").title()
    return f"{label} detected at {camera}"


def _build_fcm_data_payload(
    ce_id: str,
    phase: str,
    title: str,
    message: str,
    description: str,
    deep_link: str,
    clear_notification: str,
    camera: str,
    threat_level: str,
    live_frame_proxy: str,
    hosted_snapshot: str,
    notification_gif: str,
    hosted_clip: str,
) -> dict[str, str]:
    """Build FCM data dict per MOBILE_API_CONTRACT.md §9; all values must be strings."""
    return {
        "ce_id": ce_id,
        "phase": phase,
        "clear_notification": clear_notification,
        "threat_level": threat_level,
        "camera": camera,
        "live_frame_proxy": live_frame_proxy or "",
        "hosted_snapshot": hosted_snapshot or "",
        "notification_gif": notification_gif or "",
        "title": title,
        "description": description or "",
        "message": message,
        "hosted_clip": hosted_clip or "",
        "deep_link": deep_link,
    }


class MobileAppProvider(BaseNotificationProvider):
    """Sends FCM data-only messages to the registered mobile app token.

    Token is fetched from PreferencesManager on each send so the app can
    register/update without restarting the server. A background keep-alive
    thread runs a periodic FCM dry-run (every 10 minutes) at init to warm and
    maintain the HTTP connection pool and TLS handshake.
    """

    def __init__(self, preferences_manager: "PreferencesManager") -> None:
        """Initialize with the manager that holds the FCM token.

        Args:
            preferences_manager: Used to get_fcm_token() on each send/overflow.
        """
        self.preferences_manager = preferences_manager
        thread = threading.Thread(target=self._warmup_connection, daemon=True)
        thread.start()
        logger.info("FCM keep-alive thread started.")

    def _warmup_connection(self) -> None:
        """Run a periodic FCM dry-run keep-alive to maintain the connection pool.

        Runs in a daemon thread. Uses dry_run=True so no message is delivered;
        failures (e.g. dummy token) are logged at DEBUG and ignored.
        """
        while True:
            token = self.preferences_manager.get_fcm_token()
            if not (token and str(token).strip()):
                token = "dummy_warmup_token"
            try:
                from firebase_admin import messaging

                message = messaging.Message(
                    data={"warmup": "true"},
                    token=token,
                )
                messaging.send(message, dry_run=True)
                logger.debug("FCM keep-alive succeeded.")
            except Exception as e:
                logger.debug(
                    "FCM keep-alive failed (expected for dummy token): %s",
                    e,
                )
            time.sleep(600)

    def send(
        self,
        event: NotificationEventLike,
        status: str,
        message: str | None = None,
        tag_override: str | None = None,
    ) -> NotificationResult | None:
        """Send a phase-aware FCM data message; no notification block.

        If no FCM token is registered, logs at debug and returns None.
        All payload values are strings for FCM data message requirements.
        """
        token = self.preferences_manager.get_fcm_token()
        if not token:
            logger.debug(
                "No FCM token registered, skipping mobile notification for %s",
                getattr(event, "event_id", "?"),
            )
            return None

        ce_id = _extract_ce_id(tag_override, event)
        phase = status.upper()
        title = _build_title(event)
        msg = (message or "").strip() or _build_title(event)
        description = (getattr(event, "genai_description", None) or "").strip() or msg
        deep_link = f"buffer://event_detail/{ce_id}"
        clear_notification = "true" if status == "discarded" else "false"
        camera = getattr(event, "camera", None) or "unknown"
        threat_level = str(getattr(event, "threat_level", 0))
        folder_path = getattr(event, "folder_path", None)

        live_frame_proxy = getattr(event, "live_frame_proxy", None) or ""
        if status == "new" and not live_frame_proxy:
            live_frame_proxy = f"/api/cameras/{camera}/latest.jpg"

        hosted_snapshot = getattr(event, "hosted_snapshot", None) or ""
        if (
            not hosted_snapshot
            and status in _STATUSES_WITH_CROPPED_SNAPSHOT
            and folder_path
            and str(folder_path).strip()
        ):
            hosted_snapshot = f"/files/{folder_path}/snapshot_cropped.jpg"

        notification_gif = getattr(event, "notification_gif", None) or ""
        hosted_clip = getattr(event, "hosted_clip", None) or ""

        payload = _build_fcm_data_payload(
            ce_id=ce_id,
            phase=phase,
            title=title,
            message=msg,
            description=description,
            deep_link=deep_link,
            clear_notification=clear_notification,
            camera=camera,
            threat_level=threat_level,
            live_frame_proxy=live_frame_proxy,
            hosted_snapshot=hosted_snapshot,
            notification_gif=notification_gif,
            hosted_clip=hosted_clip,
        )
        # Ensure every value is str (FCM data only accepts strings).
        data = {k: (v if isinstance(v, str) else str(v)) for k, v in payload.items()}

        try:
            from firebase_admin import exceptions as firebase_exceptions
            from firebase_admin import messaging

            messaging.send(
                messaging.Message(
                    data=data,
                    token=token,
                    android=messaging.AndroidConfig(priority="high"),
                )
            )
            logger.info(
                "FCM notification sent for %s: %s",
                ce_id,
                status,
            )
            return {"provider": "MOBILE_APP", "status": "success"}
        except firebase_exceptions.FirebaseError as e:
            logger.warning("FCM send failed for %s: %s", ce_id, e)
            return {
                "provider": "MOBILE_APP",
                "status": "failure",
                "message": str(e),
            }

    def send_overflow(self) -> NotificationResult | None:
        """Send a static overflow FCM data payload to the registered token."""
        token = self.preferences_manager.get_fcm_token()
        if not token:
            logger.debug("No FCM token registered, skipping overflow notification")
            return None

        data = {
            "phase": OVERFLOW_PHASE,
            "message": OVERFLOW_MESSAGE,
        }
        try:
            from firebase_admin import exceptions as firebase_exceptions
            from firebase_admin import messaging

            messaging.send(
                messaging.Message(
                    data=data,
                    token=token,
                    android=messaging.AndroidConfig(priority="high"),
                )
            )
            logger.info("FCM overflow notification sent")
            return {"provider": "MOBILE_APP", "status": "success"}
        except firebase_exceptions.FirebaseError as e:
            logger.warning("FCM overflow send failed: %s", e)
            return {
                "provider": "MOBILE_APP",
                "status": "failure",
                "message": str(e),
            }
