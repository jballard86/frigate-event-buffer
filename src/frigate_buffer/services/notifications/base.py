"""
Base interface for notification providers and standard result type.

Providers return NotificationResult (not transport-specific payloads) so the
timeline can show who was notified (HA, Pushover, etc.) without coupling to
one transport.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NotRequired, TypedDict

if TYPE_CHECKING:
    from frigate_buffer.models import NotificationEvent


class NotificationResult(TypedDict):
    """Standard result from a notification provider for timeline logging.

    provider and status are always set; message and payload are optional
    (e.g. HA provider can include full payload for deep links in the UI).
    """

    provider: str
    status: str  # e.g. "success", "failure"
    message: NotRequired[str | None]
    payload: NotRequired[dict]  # Optional transport-specific payload for UI display


class BaseNotificationProvider(ABC):
    """Abstract base for notification providers (HA MQTT, Pushover, etc.)."""

    @abstractmethod
    def send(
        self,
        event: "NotificationEvent",
        status: str,
        message: str | None = None,
        tag_override: str | None = None,
    ) -> NotificationResult | None:
        """Send a notification for the given event and status.

        Args:
            event: Event-like object conforming to NotificationEvent protocol.
            status: Lifecycle status (e.g. "new", "finalized", "summarized").
            message: Optional override message; provider may build its own from event.
            tag_override: Optional tag (e.g. frigate_ce_{id}) instead of
                frigate_{event_id}.

        Returns:
            NotificationResult for timeline logging, or None if provider skipped send.
        """
        ...

    @abstractmethod
    def send_overflow(self) -> NotificationResult | None:
        """Send a rate-limit / multiple-events warning via this provider's transport.

        Called by the Dispatcher when the notification queue overflows. Provider
        sends the warning (e.g. HA → MQTT, Pushover → API) and returns a result.
        Dispatcher does not touch MQTT or any transport directly.

        Returns:
            NotificationResult for timeline, or None.
        """
        ...
