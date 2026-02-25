"""Notification dispatcher: rate limiting, queue, and generic timeline logging.

Calls provider.send() / provider.send_overflow() on all enabled providers.
Does not touch MQTT; overflow is handled by each provider's send_overflow().
"""

import logging
import threading
import time
from typing import Any

from frigate_buffer.services.timeline import TimelineLogger

from frigate_buffer.services.notifications.base import (
    BaseNotificationProvider,
    NotificationResult,
)

logger = logging.getLogger("frigate-buffer")

# Rate limiting: max notifications per time window (same as legacy notifier)
MAX_NOTIFICATIONS_PER_WINDOW = 2
RATE_WINDOW_SECONDS = 5.0
MAX_QUEUE_SIZE = 10


class NotificationDispatcher:
    """Single entry point for publish_notification: rate limit, queue, call providers, log results."""

    def __init__(
        self,
        providers: list[BaseNotificationProvider],
        timeline_logger: TimelineLogger,
    ) -> None:
        self._providers = list(providers)
        self._timeline_logger = timeline_logger
        # Reference to any provider that supports mark_last_event_ended (e.g. HA MQTT)
        self._ha_provider: BaseNotificationProvider | None = None
        for p in self._providers:
            if hasattr(p, "mark_last_event_ended"):
                self._ha_provider = p
                break

        self._notification_times: list[float] = []
        self._pending_queue: list[tuple] = []  # (event, status, message, tag_override)
        self._lock = threading.Lock()
        self._overflow_sent = False
        self._queue_processor_running = False

    def _clean_old_timestamps(self) -> None:
        cutoff = time.time() - RATE_WINDOW_SECONDS
        self._notification_times[:] = [t for t in self._notification_times if t > cutoff]

    def _is_rate_limited(self) -> bool:
        self._clean_old_timestamps()
        return len(self._notification_times) >= MAX_NOTIFICATIONS_PER_WINDOW

    def _record_notification(self) -> None:
        self._notification_times.append(time.time())

    def _send_now(
        self,
        event: Any,
        status: str,
        message: str | None,
        tag_override: str | None,
    ) -> bool:
        """Call each provider, collect results, log to timeline. Returns True if at least one send was attempted."""
        results: list[NotificationResult | None] = []
        for provider in self._providers:
            try:
                r = provider.send(event, status, message=message, tag_override=tag_override)
                if r is not None:
                    results.append(r)
            except Exception as e:
                logger.exception("Provider %s send failed: %s", type(provider).__name__, e)
                results.append({"provider": type(provider).__name__, "status": "failure", "message": str(e)})

        # Always log dispatch results (event history: phase + who was notified)
        self._timeline_logger.log_dispatch_results(event, status, [r for r in results if r is not None])

        return len(results) > 0

    def _do_overflow(self) -> None:
        """Clear queue and call send_overflow() on every provider. Dispatcher does not touch MQTT."""
        for provider in self._providers:
            try:
                provider.send_overflow()
            except Exception as e:
                logger.exception("Provider %s send_overflow failed: %s", type(provider).__name__, e)

    def publish_notification(
        self,
        event: Any,
        status: str,
        message: str | None = None,
        tag_override: str | None = None,
    ) -> bool:
        """Publish via all providers with rate limiting and queue. Same signature as legacy notifier."""
        with self._lock:
            if self._is_rate_limited():
                self._pending_queue.append((event, status, message, tag_override))
                logger.debug(
                    "Rate limited, queued notification for %s (queue size: %s)",
                    getattr(event, "event_id", "?"),
                    len(self._pending_queue),
                )
                if len(self._pending_queue) > MAX_QUEUE_SIZE and not self._overflow_sent:
                    logger.warning(
                        "Queue overflow (%s items), sending overflow via all providers",
                        len(self._pending_queue),
                    )
                    self._pending_queue.clear()
                    self._overflow_sent = True
                    self._do_overflow()
                return True

            self._record_notification()

        return self._send_now(event, status, message, tag_override)

    def _process_queue(self) -> None:
        """Background thread: drain queue when rate limit allows."""
        while self._queue_processor_running:
            time.sleep(1.0)
            event, status, message, tag_override = None, None, None, None
            with self._lock:
                if not self._pending_queue:
                    continue
                if not self._is_rate_limited():
                    item = self._pending_queue.pop(0)
                    event, status, message = item[0], item[1], item[2]
                    tag_override = item[3] if len(item) > 3 else None
                    self._record_notification()
                    if len(self._pending_queue) < MAX_QUEUE_SIZE:
                        self._overflow_sent = False
            if event is not None:
                self._send_now(event, status, message, tag_override)

    def start_queue_processor(self) -> None:
        if not self._queue_processor_running:
            self._queue_processor_running = True
            threading.Thread(
                target=self._process_queue,
                daemon=True,
                name="NotificationQueueProcessor",
            ).start()
            logger.info("Started notification queue processor")

    def stop_queue_processor(self) -> None:
        self._queue_processor_running = False

    def mark_last_event_ended(self) -> None:
        """Forward to the HA provider if present (next notification = new event, no clear_tag)."""
        if self._ha_provider is not None:
            self._ha_provider.mark_last_event_ended()

    @property
    def queue_size(self) -> int:
        with self._lock:
            return len(self._pending_queue)
