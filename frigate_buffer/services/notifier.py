"""Publishes notifications to frigate/custom/notifications with rate limiting."""

import os
import json
import time
import logging
import threading
from typing import List, Optional

import paho.mqtt.client as mqtt

from frigate_buffer.models import EventState

logger = logging.getLogger('frigate-buffer')


class NotificationPublisher:
    """Publishes notifications to frigate/custom/notifications with rate limiting."""

    TOPIC = "frigate/custom/notifications"

    # Rate limiting: max notifications per time window
    MAX_NOTIFICATIONS_PER_WINDOW = 2
    RATE_WINDOW_SECONDS = 5.0
    MAX_QUEUE_SIZE = 10

    def __init__(self, mqtt_client: mqtt.Client, buffer_ip: str, flask_port: int,
                 frigate_url: str = "", storage_path: str = ""):
        self.mqtt_client = mqtt_client
        self.buffer_ip = buffer_ip
        self.flask_port = flask_port
        self.frigate_url = frigate_url.rstrip('/')
        self.storage_path = storage_path
        self.timeline_callback = None  # Optional: (event, status, payload) -> None

        # Rate limiting state
        self._notification_times: List[float] = []
        self._pending_queue: List[tuple] = []  # (event, status, message)
        self._lock = threading.Lock()
        self._overflow_sent = False
        self._queue_processor_running = False

    def _clean_old_timestamps(self):
        """Remove timestamps older than the rate window."""
        cutoff = time.time() - self.RATE_WINDOW_SECONDS
        self._notification_times = [t for t in self._notification_times if t > cutoff]

    def _is_rate_limited(self) -> bool:
        """Check if we've hit the rate limit."""
        self._clean_old_timestamps()
        return len(self._notification_times) >= self.MAX_NOTIFICATIONS_PER_WINDOW

    def _record_notification(self):
        """Record that a notification was sent."""
        self._notification_times.append(time.time())

    def _send_overflow_notification(self) -> bool:
        """Send a summary notification when queue overflows."""
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
            "timestamp": time.time()
        }

        try:
            result = self.mqtt_client.publish(
                self.TOPIC,
                json.dumps(payload),
                retain=False
            )
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info("Published overflow notification")
                return True
            else:
                logger.warning(f"Failed to publish overflow notification: rc={result.rc}")
                return False
        except Exception as e:
            logger.error(f"Error publishing overflow notification: {e}")
            return False

    def _process_queue(self):
        """Background processor for queued notifications."""
        while self._queue_processor_running:
            time.sleep(1.0)  # Check every second

            event = None
            status = None
            message = None
            tag_override = None

            with self._lock:
                if not self._pending_queue:
                    continue

                # Check if we can send (rate limit cleared)
                if not self._is_rate_limited():
                    # Pop oldest notification from queue
                    item = self._pending_queue.pop(0)
                    event, status, message = item[0], item[1], item[2]
                    tag_override = item[3] if len(item) > 3 else None
                    self._record_notification()

                    # Reset overflow flag if queue is draining
                    if len(self._pending_queue) < self.MAX_QUEUE_SIZE:
                        self._overflow_sent = False

            # Send outside the lock
            if event:
                self._send_notification_internal(event, status, message, tag_override)

    def start_queue_processor(self):
        """Start the background queue processor thread."""
        if not self._queue_processor_running:
            self._queue_processor_running = True
            threading.Thread(
                target=self._process_queue,
                daemon=True,
                name="NotificationQueueProcessor"
            ).start()
            logger.info("Started notification queue processor")

    def stop_queue_processor(self):
        """Stop the background queue processor."""
        self._queue_processor_running = False

    def publish_notification(self, event: EventState, status: str,
                            message: Optional[str] = None,
                            tag_override: Optional[str] = None) -> bool:
        """Publish event notification with rate limiting and queue management.
        tag_override: use this tag instead of frigate_{event_id} (e.g. for CE: frigate_ce_{id})"""
        with self._lock:
            # Check rate limit
            if self._is_rate_limited():
                # Add to queue
                self._pending_queue.append((event, status, message, tag_override))
                logger.debug(f"Rate limited, queued notification for {event.event_id} (queue size: {len(self._pending_queue)})")

                # Check for queue overflow
                if len(self._pending_queue) > self.MAX_QUEUE_SIZE and not self._overflow_sent:
                    logger.warning(f"Queue overflow ({len(self._pending_queue)} items), sending summary notification")
                    self._pending_queue.clear()
                    self._overflow_sent = True
                    self._send_overflow_notification()

                return True  # Queued successfully

            # Not rate limited - record and send immediately
            self._record_notification()

        # Send outside the lock
        return self._send_notification_internal(event, status, message, tag_override)

    def _send_notification_internal(self, event: EventState, status: str,
                                    message: Optional[str] = None,
                                    tag_override: Optional[str] = None) -> bool:
        """Internal method to actually send the notification to MQTT."""
        # Construct URLs for Home Assistant - always use buffer base URL for Companion app reachability
        image_url = None
        video_url = None

        buffer_base = f"http://{self.buffer_ip}:{self.flask_port}"

        if event.folder_path:
            # Support both legacy (camera/subdir) and consolidated (events/ce_id/camera) paths
            if self.storage_path and event.folder_path.startswith(self.storage_path):
                try:
                    rel = os.path.relpath(event.folder_path, self.storage_path)
                    rel = rel.replace(os.sep, '/')  # normalize for URL
                    base_url = f"{buffer_base}/files/{rel}"
                except ValueError:
                    folder_name = os.path.basename(event.folder_path)
                    camera_dir = os.path.basename(os.path.dirname(event.folder_path))
                    base_url = f"{buffer_base}/files/{camera_dir}/{folder_name}"
            else:
                folder_name = os.path.basename(event.folder_path)
                camera_dir = os.path.basename(os.path.dirname(event.folder_path))
                base_url = f"{buffer_base}/files/{camera_dir}/{folder_name}"
            if getattr(event, 'image_url_override', None):
                image_url = event.image_url_override
            elif event.snapshot_downloaded:
                image_url = f"{base_url}/snapshot.jpg"
            else:
                # Use buffer proxy to Frigate so image_url is always reachable
                image_url = f"{buffer_base}/api/events/{event.event_id}/snapshot.jpg"
            if event.clip_downloaded:
                video_url = f"{base_url}/clip.mp4"
        elif self.frigate_url:
            # Fallback: no folder yet, use buffer proxy
            image_url = f"{buffer_base}/api/events/{event.event_id}/snapshot.jpg"

        # Build title based on phase
        title = self._build_title(event)

        # Build message
        if not message:
            message = self._build_message(event, status)

        # Deep-link to specific event when folder_path available
        if event.folder_path:
            if self.storage_path and 'events' in event.folder_path.replace(os.sep, '/'):
                # Consolidated: events/ce_id/camera -> camera=events, subdir=ce_id
                parts = event.folder_path.replace(os.sep, '/').split('/')
                if 'events' in parts:
                    idx = parts.index('events')
                    if idx + 1 < len(parts):
                        ce_id = parts[idx + 1]
                        player_url = f"http://{self.buffer_ip}:{self.flask_port}/player?camera=events&subdir={ce_id}"
                    else:
                        player_url = f"http://{self.buffer_ip}:{self.flask_port}/player"
                else:
                    player_url = f"http://{self.buffer_ip}:{self.flask_port}/player"
            else:
                folder_name = os.path.basename(event.folder_path)
                camera_dir = os.path.basename(os.path.dirname(event.folder_path))
                player_url = f"http://{self.buffer_ip}:{self.flask_port}/player?camera={camera_dir}&subdir={folder_name}"
        else:
            player_url = f"http://{self.buffer_ip}:{self.flask_port}/player"

        payload = {
            "event_id": event.event_id,
            "status": status,
            "phase": event.phase.name,
            "camera": event.camera,
            "label": event.label,
            "title": title,
            "message": message,
            "image_url": image_url,
            "video_url": video_url,
            "player_url": player_url,
            "tag": tag_override or f"frigate_{event.event_id}",
            "timestamp": event.created_at,
            "threat_level": event.threat_level,
            "critical": event.threat_level >= 2
        }

        if self.timeline_callback and event.folder_path:
            try:
                self.timeline_callback(event, status, payload)
            except Exception as e:
                logger.debug(f"Timeline callback error: {e}")

        try:
            result = self.mqtt_client.publish(
                self.TOPIC,
                json.dumps(payload),
                retain=False
            )

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Published notification for {event.event_id}: {status}")
                logger.debug(f"Notification payload: {json.dumps(payload, indent=2)}")
                return True
            else:
                logger.warning(f"Failed to publish notification: rc={result.rc}")
                return False
        except Exception as e:
            logger.error(f"Error publishing notification: {e}")
            return False

    def _build_title(self, event: EventState) -> str:
        """Build notification title based on event state."""
        if event.genai_title:
            return event.genai_title

        camera_display = event.camera.replace('_', ' ').title()
        label_display = event.label.title()

        return f"{label_display} detected at {camera_display}"

    def _build_message(self, event: EventState, status: str) -> str:
        """Build notification message combining status context with best available description."""
        # Best available description at this moment
        best_desc = event.genai_description or event.ai_description
        camera_display = event.camera.replace('_', ' ').title()
        label_display = event.label.title()
        fallback = f"{label_display} detected at {camera_display}"

        if status == "summarized" and event.review_summary:
            lines = [l.strip() for l in event.review_summary.split('\n')
                     if l.strip() and not l.strip().startswith('#')]
            excerpt = lines[0] if lines else "Review summary available"
            return excerpt[:200] + ("..." if len(excerpt) > 200 else "")

        if status == "clip_ready":
            desc = best_desc or fallback
            if event.clip_downloaded:
                return f"Video available. {desc}"
            else:
                return f"Video unavailable. {desc}"

        if status == "finalized":
            return best_desc or f"Event complete: {fallback}"

        if status == "described":
            return event.ai_description or f"{fallback} (details updating)"

        # new, snapshot_ready, or any other status
        return best_desc or fallback
