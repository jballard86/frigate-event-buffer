"""
State-Aware Orchestrator - Main coordinator for Frigate Event Buffer.

Coordinates MQTT, file management, event consolidation, notifications, and the web server.
"""

import os
import json
import time
import logging
import threading
from datetime import date, datetime, timedelta
from typing import Optional, List, Any

import requests
import schedule
from urllib.parse import urlparse, urlunparse

from frigate_buffer.models import EventState, _is_no_concerns
from frigate_buffer.managers.file import FileManager
from frigate_buffer.managers.state import EventStateManager
from frigate_buffer.services.video import VideoService
from frigate_buffer.services.download import DownloadService
from frigate_buffer.managers.consolidation import ConsolidatedEventManager
from frigate_buffer.managers.reviews import DailyReviewManager
from frigate_buffer.managers.zone_filter import SmartZoneFilter
from frigate_buffer.services.notifier import NotificationPublisher
from frigate_buffer.services.timeline import TimelineLogger
from frigate_buffer.services.mqtt_client import MqttClientWrapper
from frigate_buffer.services.lifecycle import EventLifecycleService

logger = logging.getLogger('frigate-buffer')


class StateAwareOrchestrator:
    """Main orchestrator coordinating all components."""

    def __init__(self, config: dict):
        self.config = config
        self._shutdown = False
        self._start_time = time.time()

        # Initialize components (file_manager first - needed by consolidated_manager)
        self.state_manager = EventStateManager()
        self.video_service = VideoService(config.get('FFMPEG_TIMEOUT', VideoService.DEFAULT_FFMPEG_TIMEOUT))
        self.download_service = DownloadService(
            config['FRIGATE_URL'],
            self.video_service
        )
        self.file_manager = FileManager(
            config['STORAGE_PATH'],
            config['RETENTION_DAYS']
        )

        # Initialize ConsolidatedEventManager with None callback first to break circular dependency
        self.consolidated_manager = ConsolidatedEventManager(
            self.file_manager,
            event_gap_seconds=config.get('EVENT_GAP_SECONDS', 120),
            on_close_callback=None
        )

        self.timeline_logger = TimelineLogger(self.file_manager, self.consolidated_manager)

        self.mqtt_wrapper = MqttClientWrapper(
            broker=config['MQTT_BROKER'],
            port=config['MQTT_PORT'],
            username=config.get('MQTT_USER'),
            password=config.get('MQTT_PASSWORD'),
            on_message_callback=self._on_mqtt_message,
        )

        self.notifier = NotificationPublisher(
            self.mqtt_wrapper.client,
            config['BUFFER_IP'],
            config['FLASK_PORT'],
            config.get('FRIGATE_URL', ''),
            storage_path=config['STORAGE_PATH']
        )
        self.notifier.timeline_callback = self.timeline_logger.log_ha

        self.zone_filter = SmartZoneFilter(config)

        self.lifecycle_service = EventLifecycleService(
            config,
            self.state_manager,
            self.file_manager,
            self.consolidated_manager,
            self.video_service,
            self.download_service,
            self.notifier,
            self.timeline_logger
        )

        # Update callback after lifecycle service creation
        self.consolidated_manager.on_close_callback = self.lifecycle_service.finalize_consolidated_event

        # Daily review manager (Frigate review summarize API)
        self.daily_review_manager = DailyReviewManager(
            config['STORAGE_PATH'],
            config['FRIGATE_URL'],
            config.get('DAILY_REVIEW_RETENTION_DAYS', 90)
        )

        # Flask app (lazy import to avoid circular deps)
        from frigate_buffer.web.server import create_app
        self.flask_app = create_app(self)

        # Scheduler thread
        self._scheduler_thread = None

        # Request counter for periodic logging
        self._request_count = 0
        self._request_count_lock = threading.Lock()

        # Cache for storage stats
        self._cached_storage_stats = {
            'clips': 0,
            'snapshots': 0,
            'descriptions': 0,
            'total': 0,
            'by_camera': {}
        }

        # Last cleanup tracking delegated to lifecycle service via properties

    @property
    def _last_cleanup_time(self):
        return self.lifecycle_service.last_cleanup_time

    @_last_cleanup_time.setter
    def _last_cleanup_time(self, value):
        self.lifecycle_service.last_cleanup_time = value

    @property
    def _last_cleanup_deleted(self):
        return self.lifecycle_service.last_cleanup_deleted

    @_last_cleanup_deleted.setter
    def _last_cleanup_deleted(self, value):
        self.lifecycle_service.last_cleanup_deleted = value

    def _fetch_ha_state(self, ha_url: str, ha_token: str, entity_id: str):
        """Fetch entity state from Home Assistant REST API. Returns state value or None on error."""
        base = ha_url.rstrip('/')
        path = "/states/" if base.endswith("/api") else "/api/states/"
        url = f"{base}{path}{entity_id}"
        try:
            resp = requests.get(url, headers={"Authorization": f"Bearer {ha_token}"}, timeout=5)
            if resp.ok:
                data = resp.json()
                return data.get("state")
        except requests.RequestException as e:
            logger.warning(f"Error fetching HA state for {entity_id} from {url}: {e}")
        return None

    def _on_mqtt_message(self, client, userdata, msg):
        """Route incoming MQTT messages to appropriate handlers."""
        logger.debug(f"MQTT message received: {msg.topic} ({len(msg.payload)} bytes)")

        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            topic = msg.topic

            if topic == "frigate/events":
                self._handle_frigate_event(payload)
            elif "/tracked_object_update" in topic:
                self._handle_tracked_update(payload, topic)
            elif topic == "frigate/reviews":
                self._handle_review(payload)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {msg.topic}: {e}")
        except Exception as e:
            logger.exception(f"Error processing message from {msg.topic}: {e}")

    def _handle_frigate_event(self, payload: dict):
        """Process frigate/events messages with camera/label filtering and Smart Zone Filtering."""
        event_type = payload.get("type")
        after_data = payload.get("after", {})

        event_id = after_data.get("id")
        camera = after_data.get("camera")
        label = after_data.get("label")
        sub_label = after_data.get("sub_label")
        start_time = after_data.get("start_time", time.time())
        entered_zones = after_data.get("entered_zones") or []

        if not event_id:
            logger.debug("Skipping event: no event_id in payload")
            return

        camera_label_map = self.config.get('CAMERA_LABEL_MAP', {})

        if camera_label_map:
            if camera not in camera_label_map:
                logger.debug(f"Filtered out event from camera '{camera}' (not configured)")
                return

            allowed_labels_for_camera = camera_label_map[camera]
            if allowed_labels_for_camera and label not in allowed_labels_for_camera:
                logger.debug(f"Filtered out '{label}' on '{camera}' (allowed: {allowed_labels_for_camera})")
                return

        if event_type == "end":
            self._handle_event_end(
                event_id=event_id,
                end_time=after_data.get("end_time", time.time()),
                has_clip=after_data.get("has_clip", False),
                has_snapshot=after_data.get("has_snapshot", False),
                mqtt_payload=payload
            )
            return

        # type: new or update - unified Smart Zone Filtering path
        if event_type not in ("new", "update"):
            logger.debug(f"Skipping frigate/events type: {event_type}")
            return

        # Already tracked - log MQTT to timeline, then return (no new creation)
        event = self.state_manager.get_event(event_id)
        if event:
            folder = self.timeline_logger.folder_for_event(event)
            if folder:
                mqtt_type = payload.get("type", "update")
                self.timeline_logger.log_mqtt(folder, "frigate/events", payload, f"Event {mqtt_type} (from Frigate)")
            return

        if not self.zone_filter.should_start_event(camera, label or "", sub_label, entered_zones):
            logger.debug(f"Ignoring {event_id} (smart zone filter: not in tracked zones, entered={entered_zones})")
            return

        self._handle_event_new(
            event_id=event_id,
            camera=camera,
            label=label or "unknown",
            start_time=start_time,
            mqtt_payload=payload
        )

    def _handle_event_new(self, event_id: str, camera: str, label: str,
                          start_time: float, mqtt_payload: Optional[dict] = None):
        """Handle new event detection (Phase 1). Delegates to lifecycle service."""
        self.lifecycle_service.handle_event_new(event_id, camera, label, start_time, mqtt_payload)

    def _handle_event_end(self, event_id: str, end_time: float,
                          has_clip: bool, has_snapshot: bool,
                          mqtt_payload: Optional[dict] = None):
        """Handle event end - trigger downloads/transcoding."""
        logger.info(f"Event ended: {event_id}")

        event = self.state_manager.mark_event_ended(
            event_id, end_time, has_clip, has_snapshot
        )

        if event and self.timeline_logger.folder_for_event(event) and mqtt_payload:
            self.timeline_logger.log_mqtt(
                self.timeline_logger.folder_for_event(event), "frigate/events",
                mqtt_payload, "Event end (from Frigate)"
            )

        if not event or not event.folder_path:
            logger.warning(f"Unknown event ended: {event_id}")
            return

        threading.Thread(
            target=self._process_event_end,
            args=(event,),
            daemon=True
        ).start()

    def _process_event_end(self, event: EventState):
        """Background processing when event ends. Delegates to lifecycle service."""
        self.lifecycle_service.process_event_end(event)

    def _handle_tracked_update(self, payload: dict, topic: str):
        """Handle AI description update (Phase 2)."""
        update_type = payload.get("type")
        if update_type and update_type != "description":
            logger.debug(f"Skipping tracked update type: {update_type}")
            return

        parts = topic.split("/")
        camera = parts[1] if len(parts) >= 2 else "unknown"

        event_id = payload.get("id")
        description = payload.get("description")

        if not event_id or not description:
            logger.debug(f"Skipping tracked update: event_id={event_id}, has_description={bool(description)}")
            return

        event = self.state_manager.get_event(event_id)
        if event and self.timeline_logger.folder_for_event(event):
            self.timeline_logger.log_mqtt(
                self.timeline_logger.folder_for_event(event), topic,
                payload, "Tracked object update (AI description)"
            )

        logger.info(f"Tracked update for {event_id}: {description[:50]}..." if len(str(description)) > 50 else f"Tracked update for {event_id}: {description}")

        if self.state_manager.set_ai_description(event_id, description):
            event = self.state_manager.get_event(event_id)
            if event:
                if event.folder_path:
                    self.file_manager.write_summary(event.folder_path, event)
                self.notifier.publish_notification(event, "described")

    def _handle_review(self, payload: dict):
        """Handle review/genai update (Phase 3)."""
        event_type = payload.get("type")

        if event_type not in ["update", "end", "genai"]:
            logger.debug(f"Skipping review with type: {event_type}")
            return

        review_data = payload.get("after", {}) or payload.get("before", {})
        data = review_data.get("data", {})
        detections = data.get("detections", [])
        severity = review_data.get("severity", "detection")
        genai = data.get("metadata") or data.get("genai") or {}

        logger.debug(f"Processing review: type={event_type}, {len(detections)} detections, severity={severity}")

        for event_id in detections:
            event = self.state_manager.get_event(event_id)
            if event and self.timeline_logger.folder_for_event(event):
                self.timeline_logger.log_mqtt(
                    self.timeline_logger.folder_for_event(event), "frigate/reviews",
                    payload, f"Review update (type={event_type})"
                )
            title = genai.get("title")
            description = genai.get("shortSummary") or genai.get("description")
            scene = genai.get("scene")
            threat_level = int(genai.get("potential_threat_level", 0))

            if title or description:
                logger.info(f"Review for {event_id}: title={title or 'N/A'}, threat_level={threat_level}")
            else:
                logger.debug(f"Review for {event_id}: title=N/A, threat_level={threat_level}")

            if not title and not description:
                logger.debug(f"Skipping finalization for {event_id}: no GenAI data yet")
                continue

            if self.state_manager.set_genai_metadata(
                event_id,
                title,
                description,
                severity,
                threat_level,
                scene=scene
            ):
                event = self.state_manager.get_event(event_id)
                if event:
                    self.consolidated_manager.update_best(
                        event_id, title=title, description=description, threat_level=threat_level
                    )

                    if event.folder_path:
                        event.summary_written = self.file_manager.write_summary(
                            event.folder_path, event
                        )
                        self.file_manager.write_metadata_json(event.folder_path, event)

                    ce = self.consolidated_manager.get_by_frigate_event(event_id)
                    if ce and ce.finalized_sent:
                        logger.debug(f"Suppressing finalized for {event_id} (CE {ce.consolidated_id} already sent)")
                    else:
                        if ce:
                            ce.finalized_sent = True
                            primary = self.state_manager.get_event(ce.primary_event_id)
                            media_folder = os.path.join(ce.folder_path, self.file_manager.sanitize_camera_name(ce.primary_camera or "")) if ce.primary_camera else ce.folder_path
                            if primary:
                                ce.snapshot_downloaded = primary.snapshot_downloaded
                                ce.clip_downloaded = primary.clip_downloaded
                            notify_target = type('NotifyTarget', (), {
                                'event_id': ce.consolidated_id, 'camera': ce.camera, 'label': ce.label,
                                'folder_path': primary.folder_path if primary else media_folder,
                                'created_at': ce.start_time, 'end_time': ce.end_time, 'phase': ce.phase,
                                'genai_title': ce.best_title, 'genai_description': ce.best_description,
                                'ai_description': None, 'review_summary': None,
                                'threat_level': ce.best_threat_level, 'severity': ce.severity,
                                'snapshot_downloaded': ce.snapshot_downloaded,
                                'clip_downloaded': ce.clip_downloaded,
                            })()
                        else:
                            notify_target = event
                        self.notifier.publish_notification(notify_target, "finalized",
                                                          tag_override=f"frigate_{ce.consolidated_id}" if ce else None)

                    if not ce:
                        threading.Thread(
                            target=self._fetch_and_store_review_summary,
                            args=(event,),
                            daemon=True
                        ).start()

    def _fetch_and_store_review_summary(self, event: EventState):
        """Background: fetch review summary from Frigate API, store, notify, then cleanup."""
        try:
            max_wait = 30
            waited = 0
            while event.end_time is None and waited < max_wait:
                time.sleep(2)
                waited += 2

            if event.end_time is None:
                logger.warning(f"No end_time for {event.event_id} after {max_wait}s, using created_at + 60s as fallback")
                effective_end = event.created_at + 60
            else:
                effective_end = event.end_time

            time.sleep(5)

            padding_before = self.config.get('SUMMARY_PADDING_BEFORE', 15)
            padding_after = self.config.get('SUMMARY_PADDING_AFTER', 15)

            padded_start = int(event.created_at - padding_before)
            padded_end = int(effective_end + padding_after)
            url = f"{self.config.get('FRIGATE_URL', '')}/api/review/summarize/start/{padded_start}/end/{padded_end}"
            params = {
                "start": padded_start,
                "end": padded_end,
                "padding_before": padding_before,
                "padding_after": padding_after
            }

            if self.timeline_logger.folder_for_event(event):
                self.timeline_logger.log_frigate_api(
                    self.timeline_logger.folder_for_event(event), "out",
                    "Review summarize request (to Frigate API)",
                    {"url": url, "params": params}
                )

            summary = self.download_service.fetch_review_summary(
                event.created_at, effective_end,
                padding_before, padding_after
            )

            if self.timeline_logger.folder_for_event(event):
                self.timeline_logger.log_frigate_api(
                    self.timeline_logger.folder_for_event(event), "in",
                    "Review summarize response (from Frigate API)",
                    {"url": url, "params": params, "response": summary or "(empty or error)"}
                )

            if summary:
                self.state_manager.set_review_summary(event.event_id, summary)

                write_folder = self.timeline_logger.folder_for_event(event) or event.folder_path
                if write_folder:
                    event.review_summary_written = self.file_manager.write_review_summary(
                        write_folder, summary
                    )
                    self.file_manager.write_summary(write_folder, event)
                    self.file_manager.write_metadata_json(write_folder, event)

                if not _is_no_concerns(summary):
                    self.notifier.publish_notification(event, "summarized")
                else:
                    logger.info(f"Skipping summarized notification for {event.event_id} (no concerns)")
            else:
                logger.warning(f"No review summary obtained for {event.event_id}")

        except Exception as e:
            logger.exception(f"Error fetching review summary for {event.event_id}: {e}")
        finally:
            threading.Timer(
                60.0,
                lambda eid=event.event_id: self.state_manager.remove_event(eid)
            ).start()

    def _run_scheduler(self):
        """Background thread for scheduled tasks."""
        cleanup_hours = self.config.get('CLEANUP_INTERVAL_HOURS', 1)
        schedule.every(cleanup_hours).hours.do(self._hourly_cleanup)
        schedule.every(5).minutes.do(self._log_request_stats)
        schedule.every(5).minutes.do(self._update_storage_stats)

        review_hour = self.config.get('DAILY_REVIEW_SCHEDULE_HOUR', 1)
        schedule.every().day.at(f"{review_hour:02d}:00").do(self._daily_review_job)
        logger.info(f"Scheduled daily review at {review_hour:02d}:00")
        logger.info(f"Scheduled cleanup every {cleanup_hours} hour(s)")

        while not self._shutdown:
            schedule.run_pending()
            time.sleep(60)

    def _log_request_stats(self):
        """Log API request count every 5 minutes."""
        with self._request_count_lock:
            count = self._request_count
            self._request_count = 0
        active = len(self.state_manager._events)
        logger.info(f"API stats (5m): {count} requests, {active} active events, MQTT {'connected' if self.mqtt_wrapper.mqtt_connected else 'disconnected'}")

    def _update_storage_stats(self):
        """Update cached storage stats (background task)."""
        logger.debug("Updating storage stats...")
        try:
            self._cached_storage_stats = self.file_manager.compute_storage_stats()
            logger.debug("Storage stats updated")
        except Exception as e:
            logger.error(f"Failed to update storage stats: {e}")

    def _daily_review_job(self):
        """Fetch yesterday's review from Frigate and save. Runs at configured hour (default 1am)."""
        yesterday = date.today() - timedelta(days=1)
        logger.info(f"Running daily review job for {yesterday}")
        result = self.daily_review_manager.fetch_and_save(yesterday)
        if result:
            logger.info("Daily review saved successfully")
        else:
            logger.warning("Daily review fetch failed or returned no data")
        deleted = self.daily_review_manager.cleanup_old()
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old daily reviews")

    def _hourly_cleanup(self):
        """Hourly cleanup task. Delegates to lifecycle service."""
        self.lifecycle_service.run_cleanup()

    def start(self):
        """Start all components."""
        logger.info("=" * 60)
        logger.info("Starting State-Aware Orchestrator")
        logger.info("=" * 60)
        logger.info(f"MQTT Broker: {self.config['MQTT_BROKER']}:{self.config['MQTT_PORT']}")

        frigate_url = self.config['FRIGATE_URL']
        try:
            parsed = urlparse(frigate_url)
            if parsed.password:
                safe_netloc = f"{parsed.username or ''}:***@{parsed.hostname}"
                if parsed.port:
                    safe_netloc += f":{parsed.port}"
                parsed = parsed._replace(netloc=safe_netloc)
            logger.info(f"Frigate URL: {urlunparse(parsed)}")
        except Exception:
            logger.info(f"Frigate URL: (hidden)")

        logger.info(f"Storage Path: {self.config['STORAGE_PATH']}")
        logger.info(f"Retention: {self.config['RETENTION_DAYS']} days")
        logger.info(f"FFmpeg Timeout: {self.config.get('FFMPEG_TIMEOUT', VideoService.DEFAULT_FFMPEG_TIMEOUT)}s")
        logger.info(f"Log Level: {self.config.get('LOG_LEVEL', 'INFO')}")

        camera_label_map = self.config.get('CAMERA_LABEL_MAP', {})
        if camera_label_map:
            logger.info("Camera/Label Configuration:")
            for camera, labels in camera_label_map.items():
                if labels:
                    logger.info(f"  {camera}: {labels}")
                else:
                    logger.info(f"  {camera}: ALL labels")
        else:
            logger.info("Camera/Label Filtering: DISABLED (all cameras and labels allowed)")

        event_filters = self.config.get('CAMERA_EVENT_FILTERS', {})
        if event_filters:
            logger.info("Smart Zone Filtering:")
            for camera, filt in event_filters.items():
                zones = filt.get('tracked_zones') or []
                exc = filt.get('exceptions') or []
                parts = []
                if zones:
                    parts.append(f"tracked_zones={zones}")
                if exc:
                    parts.append(f"exceptions={exc}")
                if parts:
                    logger.info(f"  {camera}: {', '.join(parts)}")
        else:
            logger.info("Smart Zone Filtering: DISABLED")

        logger.info("=" * 60)

        self.mqtt_wrapper.start()

        self.notifier.start_queue_processor()

        self._scheduler_thread = threading.Thread(
            target=self._run_scheduler,
            daemon=True
        )
        self._scheduler_thread.start()

        # Update storage stats immediately (background)
        threading.Thread(
            target=self._update_storage_stats,
            daemon=True
        ).start()

        logger.info(f"Starting Flask on port {self.config['FLASK_PORT']}...")
        self.flask_app.run(
            host='0.0.0.0',
            port=self.config['FLASK_PORT'],
            threaded=True
        )

    def stop(self):
        """Graceful shutdown."""
        logger.info("Shutting down orchestrator...")
        self._shutdown = True
        self.notifier.stop_queue_processor()
        self.mqtt_wrapper.stop()
