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

import paho.mqtt.client as mqtt
import requests
import schedule

from frigate_buffer.models import EventState, _is_no_concerns
from frigate_buffer.managers.file import FileManager
from frigate_buffer.managers.state import EventStateManager
from frigate_buffer.managers.consolidation import ConsolidatedEventManager
from frigate_buffer.managers.reviews import DailyReviewManager
from frigate_buffer.services.notifier import NotificationPublisher

logger = logging.getLogger('frigate-buffer')


class StateAwareOrchestrator:
    """Main orchestrator coordinating all components."""

    MQTT_TOPICS = [
        ("frigate/events", 0),
        ("frigate/+/tracked_object_update", 0),
        ("frigate/reviews", 0)
    ]

    def __init__(self, config: dict):
        self.config = config
        self._shutdown = False
        self._start_time = time.time()

        # Initialize components (file_manager first - needed by consolidated_manager)
        self.state_manager = EventStateManager()
        self.file_manager = FileManager(
            config['STORAGE_PATH'],
            config['FRIGATE_URL'],
            config['RETENTION_DAYS'],
            config.get('FFMPEG_TIMEOUT', 60)
        )
        self.consolidated_manager = ConsolidatedEventManager(
            self.file_manager,
            event_gap_seconds=config.get('EVENT_GAP_SECONDS', 120),
            on_close_callback=self._on_consolidated_event_close
        )

        # Setup MQTT client
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="frigate-event-buffer")
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        self.mqtt_client.on_message = self._on_mqtt_message
        self.mqtt_client.reconnect_delay_set(min_delay=1, max_delay=120)
        self.mqtt_connected = False

        # Notification publisher (initialized after MQTT setup)
        self.notifier = NotificationPublisher(
            self.mqtt_client,
            config['BUFFER_IP'],
            config['FLASK_PORT'],
            config.get('FRIGATE_URL', ''),
            storage_path=config['STORAGE_PATH']
        )
        self.notifier.timeline_callback = self._timeline_log_ha

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

        # Last cleanup tracking (for stats dashboard)
        self._last_cleanup_time: Optional[float] = None
        self._last_cleanup_deleted: int = 0

    def _on_mqtt_connect(self, client, userdata, flags, reason_code, properties):
        """Handle MQTT connection."""
        if reason_code == 0:
            self.mqtt_connected = True
            logger.info(f"Connected to MQTT broker {self.config['MQTT_BROKER']}")

            for topic, qos in self.MQTT_TOPICS:
                client.subscribe(topic, qos)
                logger.info(f"Subscribed to: {topic}")
        else:
            logger.error(f"MQTT connection failed with code: {reason_code}")

    def _on_mqtt_disconnect(self, client, userdata, flags, reason_code, properties):
        """Handle MQTT disconnection."""
        self.mqtt_connected = False
        if reason_code != 0:
            logger.warning(f"Unexpected MQTT disconnect (rc={reason_code}), reconnecting...")
        else:
            logger.info("MQTT disconnected")

    def _timeline_folder(self, event) -> Optional[str]:
        """Folder for timeline (CE root if consolidated, else event folder)."""
        if not event:
            return None
        ce = self.consolidated_manager.get_by_frigate_event(event.event_id)
        return ce.folder_path if ce else event.folder_path

    def _timeline_log_ha(self, event, status: str, payload: dict) -> None:
        """Log HA notification payload to event timeline."""
        folder = self._timeline_folder(event)
        if folder:
            self.file_manager.append_timeline_entry(folder, {
                "source": "ha_notification",
                "direction": "out",
                "label": f"Sent to Home Assistant: {status}",
                "data": payload
            })

    def _timeline_log_mqtt(self, folder_path: str, topic: str, payload: dict, label: str) -> None:
        """Log MQTT payload from Frigate to event timeline."""
        if folder_path:
            self.file_manager.append_timeline_entry(folder_path, {
                "source": "frigate_mqtt",
                "direction": "in",
                "label": label,
                "data": {"topic": topic, "payload": payload}
            })

    def _timeline_log_frigate_api(self, folder_path: str, direction: str,
                                   label: str, data: dict) -> None:
        """Log Frigate API request/response to event timeline. direction: 'in' or 'out'."""
        if folder_path:
            self.file_manager.append_timeline_entry(folder_path, {
                "source": "frigate_api",
                "direction": direction,
                "label": label,
                "data": data
            })

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
        except Exception:
            pass
        return None

    def _normalize_sub_label(self, sub_label: Any) -> Optional[str]:
        """Extract matchable string from Frigate sub_label (format varies by version).
        Handles: None, string, [name, score], empty values, unexpected types.
        """
        if sub_label is None:
            return None
        if isinstance(sub_label, str):
            return sub_label.strip() if sub_label.strip() else None
        if isinstance(sub_label, (list, tuple)) and len(sub_label) > 0:
            first = sub_label[0]
            if isinstance(first, str):
                return first.strip() if first.strip() else None
            if first is not None:
                return str(first).strip() or None
        return None

    def _should_start_event(self, camera: str, label: str, sub_label: Any,
                            entered_zones: List[str]) -> bool:
        """Smart Zone Filtering: decide if we should create an event.
        Returns True to start, False to ignore (defer).
        No event_filters for camera -> legacy behavior (always start).
        """
        filters = self.config.get('CAMERA_EVENT_FILTERS', {}).get(camera)
        if not filters:
            return True

        exceptions = filters.get('exceptions') or []
        tracked_zones = filters.get('tracked_zones') or []

        # 1. Exceptions: create event regardless of zone
        if exceptions:
            exc_set = {e.strip().lower() for e in exceptions if e}
            if label and label.strip().lower() in exc_set:
                return True
            norm = self._normalize_sub_label(sub_label)
            if norm and norm.lower() in exc_set:
                return True

        # 2. Tracked zones: create ONLY when object enters a tracked zone
        if not tracked_zones:
            return True
        entered = entered_zones or []
        if not entered:
            return False
        tracked_set = {z.strip().lower() for z in tracked_zones if z}
        entered_lower = [z.strip().lower() for z in entered if z]
        return any(z in tracked_set for z in entered_lower)

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
            folder = self._timeline_folder(event)
            if folder:
                mqtt_type = payload.get("type", "update")
                self._timeline_log_mqtt(folder, "frigate/events", payload, f"Event {mqtt_type} (from Frigate)")
            return

        if not self._should_start_event(camera, label or "", sub_label, entered_zones):
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
        """Handle new event detection (Phase 1). Uses events/{ce_id}/{camera}/ storage."""
        logger.info(f"New event: {event_id} - {label} on {camera}")

        event = self.state_manager.create_event(event_id, camera, label, start_time)

        ce, is_new, camera_folder = self.consolidated_manager.get_or_create(
            event_id, camera, label, start_time
        )
        event.folder_path = camera_folder

        if mqtt_payload:
            mqtt_type = mqtt_payload.get("type", "new")
            self._timeline_log_mqtt(
                ce.folder_path, "frigate/events",
                mqtt_payload, f"Event {mqtt_type} (from Frigate)"
            )

        if not is_new:
            logger.info(f"Event {event_id} grouped into consolidated event {ce.consolidated_id}, suppressing duplicate notification")
            def _download_grouped_snapshot():
                self.file_manager.download_snapshot(event_id, camera_folder)
            threading.Thread(target=_download_grouped_snapshot, daemon=True).start()
            return

        delay = self.config.get('NOTIFICATION_DELAY', 5)
        ce_tag = f"frigate_{ce.consolidated_id}" if ce else None
        threading.Thread(
            target=self._send_initial_notification,
            args=(event, delay, ce_tag),
            daemon=True
        ).start()

    def _send_initial_notification(self, event: EventState, delay: float,
                                   tag_override: Optional[str] = None):
        """Send notification immediately, then fetch snapshot and silently update."""
        try:
            self.notifier.publish_notification(event, "new", tag_override=tag_override)

            if delay > 0:
                time.sleep(delay)

            if event.folder_path:
                event.snapshot_downloaded = self.file_manager.download_snapshot(
                    event.event_id, event.folder_path
                )
                if event.snapshot_downloaded:
                    self.notifier.publish_notification(event, "snapshot_ready", tag_override=tag_override)
        except Exception as e:
            logger.error(f"Error in initial notification flow for {event.event_id}: {e}")

    def _handle_event_end(self, event_id: str, end_time: float,
                          has_clip: bool, has_snapshot: bool,
                          mqtt_payload: Optional[dict] = None):
        """Handle event end - trigger downloads/transcoding."""
        logger.info(f"Event ended: {event_id}")

        event = self.state_manager.mark_event_ended(
            event_id, end_time, has_clip, has_snapshot
        )

        if event and self._timeline_folder(event) and mqtt_payload:
            self._timeline_log_mqtt(
                self._timeline_folder(event), "frigate/events",
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

    def _on_consolidated_event_close(self, ce_id: str):
        """Called when CE close timer fires. Export clips, fetch summary, send notifications."""
        with self.consolidated_manager._lock:
            ce = self.consolidated_manager._events.get(ce_id)
        if not ce or not ce.closed:
            return

        export_before = self.config.get('EXPORT_BUFFER_BEFORE', 5)
        export_after = self.config.get('EXPORT_BUFFER_AFTER', 30)
        padding_before = self.config.get('SUMMARY_PADDING_BEFORE', 15)
        padding_after = self.config.get('SUMMARY_PADDING_AFTER', 15)
        start_ts = int(ce.start_time - export_before)
        end_ts = int((ce.end_time_max or ce.last_activity_time) + export_after)

        first_clip_path = None
        primary_cam = ce.primary_camera or (ce.cameras[0] if ce.cameras else None)
        for camera in ce.cameras:
            camera_folder = self.file_manager.ensure_consolidated_camera_folder(ce.folder_path, camera)
            self._timeline_log_frigate_api(
                ce.folder_path, 'out',
                f'Clip export for {camera} (CE close)',
                {'url': f"{self.config.get('FRIGATE_URL', '')}/api/export/{camera}/start/{start_ts}/end/{end_ts}"}
            )
            result = self.file_manager.export_and_transcode_clip(
                ce.consolidated_id, camera_folder, camera,
                ce.start_time, ce.end_time_max or ce.last_activity_time,
                export_before, export_after
            )
            ok = result.get("success", False)
            timeline_data = {"success": ok, "frigate_response": result.get("frigate_response")}
            if "fallback" in result:
                timeline_data["fallback"] = result["fallback"]
            self._timeline_log_frigate_api(ce.folder_path, 'in', f'Clip export response for {camera}', timeline_data)
            if ok and first_clip_path is None:
                first_clip_path = os.path.join(camera_folder, 'clip.mp4')
        # Placeholder fallback: if all exports failed, try per-event clip for primary camera
        if first_clip_path is None and primary_cam and ce.primary_event_id:
            primary_folder = self.file_manager.ensure_consolidated_camera_folder(ce.folder_path, primary_cam)
            ok = self.file_manager.download_and_transcode_clip(ce.primary_event_id, primary_folder)
            if ok:
                first_clip_path = os.path.join(primary_folder, 'clip.mp4')
                self._timeline_log_frigate_api(ce.folder_path, 'in', 'Placeholder clip (events API fallback)', {'success': True})

        if first_clip_path:
            gif_path = os.path.join(ce.folder_path, 'notification.gif')
            if self.file_manager.generate_gif_from_clip(first_clip_path, gif_path):
                ce.snapshot_downloaded = True
            ce.clip_downloaded = True

        padded_start = int(ce.start_time - padding_before)
        padded_end = int((ce.end_time_max or ce.last_activity_time) + padding_after)
        self._timeline_log_frigate_api(
            ce.folder_path, 'out',
            'Review summarize (CE close)',
            {'url': f"{self.config.get('FRIGATE_URL', '')}/api/review/summarize/start/{padded_start}/end/{padded_end}"}
        )
        summary = self.file_manager.fetch_review_summary(
            ce.start_time, ce.end_time_max or ce.last_activity_time,
            padding_before, padding_after
        )
        self._timeline_log_frigate_api(
            ce.folder_path, 'in',
            'Review summarize response',
            {'response': summary or '(empty or error)'}
        )
        if summary:
            self.file_manager.write_review_summary(ce.folder_path, summary)

        primary_cam = ce.primary_camera or (ce.cameras[0] if ce.cameras else None)
        media_folder = os.path.join(ce.folder_path, self.file_manager.sanitize_camera_name(primary_cam or "")) if primary_cam else ce.folder_path
        buf_base = f"http://{self.config['BUFFER_IP']}:{self.config['FLASK_PORT']}"
        gif_url = f"{buf_base}/files/events/{ce.folder_name}/notification.gif" if os.path.exists(os.path.join(ce.folder_path, 'notification.gif')) else None
        notify_target = type('NotifyTarget', (), {
            'event_id': ce.consolidated_id, 'camera': ce.camera, 'label': ce.label,
            'folder_path': media_folder, 'created_at': ce.start_time,
            'end_time': ce.end_time, 'phase': ce.phase,
            'genai_title': ce.best_title, 'genai_description': ce.best_description,
            'ai_description': None, 'review_summary': summary,
            'threat_level': ce.best_threat_level, 'severity': ce.severity,
            'snapshot_downloaded': ce.snapshot_downloaded,
            'clip_downloaded': ce.clip_downloaded,
            'image_url_override': gif_url,
        })()

        if not ce.clip_ready_sent:
            ce.clip_ready_sent = True
            self.notifier.publish_notification(notify_target, "clip_ready")
        if not ce.finalized_sent and (ce.best_title or ce.best_description):
            ce.finalized_sent = True
            self.notifier.publish_notification(notify_target, "finalized")
        if summary and not _is_no_concerns(summary):
            self.notifier.publish_notification(notify_target, "summarized")

        for fid in ce.frigate_event_ids:
            self.state_manager.remove_event(fid)
        self.consolidated_manager.remove(ce_id)
        logger.info(f"Consolidated event {ce_id} closed and cleaned up")

    def _process_event_end(self, event: EventState):
        """Background processing when event ends. For consolidated events, defers clip export to CE close."""
        try:
            if event.has_snapshot:
                event.snapshot_downloaded = self.file_manager.download_snapshot(
                    event.event_id, event.folder_path
                )

            ce = self.consolidated_manager.get_by_frigate_event(event.event_id)
            if ce:
                self.consolidated_manager.update_activity(
                    event.event_id,
                    activity_time=time.time(),
                    end_time=event.end_time or event.created_at
                )
                self.consolidated_manager.schedule_close_timer(ce.consolidated_id)
                self.file_manager.write_summary(self._timeline_folder(event) or event.folder_path, event)
                active_ids = self.state_manager.get_active_event_ids()
                active_ce_folders = [c.folder_name for c in self.consolidated_manager.get_all()]
                deleted = self.file_manager.cleanup_old_events(active_ids, active_ce_folders)
                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} old event folders")
                return

            if event.has_clip and event.folder_path:
                export_before = self.config.get('EXPORT_BUFFER_BEFORE', 5)
                export_after = self.config.get('EXPORT_BUFFER_AFTER', 30)
                start_ts = int(event.created_at - export_before)
                end_ts = int((event.end_time or event.created_at) + export_after)

                self._timeline_log_frigate_api(
                    self._timeline_folder(event), 'out',
                    'Clip export request (to Frigate API)',
                    {
                        'url': f"{self.config.get('FRIGATE_URL', '')}/api/export/{event.camera}/start/{start_ts}/end/{end_ts}",
                        'params': {'camera': event.camera, 'start': start_ts, 'end': end_ts},
                    },
                )

                result = self.file_manager.export_and_transcode_clip(
                    event.event_id,
                    event.folder_path,
                    camera=event.camera,
                    start_time=event.created_at,
                    end_time=event.end_time or event.created_at,
                    export_buffer_before=export_before,
                    export_buffer_after=export_after,
                )
                event.clip_downloaded = result.get("success", False)

                timeline_data = {"success": event.clip_downloaded, "frigate_response": result.get("frigate_response")}
                if "fallback" in result:
                    timeline_data["fallback"] = result["fallback"]
                self._timeline_log_frigate_api(
                    self._timeline_folder(event), 'in',
                    'Clip export response (from Frigate API)',
                    timeline_data,
                )

            self.file_manager.write_summary(event.folder_path, event)

            ce = self.consolidated_manager.get_by_frigate_event(event.event_id)
            should_send_clip = True
            if ce:
                if ce.primary_event_id != event.event_id:
                    should_send_clip = False
                    logger.debug(f"Suppressing clip_ready for {event.event_id} (non-primary in CE {ce.consolidated_id})")
                elif ce.clip_ready_sent:
                    should_send_clip = False
                else:
                    ce.clip_ready_sent = True
            if should_send_clip:
                self.notifier.publish_notification(event, "clip_ready")

            active_ids = self.state_manager.get_active_event_ids()
            active_ce_folders = [ce.folder_name for ce in self.consolidated_manager.get_all()]
            deleted = self.file_manager.cleanup_old_events(active_ids, active_ce_folders)
            self._last_cleanup_time = time.time()
            self._last_cleanup_deleted = deleted
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old event folders")

        except Exception as e:
            logger.exception(f"Error processing event end: {e}")

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
        if event and self._timeline_folder(event):
            self._timeline_log_mqtt(
                self._timeline_folder(event), topic,
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
            if event and self._timeline_folder(event):
                self._timeline_log_mqtt(
                    self._timeline_folder(event), "frigate/reviews",
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

            if self._timeline_folder(event):
                self._timeline_log_frigate_api(
                    self._timeline_folder(event), "out",
                    "Review summarize request (to Frigate API)",
                    {"url": url, "params": params}
                )

            summary = self.file_manager.fetch_review_summary(
                event.created_at, effective_end,
                padding_before, padding_after
            )

            if self._timeline_folder(event):
                self._timeline_log_frigate_api(
                    self._timeline_folder(event), "in",
                    "Review summarize response (from Frigate API)",
                    {"url": url, "params": params, "response": summary or "(empty or error)"}
                )

            if summary:
                self.state_manager.set_review_summary(event.event_id, summary)

                write_folder = self._timeline_folder(event) or event.folder_path
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
        logger.info(f"API stats (5m): {count} requests, {active} active events, MQTT {'connected' if self.mqtt_connected else 'disconnected'}")

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
        """Hourly cleanup task."""
        logger.info("Running scheduled cleanup...")
        active_ids = self.state_manager.get_active_event_ids()
        active_ce_folders = [ce.folder_name for ce in self.consolidated_manager.get_all()]
        deleted = self.file_manager.cleanup_old_events(active_ids, active_ce_folders)
        self._last_cleanup_time = time.time()
        self._last_cleanup_deleted = deleted
        logger.info(f"Scheduled cleanup complete. Deleted {deleted} folders.")

    def start(self):
        """Start all components."""
        logger.info("=" * 60)
        logger.info("Starting State-Aware Orchestrator")
        logger.info("=" * 60)
        logger.info(f"MQTT Broker: {self.config['MQTT_BROKER']}:{self.config['MQTT_PORT']}")
        logger.info(f"Frigate URL: {self.config['FRIGATE_URL']}")
        logger.info(f"Storage Path: {self.config['STORAGE_PATH']}")
        logger.info(f"Retention: {self.config['RETENTION_DAYS']} days")
        logger.info(f"FFmpeg Timeout: {self.config.get('FFMPEG_TIMEOUT', 60)}s")
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

        try:
            self.mqtt_client.connect_async(
                self.config['MQTT_BROKER'],
                self.config['MQTT_PORT'],
                keepalive=60
            )
            self.mqtt_client.loop_start()
        except Exception as e:
            logger.error(f"Failed to start MQTT client: {e}")

        self.notifier.start_queue_processor()

        self._scheduler_thread = threading.Thread(
            target=self._run_scheduler,
            daemon=True
        )
        self._scheduler_thread.start()

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
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
