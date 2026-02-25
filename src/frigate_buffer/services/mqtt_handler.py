"""
MQTT message parsing and dispatch for Frigate events, tracked_object_update, and reviews.

Decodes JSON, routes by topic, and delegates to handlers that use state_manager,
zone_filter, lifecycle_service, timeline_logger, notifier, file_manager, and download_service.
Keeps this logic out of the orchestrator so it only wires the handler.
"""

import json
import logging
import os
import threading
import time
from typing import Any

from frigate_buffer.logging_utils import should_suppress_review_debug_logs
from frigate_buffer.models import EventState

logger = logging.getLogger("frigate-buffer")


class MqttMessageHandler:
    """Handles incoming MQTT messages: frigate/events, tracked_object_update, frigate/reviews."""

    def __init__(
        self,
        config: dict[str, Any],
        state_manager: Any,
        zone_filter: Any,
        lifecycle_service: Any,
        timeline_logger: Any,
        notifier: Any,
        file_manager: Any,
        consolidated_manager: Any,
        download_service: Any,
    ) -> None:
        self._config = config
        self._state_manager = state_manager
        self._zone_filter = zone_filter
        self._lifecycle_service = lifecycle_service
        self._timeline_logger = timeline_logger
        self._notifier = notifier
        self._file_manager = file_manager
        self._consolidated_manager = consolidated_manager
        self._download_service = download_service

    def on_message(self, client: Any, userdata: Any, msg: Any) -> None:
        """Route incoming MQTT messages by topic. Called by MqttClientWrapper."""
        logger.debug("MQTT message received: %s (%s bytes)", msg.topic, len(msg.payload))
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            topic = msg.topic
            if topic == "frigate/events":
                self._handle_frigate_event(payload)
            elif "/tracked_object_update" in topic:
                self._handle_tracked_update(payload, topic)
            elif topic == "frigate/reviews":
                self._handle_review(payload)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in %s: %s", msg.topic, e)
        except Exception as e:
            logger.exception("Error processing message from %s: %s", msg.topic, e)

    def _handle_frigate_event(self, payload: dict) -> None:
        """Process frigate/events with camera/label and Smart Zone filtering."""
        event_type = payload.get("type")
        after_data = payload.get("after", {})
        event_id = after_data.get("id")
        camera = after_data.get("camera")
        label = after_data.get("label")
        sub_label = after_data.get("sub_label")
        start_time = after_data.get("start_time", time.time())
        entered_zones = after_data.get("entered_zones") or []
        current_zones = after_data.get("current_zones") or []

        if not event_id:
            logger.debug("Skipping event: no event_id in payload")
            return

        camera_label_map = self._config.get("CAMERA_LABEL_MAP", {})
        if camera_label_map:
            if camera not in camera_label_map:
                logger.debug("Filtered out event from camera '%s' (not configured)", camera)
                return
            allowed_labels_for_camera = camera_label_map[camera]
            if allowed_labels_for_camera and label not in allowed_labels_for_camera:
                logger.debug(
                    "Filtered out '%s' on '%s' (allowed: %s)",
                    label,
                    camera,
                    allowed_labels_for_camera,
                )
                return

        match event_type:
            case "end":
                self._handle_event_end(
                    event_id=event_id,
                    end_time=after_data.get("end_time", time.time()),
                    has_clip=after_data.get("has_clip", False),
                    has_snapshot=after_data.get("has_snapshot", False),
                    mqtt_payload=payload,
                )
                return
            case "new" | "update":
                pass
            case _:
                logger.debug("Skipping frigate/events type: %s", event_type)
                return

        event = self._state_manager.get_event(event_id)
        if event:
            folder = self._timeline_logger.folder_for_event(event)
            if folder:
                mqtt_type = payload.get("type", "update")
                self._timeline_logger.log_mqtt(
                    folder, "frigate/events", payload, f"Event {mqtt_type} (from Frigate)"
                )
            return

        if not self._zone_filter.should_start_event(
            camera, label or "", sub_label, entered_zones, current_zones
        ):
            logger.debug(
                "Ignoring %s (smart zone filter: not in tracked zones, entered=%s, current=%s)",
                event_id,
                entered_zones,
                current_zones,
            )
            return

        self._handle_event_new(
            event_id=event_id,
            camera=camera,
            label=label or "unknown",
            start_time=start_time,
            mqtt_payload=payload,
        )

    def _handle_event_new(
        self,
        event_id: str,
        camera: str,
        label: str,
        start_time: float,
        mqtt_payload: dict | None = None,
    ) -> None:
        """Handle new event detection. Delegates to lifecycle service."""
        self._lifecycle_service.handle_event_new(
            event_id, camera, label, start_time, mqtt_payload
        )

    def _handle_event_end(
        self,
        event_id: str,
        end_time: float,
        has_clip: bool,
        has_snapshot: bool,
        mqtt_payload: dict | None = None,
    ) -> None:
        """Handle event end: mark ended, log, then process in background thread."""
        event = self._state_manager.get_event(event_id)
        if event and event.end_time is not None:
            logger.debug("Duplicate event end for %s (already ended), skipping", event_id)
            return
        logger.info("Event ended: %s", event_id)
        event = self._state_manager.mark_event_ended(
            event_id, end_time, has_clip, has_snapshot
        )
        if event and self._timeline_logger.folder_for_event(event) and mqtt_payload:
            self._timeline_logger.log_mqtt(
                self._timeline_logger.folder_for_event(event),
                "frigate/events",
                mqtt_payload,
                "Event end (from Frigate)",
            )
        if not event or not event.folder_path:
            logger.debug("Unknown event ended: %s", event_id)
            return
        threading.Thread(
            target=self._process_event_end,
            args=(event,),
            daemon=True,
        ).start()

    def _process_event_end(self, event: EventState) -> None:
        """Background: delegate to lifecycle service."""
        self._lifecycle_service.process_event_end(event)

    def _handle_tracked_update(self, payload: dict, topic: str) -> None:
        """Handle tracked_object_update: frame metadata and/or AI description."""
        parts = topic.split("/")
        camera = parts[1] if len(parts) >= 2 else "unknown"
        event_id = payload.get("id")
        if event_id:
            after = payload.get("after") or payload.get("before") or {}
            has_frame_data = (
                after.get("frame_time") is not None
                or "box" in after
                or after.get("area") is not None
                or after.get("score") is not None
            )
            if has_frame_data and self._state_manager.get_event(event_id):
                region = after.get("region")
                fw = (
                    (region[2] - region[0])
                    if isinstance(region, (list, tuple)) and len(region) >= 4
                    else None
                )
                fh = (
                    (region[3] - region[1])
                    if isinstance(region, (list, tuple)) and len(region) >= 4
                    else None
                )
                self._state_manager.add_frame_metadata(
                    event_id,
                    frame_time=float(after.get("frame_time") or 0),
                    box=after.get("box"),
                    area=float(after.get("area") or 0),
                    score=float(after.get("score") or 0),
                    frame_width=fw,
                    frame_height=fh,
                )
        update_type = payload.get("type")
        if update_type and update_type != "description":
            return
        if self._config.get("AI_MODE") == "external_api":
            return
        description = payload.get("description")
        if not event_id or not description:
            return
        event = self._state_manager.get_event(event_id)
        if event and self._timeline_logger.folder_for_event(event):
            self._timeline_logger.log_mqtt(
                self._timeline_logger.folder_for_event(event),
                topic,
                payload,
                "Tracked object update (AI description)",
            )
        logger.info(
            "Tracked update for %s: %s...",
            event_id,
            description[:50] if len(str(description)) > 50 else description,
        )
        if self._state_manager.set_ai_description(event_id, description):
            event = self._state_manager.get_event(event_id)
            if event:
                if event.folder_path:
                    self._file_manager.write_summary(event.folder_path, event)
                self._notifier.publish_notification(event, "described")

    def _handle_review(self, payload: dict) -> None:
        """Handle frigate/reviews: GenAI update (Phase 3)."""
        if self._config.get("AI_MODE") == "external_api":
            return
        event_type = payload.get("type")
        match event_type:
            case "update" | "end" | "genai":
                pass
            case _:
                logger.debug("Skipping review with type: %s", event_type)
                return
        review_data = payload.get("after", {}) or payload.get("before", {})
        data = review_data.get("data", {})
        detections = data.get("detections", [])
        severity = review_data.get("severity", "detection")
        genai = data.get("metadata") or data.get("genai") or {}
        if not should_suppress_review_debug_logs():
            logger.debug(
                "Processing review: type=%s, %s detections, severity=%s",
                event_type,
                len(detections),
                severity,
            )
        for event_id in detections:
            event = self._state_manager.get_event(event_id)
            if event and self._timeline_logger.folder_for_event(event):
                self._timeline_logger.log_mqtt(
                    self._timeline_logger.folder_for_event(event),
                    "frigate/reviews",
                    payload,
                    f"Review update (type={event_type})",
                )
            title = genai.get("title")
            description = genai.get("shortSummary") or genai.get("description")
            scene = genai.get("scene")
            threat_level = int(genai.get("potential_threat_level", 0))
            if title or description:
                logger.info(
                    "Review for %s: title=%s, threat_level=%s",
                    event_id,
                    title or "N/A",
                    threat_level,
                )
            else:
                if not should_suppress_review_debug_logs():
                    logger.debug(
                        "Review for %s: title=N/A, threat_level=%s",
                        event_id,
                        threat_level,
                    )
            if not title and not description:
                if not should_suppress_review_debug_logs():
                    logger.debug(
                        "Skipping finalization for %s: no GenAI data yet", event_id
                    )
                continue
            if self._state_manager.set_genai_metadata(
                event_id, title, description, severity, threat_level, scene=scene
            ):
                event = self._state_manager.get_event(event_id)
                if event:
                    self._consolidated_manager.set_final_from_frigate(
                        event_id,
                        title=title,
                        description=description,
                        threat_level=threat_level,
                    )
                    if event.folder_path:
                        event.summary_written = self._file_manager.write_summary(
                            event.folder_path, event
                        )
                        self._file_manager.write_metadata_json(
                            event.folder_path, event
                        )
                    ce = self._consolidated_manager.get_by_frigate_event(event_id)
                    if ce and ce.finalized_sent:
                        logger.debug(
                            "Suppressing finalized for %s (CE %s already sent)",
                            event_id,
                            ce.consolidated_id,
                        )
                    else:
                        if ce:
                            ce.finalized_sent = True
                            primary = self._state_manager.get_event(
                                ce.primary_event_id
                            )
                            media_folder = (
                                os.path.join(
                                    ce.folder_path,
                                    self._file_manager.sanitize_camera_name(
                                        ce.primary_camera or ""
                                    ),
                                )
                                if ce.primary_camera
                                else ce.folder_path
                            )
                            if primary:
                                ce.snapshot_downloaded = primary.snapshot_downloaded
                                ce.clip_downloaded = primary.clip_downloaded
                            notify_target = type(
                                "NotifyTarget",
                                (),
                                {
                                    "event_id": ce.consolidated_id,
                                    "camera": ce.camera,
                                    "label": ce.label,
                                    "folder_path": primary.folder_path
                                    if primary
                                    else media_folder,
                                    "created_at": ce.start_time,
                                    "end_time": ce.end_time,
                                    "phase": ce.phase,
                                    "genai_title": ce.final_title,
                                    "genai_description": ce.final_description,
                                    "ai_description": None,
                                    "review_summary": None,
                                    "threat_level": ce.final_threat_level,
                                    "severity": ce.severity,
                                    "snapshot_downloaded": ce.snapshot_downloaded,
                                    "clip_downloaded": ce.clip_downloaded,
                                    "image_url_override": getattr(
                                        primary, "image_url_override", None
                                    )
                                    if primary
                                    else None,
                                },
                            )()
                        else:
                            # CE already removed; build CE-shaped target so pipeline stays CE-only (e.g. 1-camera CE)
                            notify_target = type(
                                "NotifyTarget",
                                (),
                                {
                                    "event_id": event.event_id,
                                    "camera": event.camera,
                                    "label": event.label,
                                    "folder_path": event.folder_path or "",
                                    "created_at": event.created_at,
                                    "end_time": event.end_time or event.created_at,
                                    "phase": getattr(event, "phase", None),
                                    "genai_title": title,
                                    "genai_description": description,
                                    "ai_description": None,
                                    "review_summary": None,
                                    "threat_level": threat_level,
                                    "severity": getattr(event, "severity", None) or "detection",
                                    "snapshot_downloaded": event.snapshot_downloaded,
                                    "clip_downloaded": event.clip_downloaded,
                                    "image_url_override": getattr(event, "image_url_override", None),
                                },
                            )()
                        self._notifier.publish_notification(
                            notify_target,
                            "finalized",
                            tag_override=f"frigate_{ce.consolidated_id}" if ce else f"frigate_{event.event_id}",
                        )
