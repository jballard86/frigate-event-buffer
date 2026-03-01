"""
Event Lifecycle Service - event creation, termination, and consolidation lifecycle.
"""

import logging
import os
import threading
import time
from collections.abc import Callable

from frigate_buffer.models import (
    EventPhase,
    EventState,
    _is_no_concerns,
)

logger = logging.getLogger("frigate-buffer")


class EventLifecycleService:
    """Service managing the lifecycle of events (new, end, consolidate, close)."""

    def __init__(
        self,
        config,
        state_manager,
        file_manager,
        consolidated_manager,
        video_service,
        download_service,
        notifier,
        timeline_logger,
        on_ce_ready_for_analysis: Callable[[str, str, float, dict], None] | None = None,
        on_quick_title_trigger: Callable[..., None] | None = None,
    ):
        self.config = config
        self.state_manager = state_manager
        self.file_manager = file_manager
        self.consolidated_manager = consolidated_manager
        self.video_service = video_service
        self.download_service = download_service
        self.notifier = notifier
        self.timeline_logger = timeline_logger
        self.on_ce_ready_for_analysis = on_ce_ready_for_analysis
        self.on_quick_title_trigger = on_quick_title_trigger

        self.last_cleanup_time: float | None = None
        self.last_cleanup_deleted: int = 0

    def handle_event_new(
        self,
        event_id: str,
        camera: str,
        label: str,
        start_time: float,
        mqtt_payload: dict | None = None,
    ):
        """Handle new event detection (Phase 1). Uses events/{ce_id}/{camera}/."""
        logger.info(f"New event: {event_id} - {label} on {camera}")

        event = self.state_manager.create_event(event_id, camera, label, start_time)

        ce, is_new, camera_folder = self.consolidated_manager.get_or_create(
            event_id, camera, label, start_time
        )
        event.folder_path = camera_folder

        if mqtt_payload:
            mqtt_type = mqtt_payload.get("type", "new")
            self.timeline_logger.log_mqtt(
                ce.folder_path,
                "frigate/events",
                mqtt_payload,
                f"Event {mqtt_type} (from Frigate)",
            )

        if not is_new:
            logger.info(
                "Event %s grouped into CE %s, suppressing duplicate notification",
                event_id,
                ce.consolidated_id,
            )

            def _download_grouped_snapshot():
                self.download_service.download_snapshot(event_id, camera_folder)

            threading.Thread(target=_download_grouped_snapshot, daemon=True).start()
            return

        # Phase 1: canned title and live frame for initial alert
        camera_display = camera.replace("_", " ").title()
        event.genai_title = f"Motion Detected on {camera_display}"
        if event.folder_path:
            self.file_manager.write_metadata_json(event.folder_path, event)
        buffer_base = f"http://{self.config['BUFFER_IP']}:{self.config['FLASK_PORT']}"
        event.image_url_override = f"{buffer_base}/api/cameras/{camera}/latest.jpg"

        ce_tag = f"frigate_{ce.consolidated_id}" if ce else None
        threading.Thread(
            target=self._send_initial_notification, args=(event, ce_tag), daemon=True
        ).start()

        # Quick-title: after delay, run AI title pipeline (single image from latest.jpg)
        if self.on_quick_title_trigger and self.config.get("QUICK_TITLE_ENABLED", True):
            delay_sec = max(0, int(self.config.get("QUICK_TITLE_DELAY_SECONDS", 4)))
            if delay_sec > 0:
                threading.Thread(
                    target=self._run_quick_title_after_delay,
                    args=(
                        event_id,
                        camera,
                        label,
                        ce.consolidated_id,
                        camera_folder,
                        ce_tag,
                    ),
                    daemon=True,
                ).start()

    def _run_quick_title_after_delay(
        self,
        event_id: str,
        camera: str,
        label: str,
        ce_id: str,
        camera_folder_path: str,
        tag_override: str | None,
    ) -> None:
        """Sleep then invoke orchestrator callback for quick AI title (non-blocking)."""
        delay_sec = max(0, int(self.config.get("QUICK_TITLE_DELAY_SECONDS", 4)))
        if delay_sec > 0:
            time.sleep(delay_sec)
        try:
            if self.on_quick_title_trigger:
                self.on_quick_title_trigger(
                    event_id=event_id,
                    camera=camera,
                    label=label,
                    ce_id=ce_id,
                    camera_folder_path=camera_folder_path,
                    tag_override=tag_override,
                )
        except Exception as e:
            logger.error(f"Quick title trigger failed for {event_id}: {e}")

    def _send_initial_notification(
        self, event: EventState, tag_override: str | None = None
    ):
        """Send initial notification with canned title and live frame (latest.jpg)."""
        try:
            self.notifier.publish_notification(event, "new", tag_override=tag_override)
        except Exception as e:
            logger.error("Error in initial notification for %s: %s", event.event_id, e)

    def _discard_short_event(self, event: EventState) -> None:
        """Discard event under minimum_event_seconds: delete data, remove from state/CE,
        publish discarded notification so HA can clear the phone notification."""
        logger.info(
            "Discarding short event %s (under minimum_event_seconds)", event.event_id
        )
        if event.folder_path:
            self.file_manager.delete_event_folder(event.folder_path)
        ce = self.consolidated_manager.get_by_frigate_event(event.event_id)
        if ce:
            ce_folder = self.consolidated_manager.remove_event_from_ce(event.event_id)
            if ce_folder:
                self.file_manager.delete_event_folder(ce_folder)
        self.state_manager.remove_event(event.event_id)
        discard_target = type(
            "NotifyTarget",
            (),
            {
                "event_id": event.event_id,
                "camera": event.camera,
                "label": event.label,
                "created_at": event.created_at,
                "phase": EventPhase.NEW,
                "threat_level": 0,
                "clip_downloaded": False,
                "snapshot_downloaded": False,
                "folder_path": None,
                "genai_title": None,
                "genai_description": None,
                "review_summary": None,
                "ai_description": None,
                "image_url_override": None,
            },
        )()
        self.notifier.publish_notification(
            discard_target,
            "discarded",
            message="Event discarded (under minimum duration)",
            tag_override=f"frigate_{event.event_id}",
        )
        self.notifier.mark_last_event_ended()

    def process_event_end(self, event: EventState):
        """When event ends, for CEs defers clip export to CE close."""
        try:
            min_sec = self.config.get("MINIMUM_EVENT_SECONDS", 5)
            max_sec = self.config.get("MAX_EVENT_LENGTH_SECONDS", 120)
            end_ts = event.end_time or event.created_at
            duration = end_ts - event.created_at
            if duration < min_sec:
                self._discard_short_event(event)
                return
            if duration >= max_sec:
                logger.info(
                    "Canceled event %s (duration >= max_event_length_seconds)",
                    event.event_id,
                )
                if event.folder_path:
                    self.file_manager.write_canceled_summary(event.folder_path)
                    self.notifier.publish_notification(
                        event,
                        "canceled",
                        message="Event canceled see event viewer for details",
                        tag_override=f"frigate_{event.event_id}",
                    )
                    try:
                        new_path = self.file_manager.rename_event_folder(
                            event.folder_path
                        )
                        event.folder_path = new_path
                    except ValueError:
                        pass
                self.notifier.mark_last_event_ended()
                self.run_cleanup()
                return

            if event.has_snapshot:
                event.snapshot_downloaded = self.download_service.download_snapshot(
                    event.event_id, event.folder_path
                )

            ce = self.consolidated_manager.get_by_frigate_event(event.event_id)
            if ce:
                self.consolidated_manager.update_activity(
                    event.event_id,
                    activity_time=time.time(),
                    end_time=event.end_time or event.created_at,
                )
                # All CEs use event_gap_seconds for close
                self.consolidated_manager.schedule_close_timer(
                    ce.consolidated_id, delay_seconds=None
                )
                summary_folder = (
                    self.timeline_logger.folder_for_event(event) or event.folder_path
                )
                self.file_manager.write_summary(summary_folder, event)

                self.run_cleanup()
                return

            # Event not in CE: invariant violation (all events go through CE pipeline)
            self.run_cleanup()

        except Exception as e:
            logger.exception(f"Error processing event end: {e}")

    def finalize_consolidated_event(self, ce_id: str):
        """Called to close CE. Export clips, fetch summary, send notifications."""
        # Attempt to mark as closing to prevent new additions
        if not self.consolidated_manager.mark_closing(ce_id):
            return

        with self.consolidated_manager._lock:
            ce = self.consolidated_manager._events.get(ce_id)

        if not ce:
            return

        export_before = self.config.get("EXPORT_BUFFER_BEFORE", 5)
        export_after = self.config.get("EXPORT_BUFFER_AFTER", 30)
        padding_before = self.config.get("SUMMARY_PADDING_BEFORE", 15)
        padding_after = self.config.get("SUMMARY_PADDING_AFTER", 15)
        camera_events = {}
        for fid in ce.frigate_event_ids:
            event = self.state_manager.get_event(fid)
            if event:
                if event.camera not in camera_events:
                    camera_events[event.camera] = []
                camera_events[event.camera].append(event)
            else:
                logger.warning(
                    "Event %s in CE %s not found in state manager", fid, ce_id
                )

        first_clip_path = None
        primary_cam = ce.primary_camera or (ce.cameras[0] if ce.cameras else None)

        # Global start/end times across all cameras for unified export duration
        all_events = []
        for events in camera_events.values():
            all_events.extend(events)

        if all_events:
            global_start_times = [e.created_at for e in all_events]
            global_min_start = min(global_start_times)

            global_end_times = []
            for e in all_events:
                if e.end_time:
                    global_end_times.append(e.end_time)
                else:
                    # Fallback if end_time is missing (should be rare at close time)
                    global_end_times.append(e.created_at + export_after)
            global_max_end = max(global_end_times)
        else:
            global_min_start = ce.start_time
            global_max_end = ce.end_time_max or ce.last_activity_time

        def get_duration(e):
            end = e.end_time if e.end_time else (e.created_at + export_after)
            return end - e.created_at

        max_sec = self.config.get("MAX_EVENT_LENGTH_SECONDS", 120)
        if any(get_duration(e) >= max_sec for e in all_events):
            logger.info(
                "Canceled consolidated event %s (max event length exceeded)", ce_id
            )
            self.file_manager.write_canceled_summary(ce.folder_path)
            try:
                new_path = self.file_manager.rename_event_folder(ce.folder_path)
                ce.folder_path = new_path
                ce.folder_name = os.path.basename(new_path)
            except ValueError:
                pass
            primary_cam = ce.primary_camera or (ce.cameras[0] if ce.cameras else None)
            if primary_cam:
                media_folder = os.path.join(
                    ce.folder_path,
                    self.file_manager.sanitize_camera_name(primary_cam),
                )
            else:
                media_folder = ce.folder_path
            notify_target = type(
                "NotifyTarget",
                (),
                {
                    "event_id": ce.consolidated_id,
                    "camera": ce.camera or "events",
                    "label": ce.label or "unknown",
                    "folder_path": media_folder,
                    "created_at": ce.start_time,
                    "end_time": ce.end_time,
                    "phase": ce.phase,
                    "genai_title": None,
                    "genai_description": None,
                    "ai_description": None,
                    "review_summary": None,
                    "threat_level": 0,
                    "severity": None,
                    "snapshot_downloaded": False,
                    "clip_downloaded": False,
                    "cameras": ce.cameras,
                },
            )()
            self.notifier.publish_notification(
                notify_target,
                "canceled",
                message="Event canceled see event viewer for details",
                tag_override=f"frigate_{ce.consolidated_id}",
            )
            for fid in ce.frigate_event_ids:
                self.state_manager.remove_event(fid)
            self.consolidated_manager.remove(ce_id)
            self.notifier.mark_last_event_ended()
            logger.info("Consolidated event %s closed (canceled)", ce_id)
            return

        start_ts = int(global_min_start - export_before)
        end_ts = int(global_max_end + export_after)

        # Export each camera clip and generate detection sidecar (all CEs)
        clips_for_sidecar: list[tuple[str, str, str]] = []
        for camera, events in camera_events.items():
            camera_folder = self.file_manager.ensure_consolidated_camera_folder(
                ce.folder_path, camera
            )
            representative_event = max(events, key=get_duration)
            rep_id = representative_event.event_id
            self.timeline_logger.log_frigate_api(
                ce.folder_path,
                "out",
                f"Clip export for {camera} (CE close)",
                {
                    "url": "{}/api/export/{}/start/{}/end/{}".format(
                        self.config.get("FRIGATE_URL", ""),
                        camera,
                        start_ts,
                        end_ts,
                    ),
                    "representative_id": rep_id,
                },
            )
            result = self.download_service.export_and_download_clip(
                rep_id,
                camera_folder,
                camera,
                global_min_start,
                global_max_end,
                export_before,
                export_after,
            )
            ok = result.get("success", False)
            clip_path = result.get("clip_path")
            if ok and clip_path:
                if first_clip_path is None:
                    first_clip_path = clip_path
                sidecar_path = os.path.join(camera_folder, "detection.json")
                clips_for_sidecar.append((camera, clip_path, sidecar_path))
            timeline_data = {
                "success": ok,
                "frigate_response": result.get("frigate_response"),
            }
            if "fallback" in result:
                timeline_data["fallback"] = result["fallback"]
            self.timeline_logger.log_frigate_api(
                ce.folder_path,
                "in",
                f"Clip export response for {camera}",
                timeline_data,
            )

        if clips_for_sidecar:
            self.video_service.generate_detection_sidecars_for_cameras(
                clips_for_sidecar, self.config
            )

        # Placeholder fallback: if all exports failed, try per-event clip for primary
        if first_clip_path is None and primary_cam and ce.primary_event_id:
            primary_folder = self.file_manager.ensure_consolidated_camera_folder(
                ce.folder_path, primary_cam
            )
            ok = self.download_service.download_and_transcode_clip(
                ce.primary_event_id, primary_folder, primary_cam
            )
            if ok:
                from frigate_buffer.services.query import resolve_clip_in_folder

                clip_basename = resolve_clip_in_folder(primary_folder)
                first_clip_path = (
                    os.path.join(primary_folder, clip_basename)
                    if clip_basename
                    else None
                )
                if first_clip_path:
                    self.timeline_logger.log_frigate_api(
                        ce.folder_path,
                        "in",
                        "Placeholder clip (events API fallback)",
                        {"success": True},
                    )

        # All CEs: run target-centric analysis once via on_ce_ready_for_analysis
        if (
            self.on_ce_ready_for_analysis
            and self.config.get("GEMINI", {}).get("enabled")
            and first_clip_path
        ):
            self.on_ce_ready_for_analysis(
                ce_id,
                ce.folder_path,
                ce.start_time,
                {
                    "camera": ce.camera,
                    "label": ce.label,
                    "end_time": ce.end_time,
                    "primary_camera": ce.primary_camera,
                    "cameras": ce.cameras,
                    "_start_time": ce.start_time,
                },
            )

        compilation_path = None
        if first_clip_path:
            # Trigger video compilation
            try:
                from frigate_buffer.services.video_compilation import compile_ce_video

                compilation_path = compile_ce_video(
                    ce.folder_path,
                    float(end_ts - start_ts),
                    self.config,
                    primary_cam,
                )
            except Exception as e:
                logger.error("Error executing compilation hook for %s: %s", ce_id, e)

        gif_input_path = None
        if compilation_path and os.path.isfile(compilation_path):
            gif_input_path = compilation_path
        elif first_clip_path:
            gif_input_path = first_clip_path

        if gif_input_path:
            gif_path = os.path.join(ce.folder_path, "notification.gif")
            if self.video_service.generate_gif_from_clip(gif_input_path, gif_path):
                ce.snapshot_downloaded = True
            ce.clip_downloaded = True

        summary = None
        if self.config.get("AI_MODE") != "external_api":
            padded_start = int(ce.start_time - padding_before)
            padded_end = int((ce.end_time_max or ce.last_activity_time) + padding_after)
            self.timeline_logger.log_frigate_api(
                ce.folder_path,
                "out",
                "Review summarize (CE close)",
                {
                    "url": "{}/api/review/summarize/start/{}/end/{}".format(
                        self.config.get("FRIGATE_URL", ""),
                        padded_start,
                        padded_end,
                    )
                },
            )
            summary = self.download_service.fetch_review_summary(
                ce.start_time,
                ce.end_time_max or ce.last_activity_time,
                padding_before,
                padding_after,
            )
            self.timeline_logger.log_frigate_api(
                ce.folder_path,
                "in",
                "Review summarize response",
                {"response": summary or "(empty or error)"},
            )
            if summary:
                self.file_manager.write_review_summary(ce.folder_path, summary)

        primary_cam = ce.primary_camera or (ce.cameras[0] if ce.cameras else None)
        if primary_cam:
            media_folder = os.path.join(
                ce.folder_path,
                self.file_manager.sanitize_camera_name(primary_cam),
            )
        else:
            media_folder = ce.folder_path
        buf_base = f"http://{self.config['BUFFER_IP']}:{self.config['FLASK_PORT']}"
        gif_path_joined = os.path.join(ce.folder_path, "notification.gif")
        gif_url = (
            f"{buf_base}/files/events/{ce.folder_name}/notification.gif"
            if os.path.exists(gif_path_joined)
            else None
        )
        rel_media = os.path.relpath(
            media_folder, self.file_manager.storage_path
        ).replace("\\", "/")
        hosted_snapshot_path = f"/files/{rel_media}/snapshot_cropped.jpg"
        if os.path.exists(gif_path_joined):
            notification_gif_path = "/files/" + os.path.relpath(
                gif_path_joined, self.file_manager.storage_path
            ).replace("\\", "/")
        else:
            notification_gif_path = ""
        summary_basename = f"{ce.folder_name}_summary.mp4"
        summary_path = os.path.join(ce.folder_path, summary_basename)
        if os.path.isfile(summary_path):
            hosted_clip_path = "/files/" + os.path.relpath(
                summary_path, self.file_manager.storage_path
            ).replace("\\", "/")
        elif first_clip_path:
            hosted_clip_path = "/files/" + os.path.relpath(
                first_clip_path, self.file_manager.storage_path
            ).replace("\\", "/")
        else:
            hosted_clip_path = ""
        notify_target = type(
            "NotifyTarget",
            (),
            {
                "event_id": ce.consolidated_id,
                "camera": ce.camera,
                "label": ce.label,
                "folder_path": media_folder,
                "created_at": ce.start_time,
                "end_time": ce.end_time,
                "phase": ce.phase,
                "genai_title": ce.final_title,
                "genai_description": ce.final_description,
                "ai_description": None,
                "review_summary": summary,
                "threat_level": ce.final_threat_level,
                "severity": ce.severity,
                "snapshot_downloaded": ce.snapshot_downloaded,
                "clip_downloaded": ce.clip_downloaded,
                "image_url_override": gif_url,
                "notification_gif": notification_gif_path,
                "hosted_clip": hosted_clip_path,
                "hosted_snapshot": hosted_snapshot_path,
                "cameras": ce.cameras,
            },
        )()

        if not ce.clip_ready_sent:
            ce.clip_ready_sent = True
            self.notifier.publish_notification(notify_target, "clip_ready")
        if self.config.get("AI_MODE") != "external_api":
            if not ce.finalized_sent and (ce.final_title or ce.final_description):
                ce.finalized_sent = True
                self.notifier.publish_notification(notify_target, "finalized")
            if summary and not _is_no_concerns(summary):
                self.notifier.publish_notification(notify_target, "summarized")

        for fid in ce.frigate_event_ids:
            self.state_manager.remove_event(fid)
        self.consolidated_manager.remove(ce_id)
        self.notifier.mark_last_event_ended()
        logger.info("Consolidated event %s closed and cleaned up", ce_id)

    def run_cleanup(self):
        """Cleanup old event folders based on retention policy."""
        active_ids = self.state_manager.get_active_event_ids()
        active_ce_folders = list(self.consolidated_manager.get_active_ce_folders())
        deleted = self.file_manager.cleanup_old_events(active_ids, active_ce_folders)
        self.last_cleanup_time = time.time()
        self.last_cleanup_deleted = deleted
        if deleted > 0:
            logger.info("Cleaned up %s old event folders", deleted)
        return deleted
