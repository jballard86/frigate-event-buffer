"""
Event Lifecycle Service - Handles event creation, termination, and consolidation lifecycle.
"""

import time
import logging
import threading
import os
from typing import Optional

from frigate_buffer.models import EventState, ConsolidatedEvent, _is_no_concerns

logger = logging.getLogger('frigate-buffer')


class EventLifecycleService:
    """Service managing the lifecycle of events (new, end, consolidate, close)."""

    def __init__(self, config, state_manager, file_manager, consolidated_manager,
                 video_service, download_service, notifier, timeline_logger):
        self.config = config
        self.state_manager = state_manager
        self.file_manager = file_manager
        self.consolidated_manager = consolidated_manager
        self.video_service = video_service
        self.download_service = download_service
        self.notifier = notifier
        self.timeline_logger = timeline_logger

        self.last_cleanup_time: Optional[float] = None
        self.last_cleanup_deleted: int = 0

    def handle_event_new(self, event_id: str, camera: str, label: str,
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
            self.timeline_logger.log_mqtt(
                ce.folder_path, "frigate/events",
                mqtt_payload, f"Event {mqtt_type} (from Frigate)"
            )

        if not is_new:
            logger.info(f"Event {event_id} grouped into consolidated event {ce.consolidated_id}, suppressing duplicate notification")
            def _download_grouped_snapshot():
                self.download_service.download_snapshot(event_id, camera_folder)
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
                event.snapshot_downloaded = self.download_service.download_snapshot(
                    event.event_id, event.folder_path
                )
                if event.snapshot_downloaded:
                    self.notifier.publish_notification(event, "snapshot_ready", tag_override=tag_override)
        except Exception as e:
            logger.error(f"Error in initial notification flow for {event.event_id}: {e}")

    def process_event_end(self, event: EventState):
        """Background processing when event ends. For consolidated events, defers clip export to CE close."""
        try:
            if event.has_snapshot:
                event.snapshot_downloaded = self.download_service.download_snapshot(
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
                self.file_manager.write_summary(self.timeline_logger.folder_for_event(event) or event.folder_path, event)

                self.run_cleanup()
                return

            if event.has_clip and event.folder_path:
                export_before = self.config.get('EXPORT_BUFFER_BEFORE', 5)
                export_after = self.config.get('EXPORT_BUFFER_AFTER', 30)
                start_ts = int(event.created_at - export_before)
                end_ts = int((event.end_time or event.created_at) + export_after)

                self.timeline_logger.log_frigate_api(
                    self.timeline_logger.folder_for_event(event), 'out',
                    'Clip export request (to Frigate API)',
                    {
                        'url': f"{self.config.get('FRIGATE_URL', '')}/api/export/{event.camera}/start/{start_ts}/end/{end_ts}",
                        'params': {'camera': event.camera, 'start': start_ts, 'end': end_ts},
                    },
                )

                result = self.download_service.export_and_transcode_clip(
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
                self.timeline_logger.log_frigate_api(
                    self.timeline_logger.folder_for_event(event), 'in',
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

        export_before = self.config.get('EXPORT_BUFFER_BEFORE', 5)
        export_after = self.config.get('EXPORT_BUFFER_AFTER', 30)
        padding_before = self.config.get('SUMMARY_PADDING_BEFORE', 15)
        padding_after = self.config.get('SUMMARY_PADDING_AFTER', 15)
        camera_events = {}
        for fid in ce.frigate_event_ids:
            event = self.state_manager.get_event(fid)
            if event:
                if event.camera not in camera_events:
                    camera_events[event.camera] = []
                camera_events[event.camera].append(event)
            else:
                logger.warning(f"Event {fid} in consolidated event {ce_id} not found in state manager")

        first_clip_path = None
        primary_cam = ce.primary_camera or (ce.cameras[0] if ce.cameras else None)

        # Calculate global start/end times across all cameras for unified export duration
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

        for camera, events in camera_events.items():
            camera_folder = self.file_manager.ensure_consolidated_camera_folder(ce.folder_path, camera)

            def get_duration(e):
                end = e.end_time if e.end_time else (e.created_at + export_after)
                return end - e.created_at

            representative_event = max(events, key=get_duration)
            rep_id = representative_event.event_id

            # For timeline logging
            start_ts = int(global_min_start - export_before)
            end_ts = int(global_max_end + export_after)

            self.timeline_logger.log_frigate_api(
                ce.folder_path, 'out',
                f'Clip export for {camera} (CE close)',
                {'url': f"{self.config.get('FRIGATE_URL', '')}/api/export/{camera}/start/{start_ts}/end/{end_ts}",
                 'representative_id': rep_id}
            )
            result = self.download_service.export_and_transcode_clip(
                rep_id, camera_folder, camera,
                global_min_start, global_max_end,
                export_before, export_after
            )
            ok = result.get("success", False)
            timeline_data = {"success": ok, "frigate_response": result.get("frigate_response")}
            if "fallback" in result:
                timeline_data["fallback"] = result["fallback"]
            self.timeline_logger.log_frigate_api(ce.folder_path, 'in', f'Clip export response for {camera}', timeline_data)
            if ok and first_clip_path is None:
                first_clip_path = os.path.join(camera_folder, 'clip.mp4')
        # Placeholder fallback: if all exports failed, try per-event clip for primary camera
        if first_clip_path is None and primary_cam and ce.primary_event_id:
            primary_folder = self.file_manager.ensure_consolidated_camera_folder(ce.folder_path, primary_cam)
            ok = self.download_service.download_and_transcode_clip(ce.primary_event_id, primary_folder)
            if ok:
                first_clip_path = os.path.join(primary_folder, 'clip.mp4')
                self.timeline_logger.log_frigate_api(ce.folder_path, 'in', 'Placeholder clip (events API fallback)', {'success': True})

        if first_clip_path:
            gif_path = os.path.join(ce.folder_path, 'notification.gif')
            if self.video_service.generate_gif_from_clip(first_clip_path, gif_path):
                ce.snapshot_downloaded = True
            ce.clip_downloaded = True

        padded_start = int(ce.start_time - padding_before)
        padded_end = int((ce.end_time_max or ce.last_activity_time) + padding_after)
        self.timeline_logger.log_frigate_api(
            ce.folder_path, 'out',
            'Review summarize (CE close)',
            {'url': f"{self.config.get('FRIGATE_URL', '')}/api/review/summarize/start/{padded_start}/end/{padded_end}"}
        )
        summary = self.download_service.fetch_review_summary(
            ce.start_time, ce.end_time_max or ce.last_activity_time,
            padding_before, padding_after
        )
        self.timeline_logger.log_frigate_api(
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

    def run_cleanup(self):
        """Cleanup old event folders based on retention policy."""
        active_ids = self.state_manager.get_active_event_ids()
        active_ce_folders = [ce.folder_name for ce in self.consolidated_manager.get_all()]
        deleted = self.file_manager.cleanup_old_events(active_ids, active_ce_folders)
        self.last_cleanup_time = time.time()
        self.last_cleanup_deleted = deleted
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old event folders")
        return deleted
