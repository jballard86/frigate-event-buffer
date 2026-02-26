"""
State-Aware Orchestrator - Main coordinator for Frigate Event Buffer.

Coordinates MQTT, file management, event consolidation, notifications, and the web server.
"""

import logging
import os
import threading
import time
from datetime import date, datetime, timedelta
from urllib.parse import urlparse, urlunparse

import schedule

from frigate_buffer.managers.consolidation import ConsolidatedEventManager
from frigate_buffer.managers.file import FileManager
from frigate_buffer.managers.state import EventStateManager
from frigate_buffer.managers.zone_filter import SmartZoneFilter
from frigate_buffer.models import EventPhase
from frigate_buffer.services.ai_analyzer import GeminiAnalysisService
from frigate_buffer.services.daily_reporter import DailyReporterService
from frigate_buffer.services.download import (
    DEFAULT_EVENTS_CLIP_TIMEOUT,
    DEFAULT_EXPORT_DOWNLOAD_TIMEOUT,
    DownloadService,
)
from frigate_buffer.services.frigate_export_watchdog import (
    run_once as export_watchdog_run_once,
)
from frigate_buffer.services.ha_storage_stats import StorageStatsAndHaHelper
from frigate_buffer.services.lifecycle import EventLifecycleService
from frigate_buffer.services.mqtt_client import MqttClientWrapper
from frigate_buffer.services.mqtt_handler import MqttMessageHandler
from frigate_buffer.services.notifications import (
    HomeAssistantMqttProvider,
    NotificationDispatcher,
    PushoverProvider,
)
from frigate_buffer.services.query import EventQueryService
from frigate_buffer.services.quick_title_service import QuickTitleService
from frigate_buffer.services.timeline import TimelineLogger
from frigate_buffer.services.video import VideoService

logger = logging.getLogger("frigate-buffer")


class StateAwareOrchestrator:
    """Main orchestrator coordinating all components."""

    def __init__(self, config: dict):
        self.config = config
        self._shutdown = False
        self._start_time = time.time()

        # Initialize components (file_manager first - needed by consolidated_manager)
        self.state_manager = EventStateManager()
        self.video_service = VideoService(
            config.get("FFMPEG_TIMEOUT", VideoService.DEFAULT_FFMPEG_TIMEOUT)
        )
        self._sidecar_generation_lock = threading.Lock()
        self.video_service.set_sidecar_app_lock(self._sidecar_generation_lock)
        self.download_service = DownloadService(
            config["FRIGATE_URL"],
            self.video_service,
            export_download_timeout=int(
                config.get("EXPORT_DOWNLOAD_TIMEOUT", DEFAULT_EXPORT_DOWNLOAD_TIMEOUT)
            ),
            events_clip_timeout=int(
                config.get("EVENTS_CLIP_TIMEOUT", DEFAULT_EVENTS_CLIP_TIMEOUT)
            ),
            config=config,
        )
        self.file_manager = FileManager(
            config["STORAGE_PATH"], config["RETENTION_DAYS"]
        )

        # Initialize ConsolidatedEventManager with None callback first to break circular dependency
        self.consolidated_manager = ConsolidatedEventManager(
            self.file_manager,
            event_gap_seconds=config.get("EVENT_GAP_SECONDS", 120),
            on_close_callback=None,
        )

        self.timeline_logger = TimelineLogger(
            self.file_manager, self.consolidated_manager
        )

        self._mqtt_handler = None  # Set after lifecycle_service so handler can be built

        self.mqtt_wrapper = MqttClientWrapper(
            broker=config["MQTT_BROKER"],
            port=config["MQTT_PORT"],
            username=config.get("MQTT_USER"),
            password=config.get("MQTT_PASSWORD"),
            on_message_callback=self._on_mqtt_message,
        )

        self.notifier = self._create_notifier()

        self.zone_filter = SmartZoneFilter(config)

        # AI analyzer (Gemini proxy) - only when enabled; returns result to orchestrator (no MQTT publish)
        gemini = config.get("GEMINI") or {}
        proxy_url = config.get("GEMINI_PROXY_URL") or gemini.get("proxy_url") or ""
        api_key = gemini.get("api_key") or os.getenv("GEMINI_API_KEY") or ""
        if gemini.get("enabled") and proxy_url and api_key:
            self.ai_analyzer = GeminiAnalysisService(config)
            max_concurrent = max(
                1, int(config.get("GEMINI_MAX_CONCURRENT_ANALYSES", 3))
            )
            self._gemini_analysis_semaphore = threading.Semaphore(max_concurrent)
        else:
            self.ai_analyzer = None
            self._gemini_analysis_semaphore = None

        on_ce_ready = None
        if self.ai_analyzer and config.get("AI_MODE") == "external_api":

            def _on_ce_ready_for_analysis(
                ce_id: str, ce_folder_path: str, ce_start_time: float, ce_info: dict
            ):
                def _run():
                    if not self.ai_analyzer or not self._gemini_analysis_semaphore:
                        return
                    self._gemini_analysis_semaphore.acquire()
                    try:
                        result = self.ai_analyzer.analyze_multi_clip_ce(
                            ce_id,
                            ce_folder_path,
                            ce_start_time,
                            primary_camera=ce_info.get("primary_camera"),
                        )
                        if result:
                            self._handle_ce_analysis_result(
                                ce_id, ce_folder_path, result, ce_info
                            )
                    finally:
                        self._gemini_analysis_semaphore.release()

                threading.Thread(target=_run, daemon=True).start()

            on_ce_ready = _on_ce_ready_for_analysis

        on_quick_title = None
        if (
            self.ai_analyzer
            and config.get("QUICK_TITLE_ENABLED", True)
            and config.get("AI_MODE") == "external_api"
        ):
            quick_title_svc = QuickTitleService(
                config,
                self.state_manager,
                self.file_manager,
                self.consolidated_manager,
                self.video_service,
                self.ai_analyzer,
                self.notifier,
            )
            on_quick_title = quick_title_svc.run_quick_title

        self.lifecycle_service = EventLifecycleService(
            config,
            self.state_manager,
            self.file_manager,
            self.consolidated_manager,
            self.video_service,
            self.download_service,
            self.notifier,
            self.timeline_logger,
            on_ce_ready_for_analysis=on_ce_ready,
            on_quick_title_trigger=on_quick_title,
        )

        # Update callback after lifecycle service creation
        self.consolidated_manager.on_close_callback = (
            self.lifecycle_service.finalize_consolidated_event
        )

        # MQTT handler: parses and dispatches messages; wired so wrapper calls it
        self._mqtt_handler = MqttMessageHandler(
            config,
            self.state_manager,
            self.zone_filter,
            self.lifecycle_service,
            self.timeline_logger,
            self.notifier,
            self.file_manager,
            self.consolidated_manager,
            self.download_service,
        )

        # Daily reporter (AI report from analysis_result.json) - only when AI analyzer enabled
        if self.ai_analyzer:
            self.daily_reporter = DailyReporterService(
                config, config["STORAGE_PATH"], self.ai_analyzer
            )
        else:
            self.daily_reporter = None

        # Query service for event lists (shared so test_routes can evict test_events cache)
        self.query_service = EventQueryService(config["STORAGE_PATH"])

        # Flask app (lazy import to avoid circular deps)
        from frigate_buffer.web.server import create_app

        self.flask_app = create_app(self)

        # Scheduler thread
        self._scheduler_thread = None

        # Request counter for periodic logging
        self._request_count = 0
        self._request_count_lock = threading.Lock()

        # HA state fetch and storage-stats cache (stats page and scheduler)
        self.stats_helper = StorageStatsAndHaHelper(config)

        # Last cleanup tracking delegated to lifecycle service via properties

    def _on_mqtt_message(self, client, userdata, message):
        """Dispatch MQTT message to handler when it is set (avoids None on_message call)."""
        if self._mqtt_handler is not None:
            self._mqtt_handler.on_message(client, userdata, message)

    def _create_notifier(self) -> NotificationDispatcher:
        """Build notification dispatcher with providers enabled by config."""
        providers: list = []
        if self.config.get("NOTIFICATIONS_HOME_ASSISTANT_ENABLED", True):
            providers.append(
                HomeAssistantMqttProvider(
                    self.mqtt_wrapper.client,
                    self.config["BUFFER_IP"],
                    self.config["FLASK_PORT"],
                    self.config.get("FRIGATE_URL", ""),
                    storage_path=self.config["STORAGE_PATH"],
                )
            )
        po_config = self.config.get("pushover", {})
        if (
            po_config.get("enabled")
            and po_config.get("pushover_api_token")
            and po_config.get("pushover_user_key")
        ):
            player_base_url = (
                f"http://{self.config.get('BUFFER_IP')}:{self.config.get('FLASK_PORT')}"
            )
            providers.append(
                PushoverProvider(po_config, player_base_url=player_base_url)
            )
        return NotificationDispatcher(
            providers=providers,
            timeline_logger=self.timeline_logger,
        )

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

    def get_storage_stats(self) -> dict:
        """Return cached storage stats for the stats page. See StorageStatsAndHaHelper.get()."""
        return self.stats_helper.get()

    def fetch_ha_state(self, ha_url: str, ha_token: str, entity_id: str) -> str | None:
        """Fetch entity state from Home Assistant. Delegates to stats_helper."""
        return self.stats_helper.fetch_ha_state(ha_url, ha_token, entity_id)

    def _handle_analysis_result(self, event_id: str, result: dict):
        """Handle returned analysis from GeminiAnalysisService: update state, write files, POST to Frigate, notify HA."""
        title = result.get("title") or ""
        description = result.get("shortSummary") or result.get("description") or ""
        scene = result.get("scene") or ""
        threat_level = int(result.get("potential_threat_level", 0))
        severity = "detection"
        if not title and not description:
            logger.debug(
                f"Analysis result for {event_id} has no title/description, skipping finalization"
            )
            return
        if not self.state_manager.set_genai_metadata(
            event_id, title, description, severity, threat_level, scene=scene
        ):
            logger.warning(
                f"Cannot set GenAI metadata for {event_id} (event not in state?)"
            )
            # Still try to POST to Frigate so the result is not lost
            desc_for_frigate = description or scene or title
            if desc_for_frigate:
                self.download_service.post_event_description(event_id, desc_for_frigate)
            return
        event = self.state_manager.get_event(event_id)
        if event:
            self.consolidated_manager.set_final_from_frigate(
                event_id,
                title=title,
                description=description,
                threat_level=threat_level,
            )
            if event.folder_path:
                event.summary_written = self.file_manager.write_summary(
                    event.folder_path, event
                )
                self.file_manager.write_metadata_json(event.folder_path, event)
            desc_for_frigate = description or scene or title
            if desc_for_frigate:
                self.download_service.post_event_description(event_id, desc_for_frigate)
            ce = self.consolidated_manager.get_by_frigate_event(event_id)
            if ce and ce.finalized_sent:
                logger.debug(
                    f"Suppressing finalized for {event_id} (CE {ce.consolidated_id} already sent)"
                )
            else:
                if ce:
                    ce.finalized_sent = True
                    primary = (
                        self.state_manager.get_event(ce.primary_event_id)
                        if ce.primary_event_id is not None
                        else None
                    )
                    media_folder = (
                        os.path.join(
                            ce.folder_path,
                            self.file_manager.sanitize_camera_name(
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
                    # CE already removed; build CE-shaped target (e.g. 1-camera CE)
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
                            "image_url_override": getattr(
                                event, "image_url_override", None
                            ),
                        },
                    )()
                self.notifier.publish_notification(
                    notify_target,
                    "finalized",
                    tag_override=f"frigate_{ce.consolidated_id}"
                    if ce
                    else f"frigate_{event.event_id}",
                )
        logger.info(
            f"Analysis result applied for {event_id}: title={title or 'N/A'}, threat_level={threat_level}"
        )
        if self.daily_reporter and event and event.folder_path:
            try:
                unix_ts = event.created_at or 0
                event_date = date.fromtimestamp(unix_ts)
                camera = os.path.basename(os.path.dirname(event.folder_path))
                self.daily_reporter.append_event_to_daily_aggregate(
                    event_date,
                    {
                        "title": title,
                        "scene": scene,
                        "confidence": result.get("confidence", 0),
                        "threat_level": threat_level,
                        "camera": camera,
                        "time": datetime.fromtimestamp(unix_ts).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "context": result.get("context", []),
                    },
                )
            except (ValueError, OSError) as e:
                logger.debug(
                    "Could not append to daily aggregate for %s: %s", event_id, e
                )

    def _handle_ce_analysis_result(
        self, ce_id: str, ce_folder_path: str, result: dict, ce_info: dict
    ):
        """Handle multi-clip CE analysis result: write summary/metadata to CE root, notify."""
        title = result.get("title") or ""
        description = result.get("shortSummary") or result.get("description") or ""
        scene = result.get("scene") or ""
        threat_level = int(result.get("potential_threat_level", 0))
        if not title and not description:
            logger.debug(
                "CE analysis result for %s has no title/description, skipping", ce_id
            )
            return
        self.consolidated_manager.set_final_from_ce_analysis(
            ce_id, title=title, description=description, threat_level=threat_level
        )
        label = ce_info.get("label", "unknown")
        self.file_manager.write_ce_summary(
            ce_folder_path,
            ce_id,
            title,
            description,
            scene=scene,
            threat_level=threat_level,
            label=label,
            start_time=ce_info.get("_start_time", 0),
        )
        self.file_manager.write_ce_metadata_json(
            ce_folder_path,
            ce_id,
            title,
            description,
            scene=scene,
            threat_level=threat_level,
            label=label,
            start_time=ce_info.get("_start_time", 0),
            end_time=ce_info.get("end_time"),
        )
        media_folder = ce_folder_path
        if ce_info.get("primary_camera"):
            media_folder = os.path.join(
                ce_folder_path,
                self.file_manager.sanitize_camera_name(ce_info["primary_camera"]),
            )
        notify_target = type(
            "NotifyTarget",
            (),
            {
                "event_id": ce_id,
                "camera": ce_info.get("camera", "events"),
                "label": label,
                "folder_path": media_folder,
                "created_at": ce_info.get("_start_time", 0),
                "end_time": ce_info.get("end_time"),
                "phase": EventPhase.FINALIZED,
                "genai_title": title,
                "genai_description": description,
                "ai_description": None,
                "review_summary": None,
                "threat_level": threat_level,
                "severity": "detection",
                "snapshot_downloaded": True,
                "clip_downloaded": True,
                "image_url_override": None,
            },
        )()
        self.notifier.publish_notification(
            notify_target, "finalized", tag_override=f"frigate_{ce_id}"
        )
        logger.info(
            "CE analysis result applied for %s: title=%s, threat_level=%s",
            ce_id,
            title or "N/A",
            threat_level,
        )
        if self.daily_reporter:
            try:
                unix_ts = ce_info.get("_start_time", 0) or 0
                event_date = date.fromtimestamp(unix_ts)
                camera = ce_info.get("primary_camera") or os.path.basename(
                    ce_folder_path
                )
                self.daily_reporter.append_event_to_daily_aggregate(
                    event_date,
                    {
                        "title": title,
                        "scene": scene,
                        "confidence": result.get("confidence", 0),
                        "threat_level": threat_level,
                        "camera": camera,
                        "time": datetime.fromtimestamp(unix_ts).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "context": result.get("context", []),
                    },
                )
            except (ValueError, OSError) as e:
                logger.debug(
                    "Could not append CE to daily aggregate for %s: %s", ce_id, e
                )

    def _run_scheduler(self):
        """Background thread for scheduled tasks."""
        cleanup_hours = self.config.get("CLEANUP_INTERVAL_HOURS", 1)
        schedule.every(cleanup_hours).hours.do(self._hourly_cleanup)
        schedule.every(5).minutes.do(self._log_request_stats)
        schedule.every(5).minutes.do(self._update_storage_stats)

        watchdog_minutes = self.config.get("EXPORT_WATCHDOG_INTERVAL_MINUTES", 2)
        schedule.every(watchdog_minutes).minutes.do(self._run_export_watchdog)
        logger.info(f"Scheduled export watchdog every {watchdog_minutes} minute(s)")

        report_hour = self.config.get("DAILY_REPORT_SCHEDULE_HOUR", 1)
        schedule.every().day.at(f"{report_hour:02d}:00").do(self._daily_report_job)
        logger.info(f"Scheduled daily report at {report_hour:02d}:00")
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
        logger.info(
            f"API stats (5m): {count} requests, {active} active events, MQTT {'connected' if self.mqtt_wrapper.mqtt_connected else 'disconnected'}"
        )

    def _update_storage_stats(self):
        """Update cached storage stats (background task). Delegates to stats_helper."""
        self.stats_helper.update(self.file_manager)

    def _daily_report_job(self):
        """Generate yesterday's AI daily report from analysis_result.json. Runs at configured hour (default 1am)."""
        yesterday = date.today() - timedelta(days=1)
        if self.daily_reporter is None:
            logger.debug("Daily reporter disabled (AI analyzer not enabled), skipping")
            return
        logger.info(f"Running daily report job for {yesterday}")
        if self.daily_reporter.generate_report(yesterday):
            logger.info("Daily report generated successfully")
        else:
            logger.warning("Daily report generation failed or no response from proxy")
        retention_days = self.config.get("DAILY_REPORT_RETENTION_DAYS", 90)
        deleted = self.daily_reporter.cleanup_old_reports(retention_days)
        if deleted > 0:
            logger.info("Cleaned up %d old daily report(s)", deleted)

    def _hourly_cleanup(self):
        """Hourly cleanup task. Delegates to lifecycle service."""
        self.lifecycle_service.run_cleanup()

    def _run_export_watchdog(self):
        """Check event folders for completed exports, remove from Frigate, verify download links."""
        try:
            export_watchdog_run_once(self.config)
        except Exception as e:
            logger.exception(f"Export watchdog error: {e}")

    def start(self):
        """Start all components."""
        logger.info("=" * 60)
        logger.info("Starting State-Aware Orchestrator")
        logger.info("=" * 60)
        logger.info(
            f"MQTT Broker: {self.config['MQTT_BROKER']}:{self.config['MQTT_PORT']}"
        )

        frigate_url = self.config["FRIGATE_URL"]
        try:
            parsed = urlparse(frigate_url)
            if parsed.password:
                safe_netloc = f"{parsed.username or ''}:***@{parsed.hostname}"
                if parsed.port:
                    safe_netloc += f":{parsed.port}"
                parsed = parsed._replace(netloc=safe_netloc)
            logger.info(f"Frigate URL: {urlunparse(parsed)}")
        except Exception:
            logger.info("Frigate URL: (hidden)")

        logger.info(f"Storage Path: {self.config['STORAGE_PATH']}")
        logger.info(f"Retention: {self.config['RETENTION_DAYS']} days")
        logger.info(
            f"FFmpeg Timeout: {self.config.get('FFMPEG_TIMEOUT', VideoService.DEFAULT_FFMPEG_TIMEOUT)}s"
        )
        logger.info(f"Log Level: {self.config.get('LOG_LEVEL', 'INFO')}")

        camera_label_map = self.config.get("CAMERA_LABEL_MAP", {})
        if camera_label_map:
            logger.info("Camera/Label Configuration:")
            for camera, labels in camera_label_map.items():
                if labels:
                    logger.info(f"  {camera}: {labels}")
                else:
                    logger.info(f"  {camera}: ALL labels")
        else:
            logger.info(
                "Camera/Label Filtering: DISABLED (all cameras and labels allowed)"
            )

        event_filters = self.config.get("CAMERA_EVENT_FILTERS", {})
        if event_filters:
            logger.info("Smart Zone Filtering:")
            for camera, filt in event_filters.items():
                zones = filt.get("tracked_zones") or []
                exc = filt.get("exceptions") or []
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
            target=self._run_scheduler, daemon=True
        )
        self._scheduler_thread.start()

        # Update storage stats immediately (background)
        threading.Thread(target=self._update_storage_stats, daemon=True).start()

        logger.info(f"Starting Flask on port {self.config['FLASK_PORT']}...")
        self.flask_app.run(
            host="0.0.0.0", port=self.config["FLASK_PORT"], threaded=True
        )

    def stop(self):
        """Graceful shutdown."""
        logger.info("Shutting down orchestrator...")
        self._shutdown = True
        self.notifier.stop_queue_processor()
        self.mqtt_wrapper.stop()
