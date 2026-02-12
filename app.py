#!/usr/bin/env python3
"""
Frigate Event Buffer - State-Aware Orchestrator

Listens to Frigate MQTT topics, tracks events through their lifecycle,
sends Ring-style notifications to Home Assistant, and manages rolling retention.

Configuration is loaded from config.yaml with environment variable overrides.
"""

import os
import re
import sys
import json
import time
import signal
import shutil
import logging
import threading
import subprocess
import uuid
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import timedelta, date, datetime
from typing import Dict, Optional, List

import yaml
import paho.mqtt.client as mqtt
import requests
import schedule
from flask import Flask, Response, send_from_directory, jsonify, render_template, request

# =============================================================================
# CONFIGURATION
# =============================================================================

# Early logger for config loading (will be reconfigured after config is loaded)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('frigate-buffer')

# Patterns that indicate "no concerns" from GenAI review summary (skip summarized notification)
NO_CONCERNS_PATTERNS = (
    "no concerns were found during this time period",
    "no concerns were found",
    "no concerns",
)


def _is_no_concerns(summary: str) -> bool:
    """Return True if the summary indicates no concerns (skip summarized notification)."""
    normalized = (summary or "").strip().lower()
    return any(p in normalized for p in NO_CONCERNS_PATTERNS)


def load_config() -> dict:
    """Load configuration from config.yaml merged with environment variables.

    Priority (highest to lowest):
    1. Environment variables
    2. config.yaml
    3. Default values

    Note: MQTT_BROKER, FRIGATE_URL, and BUFFER_IP are REQUIRED and must be
    provided via config.yaml or environment variables.
    """
    config = {
        # Network settings - NO DEFAULTS (required from config)
        'MQTT_BROKER': None,
        'MQTT_PORT': 1883,
        'FRIGATE_URL': None,
        'BUFFER_IP': None,
        'FLASK_PORT': 5055,
        'STORAGE_PATH': '/app/storage',

        # Settings defaults
        'RETENTION_DAYS': 3,
        'CLEANUP_INTERVAL_HOURS': 1,
        'FFMPEG_TIMEOUT': 60,
        'NOTIFICATION_DELAY': 2,
        'LOG_LEVEL': 'INFO',
        'SUMMARY_PADDING_BEFORE': 15,
        'SUMMARY_PADDING_AFTER': 15,
        'STATS_REFRESH_SECONDS': 60,
        'DAILY_REVIEW_RETENTION_DAYS': 90,
        'DAILY_REVIEW_SCHEDULE_HOUR': 1,
        'EVENT_GAP_SECONDS': 120,
        'EXPORT_BUFFER_BEFORE': 5,
        'EXPORT_BUFFER_AFTER': 30,

        # Filtering defaults (empty = allow all)
        'ALLOWED_CAMERAS': [],
        'ALLOWED_LABELS': [],
        'CAMERA_LABEL_MAP': {},
    }

    # Load from config.yaml if exists
    config_paths = ['/app/config.yaml', '/app/storage/config.yaml', './config.yaml', 'config.yaml']
    config_loaded = False

    for path in config_paths:
        if os.path.exists(path):
            try:
                logger.info(f"Loading config from {path}")
                with open(path, 'r') as f:
                    yaml_config = yaml.safe_load(f) or {}

                # Build camera-to-labels mapping from per-camera config
                if 'cameras' in yaml_config and isinstance(yaml_config['cameras'], list):
                    for cam in yaml_config['cameras']:
                        if isinstance(cam, dict) and 'name' in cam:
                            camera_name = cam['name']
                            labels = cam.get('labels', []) or []
                            config['CAMERA_LABEL_MAP'][camera_name] = labels

                    # Derive flat lists for status/logging
                    config['ALLOWED_CAMERAS'] = list(config['CAMERA_LABEL_MAP'].keys())
                    config['ALLOWED_LABELS'] = list(set(
                        label for labels in config['CAMERA_LABEL_MAP'].values() for label in labels if labels
                    ))

                if 'settings' in yaml_config:
                    settings = yaml_config['settings']
                    config['RETENTION_DAYS'] = settings.get('retention_days', config['RETENTION_DAYS'])
                    config['CLEANUP_INTERVAL_HOURS'] = settings.get('cleanup_interval_hours', config['CLEANUP_INTERVAL_HOURS'])
                    config['FFMPEG_TIMEOUT'] = settings.get('ffmpeg_timeout_seconds', config['FFMPEG_TIMEOUT'])
                    config['NOTIFICATION_DELAY'] = settings.get('notification_delay_seconds', config['NOTIFICATION_DELAY'])
                    config['LOG_LEVEL'] = settings.get('log_level', config['LOG_LEVEL'])
                    config['SUMMARY_PADDING_BEFORE'] = settings.get('summary_padding_before', config['SUMMARY_PADDING_BEFORE'])
                    config['SUMMARY_PADDING_AFTER'] = settings.get('summary_padding_after', config['SUMMARY_PADDING_AFTER'])
                    config['STATS_REFRESH_SECONDS'] = settings.get('stats_refresh_seconds', config['STATS_REFRESH_SECONDS'])
                    config['DAILY_REVIEW_RETENTION_DAYS'] = settings.get('daily_review_retention_days', config['DAILY_REVIEW_RETENTION_DAYS'])
                    config['DAILY_REVIEW_SCHEDULE_HOUR'] = settings.get('daily_review_schedule_hour', config['DAILY_REVIEW_SCHEDULE_HOUR'])
                    config['EVENT_GAP_SECONDS'] = settings.get('event_gap_seconds', config['EVENT_GAP_SECONDS'])
                    config['EXPORT_BUFFER_BEFORE'] = settings.get('export_buffer_before', config['EXPORT_BUFFER_BEFORE'])
                    config['EXPORT_BUFFER_AFTER'] = settings.get('export_buffer_after', config['EXPORT_BUFFER_AFTER'])

                if 'network' in yaml_config:
                    network = yaml_config['network']
                    config['MQTT_BROKER'] = network.get('mqtt_broker', config['MQTT_BROKER'])
                    config['MQTT_PORT'] = network.get('mqtt_port', config['MQTT_PORT'])
                    config['FRIGATE_URL'] = network.get('frigate_url', config['FRIGATE_URL'])
                    config['BUFFER_IP'] = network.get('buffer_ip') or network.get('ha_ip') or config['BUFFER_IP']
                    config['FLASK_PORT'] = network.get('flask_port', config['FLASK_PORT'])
                    config['STORAGE_PATH'] = network.get('storage_path', config['STORAGE_PATH'])

                config_loaded = True
                break

            except Exception as e:
                logger.error(f"Error loading config from {path}: {e}")

    if not config_loaded:
        logger.info("No config.yaml found, using defaults")

    # Environment variables override everything (for secrets/deployment)
    config['MQTT_BROKER'] = os.getenv('MQTT_BROKER') or config['MQTT_BROKER']
    config['MQTT_PORT'] = int(os.getenv('MQTT_PORT', str(config['MQTT_PORT'])))
    frigate_url = os.getenv('FRIGATE_URL') or config['FRIGATE_URL']
    config['FRIGATE_URL'] = frigate_url.rstrip('/') if frigate_url else None
    config['BUFFER_IP'] = os.getenv('BUFFER_IP') or os.getenv('HA_IP') or config['BUFFER_IP']
    config['FLASK_PORT'] = int(os.getenv('FLASK_PORT', str(config['FLASK_PORT'])))
    config['STORAGE_PATH'] = os.getenv('STORAGE_PATH', config['STORAGE_PATH'])
    config['RETENTION_DAYS'] = int(os.getenv('RETENTION_DAYS', str(config['RETENTION_DAYS'])))
    config['LOG_LEVEL'] = os.getenv('LOG_LEVEL', config['LOG_LEVEL'])
    config['STATS_REFRESH_SECONDS'] = int(os.getenv('STATS_REFRESH_SECONDS', str(config['STATS_REFRESH_SECONDS'])))
    config['DAILY_REVIEW_RETENTION_DAYS'] = int(os.getenv('DAILY_REVIEW_RETENTION_DAYS', str(config['DAILY_REVIEW_RETENTION_DAYS'])))
    config['DAILY_REVIEW_SCHEDULE_HOUR'] = int(os.getenv('DAILY_REVIEW_SCHEDULE_HOUR', str(config['DAILY_REVIEW_SCHEDULE_HOUR'])))
    config['EVENT_GAP_SECONDS'] = int(os.getenv('EVENT_GAP_SECONDS', str(config['EVENT_GAP_SECONDS'])))
    config['EXPORT_BUFFER_BEFORE'] = int(os.getenv('EXPORT_BUFFER_BEFORE', str(config['EXPORT_BUFFER_BEFORE'])))
    config['EXPORT_BUFFER_AFTER'] = int(os.getenv('EXPORT_BUFFER_AFTER', str(config['EXPORT_BUFFER_AFTER'])))

    # Validate required settings
    missing = []
    if not config['MQTT_BROKER']:
        missing.append('MQTT_BROKER (network.mqtt_broker)')
    if not config['FRIGATE_URL']:
        missing.append('FRIGATE_URL (network.frigate_url)')
    if not config['BUFFER_IP']:
        missing.append('BUFFER_IP (network.buffer_ip)')

    if missing:
        raise ValueError(
            f"Missing required configuration: {', '.join(missing)}. "
            f"Set these in config.yaml under 'network:' or as environment variables."
        )

    return config


# =============================================================================
# ERROR BUFFER (for stats dashboard)
# =============================================================================

class ErrorBuffer:
    """Thread-safe rotating buffer of recent ERROR/WARNING log records (max 10)."""

    def __init__(self, max_size: int = 10):
        self._entries: List[dict] = []
        self._max_size = max_size
        self._lock = threading.Lock()

    def append(self, timestamp: str, level: str, message: str) -> None:
        with self._lock:
            self._entries.append({
                "ts": timestamp,
                "level": level,
                "message": message[:500] if message else ""
            })
            if len(self._entries) > self._max_size:
                self._entries.pop(0)

    def get_all(self) -> List[dict]:
        with self._lock:
            return list(reversed(self._entries))


class ErrorBufferHandler(logging.Handler):
    """Logging handler that writes ERROR/WARNING to ErrorBuffer."""

    def __init__(self, buffer: ErrorBuffer):
        super().__init__(level=logging.WARNING)
        self._buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.created))
            self._buffer.append(ts, record.levelname, record.getMessage())
        except Exception:
            self.handleError(record)


error_buffer = ErrorBuffer(max_size=10)


def setup_logging(log_level: str):
    """Configure logging with the specified level."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Reconfigure the root logger
    logging.getLogger().setLevel(level)
    logger.setLevel(level)

    # Add error buffer handler for stats dashboard
    logger.addHandler(ErrorBufferHandler(error_buffer))

    # Suppress werkzeug per-request logging (floods logs with GET /events)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

    logger.info(f"Log level set to {log_level.upper()}")


# =============================================================================
# EVENT STATE MODELS
# =============================================================================

class EventPhase(Enum):
    """Tracks the lifecycle phase of a Frigate event."""
    NEW = auto()        # Phase 1: Initial detection from frigate/events type=new
    DESCRIBED = auto()  # Phase 2: AI description received from tracked_object_update
    FINALIZED = auto()  # Phase 3: GenAI metadata received from frigate/reviews
    SUMMARIZED = auto() # Phase 4: Review summary received from Frigate API


@dataclass
class EventState:
    """Represents the current state of a tracked Frigate event."""
    event_id: str
    camera: str
    label: str
    phase: EventPhase = EventPhase.NEW
    created_at: float = field(default_factory=time.time)

    # Phase 2 data (from tracked_object_update)
    ai_description: Optional[str] = None

    # Phase 3 data (from frigate/reviews)
    genai_title: Optional[str] = None
    genai_description: Optional[str] = None
    severity: Optional[str] = None
    threat_level: int = 0  # 0=normal, 1=suspicious, 2=critical

    # Phase 4 data (from review summary API)
    review_summary: Optional[str] = None

    # File management
    folder_path: Optional[str] = None
    clip_downloaded: bool = False
    snapshot_downloaded: bool = False
    summary_written: bool = False
    review_summary_written: bool = False

    # Event end tracking
    end_time: Optional[float] = None
    has_clip: bool = False
    has_snapshot: bool = False


# =============================================================================
# EVENT STATE MANAGER
# =============================================================================

class EventStateManager:
    """Thread-safe manager for active event states across multiple cameras."""

    def __init__(self):
        self._events: Dict[str, EventState] = {}
        self._lock = threading.RLock()

    def create_event(self, event_id: str, camera: str, label: str,
                     start_time: float) -> EventState:
        """Create a new event in NEW phase."""
        with self._lock:
            if event_id in self._events:
                logger.debug(f"Event {event_id} already exists, returning existing")
                return self._events[event_id]

            event = EventState(
                event_id=event_id,
                camera=camera,
                label=label,
                created_at=start_time or time.time()
            )
            self._events[event_id] = event
            logger.info(f"Created event state: {event_id} ({label} on {camera})")
            return event

    def get_event(self, event_id: str) -> Optional[EventState]:
        """Get event by ID (thread-safe read)."""
        with self._lock:
            return self._events.get(event_id)

    def set_ai_description(self, event_id: str, description: str) -> bool:
        """Set AI description and advance to DESCRIBED phase."""
        with self._lock:
            event = self._events.get(event_id)
            if event and event.phase == EventPhase.NEW:
                event.ai_description = description
                event.phase = EventPhase.DESCRIBED
                logger.info(f"Event {event_id} advanced to DESCRIBED phase")
                logger.debug(f"AI description: {description[:100]}..." if len(description) > 100 else f"AI description: {description}")
                return True
            elif event:
                # Update description even if already past NEW phase
                event.ai_description = description
                logger.debug(f"Updated AI description for {event_id} (already past NEW phase)")
                return True
            logger.debug(f"Cannot set AI description: event {event_id} not found")
            return False

    def set_genai_metadata(self, event_id: str, title: Optional[str],
                           description: Optional[str], severity: str,
                           threat_level: int = 0) -> bool:
        """Set GenAI review metadata and advance to FINALIZED phase."""
        with self._lock:
            event = self._events.get(event_id)
            if event:
                event.genai_title = title
                event.genai_description = description
                event.severity = severity
                event.threat_level = threat_level
                event.phase = EventPhase.FINALIZED
                logger.info(f"Event {event_id} advanced to FINALIZED (threat_level={threat_level})")
                logger.debug(f"GenAI title: {title}, severity: {severity}")
                return True
            logger.debug(f"Cannot set GenAI metadata: event {event_id} not found")
            return False

    def set_review_summary(self, event_id: str, summary: str) -> bool:
        """Set review summary and advance to SUMMARIZED phase."""
        with self._lock:
            event = self._events.get(event_id)
            if event:
                event.review_summary = summary
                event.phase = EventPhase.SUMMARIZED
                logger.info(f"Event {event_id} advanced to SUMMARIZED phase")
                return True
            logger.debug(f"Cannot set review summary: event {event_id} not found")
            return False

    def mark_event_ended(self, event_id: str, end_time: float,
                         has_clip: bool, has_snapshot: bool) -> Optional[EventState]:
        """Mark event as ended, returning the event for processing."""
        with self._lock:
            event = self._events.get(event_id)
            if event:
                event.end_time = end_time
                event.has_clip = has_clip
                event.has_snapshot = has_snapshot
                logger.debug(f"Event {event_id} marked ended (clip={has_clip}, snapshot={has_snapshot})")
            return event

    def remove_event(self, event_id: str) -> Optional[EventState]:
        """Remove event from active tracking."""
        with self._lock:
            removed = self._events.pop(event_id, None)
            if removed:
                logger.info(f"Removed event from tracking: {event_id}")
            return removed

    def get_active_event_ids(self) -> List[str]:
        """Get list of all active event IDs (for cleanup protection)."""
        with self._lock:
            return list(self._events.keys())

    def get_stats(self) -> dict:
        """Get statistics about active events."""
        with self._lock:
            by_phase = {phase.name: 0 for phase in EventPhase}
            by_camera = {}

            for event in self._events.values():
                by_phase[event.phase.name] += 1
                by_camera[event.camera] = by_camera.get(event.camera, 0) + 1

            return {
                "total_active": len(self._events),
                "by_phase": by_phase,
                "by_camera": by_camera
            }


def _generate_consolidated_id(start_ts: float) -> tuple:
    """Generate our internal consolidated event ID. Returns (full_id, folder_name)."""
    ts_int = int(start_ts)
    short_uuid = uuid.uuid4().hex[:8]
    full_id = f"ce_{ts_int}_{short_uuid}"
    folder_name = f"{ts_int}_{short_uuid}"
    return full_id, folder_name


@dataclass
class ConsolidatedEvent:
    """A consolidated event grouping multiple Frigate events (same real-world activity)."""

    consolidated_id: str
    folder_name: str
    folder_path: str
    start_time: float
    last_activity_time: float
    cameras: List[str] = field(default_factory=list)
    frigate_event_ids: List[str] = field(default_factory=list)

    # Best-so-far (never regress)
    best_title: Optional[str] = None
    best_description: Optional[str] = None
    best_threat_level: int = 0

    # Primary (first) Frigate event for immediate clip/snapshot
    primary_event_id: Optional[str] = None
    primary_camera: Optional[str] = None
    snapshot_downloaded: bool = False
    clip_downloaded: bool = False

    # Legacy-compat: expose as EventState-like for notifier
    @property
    def event_id(self) -> str:
        return self.consolidated_id

    @property
    def camera(self) -> str:
        return self.primary_camera or (self.cameras[0] if self.cameras else "unknown")

    @property
    def label(self) -> str:
        return "person"  # Aggregated; use best_description for display

    @property
    def created_at(self) -> float:
        return self.start_time

    @property
    def phase(self):
        return EventPhase.SUMMARIZED  # Simplified for notification

    @property
    def genai_title(self) -> Optional[str]:
        return self.best_title

    @property
    def genai_description(self) -> Optional[str]:
        return self.best_description

    @property
    def threat_level(self) -> int:
        return self.best_threat_level

    @property
    def severity(self) -> Optional[str]:
        return "detection"

    @property
    def review_summary(self) -> Optional[str]:
        return None  # Set when we get full-event summary

    @property
    def end_time(self) -> Optional[float]:
        return self.last_activity_time

    @property
    def has_clip(self) -> bool:
        return self.clip_downloaded

    @property
    def has_snapshot(self) -> bool:
        return self.snapshot_downloaded


class ConsolidatedEventManager:
    """Manages consolidated events (time-gap grouped)."""

    def __init__(self, event_gap_seconds: int = 120):
        self._events: Dict[str, ConsolidatedEvent] = {}
        self._frigate_to_ce: Dict[str, str] = {}  # frigate_event_id -> consolidated_id
        self._active_ce_id: Optional[str] = None  # Currently active (receiving new sub-events)
        self._lock = threading.RLock()
        self.event_gap_seconds = event_gap_seconds

    def get_or_create(
        self,
        event_id: str,
        camera: str,
        label: str,
        start_time: float,
        folder_path: str,
    ) -> tuple:
        """
        Get existing consolidated event or create new. Returns (ConsolidatedEvent, is_new).
        Groups by event_gap_seconds. For new events, folder_path must already exist.
        """
        now = time.time()
        with self._lock:
            if self._active_ce_id:
                ce = self._events.get(self._active_ce_id)
                if ce and (now - ce.last_activity_time) < self.event_gap_seconds:
                    ce.frigate_event_ids.append(event_id)
                    ce.last_activity_time = now
                    if camera not in ce.cameras:
                        ce.cameras.append(camera)
                    self._frigate_to_ce[event_id] = ce.consolidated_id
                    return ce, False

            # New consolidated event (folder_path provided by caller after create_consolidated_event_folder)
            full_id, folder_name = _generate_consolidated_id(start_time)
            ce = ConsolidatedEvent(
                consolidated_id=full_id,
                folder_name=folder_name,
                folder_path=folder_path,
                start_time=start_time,
                last_activity_time=now,
                cameras=[camera],
                frigate_event_ids=[event_id],
                primary_event_id=event_id,
                primary_camera=camera,
            )
            self._events[full_id] = ce
            self._frigate_to_ce[event_id] = full_id
            self._active_ce_id = full_id
            return ce, True

    def get_by_frigate_event(self, event_id: str) -> Optional[ConsolidatedEvent]:
        with self._lock:
            ce_id = self._frigate_to_ce.get(event_id)
            return self._events.get(ce_id) if ce_id else None

    def update_activity(self, event_id: str, activity_time: Optional[float] = None) -> None:
        with self._lock:
            ce_id = self._frigate_to_ce.get(event_id)
            if ce_id and ce_id in self._events:
                ce = self._events[ce_id]
                ce.last_activity_time = activity_time or time.time()

    def update_best(self, event_id: str, title: Optional[str] = None,
                    description: Optional[str] = None, threat_level: Optional[int] = None) -> None:
        with self._lock:
            ce = self.get_by_frigate_event(event_id)
            if ce:
                if title and (not ce.best_title or len(str(title)) > len(str(ce.best_title or ""))):
                    ce.best_title = title
                if description and (not ce.best_description or len(str(description)) > len(str(ce.best_description or ""))):
                    ce.best_description = description
                if threat_level is not None and threat_level > ce.best_threat_level:
                    ce.best_threat_level = threat_level

    def mark_inactive(self, consolidated_id: str) -> None:
        with self._lock:
            if self._active_ce_id == consolidated_id:
                self._active_ce_id = None

    def remove(self, consolidated_id: str) -> Optional[ConsolidatedEvent]:
        with self._lock:
            ce = self._events.pop(consolidated_id, None)
            if ce:
                for fid in ce.frigate_event_ids:
                    self._frigate_to_ce.pop(fid, None)
                if self._active_ce_id == consolidated_id:
                    self._active_ce_id = None
            return ce

    def get_active_consolidated_ids(self) -> List[str]:
        with self._lock:
            return list(self._events.keys())

    def get_all(self) -> List[ConsolidatedEvent]:
        with self._lock:
            return list(self._events.values())


# =============================================================================
# FILE MANAGER
# =============================================================================

class FileManager:
    """Handles file operations: folder creation, downloads, transcoding, cleanup."""

    def __init__(self, storage_path: str, frigate_url: str, retention_days: int,
                 ffmpeg_timeout: int = 60):
        self.storage_path = storage_path
        self.frigate_url = frigate_url
        self.retention_days = retention_days
        self.ffmpeg_timeout = ffmpeg_timeout

        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        logger.info(f"FileManager initialized: {storage_path}")
        logger.debug(f"FFmpeg timeout: {ffmpeg_timeout}s, Retention: {retention_days} days")

    def sanitize_camera_name(self, camera: str) -> str:
        """Sanitize camera name for filesystem use."""
        # Lowercase, replace spaces with underscores, remove special chars
        sanitized = camera.lower().replace(' ', '_')
        sanitized = re.sub(r'[^a-z0-9_]', '', sanitized)
        return sanitized or 'unknown'

    def create_event_folder(self, event_id: str, camera: str, timestamp: float) -> str:
        """Create folder for event: {camera}/{timestamp}_{event_id} (legacy)"""
        sanitized_camera = self.sanitize_camera_name(camera)
        folder_name = f"{int(timestamp)}_{event_id}"
        camera_path = os.path.join(self.storage_path, sanitized_camera)
        folder_path = os.path.join(camera_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"Created folder: {sanitized_camera}/{folder_name}")
        return folder_path

    def create_consolidated_event_folder(self, folder_name: str) -> str:
        """Create folder for consolidated event: events/{folder_name}"""
        events_dir = os.path.join(self.storage_path, "events")
        folder_path = os.path.join(events_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"Created consolidated folder: events/{folder_name}")
        return folder_path

    def download_snapshot(self, event_id: str, folder_path: str) -> bool:
        """Download snapshot from Frigate API."""
        url = f"{self.frigate_url}/api/events/{event_id}/snapshot.jpg"
        logger.debug(f"Downloading snapshot from {url}")

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            snapshot_path = os.path.join(folder_path, "snapshot.jpg")
            with open(snapshot_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"Downloaded snapshot for {event_id} ({len(response.content)} bytes)")
            return True
        except requests.exceptions.Timeout:
            logger.error(f"Timeout downloading snapshot for {event_id}")
            return False
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error downloading snapshot for {event_id}: {e}")
            return False
        except Exception as e:
            logger.exception(f"Failed to download snapshot for {event_id}: {e}")
            return False

    def _terminate_process_gracefully(self, process, event_id: str, timeout: float = 5.0):
        """Gracefully terminate a process: SIGTERM first, then SIGKILL if needed."""
        if process is None or process.poll() is not None:
            return  # Already dead

        logger.debug(f"Sending SIGTERM to FFmpeg for {event_id}")
        try:
            process.terminate()  # SIGTERM - allows graceful shutdown
        except OSError:
            return  # Process already gone

        try:
            process.wait(timeout=timeout)  # Wait for graceful exit
            logger.debug(f"FFmpeg for {event_id} terminated gracefully")
        except subprocess.TimeoutExpired:
            logger.warning(f"FFmpeg for {event_id} didn't respond to SIGTERM, sending SIGKILL")
            try:
                process.kill()  # SIGKILL - force kill
                process.wait()  # Reap zombie
            except OSError:
                pass  # Process already gone

    def download_and_transcode_clip(self, event_id: str, folder_path: str) -> bool:
        """Download clip from Frigate and transcode to H.264 with timeout protection."""
        temp_path = os.path.join(folder_path, "clip_original.mp4")
        final_path = os.path.join(folder_path, "clip.mp4")
        process = None

        try:
            # Download original clip (retry on HTTP 400 — Frigate may not have clip ready yet)
            url = f"{self.frigate_url}/api/events/{event_id}/clip.mp4"
            response = None

            for attempt in range(1, 4):
                logger.debug(f"Downloading clip from {url} (attempt {attempt}/3)")
                try:
                    response = requests.get(url, timeout=120, stream=True)
                    response.raise_for_status()
                    break
                except requests.exceptions.HTTPError:
                    if response is not None and response.status_code == 400 and attempt < 3:
                        logger.warning(f"Clip not ready for {event_id} (HTTP 400), retrying in 5s ({attempt}/3)")
                        time.sleep(5)
                        continue
                    raise

            bytes_downloaded = 0
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bytes_downloaded += len(chunk)

            logger.debug(f"Downloaded clip for {event_id} ({bytes_downloaded} bytes), starting transcode...")

            # Transcode with timeout protection
            command = [
                'ffmpeg', '-y',
                '-i', temp_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-c:a', 'aac',
                '-movflags', '+faststart',
                final_path
            ]

            logger.debug(f"FFmpeg command: {' '.join(command)}")

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            try:
                stdout, stderr = process.communicate(timeout=self.ffmpeg_timeout)

                if process.returncode == 0:
                    os.remove(temp_path)
                    final_size = os.path.getsize(final_path) if os.path.exists(final_path) else 0
                    logger.info(f"Transcoded clip for {event_id} ({final_size} bytes)")
                    return True
                else:
                    logger.error(f"FFmpeg error for {event_id}: {stderr.decode()[:500]}")
                    if os.path.exists(temp_path):
                        os.rename(temp_path, final_path)
                        logger.warning(f"Using original clip for {event_id} due to transcode failure")
                    return True

            except subprocess.TimeoutExpired:
                logger.error(f"FFmpeg timeout ({self.ffmpeg_timeout}s) for {event_id}")
                self._terminate_process_gracefully(process, event_id)
                if os.path.exists(temp_path):
                    os.rename(temp_path, final_path)
                    logger.warning(f"Using original clip for {event_id} due to timeout")
                return True

        except requests.exceptions.Timeout:
            logger.error(f"Timeout downloading clip for {event_id}")
            self._terminate_process_gracefully(process, event_id)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error downloading clip for {event_id}: {e}")
            self._terminate_process_gracefully(process, event_id)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
        except Exception as e:
            logger.exception(f"Failed to download/transcode clip for {event_id}: {e}")
            self._terminate_process_gracefully(process, event_id)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

    def write_summary(self, folder_path: str, event: EventState) -> bool:
        """Write summary.txt with event metadata."""
        try:
            summary_path = os.path.join(folder_path, "summary.txt")

            timestamp_str = time.strftime(
                '%Y-%m-%d %H:%M:%S',
                time.localtime(event.created_at)
            )

            lines = [
                f"Event ID: {event.event_id}",
                f"Camera: {event.camera}",
                f"Label: {event.label}",
                f"Timestamp: {timestamp_str}",
                f"Phase: {event.phase.name}",
                "",
            ]

            if event.genai_title:
                lines.append(f"Title: {event.genai_title}")

            if event.genai_description:
                lines.append(f"Description: {event.genai_description}")
            elif event.ai_description:
                lines.append(f"AI Description: {event.ai_description}")

            if event.severity:
                lines.append(f"Severity: {event.severity}")

            if event.threat_level > 0:
                level_names = {0: "Normal", 1: "Suspicious", 2: "Critical"}
                lines.append(f"Threat Level: {event.threat_level} ({level_names.get(event.threat_level, 'Unknown')})")

            if event.review_summary:
                lines.append("")
                lines.append("Review Summary: See review_summary.md")

            with open(summary_path, 'w') as f:
                f.write('\n'.join(lines))

            logger.debug(f"Written summary for {event.event_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to write summary: {e}")
            return False

    def write_review_summary(self, folder_path: str, summary_markdown: str) -> bool:
        """Write review_summary.md with the Frigate review summary."""
        try:
            summary_path = os.path.join(folder_path, "review_summary.md")
            with open(summary_path, 'w') as f:
                f.write(summary_markdown)
            logger.debug(f"Written review summary to {folder_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to write review summary: {e}")
            return False

    def append_timeline_entry(self, folder_path: str, entry: dict) -> None:
        """Append an entry to notification_timeline.json in the event folder."""
        timeline_path = os.path.join(folder_path, "notification_timeline.json")
        entry = dict(entry)
        entry["ts"] = entry.get("ts") or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        try:
            data = {"event_id": None, "entries": []}
            if os.path.exists(timeline_path):
                with open(timeline_path, 'r') as f:
                    data = json.load(f)
            if not data.get("event_id"):
                folder_name = os.path.basename(folder_path)
                parts = folder_name.split("_", 1)
                if len(parts) > 1:
                    data["event_id"] = parts[1]
                elif entry.get("data", {}).get("event_id"):
                    data["event_id"] = entry["data"]["event_id"]
            data["entries"] = data.get("entries", [])
            data["entries"].append(entry)
            with open(timeline_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to append timeline entry: {e}")

    def write_metadata_json(self, folder_path: str, event: EventState) -> bool:
        """Write machine-readable metadata.json for the event."""
        try:
            meta_path = os.path.join(folder_path, "metadata.json")
            metadata = {
                "event_id": event.event_id,
                "camera": event.camera,
                "label": event.label,
                "created_at": event.created_at,
                "end_time": event.end_time,
                "threat_level": event.threat_level,
                "severity": event.severity,
                "genai_title": event.genai_title,
                "genai_description": event.genai_description,
                "phase": event.phase.name,
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to write metadata.json: {e}")
            return False

    def fetch_review_summary(self, start_ts: float, end_ts: float,
                             padding_before: float, padding_after: float) -> Optional[str]:
        """Fetch review summary from Frigate API with time padding."""
        padded_start = int(start_ts - padding_before)
        padded_end = int(end_ts + padding_after)

        url = f"{self.frigate_url}/api/review/summarize/start/{padded_start}/end/{padded_end}"
        logger.info(f"Fetching review summary: {url}")

        try:
            response = requests.post(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            summary = data.get("summary", "")
            if summary:
                logger.info(f"Review summary received ({len(summary)} chars)")
                return summary
            else:
                logger.warning("Review summary API returned empty summary")
                return None
        except requests.exceptions.Timeout:
            logger.error("Timeout fetching review summary")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching review summary: {e}")
            return None
        except Exception as e:
            logger.exception(f"Failed to fetch review summary: {e}")
            return None

    def cleanup_old_events(self, active_event_ids: List[str]) -> int:
        """Delete folders older than retention period. Returns count deleted."""
        now = time.time()
        cutoff = now - (self.retention_days * 86400)
        deleted_count = 0

        logger.debug(f"Running cleanup: cutoff={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cutoff))}")

        try:
            # Iterate through camera subdirectories
            for camera_dir in os.listdir(self.storage_path):
                camera_path = os.path.join(self.storage_path, camera_dir)

                if not os.path.isdir(camera_path):
                    continue

                # Check if this is a camera folder (contains event folders)
                # or a legacy flat event folder (for migration period)
                first_item = camera_dir.split('_', 1)
                if len(first_item) > 1 and first_item[0].isdigit():
                    # Legacy flat structure: {timestamp}_{event_id}
                    try:
                        ts = float(first_item[0])
                        event_id = first_item[1]

                        if event_id and event_id in active_event_ids:
                            continue

                        if ts < cutoff:
                            shutil.rmtree(camera_path)
                            logger.info(f"Cleaned up legacy event: {camera_dir}")
                            deleted_count += 1
                    except (ValueError, IndexError):
                        pass
                    continue

                # New structure: iterate through event folders in camera dir
                for event_dir in os.listdir(camera_path):
                    event_path = os.path.join(camera_path, event_dir)

                    if not os.path.isdir(event_path):
                        continue

                    try:
                        parts = event_dir.split('_', 1)
                        ts = float(parts[0])
                        event_id = parts[1] if len(parts) > 1 else None

                        # Skip if event is still active
                        if event_id and event_id in active_event_ids:
                            logger.debug(f"Skipping active event: {camera_dir}/{event_dir}")
                            continue

                        # Delete if older than cutoff
                        if ts < cutoff:
                            shutil.rmtree(event_path)
                            logger.info(f"Cleaned up old event: {camera_dir}/{event_dir}")
                            deleted_count += 1

                    except (ValueError, IndexError):
                        logger.debug(f"Skipping malformed folder: {camera_dir}/{event_dir}")
                        continue

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

        return deleted_count

    def compute_storage_stats(self) -> dict:
        """Compute storage usage by camera and type. Returns bytes."""
        clips = 0
        snapshots = 0
        descriptions = 0
        by_camera = {}

        try:
            for camera_dir in os.listdir(self.storage_path):
                camera_path = os.path.join(self.storage_path, camera_dir)

                if not os.path.isdir(camera_path):
                    continue
                if camera_dir.split('_')[0].isdigit():
                    continue

                cam_clips = cam_snapshots = cam_descriptions = 0

                for event_dir in os.listdir(camera_path):
                    event_path = os.path.join(camera_path, event_dir)
                    if not os.path.isdir(event_path):
                        continue

                    clip_path = os.path.join(event_path, 'clip.mp4')
                    snapshot_path = os.path.join(event_path, 'snapshot.jpg')
                    for f in ('summary.txt', 'review_summary.md', 'metadata.json'):
                        p = os.path.join(event_path, f)
                        if os.path.exists(p):
                            cam_descriptions += os.path.getsize(p)
                    if os.path.exists(clip_path):
                        cam_clips += os.path.getsize(clip_path)
                    if os.path.exists(snapshot_path):
                        cam_snapshots += os.path.getsize(snapshot_path)

                cam_total = cam_clips + cam_snapshots + cam_descriptions
                if cam_total > 0:
                    by_camera[camera_dir] = {
                        'clips': cam_clips,
                        'snapshots': cam_snapshots,
                        'descriptions': cam_descriptions,
                        'total': cam_total
                    }
                clips += cam_clips
                snapshots += cam_snapshots
                descriptions += cam_descriptions

        except Exception as e:
            logger.error(f"Error computing storage stats: {e}")

        return {
            'clips': clips,
            'snapshots': snapshots,
            'descriptions': descriptions,
            'total': clips + snapshots + descriptions,
            'by_camera': by_camera
        }


# =============================================================================
# DAILY REVIEW MANAGER
# =============================================================================

class DailyReviewManager:
    """Fetches and stores Frigate daily review summaries (POST /api/review/summarize/start/{start}/end/{end})."""

    def __init__(self, storage_path: str, frigate_url: str, retention_days: int):
        self.reviews_dir = os.path.join(storage_path, 'daily_reviews')
        self.frigate_url = frigate_url.rstrip('/')
        self.retention_days = retention_days
        os.makedirs(self.reviews_dir, exist_ok=True)
        logger.info(f"DailyReviewManager: {self.reviews_dir}, retention={retention_days} days")

    def _date_to_ts_range(self, d: date, end_now: bool = False) -> tuple:
        """Return (start_ts, end_ts) for a date. end_now=True uses current time for end."""
        start_dt = datetime.combine(d, datetime.min.time())
        end_dt = datetime.now() if end_now else datetime.combine(d, datetime.max.time().replace(microsecond=0))
        return (int(start_dt.timestamp()), int(end_dt.timestamp()))

    def _date_str(self, d: date) -> str:
        return d.strftime('%Y-%m-%d')

    def _path_for_date(self, d: date) -> str:
        return os.path.join(self.reviews_dir, f"{self._date_str(d)}.json")

    def fetch_from_frigate(self, start_ts: int, end_ts: int) -> Optional[dict]:
        """Fetch review summary from Frigate API."""
        url = f"{self.frigate_url}/api/review/summarize/start/{start_ts}/end/{end_ts}"
        logger.info(f"Fetching daily review from Frigate: {url}")
        try:
            resp = requests.post(url, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            if data.get('success') and 'summary' in data:
                return data
            logger.warning(f"Frigate returned success=false or no summary: {data}")
            return None
        except requests.exceptions.Timeout:
            logger.error("Timeout fetching daily review from Frigate")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching daily review: {e}")
            return None
        except Exception as e:
            logger.exception(f"Error fetching daily review: {e}")
            return None

    def fetch_and_save(self, d: date, end_now: bool = False) -> Optional[dict]:
        """Fetch from Frigate and save to disk. Returns the response dict or None."""
        start_ts, end_ts = self._date_to_ts_range(d, end_now=end_now)
        data = self.fetch_from_frigate(start_ts, end_ts)
        if data:
            data['start_ts'] = start_ts
            data['end_ts'] = end_ts
            data['date'] = self._date_str(d)
            data['end_now'] = end_now
            path = self._path_for_date(d)
            if end_now:
                path = path.replace('.json', '_partial.json')
            try:
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Saved daily review to {path}")
                return data
            except Exception as e:
                logger.error(f"Failed to save daily review: {e}")
        return None

    def get_cached(self, d: date, allow_partial: bool = False) -> Optional[dict]:
        """Get cached review for date. allow_partial also checks _partial.json for today."""
        path = self._path_for_date(d)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading cached review: {e}")
        if allow_partial:
            partial_path = path.replace('.json', '_partial.json')
            if os.path.exists(partial_path):
                try:
                    with open(partial_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Error reading partial review: {e}")
        return None

    def get_or_fetch(self, d: date, force_refresh: bool = False, end_now: bool = False) -> Optional[dict]:
        """Get cached review or fetch from Frigate. end_now only applies when force_refresh."""
        if not force_refresh:
            cached = self.get_cached(d, allow_partial=(d == date.today()))
            if cached:
                return cached
        return self.fetch_and_save(d, end_now=end_now)

    def list_dates(self) -> List[str]:
        """Return sorted list of available date strings (YYYY-MM-DD), newest first."""
        dates = set()
        for f in os.listdir(self.reviews_dir):
            if not f.endswith('.json'):
                continue
            date_str = f.replace('_partial.json', '').replace('.json', '')
            if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
                dates.add(date_str)
        return sorted(dates, reverse=True)

    def cleanup_old(self) -> int:
        """Remove reviews older than retention. Returns count deleted."""
        cutoff = date.today() - timedelta(days=self.retention_days)
        deleted = 0
        for f in os.listdir(self.reviews_dir):
            if not f.endswith('.json'):
                continue
            date_str = f.replace('.json', '').replace('_partial', '')
            if len(date_str) != 10:
                continue
            try:
                d = datetime.strptime(date_str, '%Y-%m-%d').date()
                if d < cutoff:
                    path = os.path.join(self.reviews_dir, f)
                    os.remove(path)
                    deleted += 1
                    logger.info(f"Cleaned up old daily review: {f}")
            except ValueError:
                pass
        return deleted


# =============================================================================
# NOTIFICATION PUBLISHER
# =============================================================================

class NotificationPublisher:
    """Publishes notifications to frigate/custom/notifications with rate limiting."""

    TOPIC = "frigate/custom/notifications"

    # Rate limiting: max notifications per time window
    MAX_NOTIFICATIONS_PER_WINDOW = 2
    RATE_WINDOW_SECONDS = 5.0
    MAX_QUEUE_SIZE = 10

    def __init__(self, mqtt_client: mqtt.Client, buffer_ip: str, flask_port: int,
                 frigate_url: str = ""):
        self.mqtt_client = mqtt_client
        self.buffer_ip = buffer_ip
        self.flask_port = flask_port
        self.frigate_url = frigate_url.rstrip('/')
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

            with self._lock:
                if not self._pending_queue:
                    continue

                # Check if we can send (rate limit cleared)
                if not self._is_rate_limited():
                    # Pop oldest notification from queue
                    event, status, message = self._pending_queue.pop(0)
                    self._record_notification()

                    # Reset overflow flag if queue is draining
                    if len(self._pending_queue) < self.MAX_QUEUE_SIZE:
                        self._overflow_sent = False

            # Send outside the lock
            if event:
                self._send_notification_internal(event, status, message)

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
                            message: Optional[str] = None) -> bool:
        """Publish event notification with rate limiting and queue management."""
        with self._lock:
            # Check rate limit
            if self._is_rate_limited():
                # Add to queue
                self._pending_queue.append((event, status, message))
                logger.debug(f"Rate limited, queued notification for {event.event_id} (queue size: {len(self._pending_queue)})")

                # Check for queue overflow
                if len(self._pending_queue) > self.MAX_QUEUE_SIZE and not self._overflow_sent:
                    logger.warning(f"Queue overflow ({len(self._pending_queue)} items), sending summary notification")
                    self._pending_queue.clear()
                    self._overflow_sent = True
                    self._send_overflow_notification()

                return True  # Queued successfully

            # Not rate limited - send immediately
            self._record_notification()

        # Send outside the lock
        return self._send_notification_internal(event, status, message)

    def _send_notification_internal(self, event: EventState, status: str,
                                    message: Optional[str] = None) -> bool:
        """Internal method to actually send the notification to MQTT."""
        # Construct URLs for Home Assistant - always use buffer base URL for Companion app reachability
        image_url = None
        video_url = None

        buffer_base = f"http://{self.buffer_ip}:{self.flask_port}"

        if event.folder_path:
            folder_name = os.path.basename(event.folder_path)
            camera_dir = os.path.basename(os.path.dirname(event.folder_path))
            base_url = f"{buffer_base}/files/{camera_dir}/{folder_name}"
            if event.snapshot_downloaded:
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
            "tag": f"frigate_{event.event_id}",
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


# =============================================================================
# STATE-AWARE ORCHESTRATOR
# =============================================================================

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

        # Initialize components
        self.state_manager = EventStateManager()
        self.file_manager = FileManager(
            config['STORAGE_PATH'],
            config['FRIGATE_URL'],
            config['RETENTION_DAYS'],
            config.get('FFMPEG_TIMEOUT', 60)
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
            config.get('FRIGATE_URL', '')
        )
        self.notifier.timeline_callback = self._timeline_log_ha

        # Daily review manager (Frigate review summarize API)
        self.daily_review_manager = DailyReviewManager(
            config['STORAGE_PATH'],
            config['FRIGATE_URL'],
            config.get('DAILY_REVIEW_RETENTION_DAYS', 90)
        )

        # Flask app
        self.flask_app = self._create_flask_app()

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

            # Subscribe to all topics
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

    def _timeline_log_ha(self, event, status: str, payload: dict) -> None:
        """Log HA notification payload to event timeline."""
        if event and event.folder_path:
            self.file_manager.append_timeline_entry(event.folder_path, {
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
        """Process frigate/events messages with camera/label filtering."""
        event_type = payload.get("type")
        after_data = payload.get("after", {})

        event_id = after_data.get("id")
        camera = after_data.get("camera")
        label = after_data.get("label")

        if not event_id:
            logger.debug("Skipping event: no event_id in payload")
            return

        # Per-camera label filtering
        camera_label_map = self.config.get('CAMERA_LABEL_MAP', {})

        if camera_label_map:
            # Camera must be in the map to be processed
            if camera not in camera_label_map:
                logger.debug(f"Filtered out event from camera '{camera}' (not configured)")
                return

            # Check labels for this specific camera
            allowed_labels_for_camera = camera_label_map[camera]
            if allowed_labels_for_camera and label not in allowed_labels_for_camera:
                logger.debug(f"Filtered out '{label}' on '{camera}' (allowed: {allowed_labels_for_camera})")
                return

        if event_type == "new":
            self._handle_event_new(
                event_id=event_id,
                camera=camera,
                label=label,
                start_time=after_data.get("start_time", time.time()),
                mqtt_payload=payload
            )

        elif event_type == "end":
            self._handle_event_end(
                event_id=event_id,
                end_time=after_data.get("end_time", time.time()),
                has_clip=after_data.get("has_clip", False),
                has_snapshot=after_data.get("has_snapshot", False),
                mqtt_payload=payload
            )

    def _handle_event_new(self, event_id: str, camera: str, label: str,
                          start_time: float, mqtt_payload: Optional[dict] = None):
        """Handle new event detection (Phase 1)."""
        logger.info(f"New event: {event_id} - {label} on {camera}")

        # Create event state
        event = self.state_manager.create_event(event_id, camera, label, start_time)

        # Create folder in camera subdirectory
        folder_path = self.file_manager.create_event_folder(event_id, camera, start_time)
        event.folder_path = folder_path

        if mqtt_payload:
            self._timeline_log_mqtt(
                folder_path, "frigate/events",
                mqtt_payload, "Event new (from Frigate)"
            )

        # Delay, fetch snapshot, then notify (Ring-style: image-first notification)
        delay = self.config.get('NOTIFICATION_DELAY', 5)
        threading.Thread(
            target=self._send_initial_notification,
            args=(event, delay),
            daemon=True
        ).start()

    def _send_initial_notification(self, event: EventState, delay: float):
        """Send notification immediately, then fetch snapshot and silently update."""
        try:
            # Send notification instantly (no image yet)
            self.notifier.publish_notification(event, "new")

            # Brief delay for Frigate to select a better snapshot frame
            if delay > 0:
                time.sleep(delay)

            # Download snapshot and silently update notification with image
            if event.folder_path:
                event.snapshot_downloaded = self.file_manager.download_snapshot(
                    event.event_id, event.folder_path
                )
                if event.snapshot_downloaded:
                    # Silent update (status != "new" so HA automation won't play sound)
                    self.notifier.publish_notification(event, "snapshot_ready")
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

        if event and event.folder_path and mqtt_payload:
            self._timeline_log_mqtt(
                event.folder_path, "frigate/events",
                mqtt_payload, "Event end (from Frigate)"
            )

        if not event or not event.folder_path:
            logger.warning(f"Unknown event ended: {event_id}")
            return

        # Download in background thread
        threading.Thread(
            target=self._process_event_end,
            args=(event,),
            daemon=True
        ).start()

    def _process_event_end(self, event: EventState):
        """Background processing when event ends."""
        try:
            # Download files
            if event.has_snapshot:
                event.snapshot_downloaded = self.file_manager.download_snapshot(
                    event.event_id, event.folder_path
                )

            if event.has_clip:
                event.clip_downloaded = self.file_manager.download_and_transcode_clip(
                    event.event_id, event.folder_path
                )

            # Write initial summary
            self.file_manager.write_summary(event.folder_path, event)

            # Publish update notification
            self.notifier.publish_notification(event, "clip_ready")

            # Run cleanup check
            active_ids = self.state_manager.get_active_event_ids()
            deleted = self.file_manager.cleanup_old_events(active_ids)
            self._last_cleanup_time = time.time()
            self._last_cleanup_deleted = deleted
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old event folders")

        except Exception as e:
            logger.exception(f"Error processing event end: {e}")

    def _handle_tracked_update(self, payload: dict, topic: str):
        """Handle AI description update (Phase 2)."""
        # Frigate 0.17 sends multiple types on this topic: description, face, lpr, classification
        # Only process description-type messages for AI descriptions
        update_type = payload.get("type")
        if update_type and update_type != "description":
            logger.debug(f"Skipping tracked update type: {update_type}")
            return

        # Extract camera from topic: frigate/{camera}/tracked_object_update
        parts = topic.split("/")
        if len(parts) >= 2:
            camera = parts[1]
        else:
            camera = "unknown"

        event_id = payload.get("id")
        description = payload.get("description")

        if not event_id or not description:
            logger.debug(f"Skipping tracked update: event_id={event_id}, has_description={bool(description)}")
            return

        event = self.state_manager.get_event(event_id)
        if event and event.folder_path:
            self._timeline_log_mqtt(
                event.folder_path, topic,
                payload, "Tracked object update (AI description)"
            )

        logger.info(f"Tracked update for {event_id}: {description[:50]}..." if len(str(description)) > 50 else f"Tracked update for {event_id}: {description}")

        if self.state_manager.set_ai_description(event_id, description):
            event = self.state_manager.get_event(event_id)
            if event:
                # Update summary file
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

        # Extract detections and genai data
        data = review_data.get("data", {})
        detections = data.get("detections", [])
        severity = review_data.get("severity", "detection")

        # Frigate 0.17: GenAI data is in data.metadata (type: "genai")
        # Legacy: GenAI data was in data.genai
        genai = data.get("metadata") or data.get("genai") or {}

        logger.debug(f"Processing review: type={event_type}, {len(detections)} detections, severity={severity}")

        for event_id in detections:
            event = self.state_manager.get_event(event_id)
            if event and event.folder_path:
                self._timeline_log_mqtt(
                    event.folder_path, "frigate/reviews",
                    payload, f"Review update (type={event_type})"
                )
            # Frigate 0.17 uses "title" and "shortSummary" in metadata
            # Legacy used "title" and "description" in genai
            title = genai.get("title")
            description = genai.get("shortSummary") or genai.get("description")
            threat_level = int(genai.get("potential_threat_level", 0))

            # Only log at INFO when GenAI data is present; suppress title=N/A noise
            if title or description:
                logger.info(f"Review for {event_id}: title={title or 'N/A'}, "
                            f"threat_level={threat_level}")
            else:
                logger.debug(f"Review for {event_id}: title=N/A, threat_level={threat_level}")

            # Only finalize when actual GenAI data is present
            if not title and not description:
                logger.debug(f"Skipping finalization for {event_id}: no GenAI data yet")
                continue

            if self.state_manager.set_genai_metadata(
                event_id,
                title,
                description,
                severity,
                threat_level
            ):
                event = self.state_manager.get_event(event_id)
                if event:
                    # Write final summary and metadata
                    if event.folder_path:
                        event.summary_written = self.file_manager.write_summary(
                            event.folder_path, event
                        )
                        self.file_manager.write_metadata_json(event.folder_path, event)

                    # Publish finalized notification
                    self.notifier.publish_notification(event, "finalized")

                    # Fetch review summary in background, then schedule removal
                    threading.Thread(
                        target=self._fetch_and_store_review_summary,
                        args=(event,),
                        daemon=True
                    ).start()

    def _fetch_and_store_review_summary(self, event: EventState):
        """Background: fetch review summary from Frigate API, store, notify, then cleanup."""
        try:
            # Wait for end_time if not yet available
            max_wait = 30
            waited = 0
            while event.end_time is None and waited < max_wait:
                time.sleep(2)
                waited += 2

            if event.end_time is None:
                logger.warning(f"No end_time for {event.event_id} after {max_wait}s, "
                               f"using created_at + 60s as fallback")
                effective_end = event.created_at + 60
            else:
                effective_end = event.end_time

            # Brief delay to let Frigate's LLM finish processing
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

            if event.folder_path:
                self._timeline_log_frigate_api(
                    event.folder_path, "out",
                    "Review summarize request (to Frigate API)",
                    {"url": url, "params": params}
                )

            summary = self.file_manager.fetch_review_summary(
                event.created_at, effective_end,
                padding_before, padding_after
            )

            if event.folder_path:
                self._timeline_log_frigate_api(
                    event.folder_path, "in",
                    "Review summarize response (from Frigate API)",
                    {"url": url, "params": params, "response": summary or "(empty or error)"}
                )

            if summary:
                self.state_manager.set_review_summary(event.event_id, summary)

                if event.folder_path:
                    event.review_summary_written = self.file_manager.write_review_summary(
                        event.folder_path, summary
                    )
                    # Update summary.txt and metadata.json with new phase
                    self.file_manager.write_summary(event.folder_path, event)
                    self.file_manager.write_metadata_json(event.folder_path, event)

                # Skip summarized notification when GenAI returns "No Concerns"
                if not _is_no_concerns(summary):
                    self.notifier.publish_notification(event, "summarized")
                else:
                    logger.info(f"Skipping summarized notification for {event.event_id} (no concerns)")
            else:
                logger.warning(f"No review summary obtained for {event.event_id}")

        except Exception as e:
            logger.exception(f"Error fetching review summary for {event.event_id}: {e}")
        finally:
            # Schedule event removal after grace period
            threading.Timer(
                60.0,
                lambda eid=event.event_id: self.state_manager.remove_event(eid)
            ).start()

    def _create_flask_app(self) -> Flask:
        """Create Flask app with all endpoints."""
        app = Flask(__name__)
        storage_path = self.config['STORAGE_PATH']
        allowed_cameras = self.config.get('ALLOWED_CAMERAS', [])
        state_manager = self.state_manager
        file_manager = self.file_manager
        orchestrator = self

        @app.before_request
        def _count_request():
            with orchestrator._request_count_lock:
                orchestrator._request_count += 1

        @app.route('/player')
        def player():
            """Serve the event viewer page."""
            return render_template('player.html',
                stats_refresh_seconds=self.config.get('STATS_REFRESH_SECONDS', 60))

        @app.route('/api/events/<event_id>/snapshot.jpg')
        def proxy_snapshot(event_id):
            """Proxy snapshot from Frigate so image_url is always buffer-based (Companion app reachability)."""
            frigate_url = self.config.get('FRIGATE_URL', '').rstrip('/')
            if not frigate_url:
                return "Frigate URL not configured", 503
            url = f"{frigate_url}/api/events/{event_id}/snapshot.jpg"
            try:
                resp = requests.get(url, timeout=15, stream=True)
                resp.raise_for_status()
                return Response(
                    resp.iter_content(chunk_size=8192),
                    content_type=resp.headers.get('Content-Type', 'image/jpeg'),
                    status=resp.status_code
                )
            except requests.RequestException as e:
                logger.debug(f"Snapshot proxy error for {event_id}: {e}")
                return "Snapshot unavailable", 502

        @app.route('/daily-review')
        def daily_review_page():
            """Serve the daily review page."""
            return render_template('daily_review.html')

        @app.route('/api/daily-review/dates')
        def daily_review_dates():
            """List available daily review dates."""
            dates = self.daily_review_manager.list_dates()
            return jsonify({"dates": dates})

        @app.route('/api/daily-review/current')
        def daily_review_current():
            """Fetch current day review (midnight to now)."""
            today = date.today()
            data = self.daily_review_manager.fetch_and_save(today, end_now=True)
            if data:
                return jsonify(data)
            return jsonify({"error": "Failed to fetch current day review"}), 503

        @app.route('/api/daily-review/<date_str>')
        def daily_review_get(date_str):
            """Get cached review for date, or fetch if missing. date_str: YYYY-MM-DD."""
            try:
                d = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                return jsonify({"error": "Invalid date format"}), 400
            force = request.args.get('force') == '1'
            data = self.daily_review_manager.get_or_fetch(d, force_refresh=force)
            if data:
                return jsonify(data)
            return jsonify({"error": "Failed to fetch review"}), 503

        def _parse_summary(summary_text: str) -> dict:
            """Parse key-value pairs from summary.txt format."""
            parsed = {}
            for line in summary_text.split('\n'):
                if ':' in line:
                    key, _, value = line.partition(':')
                    parsed[key.strip()] = value.strip()
            return parsed

        def _get_events_for_camera(camera_name: str) -> list:
            """Helper to get events for a specific camera."""
            camera_path = os.path.join(storage_path, camera_name)
            events = []

            if not os.path.isdir(camera_path):
                return events

            subdirs = sorted(
                [d for d in os.listdir(camera_path)
                 if os.path.isdir(os.path.join(camera_path, d))],
                reverse=True
            )

            for subdir in subdirs:
                folder_path = os.path.join(camera_path, subdir)
                summary_path = os.path.join(folder_path, 'summary.txt')

                parts = subdir.split('_', 1)
                ts = parts[0] if len(parts) > 0 else "0"
                eid = parts[1] if len(parts) > 1 else subdir

                summary_text = "Analysis pending..."
                parsed = {}
                if os.path.exists(summary_path):
                    with open(summary_path, 'r') as f:
                        summary_text = f.read().strip()
                    parsed = _parse_summary(summary_text)

                # Read metadata.json for structured data (threat_level, etc.)
                metadata_path = os.path.join(folder_path, 'metadata.json')
                metadata = {}
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    except Exception:
                        pass

                # Read review summary markdown if available
                review_summary_path = os.path.join(folder_path, 'review_summary.md')
                review_summary = None
                if os.path.exists(review_summary_path):
                    with open(review_summary_path, 'r') as f:
                        review_summary = f.read().strip()

                clip_path = os.path.join(folder_path, 'clip.mp4')
                snapshot_path = os.path.join(folder_path, 'snapshot.jpg')
                viewed_path = os.path.join(folder_path, '.viewed')

                events.append({
                    "event_id": eid,
                    "camera": camera_name,
                    "subdir": subdir,
                    "timestamp": ts,
                    "summary": summary_text,
                    "title": metadata.get("genai_title") or parsed.get("Title"),
                    "description": metadata.get("genai_description") or parsed.get("Description") or parsed.get("AI Description"),
                    "label": metadata.get("label") or parsed.get("Label", "unknown"),
                    "severity": metadata.get("severity") or parsed.get("Severity"),
                    "threat_level": metadata.get("threat_level", 0),
                    "review_summary": review_summary,
                    "has_clip": os.path.exists(clip_path),
                    "has_snapshot": os.path.exists(snapshot_path),
                    "viewed": os.path.exists(viewed_path),
                    "hosted_clip": f"/files/{camera_name}/{subdir}/clip.mp4",
                    "hosted_snapshot": f"/files/{camera_name}/{subdir}/snapshot.jpg"
                })

            return events

        @app.route('/cameras')
        def list_cameras():
            """List available cameras from config."""
            # Get cameras that have event folders
            active_cameras = []
            try:
                for item in os.listdir(storage_path):
                    item_path = os.path.join(storage_path, item)
                    if os.path.isdir(item_path):
                        # Check if it looks like a camera folder (not a timestamp)
                        if not item.split('_')[0].isdigit():
                            active_cameras.append(item)
            except Exception as e:
                logger.error(f"Error listing cameras: {e}")

            # Merge with allowed cameras from config
            all_cameras = list(set(active_cameras + [
                file_manager.sanitize_camera_name(c) for c in allowed_cameras
            ]))
            all_cameras.sort()

            return jsonify({
                "cameras": all_cameras,
                "default": all_cameras[0] if all_cameras else None
            })

        def _filter_events(events: list) -> list:
            """Apply ?filter= query param: unreviewed (default), reviewed, all."""
            f = request.args.get('filter', 'unreviewed')
            if f == 'reviewed':
                return [e for e in events if e.get('viewed')]
            elif f == 'all':
                return events
            else:  # unreviewed (default)
                return [e for e in events if not e.get('viewed')]

        @app.route('/events/<camera>')
        def list_camera_events(camera):
            """List events for a specific camera."""
            # Run cleanup
            active_ids = state_manager.get_active_event_ids()
            deleted = file_manager.cleanup_old_events(active_ids)
            self._last_cleanup_time = time.time()
            self._last_cleanup_deleted = deleted

            sanitized = file_manager.sanitize_camera_name(camera)
            events = _filter_events(_get_events_for_camera(sanitized))

            return jsonify({
                "camera": sanitized,
                "events": events
            })

        @app.route('/events')
        def list_events():
            """List all events across all cameras (global view)."""
            # Run cleanup
            active_ids = state_manager.get_active_event_ids()
            deleted = file_manager.cleanup_old_events(active_ids)
            self._last_cleanup_time = time.time()
            self._last_cleanup_deleted = deleted

            all_events = []
            cameras_found = []

            try:
                # Iterate through camera subdirectories
                for camera_dir in os.listdir(storage_path):
                    camera_path = os.path.join(storage_path, camera_dir)

                    if not os.path.isdir(camera_path):
                        continue

                    # Skip legacy flat structure (timestamp_eventid)
                    if camera_dir.split('_')[0].isdigit():
                        continue

                    cameras_found.append(camera_dir)
                    events = _get_events_for_camera(camera_dir)
                    all_events.extend(events)

                # Sort all events by timestamp descending
                all_events.sort(key=lambda x: x['timestamp'], reverse=True)

            except Exception as e:
                logger.error(f"Error listing events: {e}")
                return jsonify({"error": str(e)}), 500

            filtered = _filter_events(all_events)
            return jsonify({
                "cameras": sorted(cameras_found),
                "total_count": len(filtered),
                "events": filtered
            })

        @app.route('/delete/<path:subdir>', methods=['POST'])
        def delete_event(subdir):
            """Delete a specific event folder."""
            base_dir = os.path.realpath(storage_path)
            # Securely join the path and resolve it to prevent traversal.
            folder_path = os.path.realpath(os.path.join(base_dir, subdir))

            # The resolved path must be inside the storage directory and not be the directory itself.
            if folder_path.startswith(base_dir) and folder_path != base_dir:
                if os.path.exists(folder_path) and os.path.isdir(folder_path):
                    try:
                        shutil.rmtree(folder_path)
                        logger.info(f"User manually deleted: {subdir}")
                        return jsonify({
                            "status": "success",
                            "message": f"Deleted folder: {subdir}"
                        }), 200
                    except Exception as e:
                        logger.error(f"Error deleting {subdir}: {e}")
                        return jsonify({
                            "status": "error",
                            "message": str(e)
                        }), 500
                else:
                    return jsonify({"status": "error", "message": "Folder not found"}), 404

            return jsonify({
                "status": "error",
                "message": "Invalid folder or path"
            }), 400

        @app.route('/viewed/<camera>/<subdir>', methods=['POST'])
        def mark_viewed(camera, subdir):
            """Mark a specific event as viewed."""
            base_dir = os.path.realpath(storage_path)
            folder_path = os.path.realpath(os.path.join(base_dir, camera, subdir))

            if not folder_path.startswith(base_dir) or folder_path == base_dir:
                return jsonify({"status": "error", "message": "Invalid path"}), 400

            if not os.path.isdir(folder_path):
                return jsonify({"status": "error", "message": "Event not found"}), 404

            viewed_path = os.path.join(folder_path, '.viewed')
            try:
                open(viewed_path, 'a').close()
                return jsonify({"status": "success"}), 200
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 500

        @app.route('/viewed/<camera>/<subdir>', methods=['DELETE'])
        def unmark_viewed(camera, subdir):
            """Remove viewed marker from a specific event."""
            base_dir = os.path.realpath(storage_path)
            folder_path = os.path.realpath(os.path.join(base_dir, camera, subdir))

            if not folder_path.startswith(base_dir) or folder_path == base_dir:
                return jsonify({"status": "error", "message": "Invalid path"}), 400

            viewed_path = os.path.join(folder_path, '.viewed')
            if os.path.exists(viewed_path):
                os.remove(viewed_path)
            return jsonify({"status": "success"}), 200

        @app.route('/viewed/all', methods=['POST'])
        def mark_all_viewed():
            """Mark ALL events across all cameras as viewed."""
            count = 0
            try:
                for camera_dir in os.listdir(storage_path):
                    camera_path = os.path.join(storage_path, camera_dir)
                    if not os.path.isdir(camera_path) or camera_dir.split('_')[0].isdigit():
                        continue
                    for subdir in os.listdir(camera_path):
                        folder_path = os.path.join(camera_path, subdir)
                        if os.path.isdir(folder_path):
                            viewed_path = os.path.join(folder_path, '.viewed')
                            if not os.path.exists(viewed_path):
                                open(viewed_path, 'a').close()
                                count += 1
            except Exception as e:
                logger.error(f"Error marking all viewed: {e}")
                return jsonify({"status": "error", "message": str(e)}), 500

            return jsonify({"status": "success", "marked": count}), 200

        @app.route('/events/<camera>/<subdir>/timeline')
        def event_timeline(camera, subdir):
            """Serve the notification timeline page for an event (most recent first)."""
            base_dir = os.path.realpath(storage_path)
            folder_path = os.path.realpath(os.path.join(base_dir, camera, subdir))

            if not folder_path.startswith(base_dir) or folder_path == base_dir:
                return "Invalid path", 400

            if not os.path.isdir(folder_path):
                return "Event not found", 404

            timeline_path = os.path.join(folder_path, "notification_timeline.json")
            timeline_data = {"event_id": None, "entries": []}
            if os.path.exists(timeline_path):
                try:
                    with open(timeline_path, 'r') as f:
                        timeline_data = json.load(f)
                except Exception as e:
                    logger.debug(f"Error reading timeline: {e}")

            # Sort entries most recent first (by ts)
            entries = timeline_data.get("entries", [])
            entries.sort(key=lambda e: e.get("ts", ""), reverse=True)

            # List all files in event folder for download links
            event_files = []
            try:
                for f in os.listdir(folder_path):
                    fp = os.path.join(folder_path, f)
                    if os.path.isfile(fp):
                        event_files.append(f)
                event_files.sort()
            except OSError:
                pass

            return render_template(
                'timeline.html',
                event_id=timeline_data.get("event_id", subdir),
                camera=camera,
                subdir=subdir,
                entries=entries,
                event_files=event_files
            )

        @app.route('/files/<path:filename>')
        def serve_file(filename):
            """Serve stored files (clips are already transcoded to H.264)."""
            directory = os.path.realpath(storage_path)
            # Securely join the path and resolve it to prevent traversal.
            safe_path = os.path.realpath(os.path.join(directory, filename))

            # The resolved path must be inside the storage directory.
            if not safe_path.startswith(directory):
                return "File not found", 404

            # Use the directory and the original relative filename.
            return send_from_directory(directory, filename)

        @app.route('/stats')
        def stats():
            """Return stats for the player dashboard (events, storage, errors, system)."""
            now = time.time()
            day_start = now - 86400
            week_start = now - 604800
            month_start = now - 2592000

            events_today = events_week = events_month = 0
            total_reviewed = total_unreviewed = 0
            by_camera = {}
            most_recent = None

            try:
                for camera_dir in os.listdir(storage_path):
                    camera_path = os.path.join(storage_path, camera_dir)
                    if not os.path.isdir(camera_path) or camera_dir.split('_')[0].isdigit():
                        continue

                    count = 0
                    for event_dir in os.listdir(camera_path):
                        event_path = os.path.join(camera_path, event_dir)
                        if not os.path.isdir(event_path):
                            continue
                        try:
                            parts = event_dir.split('_', 1)
                            ts = float(parts[0])
                        except (ValueError, IndexError):
                            continue

                        viewed = os.path.exists(os.path.join(event_path, '.viewed'))
                        if viewed:
                            total_reviewed += 1
                        else:
                            total_unreviewed += 1

                        count += 1
                        if ts >= day_start:
                            events_today += 1
                        if ts >= week_start:
                            events_week += 1
                        if ts >= month_start:
                            events_month += 1

                        if most_recent is None or ts > most_recent['timestamp']:
                            most_recent = {
                                'event_id': parts[1] if len(parts) > 1 else event_dir,
                                'camera': camera_dir,
                                'subdir': event_dir,
                                'timestamp': ts
                            }

                    by_camera[camera_dir] = count
            except Exception as e:
                logger.error(f"Error scanning events for stats: {e}")

            storage_raw = file_manager.compute_storage_stats()
            mb = 1024 * 1024

            def fmt_size(b):
                if b >= 1024 * mb:
                    return {'gb': round(b / (1024 * mb), 2), 'mb': None}
                return {'mb': round(b / mb, 2), 'gb': None}

            by_camera_storage = {}
            for cam, data in storage_raw.get('by_camera', {}).items():
                total = data['total']
                by_camera_storage[cam] = fmt_size(total)

            total_bytes = storage_raw.get('total', 0)
            breakdown = {
                'clips_mb': round(storage_raw.get('clips', 0) / mb, 2),
                'snapshots_mb': round(storage_raw.get('snapshots', 0) / mb, 2),
                'descriptions_mb': round(storage_raw.get('descriptions', 0) / mb, 2)
            }

            most_recent_out = None
            if most_recent:
                most_recent_out = {
                    'event_id': most_recent['event_id'],
                    'camera': most_recent['camera'],
                    'url': '/player?filter=all',
                    'timestamp': most_recent['timestamp']
                }

            last_cleanup = None
            if self._last_cleanup_time is not None:
                last_cleanup = {
                    'at': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self._last_cleanup_time)),
                    'deleted': self._last_cleanup_deleted
                }

            return jsonify({
                'events': {
                    'today': events_today,
                    'this_week': events_week,
                    'this_month': events_month,
                    'total_reviewed': total_reviewed,
                    'total_unreviewed': total_unreviewed,
                    'by_camera': by_camera
                },
                'storage': {
                    'total_mb': round(total_bytes / mb, 2),
                    'total_gb': round(total_bytes / (1024 * mb), 2) if total_bytes >= 1024 * mb else None,
                    'by_camera': by_camera_storage,
                    'breakdown': breakdown
                },
                'errors': error_buffer.get_all(),
                'last_cleanup': last_cleanup,
                'most_recent': most_recent_out,
                'system': {
                    'uptime_seconds': int(time.time() - self._start_time),
                    'mqtt_connected': self.mqtt_connected,
                    'active_events': len(self.state_manager.get_active_event_ids()),
                    'retention_days': self.config['RETENTION_DAYS'],
                    'cleanup_interval_hours': self.config.get('CLEANUP_INTERVAL_HOURS', 1),
                    'storage_path': self.config['STORAGE_PATH'],
                    'stats_refresh_seconds': self.config.get('STATS_REFRESH_SECONDS', 60)
                }
            })

        @app.route('/status')
        def status():
            """Return orchestrator status for monitoring."""
            uptime_seconds = time.time() - self._start_time
            uptime_str = str(timedelta(seconds=int(uptime_seconds)))

            return jsonify({
                # For HA binary sensor
                "online": True,
                "mqtt_connected": self.mqtt_connected,

                # Monitoring info
                "uptime_seconds": uptime_seconds,
                "uptime": uptime_str,
                "started_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self._start_time)),

                # Active events
                "active_events": state_manager.get_stats(),

                # Configuration (non-sensitive)
                "config": {
                    "retention_days": self.config['RETENTION_DAYS'],
                    "log_level": self.config.get('LOG_LEVEL', 'INFO'),
                    "ffmpeg_timeout": self.config.get('FFMPEG_TIMEOUT', 60),
                    "summary_padding_before": self.config.get('SUMMARY_PADDING_BEFORE', 15),
                    "summary_padding_after": self.config.get('SUMMARY_PADDING_AFTER', 15)
                }
            })

        return app

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
            time.sleep(60)  # Check every minute

    def _log_request_stats(self):
        """Log API request count every 5 minutes."""
        with self._request_count_lock:
            count = self._request_count
            self._request_count = 0
        active = self.state_manager.get_active_event_count() if hasattr(self.state_manager, 'get_active_event_count') else len(self.state_manager._events)
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
        deleted = self.file_manager.cleanup_old_events(active_ids)
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

        logger.info("=" * 60)

        # Start MQTT
        try:
            self.mqtt_client.connect_async(
                self.config['MQTT_BROKER'],
                self.config['MQTT_PORT'],
                keepalive=60
            )
            self.mqtt_client.loop_start()
        except Exception as e:
            logger.error(f"Failed to start MQTT client: {e}")

        # Start notification queue processor
        self.notifier.start_queue_processor()

        # Start scheduler thread
        self._scheduler_thread = threading.Thread(
            target=self._run_scheduler,
            daemon=True
        )
        self._scheduler_thread.start()

        # Start Flask (blocking)
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


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

orchestrator = None

def signal_handler(sig, frame):
    """Handle shutdown signals."""
    global orchestrator
    logger.info(f"Received signal {sig}, shutting down...")
    if orchestrator:
        orchestrator.stop()
    sys.exit(0)


if __name__ == '__main__':
    # Load configuration
    CONFIG = load_config()

    # Setup logging with configured level
    setup_logging(CONFIG.get('LOG_LEVEL', 'INFO'))

    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Create and start orchestrator
    orchestrator = StateAwareOrchestrator(CONFIG)
    orchestrator.start()
