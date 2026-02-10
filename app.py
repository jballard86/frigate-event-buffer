#!/usr/bin/env python3
"""
Frigate Event Buffer - State-Aware Orchestrator

Listens to Frigate MQTT topics, tracks events through their lifecycle,
sends Ring-style notifications to Home Assistant, and manages rolling retention.

Configuration is loaded from config.yaml with environment variable overrides.
"""

import os
import sys
import json
import time
import signal
import shutil
import logging
import threading
import subprocess
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, Optional, List

import yaml
import paho.mqtt.client as mqtt
import requests
import schedule
from flask import Flask, send_from_directory, jsonify

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


def load_config() -> dict:
    """Load configuration from config.yaml merged with environment variables.

    Priority (highest to lowest):
    1. Environment variables
    2. config.yaml
    3. Default values

    Note: MQTT_BROKER, FRIGATE_URL, and HA_IP are REQUIRED and must be
    provided via config.yaml or environment variables.
    """
    config = {
        # Network settings - NO DEFAULTS (required from config)
        'MQTT_BROKER': None,
        'MQTT_PORT': 1883,
        'FRIGATE_URL': None,
        'HA_IP': None,
        'FLASK_PORT': 5055,
        'STORAGE_PATH': '/app/storage',

        # Settings defaults
        'RETENTION_DAYS': 3,
        'CLEANUP_INTERVAL_HOURS': 1,
        'FFMPEG_TIMEOUT': 60,
        'LOG_LEVEL': 'INFO',

        # Filtering defaults (empty = allow all)
        'ALLOWED_CAMERAS': [],
        'ALLOWED_LABELS': [],
    }

    # Load from config.yaml if exists
    config_paths = ['/app/config.yaml', './config.yaml', 'config.yaml']
    config_loaded = False

    for path in config_paths:
        if os.path.exists(path):
            try:
                logger.info(f"Loading config from {path}")
                with open(path, 'r') as f:
                    yaml_config = yaml.safe_load(f) or {}

                # Map YAML structure to flat config
                if 'cameras' in yaml_config:
                    config['ALLOWED_CAMERAS'] = yaml_config['cameras'].get('allowed', []) or []

                if 'labels' in yaml_config:
                    config['ALLOWED_LABELS'] = yaml_config['labels'].get('allowed', []) or []

                if 'settings' in yaml_config:
                    settings = yaml_config['settings']
                    config['RETENTION_DAYS'] = settings.get('retention_days', config['RETENTION_DAYS'])
                    config['CLEANUP_INTERVAL_HOURS'] = settings.get('cleanup_interval_hours', config['CLEANUP_INTERVAL_HOURS'])
                    config['FFMPEG_TIMEOUT'] = settings.get('ffmpeg_timeout_seconds', config['FFMPEG_TIMEOUT'])
                    config['LOG_LEVEL'] = settings.get('log_level', config['LOG_LEVEL'])

                if 'network' in yaml_config:
                    network = yaml_config['network']
                    config['MQTT_BROKER'] = network.get('mqtt_broker', config['MQTT_BROKER'])
                    config['MQTT_PORT'] = network.get('mqtt_port', config['MQTT_PORT'])
                    config['FRIGATE_URL'] = network.get('frigate_url', config['FRIGATE_URL'])
                    config['HA_IP'] = network.get('ha_ip', config['HA_IP'])
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
    config['HA_IP'] = os.getenv('HA_IP') or config['HA_IP']
    config['FLASK_PORT'] = int(os.getenv('FLASK_PORT', str(config['FLASK_PORT'])))
    config['STORAGE_PATH'] = os.getenv('STORAGE_PATH', config['STORAGE_PATH'])
    config['RETENTION_DAYS'] = int(os.getenv('RETENTION_DAYS', str(config['RETENTION_DAYS'])))
    config['LOG_LEVEL'] = os.getenv('LOG_LEVEL', config['LOG_LEVEL'])

    # Validate required settings
    missing = []
    if not config['MQTT_BROKER']:
        missing.append('MQTT_BROKER (network.mqtt_broker)')
    if not config['FRIGATE_URL']:
        missing.append('FRIGATE_URL (network.frigate_url)')
    if not config['HA_IP']:
        missing.append('HA_IP (network.ha_ip)')

    if missing:
        raise ValueError(
            f"Missing required configuration: {', '.join(missing)}. "
            f"Set these in config.yaml under 'network:' or as environment variables."
        )

    return config


def setup_logging(log_level: str):
    """Configure logging with the specified level."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Reconfigure the root logger
    logging.getLogger().setLevel(level)
    logger.setLevel(level)

    logger.info(f"Log level set to {log_level.upper()}")


# =============================================================================
# EVENT STATE MODELS
# =============================================================================

class EventPhase(Enum):
    """Tracks the lifecycle phase of a Frigate event."""
    NEW = auto()        # Phase 1: Initial detection from frigate/events type=new
    DESCRIBED = auto()  # Phase 2: AI description received from tracked_object_update
    FINALIZED = auto()  # Phase 3: GenAI metadata received from frigate/reviews


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

    # File management
    folder_path: Optional[str] = None
    clip_downloaded: bool = False
    snapshot_downloaded: bool = False
    summary_written: bool = False

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
                           description: Optional[str], severity: str) -> bool:
        """Set GenAI review metadata and advance to FINALIZED phase."""
        with self._lock:
            event = self._events.get(event_id)
            if event:
                event.genai_title = title
                event.genai_description = description
                event.severity = severity
                event.phase = EventPhase.FINALIZED
                logger.info(f"Event {event_id} advanced to FINALIZED phase")
                logger.debug(f"GenAI title: {title}, severity: {severity}")
                return True
            logger.debug(f"Cannot set GenAI metadata: event {event_id} not found")
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

    def create_event_folder(self, event_id: str, timestamp: float) -> str:
        """Create folder for event: {timestamp}_{event_id}"""
        folder_name = f"{int(timestamp)}_{event_id}"
        folder_path = os.path.join(self.storage_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        logger.info(f"Created folder: {folder_name}")
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
            # Download original clip
            url = f"{self.frigate_url}/api/events/{event_id}/clip.mp4"
            logger.debug(f"Downloading clip from {url}")
            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()

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

            with open(summary_path, 'w') as f:
                f.write('\n'.join(lines))

            logger.debug(f"Written summary for {event.event_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to write summary: {e}")
            return False

    def cleanup_old_events(self, active_event_ids: List[str]) -> int:
        """Delete folders older than retention period. Returns count deleted."""
        now = time.time()
        cutoff = now - (self.retention_days * 86400)
        deleted_count = 0

        logger.debug(f"Running cleanup: cutoff={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cutoff))}")

        try:
            for subdir in os.listdir(self.storage_path):
                folder_path = os.path.join(self.storage_path, subdir)

                if not os.path.isdir(folder_path):
                    continue

                try:
                    parts = subdir.split('_', 1)
                    ts = float(parts[0])
                    event_id = parts[1] if len(parts) > 1 else None

                    # Skip if event is still active
                    if event_id and event_id in active_event_ids:
                        logger.debug(f"Skipping active event: {subdir}")
                        continue

                    # Delete if older than cutoff
                    if ts < cutoff:
                        shutil.rmtree(folder_path)
                        logger.info(f"Cleaned up old event: {subdir}")
                        deleted_count += 1

                except (ValueError, IndexError):
                    # Malformed folder name - skip
                    logger.debug(f"Skipping malformed folder name: {subdir}")
                    continue

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

        return deleted_count


# =============================================================================
# NOTIFICATION PUBLISHER
# =============================================================================

class NotificationPublisher:
    """Publishes notifications to frigate/custom/notifications."""

    TOPIC = "frigate/custom/notifications"

    def __init__(self, mqtt_client: mqtt.Client, ha_ip: str, flask_port: int):
        self.mqtt_client = mqtt_client
        self.ha_ip = ha_ip
        self.flask_port = flask_port

    def publish_notification(self, event: EventState, status: str,
                            message: Optional[str] = None) -> bool:
        """Publish event notification to MQTT."""

        # Construct URLs for Home Assistant
        image_url = None
        video_url = None

        if event.folder_path:
            folder_name = os.path.basename(event.folder_path)
            base_url = f"http://{self.ha_ip}:{self.flask_port}/files/{folder_name}"
            image_url = f"{base_url}/snapshot.jpg"
            video_url = f"{base_url}/clip.mp4"

        # Build title based on phase
        title = self._build_title(event)

        # Build message
        if not message:
            message = self._build_message(event, status)

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
            "tag": f"frigate_{event.event_id}",
            "timestamp": event.created_at
        }

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
        """Build notification message."""
        if event.genai_description:
            return event.genai_description

        if event.ai_description:
            return event.ai_description

        camera_display = event.camera.replace('_', ' ').title()
        label_display = event.label.title()

        return f"{label_display} detected by {camera_display}"


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
        self.mqtt_client = mqtt.Client(client_id="frigate-event-buffer")
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        self.mqtt_client.on_message = self._on_mqtt_message
        self.mqtt_client.reconnect_delay_set(min_delay=1, max_delay=120)
        self.mqtt_connected = False

        # Notification publisher (initialized after MQTT setup)
        self.notifier = NotificationPublisher(
            self.mqtt_client,
            config['HA_IP'],
            config['FLASK_PORT']
        )

        # Flask app
        self.flask_app = self._create_flask_app()

        # Scheduler thread
        self._scheduler_thread = None

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection."""
        if rc == 0:
            self.mqtt_connected = True
            logger.info(f"Connected to MQTT broker {self.config['MQTT_BROKER']}")

            # Subscribe to all topics
            for topic, qos in self.MQTT_TOPICS:
                client.subscribe(topic, qos)
                logger.info(f"Subscribed to: {topic}")
        else:
            logger.error(f"MQTT connection failed with code: {rc}")

    def _on_mqtt_disconnect(self, client, userdata, rc):
        """Handle MQTT disconnection."""
        self.mqtt_connected = False
        if rc != 0:
            logger.warning(f"Unexpected MQTT disconnect (rc={rc}), reconnecting...")
        else:
            logger.info("MQTT disconnected")

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

        # Camera filtering
        allowed_cameras = self.config.get('ALLOWED_CAMERAS', [])
        if allowed_cameras and camera not in allowed_cameras:
            logger.debug(f"Filtered out event from camera '{camera}' (allowed: {allowed_cameras})")
            return

        # Label filtering
        allowed_labels = self.config.get('ALLOWED_LABELS', [])
        if allowed_labels and label not in allowed_labels:
            logger.debug(f"Filtered out event with label '{label}' (allowed: {allowed_labels})")
            return

        if event_type == "new":
            self._handle_event_new(
                event_id=event_id,
                camera=camera,
                label=label,
                start_time=after_data.get("start_time", time.time())
            )

        elif event_type == "end":
            self._handle_event_end(
                event_id=event_id,
                end_time=after_data.get("end_time", time.time()),
                has_clip=after_data.get("has_clip", False),
                has_snapshot=after_data.get("has_snapshot", False)
            )

    def _handle_event_new(self, event_id: str, camera: str, label: str,
                          start_time: float):
        """Handle new event detection (Phase 1)."""
        logger.info(f"New event: {event_id} - {label} on {camera}")

        # Create event state
        event = self.state_manager.create_event(event_id, camera, label, start_time)

        # Create folder
        folder_path = self.file_manager.create_event_folder(event_id, start_time)
        event.folder_path = folder_path

        # Publish initial notification
        self.notifier.publish_notification(event, "new")

    def _handle_event_end(self, event_id: str, end_time: float,
                          has_clip: bool, has_snapshot: bool):
        """Handle event end - trigger downloads/transcoding."""
        logger.info(f"Event ended: {event_id}")

        event = self.state_manager.mark_event_ended(
            event_id, end_time, has_clip, has_snapshot
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
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old event folders")

        except Exception as e:
            logger.exception(f"Error processing event end: {e}")

    def _handle_tracked_update(self, payload: dict, topic: str):
        """Handle AI description update (Phase 2)."""
        # Extract camera from topic: frigate/{camera}/tracked_object_update
        parts = topic.split("/")
        if len(parts) >= 2:
            camera = parts[1]
        else:
            camera = "unknown"

        event_id = payload.get("id")
        # AI description may be in sub_label or description field
        description = payload.get("sub_label") or payload.get("description")

        if not event_id or not description:
            logger.debug(f"Skipping tracked update: event_id={event_id}, has_description={bool(description)}")
            return

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

        if event_type not in ["update", "end"]:
            logger.debug(f"Skipping review with type: {event_type}")
            return

        review_data = payload.get("after", {}) or payload.get("before", {})

        # Extract detections and genai data
        data = review_data.get("data", {})
        detections = data.get("detections", [])
        genai = data.get("genai", {})
        severity = review_data.get("severity", "detection")

        logger.debug(f"Processing review: {len(detections)} detections, severity={severity}")

        for event_id in detections:
            logger.info(f"Review for {event_id}: {genai.get('title', 'N/A')}")

            if self.state_manager.set_genai_metadata(
                event_id,
                genai.get("title"),
                genai.get("description"),
                severity
            ):
                event = self.state_manager.get_event(event_id)
                if event:
                    # Write final summary
                    if event.folder_path:
                        event.summary_written = self.file_manager.write_summary(
                            event.folder_path, event
                        )

                    # Publish final notification
                    self.notifier.publish_notification(event, "finalized")

                    # Schedule event removal after grace period
                    threading.Timer(
                        60.0,  # 1 minute grace period
                        lambda eid=event_id: self.state_manager.remove_event(eid)
                    ).start()

    def _create_flask_app(self) -> Flask:
        """Create Flask app with all endpoints."""
        app = Flask(__name__)
        storage_path = self.config['STORAGE_PATH']
        state_manager = self.state_manager
        file_manager = self.file_manager

        @app.route('/events')
        def list_events():
            """List all stored events with summaries."""
            # Run cleanup
            active_ids = state_manager.get_active_event_ids()
            file_manager.cleanup_old_events(active_ids)

            events = []
            try:
                subdirs = sorted(
                    [d for d in os.listdir(storage_path)
                     if os.path.isdir(os.path.join(storage_path, d))],
                    reverse=True
                )

                for subdir in subdirs:
                    folder_path = os.path.join(storage_path, subdir)
                    summary_path = os.path.join(folder_path, 'summary.txt')

                    parts = subdir.split('_', 1)
                    ts = parts[0] if len(parts) > 0 else "0"
                    eid = parts[1] if len(parts) > 1 else subdir

                    summary_text = "Analysis pending..."
                    if os.path.exists(summary_path):
                        with open(summary_path, 'r') as f:
                            summary_text = f.read().strip()

                    events.append({
                        "event_id": eid,
                        "timestamp": ts,
                        "summary": summary_text,
                        "hosted_clip": f"/files/{subdir}/clip.mp4",
                        "hosted_snapshot": f"/files/{subdir}/snapshot.jpg"
                    })

            except Exception as e:
                logger.error(f"Error listing events: {e}")
                return jsonify({"error": str(e)}), 500

            return jsonify({"events": events})

        @app.route('/delete/<subdir>', methods=['POST'])
        def delete_event(subdir):
            """Delete a specific event folder."""
            folder_path = os.path.join(storage_path, subdir)

            if os.path.abspath(folder_path).startswith(os.path.abspath(storage_path)):
                if os.path.exists(folder_path) and os.path.isdir(folder_path):
                    try:
                        shutil.rmtree(folder_path)
                        logger.info(f"User manually deleted: {subdir}")
                        return jsonify({
                            "status": "success",
                            "message": f"Deleted {subdir}"
                        }), 200
                    except Exception as e:
                        logger.error(f"Error deleting {subdir}: {e}")
                        return jsonify({
                            "status": "error",
                            "message": str(e)
                        }), 500

            return jsonify({
                "status": "error",
                "message": "Invalid folder or path"
            }), 400

        @app.route('/files/<path:filename>')
        def serve_file(filename):
            """Serve stored files (clips are already transcoded to H.264)."""
            file_path = os.path.join(storage_path, filename)

            if not os.path.exists(file_path):
                return "File not found", 404

            return send_from_directory(storage_path, filename)

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
                    "mqtt_broker": self.config['MQTT_BROKER'],
                    "frigate_url": self.config['FRIGATE_URL'],
                    "retention_days": self.config['RETENTION_DAYS'],
                    "allowed_cameras": self.config.get('ALLOWED_CAMERAS', []),
                    "allowed_labels": self.config.get('ALLOWED_LABELS', []),
                    "log_level": self.config.get('LOG_LEVEL', 'INFO'),
                    "ffmpeg_timeout": self.config.get('FFMPEG_TIMEOUT', 60)
                }
            })

        return app

    def _run_scheduler(self):
        """Background thread for scheduled tasks."""
        cleanup_hours = self.config.get('CLEANUP_INTERVAL_HOURS', 1)
        schedule.every(cleanup_hours).hours.do(self._hourly_cleanup)
        logger.info(f"Scheduled cleanup every {cleanup_hours} hour(s)")

        while not self._shutdown:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def _hourly_cleanup(self):
        """Hourly cleanup task."""
        logger.info("Running scheduled cleanup...")
        active_ids = self.state_manager.get_active_event_ids()
        deleted = self.file_manager.cleanup_old_events(active_ids)
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

        if self.config.get('ALLOWED_CAMERAS'):
            logger.info(f"Allowed Cameras: {self.config['ALLOWED_CAMERAS']}")
        else:
            logger.info("Allowed Cameras: ALL")

        if self.config.get('ALLOWED_LABELS'):
            logger.info(f"Allowed Labels: {self.config['ALLOWED_LABELS']}")
        else:
            logger.info("Allowed Labels: ALL")

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
