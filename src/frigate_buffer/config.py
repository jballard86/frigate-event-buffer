"""Configuration loading and validation."""

import os
import logging
import sys

import yaml
from voluptuous import Schema, Required, Optional, Any, ALLOW_EXTRA, Invalid

logger = logging.getLogger('frigate-buffer')


# Configuration Schema
CONFIG_SCHEMA = Schema({
    # List of cameras to process; only events from these cameras are ingested.
    Required('cameras'): [{
        # Frigate camera name; must match the camera key in Frigate config.
        Required('name'): str,
        # If set, only events with these object labels (e.g. person, car) are processed; empty or omit = allow all.
        Optional('labels'): [str],
        # Smart zone filter: event creation is gated by zone and exceptions (omit for legacy: all events start immediately).
        Optional('event_filters'): {
            # Only create an event when the object enters one of these Frigate zone names.
            Optional('tracked_zones'): [str],
            # Labels or sub_labels that start an event regardless of zone (e.g. person, UPS).
            Optional('exceptions'): [str],
        }
    }],
    # Network and storage; MQTT broker, Frigate URL, and buffer IP are required at runtime (config or env).
    Optional('network'): {
        Optional('mqtt_broker'): str,      # MQTT broker hostname or IP for Frigate events.
        Optional('mqtt_port'): int,        # MQTT broker port (default 1883).
        Optional('mqtt_user'): str,        # Optional MQTT username.
        Optional('mqtt_password'): str,    # Optional MQTT password.
        Optional('frigate_url'): str,      # Base URL of Frigate (e.g. http://host:5000) for API and snapshots.
        Optional('buffer_ip'): str,        # IP/hostname where this buffer is reachable (used in notification URLs).
        Optional('flask_port'): int,       # Port for the Flask web server (player, stats, API).
        Optional('storage_path'): str,     # Root path for event clips, snapshots, and exported files.
        Optional('ha_ip'): str,            # Legacy fallback for buffer_ip when building notification URLs.
    },
    # Application behavior: retention, cleanup, export, logging, review/report, and AI analysis options.
    Optional('settings'): {
        Optional('retention_days'): int,                         # Days to keep event data before cleanup.
        Optional('cleanup_interval_hours'): int,                 # How often to run retention cleanup (hours).
        Optional('export_watchdog_interval_minutes'): int,      # How often to check/remove completed exports in Frigate (minutes).
        Optional('ffmpeg_timeout_seconds'): int,                # Timeout for FFmpeg transcode; kills hung processes.
        Optional('notification_delay_seconds'): int,            # Delay before fetching snapshot after notification (lets Frigate pick a better frame).
        Optional('log_level'): Any('DEBUG', 'INFO', 'WARNING', 'ERROR'),  # Logging verbosity.
        Optional('summary_padding_before'): int,                # Seconds before event start for Frigate review summary window.
        Optional('summary_padding_after'): int,                 # Seconds after event end for Frigate review summary window.
        Optional('stats_refresh_seconds'): int,                 # Stats page auto-refresh interval (seconds).
        Optional('daily_review_retention_days'): int,           # How long to keep saved daily reviews (days).
        Optional('daily_review_schedule_hour'): int,            # Hour (0-23) to run daily review fetch for previous day.
        Optional('daily_report_schedule_hour'): int,             # Hour (0-23) to generate AI daily report from analysis results.
        Optional('report_prompt_file'): str,                    # Path to prompt file for daily report; empty = use default.
        Optional('report_known_person_name'): str,              # Placeholder value for {known_person_name} in report prompt.
        Optional('event_gap_seconds'): int,                     # Seconds of inactivity before next event starts a new consolidated group.
        Optional('minimum_event_seconds'): int,                 # Events shorter than this are discarded (data deleted, state/CE updated, discarded MQTT notification sent).
        Optional('export_buffer_before'): int,                  # Seconds to include before event start in exported clip.
        Optional('export_buffer_after'): int,                    # Seconds to include after event end in exported clip.
        Optional('final_review_image_count'): int,              # Max number of images to send to Frigate final review summary.
        Optional('gemini_max_concurrent_analyses'): int,        # Max concurrent Gemini clip analyses (throttling).
        Optional('max_concurrent_transcodes'): int,     # Max concurrent clip transcodes for multi-cam CE (overlap download + transcode).
        Optional('save_ai_frames'): bool,                        # Whether to save extracted AI analysis frames to disk.
        Optional('create_ai_analysis_zip'): bool,               # Whether to create a zip of AI analysis assets (e.g. for multi-cam).
    },
    # Home Assistant REST API; used for stats page to display Gemini cost/token entities.
    Optional('ha'): {
        Optional('base_url'): str,           # HA API base URL (e.g. http://host:8123/api); preferred over url.
        Optional('url'): str,                # Alternative to base_url for HA API endpoint.
        Optional('token'): str,              # Long-lived access token for HA API.
        Optional('gemini_cost_entity'): str, # HA entity ID for Gemini cost (e.g. input_number.gemini_daily_cost).
        Optional('gemini_tokens_entity'): str,  # HA entity ID for Gemini token count (e.g. input_number.gemini_total_tokens).
    },
    # Gemini proxy for main-app clip AI analysis; optional; API key often via env (GEMINI_API_KEY).
    Optional('gemini'): {
        Optional('proxy_url'): str,   # URL of OpenAI-compatible proxy (e.g. for Gemini); no Google fallback if unset.
        Optional('api_key'): str,    # API key for the proxy (can override via GEMINI_API_KEY env).
        Optional('model'): str,      # Model name sent to the proxy (e.g. gemini-2.5-flash-lite).
        Optional('enabled'): bool,   # Whether clip analysis via Gemini is enabled.
    },
    # Multi-cam frame extraction (main app AI analyzer + standalone multi_cam_recap); frame limits and crop options.
    Optional('multi_cam'): {
        Optional('max_multi_cam_frames_min'): int,              # Maximum frames to extract per clip (cap).
        Optional('max_multi_cam_frames_sec'): int,              # Target interval in seconds between captured frames.
        Optional('motion_threshold_px'): int,                   # Minimum pixel change to trigger high-rate capture.
        Optional('crop_width'): int,                            # Width of output crop for stitched/multi-cam frames.
        Optional('crop_height'): int,                            # Height of output crop for stitched/multi-cam frames.
        Optional('nvenc_probe_width'): int,                      # Width for NVENC preflight probe (match crop for safety).
        Optional('nvenc_probe_height'): int,                     # Height for NVENC preflight probe.
        Optional('multi_cam_system_prompt_file'): str,          # Path to system prompt file for multi-cam Gemini; empty = built-in.
        Optional('smart_crop_padding'): Any(int, float),        # Padding fraction around motion-based crop (e.g. 0.15).
        Optional('motion_crop_min_area_fraction'): Any(int, float),  # Minimum motion region area as fraction of frame to consider for crop.
        Optional('motion_crop_min_px'): int,                    # Minimum motion region area in pixels for crop.
        Optional('detection_model'): str,                         # Ultralytics model for transcode-time detection (e.g. yolov8n.pt).
        Optional('detection_device'): str,                       # Device for detection (e.g. cuda:0, cpu).
        Optional('detection_frame_interval'): int,               # Run YOLO every N frames; default 5.
        Optional('first_camera_bias_decay_seconds'): Any(int, float),   # Time constant for exponential bias decay; default 1.0.
        Optional('first_camera_bias_initial'): Any(int, float),         # Initial bias multiplier for primary camera; default 1.5.
        Optional('first_camera_bias_cap_seconds'): Any(int, float),     # Cap bias to 0 after this many seconds; 0 = no cap.
        Optional('person_area_switch_threshold'): int,                  # Allow switch when current camera area below this (px²); 0 = disable.
        Optional('camera_switch_ratio'): Any(int, float),               # New camera must have this ratio of current to switch; default 1.2.
    },
    # Extended Gemini proxy options (e.g. for multi_cam); model params; single API key via GEMINI_API_KEY, URL here or env.
    Optional('gemini_proxy'): {
        Optional('url'): str,                    # Proxy URL (no Google fallback; set explicitly or via GEMINI_PROXY_URL env).
        Optional('model'): str,                  # Model name for proxy requests.
        Optional('temperature'): Any(int, float),   # Sampling temperature (0–2 typical).
        Optional('top_p'): Any(int, float),         # Nucleus sampling parameter.
        Optional('frequency_penalty'): Any(int, float),  # Penalty for repeated tokens.
        Optional('presence_penalty'): Any(int, float),   # Penalty for token presence.
    },
}, extra=ALLOW_EXTRA)


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
        'MQTT_USER': None,
        'MQTT_PASSWORD': None,
        'FRIGATE_URL': None,
        'BUFFER_IP': None,
        'FLASK_PORT': 5055,
        'STORAGE_PATH': '/app/storage',

        # Settings defaults
        'RETENTION_DAYS': 3,
        'CLEANUP_INTERVAL_HOURS': 1,
        'EXPORT_WATCHDOG_INTERVAL_MINUTES': 2,
        'FFMPEG_TIMEOUT': 60,
        'NOTIFICATION_DELAY': 2,
        'LOG_LEVEL': 'INFO',
        'SUMMARY_PADDING_BEFORE': 15,
        'SUMMARY_PADDING_AFTER': 15,
        'STATS_REFRESH_SECONDS': 60,
        'DAILY_REVIEW_RETENTION_DAYS': 90,
        'DAILY_REVIEW_SCHEDULE_HOUR': 1,
        'DAILY_REPORT_SCHEDULE_HOUR': 1,
        'REPORT_PROMPT_FILE': '',
        'REPORT_KNOWN_PERSON_NAME': '',
        'EVENT_GAP_SECONDS': 120,
        'MINIMUM_EVENT_SECONDS': 5,
        'EXPORT_BUFFER_BEFORE': 5,
        'EXPORT_BUFFER_AFTER': 30,
        'FINAL_REVIEW_IMAGE_COUNT': 20,
        'GEMINI_MAX_CONCURRENT_ANALYSES': 3,
        'MAX_CONCURRENT_TRANSCODES': 2,
        'SAVE_AI_FRAMES': True,
        'CREATE_AI_ANALYSIS_ZIP': True,

        # Optional HA REST API (for stats page token/cost display)
        'HA_URL': None,
        'HA_TOKEN': None,
        'HA_GEMINI_COST_ENTITY': 'input_number.gemini_daily_cost',
        'HA_GEMINI_TOKENS_ENTITY': 'input_number.gemini_total_tokens',

        # Filtering defaults (empty = allow all)
        'ALLOWED_CAMERAS': [],
        'ALLOWED_LABELS': [],
        'CAMERA_LABEL_MAP': {},
        # Smart Zone Filtering: per-camera event_filters (zones_to_ignore, exceptions)
        'CAMERA_EVENT_FILTERS': {},

        # Gemini proxy (AI analysis) - optional
        'GEMINI': None,

        # Multi-cam frame extractor (optional). No Google fallback: proxy URL default "".
        'MAX_MULTI_CAM_FRAMES_MIN': 45,
        'MAX_MULTI_CAM_FRAMES_SEC': 2,
        'MOTION_THRESHOLD_PX': 50,
        'CROP_WIDTH': 1280,
        'CROP_HEIGHT': 720,
        'NVENC_PROBE_WIDTH': 1280,
        'NVENC_PROBE_HEIGHT': 720,
        'MULTI_CAM_SYSTEM_PROMPT_FILE': '',
        'SMART_CROP_PADDING': 0.15,
        'MOTION_CROP_MIN_AREA_FRACTION': 0.001,
        'MOTION_CROP_MIN_PX': 500,
        'DETECTION_MODEL': 'yolov8n.pt',
        'DETECTION_DEVICE': '',  # Empty = auto (CUDA if available else CPU)
        'DETECTION_FRAME_INTERVAL': 5,
        'FIRST_CAMERA_BIAS_DECAY_SECONDS': 1.0,
        'FIRST_CAMERA_BIAS_INITIAL': 1.5,
        'FIRST_CAMERA_BIAS_CAP_SECONDS': 0,
        'PERSON_AREA_SWITCH_THRESHOLD': 200,
        'CAMERA_SWITCH_RATIO': 1.2,

        # Gemini proxy (extended): Single API Key (GEMINI_API_KEY only). Default URL "" (no Google fallback).
        'GEMINI_PROXY_URL': '',
        'GEMINI_PROXY_MODEL': 'gemini-2.5-flash-lite',
        'GEMINI_PROXY_TEMPERATURE': 0.3,
        'GEMINI_PROXY_TOP_P': 1,
        'GEMINI_PROXY_FREQUENCY_PENALTY': 0,
        'GEMINI_PROXY_PRESENCE_PENALTY': 0,
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

                # Validate schema
                try:
                    yaml_config = CONFIG_SCHEMA(yaml_config)
                except Invalid as e:
                    logger.error(f"Invalid configuration in {path}: {e}")
                    sys.exit(1)

                # Build camera-to-labels and event_filters mapping from per-camera config
                for cam in yaml_config['cameras']:
                    camera_name = cam['name']
                    labels = cam.get('labels', []) or []
                    config['CAMERA_LABEL_MAP'][camera_name] = labels

                    # Smart Zone Filtering (optional per camera)
                    event_filters = cam.get('event_filters')
                    if event_filters:
                        zones = event_filters.get('tracked_zones')
                        exceptions = event_filters.get('exceptions')
                        config['CAMERA_EVENT_FILTERS'][camera_name] = {
                            'tracked_zones': zones if zones else [],
                            'exceptions': [str(x).strip() for x in exceptions] if exceptions else [],
                        }

                # Derive flat lists for status/logging
                config['ALLOWED_CAMERAS'] = list(config['CAMERA_LABEL_MAP'].keys())
                config['ALLOWED_LABELS'] = list(set(
                    label for labels in config['CAMERA_LABEL_MAP'].values() for label in labels if labels
                ))

                if 'settings' in yaml_config:
                    settings = yaml_config['settings']
                    config['RETENTION_DAYS'] = settings.get('retention_days', config['RETENTION_DAYS'])
                    config['CLEANUP_INTERVAL_HOURS'] = settings.get('cleanup_interval_hours', config['CLEANUP_INTERVAL_HOURS'])
                    config['EXPORT_WATCHDOG_INTERVAL_MINUTES'] = settings.get('export_watchdog_interval_minutes', config['EXPORT_WATCHDOG_INTERVAL_MINUTES'])
                    config['FFMPEG_TIMEOUT'] = settings.get('ffmpeg_timeout_seconds', config['FFMPEG_TIMEOUT'])
                    config['NOTIFICATION_DELAY'] = settings.get('notification_delay_seconds', config['NOTIFICATION_DELAY'])
                    config['LOG_LEVEL'] = settings.get('log_level', config['LOG_LEVEL'])
                    config['SUMMARY_PADDING_BEFORE'] = settings.get('summary_padding_before', config['SUMMARY_PADDING_BEFORE'])
                    config['SUMMARY_PADDING_AFTER'] = settings.get('summary_padding_after', config['SUMMARY_PADDING_AFTER'])
                    config['STATS_REFRESH_SECONDS'] = settings.get('stats_refresh_seconds', config['STATS_REFRESH_SECONDS'])
                    config['DAILY_REVIEW_RETENTION_DAYS'] = settings.get('daily_review_retention_days', config['DAILY_REVIEW_RETENTION_DAYS'])
                    config['DAILY_REVIEW_SCHEDULE_HOUR'] = settings.get('daily_review_schedule_hour', config['DAILY_REVIEW_SCHEDULE_HOUR'])
                    config['DAILY_REPORT_SCHEDULE_HOUR'] = settings.get('daily_report_schedule_hour', config['DAILY_REPORT_SCHEDULE_HOUR'])
                    config['REPORT_PROMPT_FILE'] = (settings.get('report_prompt_file') or config['REPORT_PROMPT_FILE']) or ''
                    config['REPORT_KNOWN_PERSON_NAME'] = (settings.get('report_known_person_name') or config['REPORT_KNOWN_PERSON_NAME']) or ''
                    config['EVENT_GAP_SECONDS'] = settings.get('event_gap_seconds', config['EVENT_GAP_SECONDS'])
                    config['MINIMUM_EVENT_SECONDS'] = settings.get('minimum_event_seconds', config['MINIMUM_EVENT_SECONDS'])
                    config['EXPORT_BUFFER_BEFORE'] = settings.get('export_buffer_before', config['EXPORT_BUFFER_BEFORE'])
                    config['EXPORT_BUFFER_AFTER'] = settings.get('export_buffer_after', config['EXPORT_BUFFER_AFTER'])
                    config['FINAL_REVIEW_IMAGE_COUNT'] = settings.get('final_review_image_count', config.get('FINAL_REVIEW_IMAGE_COUNT', 20))
                    config['GEMINI_MAX_CONCURRENT_ANALYSES'] = settings.get('gemini_max_concurrent_analyses', config.get('GEMINI_MAX_CONCURRENT_ANALYSES', 3))
                    config['MAX_CONCURRENT_TRANSCODES'] = settings.get('max_concurrent_transcodes', config.get('MAX_CONCURRENT_TRANSCODES', 2))
                    config['SAVE_AI_FRAMES'] = settings.get('save_ai_frames', config.get('SAVE_AI_FRAMES', True))
                    config['CREATE_AI_ANALYSIS_ZIP'] = settings.get('create_ai_analysis_zip', config.get('CREATE_AI_ANALYSIS_ZIP', True))

                if 'network' in yaml_config:
                    network = yaml_config['network']
                    config['MQTT_BROKER'] = network.get('mqtt_broker', config['MQTT_BROKER'])
                    config['MQTT_PORT'] = network.get('mqtt_port', config['MQTT_PORT'])
                    config['MQTT_USER'] = network.get('mqtt_user', config['MQTT_USER'])
                    config['MQTT_PASSWORD'] = network.get('mqtt_password', config['MQTT_PASSWORD'])
                    config['FRIGATE_URL'] = network.get('frigate_url', config['FRIGATE_URL'])
                    config['BUFFER_IP'] = network.get('buffer_ip') or network.get('ha_ip') or config['BUFFER_IP']
                    config['FLASK_PORT'] = network.get('flask_port', config['FLASK_PORT'])
                    config['STORAGE_PATH'] = network.get('storage_path', config['STORAGE_PATH'])

                if 'ha' in yaml_config:
                    ha_cfg = yaml_config['ha']
                    config['HA_URL'] = ha_cfg.get('base_url') or ha_cfg.get('url') or config['HA_URL']
                    config['HA_TOKEN'] = ha_cfg.get('token') or config['HA_TOKEN']
                    config['HA_GEMINI_COST_ENTITY'] = ha_cfg.get('gemini_cost_entity', config['HA_GEMINI_COST_ENTITY'])
                    config['HA_GEMINI_TOKENS_ENTITY'] = ha_cfg.get('gemini_tokens_entity', config['HA_GEMINI_TOKENS_ENTITY'])

                if 'gemini' in yaml_config:
                    gemini_cfg = yaml_config['gemini']
                    config['GEMINI'] = {
                        'proxy_url': gemini_cfg.get('proxy_url', ''),
                        'api_key': gemini_cfg.get('api_key', ''),
                        'model': gemini_cfg.get('model', 'gemini-2.5-flash-lite'),
                        'enabled': bool(gemini_cfg.get('enabled', False)),
                    }

                if 'multi_cam' in yaml_config:
                    mc = yaml_config['multi_cam']
                    config['MAX_MULTI_CAM_FRAMES_MIN'] = mc.get('max_multi_cam_frames_min', config['MAX_MULTI_CAM_FRAMES_MIN'])
                    config['MAX_MULTI_CAM_FRAMES_SEC'] = mc.get('max_multi_cam_frames_sec', config['MAX_MULTI_CAM_FRAMES_SEC'])
                    config['MOTION_THRESHOLD_PX'] = mc.get('motion_threshold_px', config['MOTION_THRESHOLD_PX'])
                    config['CROP_WIDTH'] = mc.get('crop_width', config['CROP_WIDTH'])
                    config['CROP_HEIGHT'] = mc.get('crop_height', config['CROP_HEIGHT'])
                    config['NVENC_PROBE_WIDTH'] = mc.get('nvenc_probe_width', config['NVENC_PROBE_WIDTH'])
                    config['NVENC_PROBE_HEIGHT'] = mc.get('nvenc_probe_height', config['NVENC_PROBE_HEIGHT'])
                    config['MULTI_CAM_SYSTEM_PROMPT_FILE'] = mc.get('multi_cam_system_prompt_file', config['MULTI_CAM_SYSTEM_PROMPT_FILE']) or ''
                    config['SMART_CROP_PADDING'] = float(mc.get('smart_crop_padding', config.get('SMART_CROP_PADDING', 0.15)))
                    config['MOTION_CROP_MIN_AREA_FRACTION'] = float(mc.get('motion_crop_min_area_fraction', config.get('MOTION_CROP_MIN_AREA_FRACTION', 0.001)))
                    config['MOTION_CROP_MIN_PX'] = int(mc.get('motion_crop_min_px', config.get('MOTION_CROP_MIN_PX', 500)))
                    config['DETECTION_MODEL'] = (mc.get('detection_model') or config.get('DETECTION_MODEL', 'yolov8n.pt')) or 'yolov8n.pt'
                    config['DETECTION_DEVICE'] = (mc.get('detection_device') or config.get('DETECTION_DEVICE', '')) or ''
                    config['DETECTION_FRAME_INTERVAL'] = mc.get('detection_frame_interval', config['DETECTION_FRAME_INTERVAL'])
                    config['FIRST_CAMERA_BIAS_DECAY_SECONDS'] = float(mc.get('first_camera_bias_decay_seconds', config['FIRST_CAMERA_BIAS_DECAY_SECONDS']))
                    config['FIRST_CAMERA_BIAS_INITIAL'] = float(mc.get('first_camera_bias_initial', config['FIRST_CAMERA_BIAS_INITIAL']))
                    config['FIRST_CAMERA_BIAS_CAP_SECONDS'] = float(mc.get('first_camera_bias_cap_seconds', config['FIRST_CAMERA_BIAS_CAP_SECONDS']))
                    config['PERSON_AREA_SWITCH_THRESHOLD'] = int(mc.get('person_area_switch_threshold', config['PERSON_AREA_SWITCH_THRESHOLD']))
                    config['CAMERA_SWITCH_RATIO'] = float(mc.get('camera_switch_ratio', config['CAMERA_SWITCH_RATIO']))
                if 'gemini_proxy' in yaml_config:
                    gp = yaml_config['gemini_proxy']
                    config['GEMINI_PROXY_URL'] = gp.get('url', config['GEMINI_PROXY_URL']) or ''
                    config['GEMINI_PROXY_MODEL'] = gp.get('model', config['GEMINI_PROXY_MODEL']) or 'gemini-2.5-flash-lite'
                    config['GEMINI_PROXY_TEMPERATURE'] = float(gp.get('temperature', config['GEMINI_PROXY_TEMPERATURE']))
                    config['GEMINI_PROXY_TOP_P'] = float(gp.get('top_p', config['GEMINI_PROXY_TOP_P']))
                    config['GEMINI_PROXY_FREQUENCY_PENALTY'] = float(gp.get('frequency_penalty', config['GEMINI_PROXY_FREQUENCY_PENALTY']))
                    config['GEMINI_PROXY_PRESENCE_PENALTY'] = float(gp.get('presence_penalty', config['GEMINI_PROXY_PRESENCE_PENALTY']))
                elif 'gemini' in yaml_config and config.get('GEMINI'):
                    # Backward compat: derive proxy URL and model from gemini when gemini_proxy not set
                    config['GEMINI_PROXY_URL'] = (config['GEMINI'].get('proxy_url') or '') or config['GEMINI_PROXY_URL']
                    config['GEMINI_PROXY_MODEL'] = (config['GEMINI'].get('model') or 'gemini-2.5-flash-lite') or config['GEMINI_PROXY_MODEL']

                config_loaded = True
                break

            except Exception as e:
                logger.error(f"Error loading config from {path}: {e}")

    if not config_loaded:
        logger.info("No config.yaml found, using defaults")
    if config.get('GEMINI') is None:
        config['GEMINI'] = {'proxy_url': '', 'api_key': '', 'model': 'gemini-2.5-flash-lite', 'enabled': False}

    # Environment variables override everything (for secrets/deployment)
    config['MQTT_BROKER'] = os.getenv('MQTT_BROKER') or config['MQTT_BROKER']
    config['MQTT_PORT'] = int(os.getenv('MQTT_PORT', str(config['MQTT_PORT'])))
    config['MQTT_USER'] = os.getenv('MQTT_USER') or config['MQTT_USER']
    config['MQTT_PASSWORD'] = os.getenv('MQTT_PASSWORD') or config['MQTT_PASSWORD']
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
    config['DAILY_REPORT_SCHEDULE_HOUR'] = int(os.getenv('DAILY_REPORT_SCHEDULE_HOUR', str(config['DAILY_REPORT_SCHEDULE_HOUR'])))
    config['REPORT_KNOWN_PERSON_NAME'] = (os.getenv('REPORT_KNOWN_PERSON_NAME') or config['REPORT_KNOWN_PERSON_NAME']) or ''
    config['EVENT_GAP_SECONDS'] = int(os.getenv('EVENT_GAP_SECONDS', str(config['EVENT_GAP_SECONDS'])))
    config['EXPORT_WATCHDOG_INTERVAL_MINUTES'] = int(os.getenv('EXPORT_WATCHDOG_INTERVAL_MINUTES', str(config['EXPORT_WATCHDOG_INTERVAL_MINUTES'])))
    config['EXPORT_BUFFER_BEFORE'] = int(os.getenv('EXPORT_BUFFER_BEFORE', str(config['EXPORT_BUFFER_BEFORE'])))
    config['EXPORT_BUFFER_AFTER'] = int(os.getenv('EXPORT_BUFFER_AFTER', str(config['EXPORT_BUFFER_AFTER'])))
    config['HA_URL'] = os.getenv('HA_URL') or config['HA_URL']
    config['HA_TOKEN'] = os.getenv('HA_TOKEN') or config['HA_TOKEN']
    _save_ai = os.getenv('SAVE_AI_FRAMES')
    if _save_ai is not None:
        config['SAVE_AI_FRAMES'] = str(_save_ai).lower() in ('true', '1', 'yes')
    _create_zip = os.getenv('CREATE_AI_ANALYSIS_ZIP')
    if _create_zip is not None:
        config['CREATE_AI_ANALYSIS_ZIP'] = str(_create_zip).lower() in ('true', '1', 'yes')

    # Gemini env overrides (api_key for secrets). Single API Key: GEMINI_API_KEY only.
    config['GEMINI'] = dict(config.get('GEMINI') or {})
    config['GEMINI'].setdefault('proxy_url', '')
    config['GEMINI'].setdefault('api_key', '')
    config['GEMINI'].setdefault('model', 'gemini-2.5-flash-lite')
    config['GEMINI'].setdefault('enabled', False)
    config['GEMINI']['api_key'] = os.getenv('GEMINI_API_KEY') or config['GEMINI'].get('api_key') or ''

    # Gemini proxy URL override (no Google fallback; default remains "")
    config['GEMINI_PROXY_URL'] = (os.getenv('GEMINI_PROXY_URL') or config.get('GEMINI_PROXY_URL') or '').strip() or ''

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
