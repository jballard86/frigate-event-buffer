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
        Optional('ffmpeg_timeout_seconds'): int,                # Timeout for FFmpeg (e.g. GIF generation); kills hung processes.
        Optional('notification_delay_seconds'): int,            # Delay before fetching snapshot after notification (lets Frigate pick a better frame).
        Optional('log_level'): Any('DEBUG', 'INFO', 'WARNING', 'ERROR'),  # Logging verbosity.
        Optional('summary_padding_before'): int,                # Seconds before event start for Frigate review summary window.
        Optional('summary_padding_after'): int,                 # Seconds after event end for Frigate review summary window.
        Optional('stats_refresh_seconds'): int,                 # Stats page auto-refresh interval (seconds).
        Optional('daily_report_retention_days'): int,          # How long to keep saved daily reports (days).
        Optional('daily_report_schedule_hour'): int,            # Hour (0-23) to generate AI daily report from analysis results.
        Optional('report_prompt_file'): str,                    # Path to prompt file for daily report; empty = use default.
        Optional('report_known_person_name'): str,              # Placeholder value for {known_person_name} in report prompt.
        Optional('event_gap_seconds'): int,                     # Seconds of inactivity before next event starts a new consolidated group.
        Optional('minimum_event_seconds'): int,                 # Events shorter than this are discarded (data deleted, state/CE updated, discarded MQTT notification sent).
        Optional('max_event_length_seconds'): int,             # Events with duration >= this are canceled: no AI/decode, folder renamed with "-canceled", notification sent; default 120 (2 min).
        Optional('export_buffer_before'): int,                  # Seconds to include before event start in exported clip.
        Optional('export_buffer_after'): int,                    # Seconds to include after event end in exported clip.
        Optional('single_camera_ce_close_delay_seconds'): int,  # When CE has one camera, delay (s) before close; 0 = close as soon as event ends. Multi-cam uses event_gap_seconds.
        Optional('gemini_max_concurrent_analyses'): int,        # Max concurrent Gemini clip analyses (throttling).
        Optional('save_ai_frames'): bool,                        # Whether to save extracted AI analysis frames to disk.
        Optional('create_ai_analysis_zip'): bool,               # Whether to create a zip of AI analysis assets (e.g. for multi-cam).
        Optional('gemini_frames_per_hour_cap'): int,             # Rolling cap: max frames sent to proxy per hour; 0 = disabled.
        Optional('quick_title_delay_seconds'): int,              # Delay (3–5s) before running quick AI title on live frame; 0 = disabled.
        Optional('quick_title_enabled'): bool,                  # When true (and Gemini enabled), run quick-title pipeline after delay.
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
    # Multi-cam frame extraction (main app AI analyzer); frame limits and crop options.
    Optional('multi_cam'): {
        Optional('max_multi_cam_frames_min'): int,              # Maximum frames to extract per clip (cap).
        Optional('max_multi_cam_frames_sec'): Any(int, float),  # Target interval in seconds between captured frames (e.g. 1, 0.5, 1.5).
        Optional('crop_width'): int,                            # Width of output crop for stitched/multi-cam frames.
        Optional('crop_height'): int,                            # Height of output crop for stitched/multi-cam frames.
        Optional('multi_cam_system_prompt_file'): str,          # Path to system prompt file for multi-cam Gemini; empty = built-in.
        Optional('smart_crop_padding'): Any(int, float),        # Padding fraction around motion-based crop (e.g. 0.15).
        Optional('detection_model'): str,                         # Ultralytics model for detection sidecar (e.g. yolov8n.pt).
        Optional('detection_device'): str,                       # Device for detection (e.g. cuda:0, cpu).
        Optional('detection_frame_interval'): int,               # Run YOLO every N frames; default 5.
        Optional('detection_imgsz'): int,                        # YOLO inference size (higher = better small objects, slower); default 640.
        # Timeline / EMA (Phase 1 dense grid + EMA + hysteresis + merge; sole assignment path)
        Optional('camera_timeline_analysis_multiplier'): Any(int, float),  # Denser analysis grid vs base step (2 or 3); default 2.
        Optional('camera_timeline_ema_alpha'): Any(int, float),            # EMA smoothing factor 0–1; default 0.4.
        Optional('camera_timeline_primary_bias_multiplier'): Any(int, float),  # Weight on primary camera area curve; default 1.2.
        Optional('camera_switch_min_segment_frames'): int,                # Min frames per segment; short runs merged into previous; default 5.
        Optional('camera_switch_hysteresis_margin'): Any(int, float),     # New camera must exceed current by this factor to switch (e.g. 1.15 = 15%); default 1.15.
        Optional('camera_timeline_final_yolo_drop_no_person'): bool,     # If true drop frames with no person after Phase 2 YOLO; default False (keep all, tag).
        Optional('decode_second_camera_cpu_only'): bool,                # If true, use CPU decode for 2nd+ cameras (contention workaround). Default false: NVDEC for all; CPU only when NVDEC fails.
        Optional('log_extraction_phase_timing'): bool,                  # Log elapsed time per extraction phase (e.g. "Opening clips: 0.5s") for debugging; default false.
        Optional('merge_frame_timeout_sec'): int,                       # Timeout (seconds) when merge waits for a camera frame; on timeout camera is dropped from active pool; default 10.
        Optional('tracking_target_frame_percent'): int,                 # When person area >= this % of reference area, use full-frame resize; default 40.
        Optional('person_area_debug'): bool,                             # Draw person area (px²) on frame bottom-right when true; default false.
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
        'DAILY_REPORT_RETENTION_DAYS': 90,
        'DAILY_REPORT_SCHEDULE_HOUR': 1,
        'REPORT_PROMPT_FILE': '',
        'REPORT_KNOWN_PERSON_NAME': '',
        'EVENT_GAP_SECONDS': 120,
        'MINIMUM_EVENT_SECONDS': 5,
        'MAX_EVENT_LENGTH_SECONDS': 120,
        'EXPORT_BUFFER_BEFORE': 5,
        'EXPORT_BUFFER_AFTER': 30,
        'SINGLE_CAMERA_CE_CLOSE_DELAY_SECONDS': 0,
        'GEMINI_MAX_CONCURRENT_ANALYSES': 3,
        'SAVE_AI_FRAMES': True,
        'CREATE_AI_ANALYSIS_ZIP': True,
        'GEMINI_FRAMES_PER_HOUR_CAP': 200,
        'QUICK_TITLE_DELAY_SECONDS': 4,
        'QUICK_TITLE_ENABLED': True,

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
        'CROP_WIDTH': 1280,
        'CROP_HEIGHT': 720,
        'MULTI_CAM_SYSTEM_PROMPT_FILE': '',
        'SMART_CROP_PADDING': 0.15,
        'DETECTION_MODEL': 'yolov8n.pt',
        'DETECTION_DEVICE': '',  # Empty = auto (CUDA if available else CPU)
        'DETECTION_FRAME_INTERVAL': 5,
        'DETECTION_IMGSZ': 640,
        # Timeline / EMA (Phase 1 dense grid + EMA + hysteresis + merge; sole path)
        'CAMERA_TIMELINE_ANALYSIS_MULTIPLIER': 2,
        'CAMERA_TIMELINE_EMA_ALPHA': 0.4,
        'CAMERA_TIMELINE_PRIMARY_BIAS_MULTIPLIER': 1.2,
        'CAMERA_SWITCH_MIN_SEGMENT_FRAMES': 5,
        'CAMERA_SWITCH_HYSTERESIS_MARGIN': 1.15,
        'CAMERA_TIMELINE_FINAL_YOLO_DROP_NO_PERSON': False,
        'DECODE_SECOND_CAMERA_CPU_ONLY': False,
        'LOG_EXTRACTION_PHASE_TIMING': False,
        'MERGE_FRAME_TIMEOUT_SEC': 10,
        'TRACKING_TARGET_FRAME_PERCENT': 40,
        'PERSON_AREA_DEBUG': False,

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
                    config['DAILY_REPORT_RETENTION_DAYS'] = settings.get('daily_report_retention_days', config['DAILY_REPORT_RETENTION_DAYS'])
                    config['DAILY_REPORT_SCHEDULE_HOUR'] = settings.get('daily_report_schedule_hour', config['DAILY_REPORT_SCHEDULE_HOUR'])
                    config['REPORT_PROMPT_FILE'] = (settings.get('report_prompt_file') or config['REPORT_PROMPT_FILE']) or ''
                    config['REPORT_KNOWN_PERSON_NAME'] = (settings.get('report_known_person_name') or config['REPORT_KNOWN_PERSON_NAME']) or ''
                    config['EVENT_GAP_SECONDS'] = settings.get('event_gap_seconds', config['EVENT_GAP_SECONDS'])
                    config['MINIMUM_EVENT_SECONDS'] = settings.get('minimum_event_seconds', config['MINIMUM_EVENT_SECONDS'])
                    config['MAX_EVENT_LENGTH_SECONDS'] = settings.get('max_event_length_seconds', config['MAX_EVENT_LENGTH_SECONDS'])
                    config['EXPORT_BUFFER_BEFORE'] = settings.get('export_buffer_before', config['EXPORT_BUFFER_BEFORE'])
                    config['EXPORT_BUFFER_AFTER'] = settings.get('export_buffer_after', config['EXPORT_BUFFER_AFTER'])
                    config['SINGLE_CAMERA_CE_CLOSE_DELAY_SECONDS'] = settings.get('single_camera_ce_close_delay_seconds', config.get('SINGLE_CAMERA_CE_CLOSE_DELAY_SECONDS', 0))
                    config['GEMINI_MAX_CONCURRENT_ANALYSES'] = settings.get('gemini_max_concurrent_analyses', config.get('GEMINI_MAX_CONCURRENT_ANALYSES', 3))
                    config['SAVE_AI_FRAMES'] = settings.get('save_ai_frames', config.get('SAVE_AI_FRAMES', True))
                    config['CREATE_AI_ANALYSIS_ZIP'] = settings.get('create_ai_analysis_zip', config.get('CREATE_AI_ANALYSIS_ZIP', True))
                    config['GEMINI_FRAMES_PER_HOUR_CAP'] = settings.get('gemini_frames_per_hour_cap', config.get('GEMINI_FRAMES_PER_HOUR_CAP', 200))
                    config['QUICK_TITLE_DELAY_SECONDS'] = settings.get('quick_title_delay_seconds', config.get('QUICK_TITLE_DELAY_SECONDS', 4))
                    config['QUICK_TITLE_ENABLED'] = settings.get('quick_title_enabled', config.get('QUICK_TITLE_ENABLED', True))

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
                    config['MAX_MULTI_CAM_FRAMES_SEC'] = float(mc.get('max_multi_cam_frames_sec', config['MAX_MULTI_CAM_FRAMES_SEC']))
                    config['CROP_WIDTH'] = mc.get('crop_width', config['CROP_WIDTH'])
                    config['CROP_HEIGHT'] = mc.get('crop_height', config['CROP_HEIGHT'])
                    config['MULTI_CAM_SYSTEM_PROMPT_FILE'] = mc.get('multi_cam_system_prompt_file', config['MULTI_CAM_SYSTEM_PROMPT_FILE']) or ''
                    config['SMART_CROP_PADDING'] = float(mc.get('smart_crop_padding', config.get('SMART_CROP_PADDING', 0.15)))
                    config['DETECTION_MODEL'] = (mc.get('detection_model') or config.get('DETECTION_MODEL', 'yolov8n.pt')) or 'yolov8n.pt'
                    config['DETECTION_DEVICE'] = (mc.get('detection_device') or config.get('DETECTION_DEVICE', '')) or ''
                    config['DETECTION_FRAME_INTERVAL'] = mc.get('detection_frame_interval', config['DETECTION_FRAME_INTERVAL'])
                    config['DETECTION_IMGSZ'] = int(mc.get('detection_imgsz', config['DETECTION_IMGSZ']))
                    config['CAMERA_TIMELINE_ANALYSIS_MULTIPLIER'] = float(mc.get('camera_timeline_analysis_multiplier', config['CAMERA_TIMELINE_ANALYSIS_MULTIPLIER']))
                    config['CAMERA_TIMELINE_EMA_ALPHA'] = float(mc.get('camera_timeline_ema_alpha', config['CAMERA_TIMELINE_EMA_ALPHA']))
                    config['CAMERA_TIMELINE_PRIMARY_BIAS_MULTIPLIER'] = float(mc.get('camera_timeline_primary_bias_multiplier', config['CAMERA_TIMELINE_PRIMARY_BIAS_MULTIPLIER']))
                    config['CAMERA_SWITCH_MIN_SEGMENT_FRAMES'] = int(mc.get('camera_switch_min_segment_frames', config['CAMERA_SWITCH_MIN_SEGMENT_FRAMES']))
                    config['CAMERA_SWITCH_HYSTERESIS_MARGIN'] = float(mc.get('camera_switch_hysteresis_margin', config['CAMERA_SWITCH_HYSTERESIS_MARGIN']))
                    config['CAMERA_TIMELINE_FINAL_YOLO_DROP_NO_PERSON'] = bool(mc.get('camera_timeline_final_yolo_drop_no_person', config['CAMERA_TIMELINE_FINAL_YOLO_DROP_NO_PERSON']))
                    config['DECODE_SECOND_CAMERA_CPU_ONLY'] = bool(mc.get('decode_second_camera_cpu_only', config['DECODE_SECOND_CAMERA_CPU_ONLY']))
                    config['LOG_EXTRACTION_PHASE_TIMING'] = bool(mc.get('log_extraction_phase_timing', config['LOG_EXTRACTION_PHASE_TIMING']))
                    config['MERGE_FRAME_TIMEOUT_SEC'] = int(mc.get('merge_frame_timeout_sec', config['MERGE_FRAME_TIMEOUT_SEC']))
                    config['TRACKING_TARGET_FRAME_PERCENT'] = int(mc.get('tracking_target_frame_percent', config['TRACKING_TARGET_FRAME_PERCENT']))
                    config['PERSON_AREA_DEBUG'] = bool(mc.get('person_area_debug', config['PERSON_AREA_DEBUG']))
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
    config['DAILY_REPORT_RETENTION_DAYS'] = int(os.getenv('DAILY_REPORT_RETENTION_DAYS', str(config['DAILY_REPORT_RETENTION_DAYS'])))
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
    _cap = os.getenv('GEMINI_FRAMES_PER_HOUR_CAP')
    if _cap is not None:
        try:
            config['GEMINI_FRAMES_PER_HOUR_CAP'] = int(_cap)
        except ValueError:
            pass

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
