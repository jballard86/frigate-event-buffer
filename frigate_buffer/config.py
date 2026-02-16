"""Configuration loading and validation."""

import os
import logging
import sys

import yaml
from voluptuous import Schema, Required, Optional, Any, ALLOW_EXTRA, Invalid

logger = logging.getLogger('frigate-buffer')


# Configuration Schema
CONFIG_SCHEMA = Schema({
    Required('cameras'): [{
        Required('name'): str,
        Optional('labels'): [str],
        Optional('event_filters'): {
            Optional('tracked_zones'): [str],
            Optional('exceptions'): [str],
        }
    }],
    Optional('network'): {
        Optional('mqtt_broker'): str,
        Optional('mqtt_port'): int,
        Optional('mqtt_user'): str,
        Optional('mqtt_password'): str,
        Optional('frigate_url'): str,
        Optional('buffer_ip'): str,
        Optional('flask_port'): int,
        Optional('storage_path'): str,
        Optional('ha_ip'): str,  # Legacy fallback
    },
    Optional('settings'): {
        Optional('retention_days'): int,
        Optional('cleanup_interval_hours'): int,
        Optional('ffmpeg_timeout_seconds'): int,
        Optional('notification_delay_seconds'): int,
        Optional('log_level'): Any('DEBUG', 'INFO', 'WARNING', 'ERROR'),
        Optional('summary_padding_before'): int,
        Optional('summary_padding_after'): int,
        Optional('stats_refresh_seconds'): int,
        Optional('daily_review_retention_days'): int,
        Optional('daily_review_schedule_hour'): int,
        Optional('event_gap_seconds'): int,
        Optional('export_buffer_before'): int,
        Optional('export_buffer_after'): int,
        Optional('final_review_image_count'): int,
    },
    Optional('ha'): {
        Optional('base_url'): str,
        Optional('url'): str,
        Optional('token'): str,
        Optional('gemini_cost_entity'): str,
        Optional('gemini_tokens_entity'): str,
    },
    Optional('gemini'): {
        Optional('proxy_url'): str,
        Optional('api_key'): str,
        Optional('model'): str,
        Optional('enabled'): bool,
    },
    Optional('multi_cam'): {
        Optional('max_multi_cam_frames_min'): int,
        Optional('max_multi_cam_frames_sec'): int,
        Optional('motion_threshold_px'): int,
        Optional('crop_width'): int,
        Optional('crop_height'): int,
        Optional('multi_cam_system_prompt_file'): str,
        Optional('smart_crop_padding'): Any(int, float),
    },
    Optional('gemini_proxy'): {
        Optional('url'): str,
        Optional('model'): str,
        Optional('temperature'): Any(int, float),
        Optional('top_p'): Any(int, float),
        Optional('frequency_penalty'): Any(int, float),
        Optional('presence_penalty'): Any(int, float),
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
        'FINAL_REVIEW_IMAGE_COUNT': 20,

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
        'MULTI_CAM_SYSTEM_PROMPT_FILE': '',
        'SMART_CROP_PADDING': 0.15,

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
                    config['FINAL_REVIEW_IMAGE_COUNT'] = settings.get('final_review_image_count', config.get('FINAL_REVIEW_IMAGE_COUNT', 20))

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
                    config['MULTI_CAM_SYSTEM_PROMPT_FILE'] = mc.get('multi_cam_system_prompt_file', config['MULTI_CAM_SYSTEM_PROMPT_FILE']) or ''
                    config['SMART_CROP_PADDING'] = float(mc.get('smart_crop_padding', config.get('SMART_CROP_PADDING', 0.15)))

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
    config['EVENT_GAP_SECONDS'] = int(os.getenv('EVENT_GAP_SECONDS', str(config['EVENT_GAP_SECONDS'])))
    config['EXPORT_BUFFER_BEFORE'] = int(os.getenv('EXPORT_BUFFER_BEFORE', str(config['EXPORT_BUFFER_BEFORE'])))
    config['EXPORT_BUFFER_AFTER'] = int(os.getenv('EXPORT_BUFFER_AFTER', str(config['EXPORT_BUFFER_AFTER'])))
    config['HA_URL'] = os.getenv('HA_URL') or config['HA_URL']
    config['HA_TOKEN'] = os.getenv('HA_TOKEN') or config['HA_TOKEN']

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
