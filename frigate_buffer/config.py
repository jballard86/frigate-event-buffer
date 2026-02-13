"""Configuration loading and validation."""

import os
import logging

import yaml

logger = logging.getLogger('frigate-buffer')


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

                # Build camera-to-labels and event_filters mapping from per-camera config
                if 'cameras' in yaml_config and isinstance(yaml_config['cameras'], list):
                    for cam in yaml_config['cameras']:
                        if isinstance(cam, dict) and 'name' in cam:
                            camera_name = cam['name']
                            labels = cam.get('labels', []) or []
                            config['CAMERA_LABEL_MAP'][camera_name] = labels

                            # Smart Zone Filtering (optional per camera)
                            event_filters = cam.get('event_filters')
                            if isinstance(event_filters, dict):
                                zones = event_filters.get('zones_to_ignore')
                                exceptions = event_filters.get('exceptions')
                                config['CAMERA_EVENT_FILTERS'][camera_name] = {
                                    'zones_to_ignore': zones if isinstance(zones, list) else [],
                                    'exceptions': [str(x).strip() for x in exceptions] if isinstance(exceptions, list) else [],
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

                if 'network' in yaml_config:
                    network = yaml_config['network']
                    config['MQTT_BROKER'] = network.get('mqtt_broker', config['MQTT_BROKER'])
                    config['MQTT_PORT'] = network.get('mqtt_port', config['MQTT_PORT'])
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
    config['HA_URL'] = os.getenv('HA_URL') or config['HA_URL']
    config['HA_TOKEN'] = os.getenv('HA_TOKEN') or config['HA_TOKEN']

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
