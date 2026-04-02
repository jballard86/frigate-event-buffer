"""Configuration loading and validation."""

import logging
import os
import sys

import yaml
from voluptuous import ALLOW_EXTRA, Invalid, Optional, Required, Schema

from frigate_buffer.config_schema.cameras import CAMERAS_LIST_SCHEMA
from frigate_buffer.config_schema.gemini_block import GEMINI_BLOCK_SCHEMA
from frigate_buffer.config_schema.gemini_proxy import GEMINI_PROXY_SCHEMA
from frigate_buffer.config_schema.ha import HA_SCHEMA
from frigate_buffer.config_schema.multi_cam import MULTI_CAM_SCHEMA
from frigate_buffer.config_schema.network import NETWORK_SCHEMA
from frigate_buffer.config_schema.notifications import NOTIFICATIONS_SCHEMA
from frigate_buffer.config_schema.settings import SETTINGS_SCHEMA
from frigate_buffer.constants import AI_MODE_EXTERNAL_API, AI_MODE_FRIGATE

logger = logging.getLogger("frigate-buffer")


# Configuration Schema
CONFIG_SCHEMA = Schema(
    {
        # List of cameras to process; only events from these cameras are ingested.
        Required("cameras"): CAMERAS_LIST_SCHEMA,
        # Network and storage; MQTT, Frigate URL, buffer IP required at runtime.
        Optional("network"): NETWORK_SCHEMA,
        Optional("settings"): SETTINGS_SCHEMA,
        Optional("ha"): HA_SCHEMA,
        Optional("notifications"): NOTIFICATIONS_SCHEMA,
        Optional("gemini"): GEMINI_BLOCK_SCHEMA,
        Optional("multi_cam"): MULTI_CAM_SCHEMA,
        Optional("gemini_proxy"): GEMINI_PROXY_SCHEMA,
    },
    extra=ALLOW_EXTRA,
)


def effective_gpu_device_index(config: dict) -> int:
    """Return the configured GPU adapter index for decode and runtime (vendor-neutral).

    Prefer ``GPU_DEVICE_INDEX``; ``CUDA_DEVICE_INDEX`` is a deprecated alias set by
    :func:`load_config` to the same value after merge.
    """
    if "GPU_DEVICE_INDEX" in config:
        return int(config["GPU_DEVICE_INDEX"])
    return int(config.get("CUDA_DEVICE_INDEX", 0))


def _finalize_gpu_vendor_and_device(config: dict) -> None:
    """Normalize GPU_VENDOR (nvidia | intel | amd), set CUDA_* mirror."""
    raw = config.get("GPU_VENDOR") or "nvidia"
    vendor = str(raw).strip().lower() if raw is not None else "nvidia"
    if not vendor:
        vendor = "nvidia"
    config["GPU_VENDOR"] = vendor
    if vendor not in ("nvidia", "intel", "amd"):
        raise ValueError(
            f"GPU_VENDOR={vendor!r} is not supported; use 'nvidia', 'intel', or "
            "'amd' (see docs/Multi_GPU_Support_Integration_Plan/)."
        )
    idx = effective_gpu_device_index(config)
    config["GPU_DEVICE_INDEX"] = idx
    config["CUDA_DEVICE_INDEX"] = idx


def _coerce_bool(value: object, default: bool) -> bool:
    """Coerce config to bool; 'false'/'true' strings become False/True.

    Avoids Python's bool('false') == True. Used for notification enabled flags.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in ("false", "0", "no", "off"):
            return False
        if lower in ("true", "1", "yes", "on"):
            return True
    return bool(value)


def load_config(*, yaml_path: str | None = None) -> dict:
    """Load configuration from config.yaml merged with environment variables.

    Priority (highest to lowest):
    1. Environment variables
    2. config.yaml
    3. Default values

    Note: MQTT_BROKER, FRIGATE_URL, and BUFFER_IP are REQUIRED via config or env.

    Args:
        yaml_path: If set, only this path is tried for YAML (for tests and snapshot
            tooling). If the file is missing, behaves like no YAML was loaded.
    """
    config = {
        # Network settings - NO DEFAULTS (required from config)
        "MQTT_BROKER": None,
        "MQTT_PORT": 1883,
        "MQTT_USER": None,
        "MQTT_PASSWORD": None,
        "FRIGATE_URL": None,
        "BUFFER_IP": None,
        "FLASK_PORT": 5055,
        "FLASK_HOST": "0.0.0.0",
        "STORAGE_PATH": "/app/storage",
        # Settings defaults
        "RETENTION_DAYS": 3,
        "CLEANUP_INTERVAL_HOURS": 1,
        "EXPORT_WATCHDOG_INTERVAL_MINUTES": 2,
        "FFMPEG_TIMEOUT": 60,
        "NOTIFICATION_DELAY": 2,
        "LOG_LEVEL": "INFO",
        "SUMMARY_PADDING_BEFORE": 15,
        "SUMMARY_PADDING_AFTER": 15,
        "STATS_REFRESH_SECONDS": 60,
        "DAILY_REPORT_RETENTION_DAYS": 90,
        "DAILY_REPORT_SCHEDULE_HOUR": 1,
        "REPORT_PROMPT_FILE": "",
        "REPORT_KNOWN_PERSON_NAME": "",
        "EVENT_GAP_SECONDS": 120,
        "MINIMUM_EVENT_SECONDS": 5,
        "MAX_EVENT_LENGTH_SECONDS": 120,
        "EXPORT_BUFFER_BEFORE": 5,
        "EXPORT_BUFFER_AFTER": 30,
        "GEMINI_MAX_CONCURRENT_ANALYSES": 3,
        "SAVE_AI_FRAMES": True,
        "CREATE_AI_ANALYSIS_ZIP": True,
        "GEMINI_FRAMES_PER_HOUR_CAP": 200,
        "QUICK_TITLE_DELAY_SECONDS": 4,
        "QUICK_TITLE_ENABLED": True,
        "AI_MODE": AI_MODE_EXTERNAL_API,
        # Optional HA REST API (for stats page token/cost display)
        "HA_URL": None,
        "HA_TOKEN": None,
        "HA_GEMINI_COST_ENTITY": "input_number.gemini_daily_cost",
        "HA_GEMINI_TOKENS_ENTITY": "input_number.gemini_total_tokens",
        # HA notifications opt-in; True adds HomeAssistantMqttProvider.
        "NOTIFICATIONS_HOME_ASSISTANT_ENABLED": False,
        # Pushover: optional; orchestrator adds PushoverProvider when enabled.
        "pushover": {},
        # Mobile app (FCM): optional; when True, Firebase init runs if
        # credentials present.
        "NOTIFICATIONS_MOBILE_APP_ENABLED": False,
        "MOBILE_APP_GOOGLE_APPLICATION_CREDENTIALS": "",
        "MOBILE_APP_FIREBASE_PROJECT_ID": "",
        # Filtering defaults (empty = allow all)
        "ALLOWED_CAMERAS": [],
        "ALLOWED_LABELS": [],
        "CAMERA_LABEL_MAP": {},
        # Smart Zone Filtering: per-camera event_filters (zones_to_ignore, exceptions)
        "CAMERA_EVENT_FILTERS": {},
        # Gemini proxy (AI analysis) - optional
        "GEMINI": None,
        # Multi-cam frame extractor. No Google fallback: proxy URL default "".
        "MAX_MULTI_CAM_FRAMES_MIN": 45,
        "MAX_MULTI_CAM_FRAMES_SEC": 2,
        "CROP_WIDTH": 1280,
        "CROP_HEIGHT": 720,
        "MULTI_CAM_SYSTEM_PROMPT_FILE": "",
        "SMART_CROP_PADDING": 0.15,
        "DETECTION_MODEL": "yolov8n.pt",
        "DETECTION_DEVICE": "",  # Empty = auto (CUDA if available else CPU)
        # GPU backend (gpu-01+); load_config mirrors CUDA_DEVICE_INDEX (legacy).
        "GPU_VENDOR": "nvidia",
        "GPU_DEVICE_INDEX": 0,
        "DETECTION_FRAME_INTERVAL": 5,
        "DETECTION_IMGSZ": 640,
        # Timeline / EMA (Phase 1 dense grid + EMA + hysteresis + merge; sole path)
        "CAMERA_TIMELINE_ANALYSIS_MULTIPLIER": 2,
        "CAMERA_TIMELINE_EMA_ALPHA": 0.4,
        "CAMERA_TIMELINE_PRIMARY_BIAS_MULTIPLIER": 1.2,
        "CAMERA_SWITCH_MIN_SEGMENT_FRAMES": 5,
        "CAMERA_SWITCH_HYSTERESIS_MARGIN": 1.15,
        "CAMERA_TIMELINE_FINAL_YOLO_DROP_NO_PERSON": False,
        "DECODE_SECOND_CAMERA_CPU_ONLY": False,
        "LOG_EXTRACTION_PHASE_TIMING": False,
        "MERGE_FRAME_TIMEOUT_SEC": 10,
        "TRACKING_TARGET_FRAME_PERCENT": 40,
        "PERSON_AREA_DEBUG": False,
        "COMPILATION_ZOOM_SMOOTH_EMA_ALPHA": 0.25,
        # Intel h264_qsv compilation encode (multi_cam.intel or env; see INSTALL).
        "INTEL_QSV_ENCODE_PRESET": "medium",
        "INTEL_QSV_ENCODE_GLOBAL_QUALITY": 24,
        # Gemini proxy (extended): Single API Key (GEMINI_API_KEY). URL default "".
        "GEMINI_PROXY_URL": "",
        "GEMINI_PROXY_MODEL": "gemini-2.5-flash-lite",
        "GEMINI_PROXY_TEMPERATURE": 0.3,
        "GEMINI_PROXY_TOP_P": 1,
        "GEMINI_PROXY_FREQUENCY_PENALTY": 0,
        "GEMINI_PROXY_PRESENCE_PENALTY": 0,
    }

    # Load from config.yaml if exists
    if yaml_path is not None:
        config_paths = [yaml_path]
    else:
        config_paths = [
            "/app/config.yaml",
            "/app/storage/config.yaml",
            "./config.yaml",
            "config.yaml",
        ]
    config_loaded = False

    for path in config_paths:
        if os.path.exists(path):
            try:
                logger.info(f"Loading config from {path}")
                with open(path) as f:
                    yaml_config = yaml.safe_load(f) or {}

                # Validate schema
                try:
                    yaml_config = CONFIG_SCHEMA(yaml_config)
                except Invalid as e:
                    logger.error(f"Invalid configuration in {path}: {e}")
                    sys.exit(1)

                # Build camera-to-labels and event_filters from per-camera config
                for cam in yaml_config["cameras"]:
                    camera_name = cam["name"]
                    labels = cam.get("labels", []) or []
                    config["CAMERA_LABEL_MAP"][camera_name] = labels

                    # Smart Zone Filtering (optional per camera)
                    event_filters = cam.get("event_filters")
                    if event_filters:
                        zones = event_filters.get("tracked_zones")
                        exceptions = event_filters.get("exceptions")
                        config["CAMERA_EVENT_FILTERS"][camera_name] = {
                            "tracked_zones": zones if zones else [],
                            "exceptions": [str(x).strip() for x in exceptions]
                            if exceptions
                            else [],
                        }

                # Derive flat lists for status/logging
                config["ALLOWED_CAMERAS"] = list(config["CAMERA_LABEL_MAP"].keys())
                config["ALLOWED_LABELS"] = list(
                    {
                        label
                        for labels in config["CAMERA_LABEL_MAP"].values()
                        for label in labels
                        if labels
                    }
                )

                if "settings" in yaml_config:
                    settings = yaml_config["settings"]
                    config["RETENTION_DAYS"] = settings.get(
                        "retention_days", config["RETENTION_DAYS"]
                    )
                    config["CLEANUP_INTERVAL_HOURS"] = settings.get(
                        "cleanup_interval_hours", config["CLEANUP_INTERVAL_HOURS"]
                    )
                    config["EXPORT_WATCHDOG_INTERVAL_MINUTES"] = settings.get(
                        "export_watchdog_interval_minutes",
                        config["EXPORT_WATCHDOG_INTERVAL_MINUTES"],
                    )
                    config["FFMPEG_TIMEOUT"] = settings.get(
                        "ffmpeg_timeout_seconds", config["FFMPEG_TIMEOUT"]
                    )
                    config["NOTIFICATION_DELAY"] = settings.get(
                        "notification_delay_seconds", config["NOTIFICATION_DELAY"]
                    )
                    config["LOG_LEVEL"] = settings.get("log_level", config["LOG_LEVEL"])
                    config["SUMMARY_PADDING_BEFORE"] = settings.get(
                        "summary_padding_before", config["SUMMARY_PADDING_BEFORE"]
                    )
                    config["SUMMARY_PADDING_AFTER"] = settings.get(
                        "summary_padding_after", config["SUMMARY_PADDING_AFTER"]
                    )
                    config["STATS_REFRESH_SECONDS"] = settings.get(
                        "stats_refresh_seconds", config["STATS_REFRESH_SECONDS"]
                    )
                    config["DAILY_REPORT_RETENTION_DAYS"] = settings.get(
                        "daily_report_retention_days",
                        config["DAILY_REPORT_RETENTION_DAYS"],
                    )
                    config["DAILY_REPORT_SCHEDULE_HOUR"] = settings.get(
                        "daily_report_schedule_hour",
                        config["DAILY_REPORT_SCHEDULE_HOUR"],
                    )
                    config["REPORT_PROMPT_FILE"] = (
                        settings.get("report_prompt_file")
                        or config["REPORT_PROMPT_FILE"]
                    ) or ""
                    config["REPORT_KNOWN_PERSON_NAME"] = (
                        settings.get("report_known_person_name")
                        or config["REPORT_KNOWN_PERSON_NAME"]
                    ) or ""
                    config["EVENT_GAP_SECONDS"] = settings.get(
                        "event_gap_seconds", config["EVENT_GAP_SECONDS"]
                    )
                    config["MINIMUM_EVENT_SECONDS"] = settings.get(
                        "minimum_event_seconds", config["MINIMUM_EVENT_SECONDS"]
                    )
                    config["MAX_EVENT_LENGTH_SECONDS"] = settings.get(
                        "max_event_length_seconds",
                        config["MAX_EVENT_LENGTH_SECONDS"],
                    )
                    config["EXPORT_BUFFER_BEFORE"] = settings.get(
                        "export_buffer_before", config["EXPORT_BUFFER_BEFORE"]
                    )
                    config["EXPORT_BUFFER_AFTER"] = settings.get(
                        "export_buffer_after", config["EXPORT_BUFFER_AFTER"]
                    )
                    config["GEMINI_MAX_CONCURRENT_ANALYSES"] = settings.get(
                        "gemini_max_concurrent_analyses",
                        config.get("GEMINI_MAX_CONCURRENT_ANALYSES", 3),
                    )
                    config["SAVE_AI_FRAMES"] = settings.get(
                        "save_ai_frames", config.get("SAVE_AI_FRAMES", True)
                    )
                    config["CREATE_AI_ANALYSIS_ZIP"] = settings.get(
                        "create_ai_analysis_zip",
                        config.get("CREATE_AI_ANALYSIS_ZIP", True),
                    )
                    config["GEMINI_FRAMES_PER_HOUR_CAP"] = settings.get(
                        "gemini_frames_per_hour_cap",
                        config.get("GEMINI_FRAMES_PER_HOUR_CAP", 200),
                    )
                    config["QUICK_TITLE_DELAY_SECONDS"] = settings.get(
                        "quick_title_delay_seconds",
                        config.get("QUICK_TITLE_DELAY_SECONDS", 4),
                    )
                    config["QUICK_TITLE_ENABLED"] = settings.get(
                        "quick_title_enabled",
                        config.get("QUICK_TITLE_ENABLED", True),
                    )
                    _ai_mode = settings.get(
                        "ai_mode", config.get("AI_MODE", AI_MODE_EXTERNAL_API)
                    )
                    config["AI_MODE"] = (
                        (_ai_mode or AI_MODE_EXTERNAL_API).strip().lower()
                        if isinstance(_ai_mode, str)
                        else AI_MODE_EXTERNAL_API
                    )

                if "network" in yaml_config:
                    network = yaml_config["network"]
                    config["MQTT_BROKER"] = network.get(
                        "mqtt_broker", config["MQTT_BROKER"]
                    )
                    config["MQTT_PORT"] = network.get("mqtt_port", config["MQTT_PORT"])
                    config["MQTT_USER"] = network.get("mqtt_user", config["MQTT_USER"])
                    config["MQTT_PASSWORD"] = network.get(
                        "mqtt_password", config["MQTT_PASSWORD"]
                    )
                    config["FRIGATE_URL"] = network.get(
                        "frigate_url", config["FRIGATE_URL"]
                    )
                    config["BUFFER_IP"] = (
                        network.get("buffer_ip")
                        or network.get("ha_ip")
                        or config["BUFFER_IP"]
                    )
                    config["FLASK_PORT"] = network.get(
                        "flask_port", config["FLASK_PORT"]
                    )
                    config["FLASK_HOST"] = network.get(
                        "flask_host", config["FLASK_HOST"]
                    )
                    config["STORAGE_PATH"] = network.get(
                        "storage_path", config["STORAGE_PATH"]
                    )

                if "ha" in yaml_config:
                    ha_cfg = yaml_config["ha"]
                    config["HA_URL"] = (
                        ha_cfg.get("base_url") or ha_cfg.get("url") or config["HA_URL"]
                    )
                    config["HA_TOKEN"] = ha_cfg.get("token") or config["HA_TOKEN"]
                    config["HA_GEMINI_COST_ENTITY"] = ha_cfg.get(
                        "gemini_cost_entity", config["HA_GEMINI_COST_ENTITY"]
                    )
                    config["HA_GEMINI_TOKENS_ENTITY"] = ha_cfg.get(
                        "gemini_tokens_entity",
                        config["HA_GEMINI_TOKENS_ENTITY"],
                    )

                if "notifications" in yaml_config:
                    notif = yaml_config["notifications"]
                    logger.debug(
                        "notifications.home_assistant raw: %s",
                        notif.get("home_assistant"),
                    )
                    if "home_assistant" in notif:
                        config["NOTIFICATIONS_HOME_ASSISTANT_ENABLED"] = _coerce_bool(
                            notif["home_assistant"].get("enabled"), False
                        )
                    if "mobile_app" in notif:
                        mob = notif["mobile_app"]
                        config["NOTIFICATIONS_MOBILE_APP_ENABLED"] = _coerce_bool(
                            mob.get("enabled"), False
                        )
                        config["MOBILE_APP_GOOGLE_APPLICATION_CREDENTIALS"] = (
                            mob.get("credentials_path") or ""
                        ).strip() or ""
                        config["MOBILE_APP_FIREBASE_PROJECT_ID"] = (
                            mob.get("project_id") or ""
                        ).strip() or ""

                # Normalize pushover block so po_config.get() never raises.
                po_raw = (yaml_config.get("notifications") or {}).get("pushover") or {}
                if isinstance(po_raw, dict) and "enabled" in po_raw:
                    config["pushover"] = {
                        **po_raw,
                        "enabled": _coerce_bool(po_raw.get("enabled"), False),
                    }
                else:
                    config["pushover"] = (
                        dict(po_raw) if isinstance(po_raw, dict) else {}
                    )

                if "gemini" in yaml_config:
                    gemini_cfg = yaml_config["gemini"]
                    config["GEMINI"] = {
                        "proxy_url": gemini_cfg.get("proxy_url", ""),
                        "api_key": gemini_cfg.get("api_key", ""),
                        "model": gemini_cfg.get("model", "gemini-2.5-flash-lite"),
                        "enabled": bool(gemini_cfg.get("enabled", False)),
                    }

                if "multi_cam" in yaml_config:
                    mc = yaml_config["multi_cam"]
                    config["MAX_MULTI_CAM_FRAMES_MIN"] = mc.get(
                        "max_multi_cam_frames_min", config["MAX_MULTI_CAM_FRAMES_MIN"]
                    )
                    config["MAX_MULTI_CAM_FRAMES_SEC"] = float(
                        mc.get(
                            "max_multi_cam_frames_sec",
                            config["MAX_MULTI_CAM_FRAMES_SEC"],
                        )
                    )
                    config["CROP_WIDTH"] = mc.get("crop_width", config["CROP_WIDTH"])
                    config["CROP_HEIGHT"] = mc.get("crop_height", config["CROP_HEIGHT"])
                    config["MULTI_CAM_SYSTEM_PROMPT_FILE"] = (
                        mc.get(
                            "multi_cam_system_prompt_file",
                            config["MULTI_CAM_SYSTEM_PROMPT_FILE"],
                        )
                        or ""
                    )
                    config["SMART_CROP_PADDING"] = float(
                        mc.get(
                            "smart_crop_padding", config.get("SMART_CROP_PADDING", 0.15)
                        )
                    )
                    config["DETECTION_MODEL"] = (
                        mc.get("detection_model")
                        or config.get("DETECTION_MODEL", "yolov8n.pt")
                    ) or "yolov8n.pt"
                    config["DETECTION_DEVICE"] = (
                        mc.get("detection_device") or config.get("DETECTION_DEVICE", "")
                    ) or ""
                    config["DETECTION_FRAME_INTERVAL"] = mc.get(
                        "detection_frame_interval", config["DETECTION_FRAME_INTERVAL"]
                    )
                    config["DETECTION_IMGSZ"] = int(
                        mc.get("detection_imgsz", config["DETECTION_IMGSZ"])
                    )
                    config["CAMERA_TIMELINE_ANALYSIS_MULTIPLIER"] = float(
                        mc.get(
                            "camera_timeline_analysis_multiplier",
                            config["CAMERA_TIMELINE_ANALYSIS_MULTIPLIER"],
                        )
                    )
                    config["CAMERA_TIMELINE_EMA_ALPHA"] = float(
                        mc.get(
                            "camera_timeline_ema_alpha",
                            config["CAMERA_TIMELINE_EMA_ALPHA"],
                        )
                    )
                    config["CAMERA_TIMELINE_PRIMARY_BIAS_MULTIPLIER"] = float(
                        mc.get(
                            "camera_timeline_primary_bias_multiplier",
                            config["CAMERA_TIMELINE_PRIMARY_BIAS_MULTIPLIER"],
                        )
                    )
                    config["CAMERA_SWITCH_MIN_SEGMENT_FRAMES"] = int(
                        mc.get(
                            "camera_switch_min_segment_frames",
                            config["CAMERA_SWITCH_MIN_SEGMENT_FRAMES"],
                        )
                    )
                    config["CAMERA_SWITCH_HYSTERESIS_MARGIN"] = float(
                        mc.get(
                            "camera_switch_hysteresis_margin",
                            config["CAMERA_SWITCH_HYSTERESIS_MARGIN"],
                        )
                    )
                    config["CAMERA_TIMELINE_FINAL_YOLO_DROP_NO_PERSON"] = bool(
                        mc.get(
                            "camera_timeline_final_yolo_drop_no_person",
                            config["CAMERA_TIMELINE_FINAL_YOLO_DROP_NO_PERSON"],
                        )
                    )
                    config["DECODE_SECOND_CAMERA_CPU_ONLY"] = bool(
                        mc.get(
                            "decode_second_camera_cpu_only",
                            config["DECODE_SECOND_CAMERA_CPU_ONLY"],
                        )
                    )
                    config["LOG_EXTRACTION_PHASE_TIMING"] = bool(
                        mc.get(
                            "log_extraction_phase_timing",
                            config["LOG_EXTRACTION_PHASE_TIMING"],
                        )
                    )
                    config["MERGE_FRAME_TIMEOUT_SEC"] = int(
                        mc.get(
                            "merge_frame_timeout_sec", config["MERGE_FRAME_TIMEOUT_SEC"]
                        )
                    )
                    config["TRACKING_TARGET_FRAME_PERCENT"] = int(
                        mc.get(
                            "tracking_target_frame_percent",
                            config["TRACKING_TARGET_FRAME_PERCENT"],
                        )
                    )
                    config["PERSON_AREA_DEBUG"] = bool(
                        mc.get("person_area_debug", config["PERSON_AREA_DEBUG"])
                    )
                    config["COMPILATION_ZOOM_SMOOTH_EMA_ALPHA"] = float(
                        mc.get(
                            "compilation_zoom_smooth_ema_alpha",
                            config["COMPILATION_ZOOM_SMOOTH_EMA_ALPHA"],
                        )
                    )
                    if mc.get("gpu_vendor") is not None:
                        gv = str(mc["gpu_vendor"]).strip().lower()
                        if gv:
                            config["GPU_VENDOR"] = gv
                    if "gpu_device_index" in mc:
                        config["GPU_DEVICE_INDEX"] = int(mc["gpu_device_index"])
                    elif "cuda_device_index" in mc:
                        config["GPU_DEVICE_INDEX"] = int(mc["cuda_device_index"])
                    _intel_mc = mc.get("intel")
                    if isinstance(_intel_mc, dict):
                        if _intel_mc.get("qsv_encode_preset") is not None:
                            _preset = str(_intel_mc["qsv_encode_preset"]).strip()
                            if _preset:
                                config["INTEL_QSV_ENCODE_PRESET"] = _preset
                        if _intel_mc.get("qsv_encode_global_quality") is not None:
                            config["INTEL_QSV_ENCODE_GLOBAL_QUALITY"] = int(
                                _intel_mc["qsv_encode_global_quality"]
                            )
                if "gemini_proxy" in yaml_config:
                    gp = yaml_config["gemini_proxy"]
                    config["GEMINI_PROXY_URL"] = (
                        gp.get("url", config["GEMINI_PROXY_URL"]) or ""
                    )
                    config["GEMINI_PROXY_MODEL"] = (
                        gp.get("model", config["GEMINI_PROXY_MODEL"])
                        or "gemini-2.5-flash-lite"
                    )
                    config["GEMINI_PROXY_TEMPERATURE"] = float(
                        gp.get("temperature", config["GEMINI_PROXY_TEMPERATURE"])
                    )
                    config["GEMINI_PROXY_TOP_P"] = float(
                        gp.get("top_p", config["GEMINI_PROXY_TOP_P"])
                    )
                    config["GEMINI_PROXY_FREQUENCY_PENALTY"] = float(
                        gp.get(
                            "frequency_penalty",
                            config["GEMINI_PROXY_FREQUENCY_PENALTY"],
                        )
                    )
                    config["GEMINI_PROXY_PRESENCE_PENALTY"] = float(
                        gp.get(
                            "presence_penalty", config["GEMINI_PROXY_PRESENCE_PENALTY"]
                        )
                    )
                elif "gemini" in yaml_config and config.get("GEMINI"):
                    # Backward compat: URL/model from gemini when gemini_proxy unset
                    config["GEMINI_PROXY_URL"] = (
                        config["GEMINI"].get("proxy_url") or ""
                    ) or config["GEMINI_PROXY_URL"]
                    config["GEMINI_PROXY_MODEL"] = (
                        config["GEMINI"].get("model") or "gemini-2.5-flash-lite"
                    ) or config["GEMINI_PROXY_MODEL"]

                config_loaded = True
                break

            except Exception as e:
                logger.error(f"Error loading config from {path}: {e}")

    if not config_loaded:
        logger.info("No config.yaml found, using defaults")
    if config.get("GEMINI") is None:
        config["GEMINI"] = {
            "proxy_url": "",
            "api_key": "",
            "model": "gemini-2.5-flash-lite",
            "enabled": False,
        }

    # Environment variables override everything (for secrets/deployment)
    config["MQTT_BROKER"] = os.getenv("MQTT_BROKER") or config["MQTT_BROKER"]
    config["MQTT_PORT"] = int(os.getenv("MQTT_PORT", str(config["MQTT_PORT"])))
    config["MQTT_USER"] = os.getenv("MQTT_USER") or config["MQTT_USER"]
    config["MQTT_PASSWORD"] = os.getenv("MQTT_PASSWORD") or config["MQTT_PASSWORD"]
    frigate_url = os.getenv("FRIGATE_URL") or config["FRIGATE_URL"]
    config["FRIGATE_URL"] = frigate_url.rstrip("/") if frigate_url else None
    config["BUFFER_IP"] = (
        os.getenv("BUFFER_IP") or os.getenv("HA_IP") or config["BUFFER_IP"]
    )
    config["FLASK_PORT"] = int(os.getenv("FLASK_PORT", str(config["FLASK_PORT"])))
    config["FLASK_HOST"] = os.getenv("FLASK_HOST", config["FLASK_HOST"])
    config["STORAGE_PATH"] = os.getenv("STORAGE_PATH", config["STORAGE_PATH"])
    config["RETENTION_DAYS"] = int(
        os.getenv("RETENTION_DAYS", str(config["RETENTION_DAYS"]))
    )
    config["LOG_LEVEL"] = os.getenv("LOG_LEVEL", config["LOG_LEVEL"])
    _env_ai = os.getenv("AI_MODE")
    if _env_ai and str(_env_ai).strip().lower() in (
        AI_MODE_FRIGATE,
        AI_MODE_EXTERNAL_API,
    ):
        config["AI_MODE"] = str(_env_ai).strip().lower()
    config["STATS_REFRESH_SECONDS"] = int(
        os.getenv("STATS_REFRESH_SECONDS", str(config["STATS_REFRESH_SECONDS"]))
    )
    config["DAILY_REPORT_RETENTION_DAYS"] = int(
        os.getenv(
            "DAILY_REPORT_RETENTION_DAYS", str(config["DAILY_REPORT_RETENTION_DAYS"])
        )
    )
    config["DAILY_REPORT_SCHEDULE_HOUR"] = int(
        os.getenv(
            "DAILY_REPORT_SCHEDULE_HOUR", str(config["DAILY_REPORT_SCHEDULE_HOUR"])
        )
    )
    config["REPORT_KNOWN_PERSON_NAME"] = (
        os.getenv("REPORT_KNOWN_PERSON_NAME") or config["REPORT_KNOWN_PERSON_NAME"]
    ) or ""
    config["EVENT_GAP_SECONDS"] = int(
        os.getenv("EVENT_GAP_SECONDS", str(config["EVENT_GAP_SECONDS"]))
    )
    config["EXPORT_WATCHDOG_INTERVAL_MINUTES"] = int(
        os.getenv(
            "EXPORT_WATCHDOG_INTERVAL_MINUTES",
            str(config["EXPORT_WATCHDOG_INTERVAL_MINUTES"]),
        )
    )
    config["EXPORT_BUFFER_BEFORE"] = int(
        os.getenv("EXPORT_BUFFER_BEFORE", str(config["EXPORT_BUFFER_BEFORE"]))
    )
    config["EXPORT_BUFFER_AFTER"] = int(
        os.getenv("EXPORT_BUFFER_AFTER", str(config["EXPORT_BUFFER_AFTER"]))
    )
    config["HA_URL"] = os.getenv("HA_URL") or config["HA_URL"]
    config["HA_TOKEN"] = os.getenv("HA_TOKEN") or config["HA_TOKEN"]
    _save_ai = os.getenv("SAVE_AI_FRAMES")
    if _save_ai is not None:
        config["SAVE_AI_FRAMES"] = str(_save_ai).lower() in ("true", "1", "yes")
    _create_zip = os.getenv("CREATE_AI_ANALYSIS_ZIP")
    if _create_zip is not None:
        config["CREATE_AI_ANALYSIS_ZIP"] = str(_create_zip).lower() in (
            "true",
            "1",
            "yes",
        )
    _cap = os.getenv("GEMINI_FRAMES_PER_HOUR_CAP")
    if _cap is not None:
        try:
            config["GEMINI_FRAMES_PER_HOUR_CAP"] = int(_cap)
        except ValueError:
            pass

    # Pushover: env overrides (e.g. PUSHOVER_USER_KEY, PUSHOVER_API_TOKEN)
    config.setdefault("pushover", {})
    if isinstance(config.get("pushover"), dict):
        config["pushover"]["pushover_user_key"] = (
            os.getenv("PUSHOVER_USER_KEY")
            or config["pushover"].get("pushover_user_key")
            or ""
        )
        config["pushover"]["pushover_api_token"] = (
            os.getenv("PUSHOVER_API_TOKEN")
            or config["pushover"].get("pushover_api_token")
            or ""
        )
        if "enabled" in config["pushover"]:
            config["pushover"]["enabled"] = _coerce_bool(
                config["pushover"]["enabled"], False
            )

    # Notifications: env override so HA can be forced off/on regardless of file.
    _env_ha = os.getenv("NOTIFICATIONS_HOME_ASSISTANT_ENABLED")
    if _env_ha is not None:
        config["NOTIFICATIONS_HOME_ASSISTANT_ENABLED"] = _coerce_bool(
            _env_ha, config["NOTIFICATIONS_HOME_ASSISTANT_ENABLED"]
        )

    # Mobile app (FCM): env overrides credentials path (GOOGLE_APPLICATION_CREDENTIALS).
    config["MOBILE_APP_GOOGLE_APPLICATION_CREDENTIALS"] = (
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        or config.get("MOBILE_APP_GOOGLE_APPLICATION_CREDENTIALS")
        or ""
    ).strip()

    # Mobile app (FCM): project ID from env GOOGLE_CLOUD_PROJECT or YAML project_id.
    config["MOBILE_APP_FIREBASE_PROJECT_ID"] = (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or config.get("MOBILE_APP_FIREBASE_PROJECT_ID")
        or ""
    ).strip()

    # Gemini env overrides (api_key for secrets). Single API Key: GEMINI_API_KEY only.
    config["GEMINI"] = dict(config.get("GEMINI") or {})
    config["GEMINI"].setdefault("proxy_url", "")
    config["GEMINI"].setdefault("api_key", "")
    config["GEMINI"].setdefault("model", "gemini-2.5-flash-lite")
    config["GEMINI"].setdefault("enabled", False)
    config["GEMINI"]["api_key"] = (
        os.getenv("GEMINI_API_KEY") or config["GEMINI"].get("api_key") or ""
    )

    # Gemini proxy URL override (no Google fallback; default remains "")
    config["GEMINI_PROXY_URL"] = (
        os.getenv("GEMINI_PROXY_URL") or config.get("GEMINI_PROXY_URL") or ""
    ).strip() or ""

    # GPU vendor/device: env overrides YAML; GPU_DEVICE_INDEX beats CUDA_* env.
    _env_gv = os.getenv("GPU_VENDOR")
    if _env_gv is not None and str(_env_gv).strip():
        config["GPU_VENDOR"] = str(_env_gv).strip().lower()
    _env_gpu_idx = os.getenv("GPU_DEVICE_INDEX")
    _env_cuda_idx = os.getenv("CUDA_DEVICE_INDEX")
    if _env_gpu_idx is not None:
        config["GPU_DEVICE_INDEX"] = int(_env_gpu_idx)
    elif _env_cuda_idx is not None:
        config["GPU_DEVICE_INDEX"] = int(_env_cuda_idx)

    _env_intel_preset = os.getenv("INTEL_QSV_ENCODE_PRESET")
    if _env_intel_preset is not None and str(_env_intel_preset).strip():
        config["INTEL_QSV_ENCODE_PRESET"] = str(_env_intel_preset).strip()
    _env_intel_gq = os.getenv("INTEL_QSV_ENCODE_GLOBAL_QUALITY")
    if _env_intel_gq is not None:
        try:
            config["INTEL_QSV_ENCODE_GLOBAL_QUALITY"] = int(_env_intel_gq)
        except ValueError:
            pass

    _finalize_gpu_vendor_and_device(config)

    # Validate required settings
    missing = []
    if not config["MQTT_BROKER"]:
        missing.append("MQTT_BROKER (network.mqtt_broker)")
    if not config["FRIGATE_URL"]:
        missing.append("FRIGATE_URL (network.frigate_url)")
    if not config["BUFFER_IP"]:
        missing.append("BUFFER_IP (network.buffer_ip)")

    if missing:
        raise ValueError(
            f"Missing required configuration: {', '.join(missing)}. "
            f"Set these in config.yaml under 'network:' or as environment variables."
        )

    po_enabled = (
        _coerce_bool(config.get("pushover", {}).get("enabled"), False)
        if isinstance(config.get("pushover"), dict)
        else False
    )
    logger.info(
        "Notifications: home_assistant=%s, pushover=%s, mobile_app=%s",
        config["NOTIFICATIONS_HOME_ASSISTANT_ENABLED"],
        po_enabled,
        config["NOTIFICATIONS_MOBILE_APP_ENABLED"],
    )

    return config
