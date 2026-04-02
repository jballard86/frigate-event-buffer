"""Voluptuous fragment for YAML ``settings``."""

from __future__ import annotations

from voluptuous import Any, Optional

SETTINGS_SCHEMA = {
    Optional("retention_days"): int,  # Days to keep event data before cleanup.
    Optional(
        "cleanup_interval_hours"
    ): int,  # How often to run retention cleanup (hours).
    Optional(
        "export_watchdog_interval_minutes"
    ): int,  # How often to check/remove completed exports in Frigate (min).
    Optional(
        "ffmpeg_timeout_seconds"
    ): int,  # Timeout for FFmpeg (e.g. GIF generation); kills hung processes.
    Optional(
        "notification_delay_seconds"
    ): int,  # Delay before snapshot (lets Frigate pick a better frame).
    Optional("log_level"): Any(
        "DEBUG", "INFO", "WARNING", "ERROR"
    ),  # Logging verbosity.
    Optional(
        "summary_padding_before"
    ): int,  # Seconds before event start for Frigate review summary.
    Optional(
        "summary_padding_after"
    ): int,  # Seconds after event end for Frigate review summary.
    Optional(
        "stats_refresh_seconds"
    ): int,  # Stats page auto-refresh interval (seconds).
    Optional(
        "daily_report_retention_days"
    ): int,  # How long to keep saved daily reports (days).
    Optional(
        "daily_report_schedule_hour"
    ): int,  # Hour (0-23) to generate AI daily report.
    Optional(
        "report_prompt_file"
    ): str,  # Path to prompt file for daily report; empty = default.
    Optional(
        "report_known_person_name"
    ): str,  # Placeholder for {known_person_name} in report prompt.
    Optional(
        "event_gap_seconds"
    ): int,  # Seconds of inactivity before next event starts new group.
    Optional(
        "minimum_event_seconds"
    ): int,  # Shorter events discarded (data deleted, MQTT sent).
    Optional(
        "max_event_length_seconds"
    ): int,  # Events >= this canceled; no AI/decode, folder -canceled. Def 120.
    Optional(
        "export_buffer_before"
    ): int,  # Seconds before event start in exported clip.
    Optional("export_buffer_after"): int,  # Seconds after event end in exported clip.
    Optional(
        "gemini_max_concurrent_analyses"
    ): int,  # Max concurrent Gemini clip analyses (throttling).
    Optional(
        "save_ai_frames"
    ): bool,  # Whether to save extracted AI analysis frames to disk.
    Optional(
        "create_ai_analysis_zip"
    ): bool,  # Whether to create zip of AI analysis assets (e.g. multi-cam).
    Optional(
        "gemini_frames_per_hour_cap"
    ): int,  # Rolling cap: max frames to proxy per hour; 0 = disabled.
    Optional(
        "quick_title_delay_seconds"
    ): int,  # Delay (3–5s) before quick AI title on live frame; 0 = disabled.
    Optional(
        "quick_title_enabled"
    ): bool,  # When true (and Gemini enabled), run quick-title after delay.
    Optional("ai_mode"): Any(
        "frigate", "external_api"
    ),  # frigate = Frigate MQTT AI; external_api = buffer Gemini/Quick Title.
}
