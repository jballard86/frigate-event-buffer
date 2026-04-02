"""Voluptuous fragment for YAML ``notifications``."""

from __future__ import annotations

from voluptuous import Any, Optional, Schema

NOTIFICATIONS_SCHEMA = {
    Optional("home_assistant"): {
        # bool or str so "false" in YAML is accepted and coerced correctly.
        Optional("enabled"): Any(bool, str),
    },
    Optional("pushover"): Any(
        None,
        Schema(
            {
                Optional("enabled"): Any(bool, str),
                Optional("pushover_user_key"): str,
                Optional("pushover_api_token"): str,
                Optional("device"): str,
                Optional("default_sound"): str,
                Optional("html"): int,
            }
        ),
    ),
    Optional("mobile_app"): {
        Optional("enabled"): Any(bool, str),
        Optional("credentials_path"): str,
        Optional("project_id"): str,
    },
}
