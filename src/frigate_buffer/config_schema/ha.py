"""Voluptuous fragment for YAML ``ha`` (Home Assistant REST API)."""

from __future__ import annotations

from voluptuous import Optional

HA_SCHEMA = {
    Optional(
        "base_url"
    ): str,  # HA API base URL (e.g. http://host:8123/api); preferred over url.
    Optional("url"): str,  # Alternative to base_url for HA API endpoint.
    Optional("token"): str,  # Long-lived access token for HA API.
    Optional("gemini_cost_entity"): str,  # HA entity ID for Gemini cost.
    Optional("gemini_tokens_entity"): str,  # HA entity ID for Gemini token count.
}
