"""Voluptuous fragment for YAML ``gemini`` (clip AI / proxy)."""

from __future__ import annotations

from voluptuous import Optional

GEMINI_BLOCK_SCHEMA = {
    Optional(
        "proxy_url"
    ): str,  # OpenAI-compatible proxy URL; no Google fallback if unset.
    Optional(
        "api_key"
    ): str,  # API key for proxy (can override via GEMINI_API_KEY env).
    Optional("model"): str,  # Model name sent to proxy (e.g. gemini-2.5-flash-lite).
    Optional("enabled"): bool,  # Whether clip analysis via Gemini is enabled.
}
