"""Voluptuous fragment for YAML ``gemini_proxy`` (extended proxy options)."""

from __future__ import annotations

from voluptuous import Any, Optional

GEMINI_PROXY_SCHEMA = {
    Optional("url"): str,  # Proxy URL (no Google fallback; or GEMINI_PROXY_URL env).
    Optional("model"): str,  # Model name for proxy requests.
    Optional("temperature"): Any(int, float),  # Sampling temperature (0–2 typical).
    Optional("top_p"): Any(int, float),  # Nucleus sampling parameter.
    Optional("frequency_penalty"): Any(int, float),  # Penalty for repeated tokens.
    Optional("presence_penalty"): Any(int, float),  # Penalty for token presence.
}
