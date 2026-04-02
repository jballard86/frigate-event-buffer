"""Voluptuous fragment for YAML ``network`` (MQTT, Frigate, Flask, storage)."""

from __future__ import annotations

from voluptuous import Optional

# Network and storage; MQTT, Frigate URL, buffer IP required at runtime.
NETWORK_SCHEMA = {
    Optional("mqtt_broker"): str,  # MQTT broker hostname or IP for Frigate events.
    Optional("mqtt_port"): int,  # MQTT broker port (default 1883).
    Optional("mqtt_user"): str,  # Optional MQTT username.
    Optional("mqtt_password"): str,  # Optional MQTT password.
    Optional("frigate_url"): str,  # Base URL of Frigate (e.g. http://host:5000).
    Optional(
        "buffer_ip"
    ): str,  # IP/hostname where buffer is reachable (notification URLs).
    Optional("flask_port"): int,  # Port for the Flask web server (player, stats, API).
    Optional(
        "flask_host"
    ): str,  # Bind address for the web server (default 0.0.0.0 for Docker).
    Optional(
        "storage_path"
    ): str,  # Root path for event clips, snapshots, exported files.
    Optional(
        "ha_ip"
    ): str,  # Legacy fallback for buffer_ip when building notification URLs.
}
