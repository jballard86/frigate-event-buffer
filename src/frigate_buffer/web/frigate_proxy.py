"""
Frigate HTTP proxy helpers: snapshot and latest.jpg.

Streams image responses from Frigate so the buffer can serve them (e.g. for
Companion app reachability). Returns either a Flask Response or (body, status_code).
"""

import re
import logging

import requests
from flask import Response

from frigate_buffer.constants import (
    FRIGATE_PROXY_LATEST_TIMEOUT,
    FRIGATE_PROXY_SNAPSHOT_TIMEOUT,
    HTTP_STREAM_CHUNK_SIZE,
)

logger = logging.getLogger("frigate-buffer")


def proxy_snapshot(
    frigate_url: str,
    event_id: str,
    chunk_size: int = HTTP_STREAM_CHUNK_SIZE,
    timeout: int = FRIGATE_PROXY_SNAPSHOT_TIMEOUT,
) -> Response | tuple[str, int]:
    """
    Proxy GET Frigate /api/events/<event_id>/snapshot.jpg.

    Returns a streaming Response on success, or (body, status_code) on error
    (503 if Frigate URL not configured, 502 on request failure).
    """
    url_base = frigate_url.rstrip("/")
    if not url_base:
        return "Frigate URL not configured", 503
    url = f"{url_base}/api/events/{event_id}/snapshot.jpg"
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        return Response(
            resp.iter_content(chunk_size=chunk_size),
            content_type=resp.headers.get("Content-Type", "image/jpeg"),
            status=resp.status_code,
        )
    except requests.RequestException as e:
        logger.debug("Snapshot proxy error for %s: %s", event_id, e)
        return "Snapshot unavailable", 502


def proxy_camera_latest(
    frigate_url: str,
    camera_name: str,
    allowed_cameras: list,
    chunk_size: int = HTTP_STREAM_CHUNK_SIZE,
    timeout: int = FRIGATE_PROXY_LATEST_TIMEOUT,
) -> Response | tuple[str, int]:
    """
    Proxy GET Frigate /api/<camera_name>/latest.jpg (live frame).

    Validates camera_name (alphanumeric, underscore, hyphen only). If
    allowed_cameras is non-empty, camera must be in the list.
    Returns a streaming Response on success, or (body, status_code) on error
    (400 invalid name, 404 camera not configured, 503 no Frigate URL, 502 request failure).
    """
    if not camera_name or camera_name != re.sub(r"[^a-zA-Z0-9_-]", "", camera_name):
        return "Invalid camera name", 400
    if allowed_cameras and camera_name not in allowed_cameras:
        return "Camera not configured", 404
    url_base = frigate_url.rstrip("/")
    if not url_base:
        return "Frigate URL not configured", 503
    url = f"{url_base}/api/{camera_name}/latest.jpg"
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        return Response(
            resp.iter_content(chunk_size=chunk_size),
            content_type=resp.headers.get("Content-Type", "image/jpeg"),
            status=resp.status_code,
        )
    except requests.RequestException as e:
        logger.debug("Latest frame proxy error for %s: %s", camera_name, e)
        return "Live frame unavailable", 502
