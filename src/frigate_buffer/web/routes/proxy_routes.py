"""Proxy blueprint: Frigate snapshot and latest.jpg."""

from flask import Blueprint

from frigate_buffer.web.frigate_proxy import proxy_camera_latest, proxy_snapshot


def create_bp(orchestrator):
    """Create proxy blueprint with routes closed over orchestrator."""
    bp = Blueprint("proxy", __name__)
    allowed_cameras = orchestrator.config.get("ALLOWED_CAMERAS", [])
    frigate_url = orchestrator.config.get("FRIGATE_URL", "")

    @bp.route("/api/events/<event_id>/snapshot.jpg")
    def route_proxy_snapshot(event_id):
        result = proxy_snapshot(frigate_url, event_id)
        if isinstance(result, tuple):
            return result[0], result[1]
        return result

    @bp.route("/api/cameras/<camera_name>/latest.jpg")
    def route_proxy_camera_latest(camera_name):
        result = proxy_camera_latest(frigate_url, camera_name, allowed_cameras)
        if isinstance(result, tuple):
            return result[0], result[1]
        return result

    return bp
