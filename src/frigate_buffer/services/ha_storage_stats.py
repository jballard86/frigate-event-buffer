"""
HA state fetch and storage-stats cache for the stats page and scheduler.

Provides a thin helper so the orchestrator and Flask server do not hold
HA HTTP or storage-stats logic inline. Used by the stats route and by
the scheduled job that refreshes storage stats every 5 minutes.
"""

import logging
import time
from typing import Any

import requests

from frigate_buffer.constants import DEFAULT_STORAGE_STATS_MAX_AGE_SECONDS

logger = logging.getLogger("frigate-buffer")

# Default timeout for Home Assistant state fetch (seconds).
HA_STATE_TIMEOUT = 5


def fetch_ha_state(ha_url: str, ha_token: str, entity_id: str) -> str | None:
    """Fetch entity state from Home Assistant REST API.

    Returns the state value (e.g. "42.5") or None on error or non-OK response.
    """
    base = ha_url.rstrip("/")
    path = "/states/" if base.endswith("/api") else "/api/states/"
    url = f"{base}{path}{entity_id}"
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {ha_token}"},
            timeout=HA_STATE_TIMEOUT,
        )
        if resp.ok:
            data = resp.json()
            return data.get("state")
    except requests.RequestException as e:
        logger.warning("Error fetching HA state for %s from %s: %s", entity_id, url, e)
    return None


class StorageStatsAndHaHelper:
    """Holds storage-stats cache and exposes HA state fetch for the stats page."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._cached: dict[str, Any] = {
            "clips": 0,
            "snapshots": 0,
            "descriptions": 0,
            "total": 0,
            "by_camera": {},
        }
        self._cached_time: float | None = None
        self._max_age_seconds = int(
            config.get(
                "STORAGE_STATS_MAX_AGE_SECONDS", DEFAULT_STORAGE_STATS_MAX_AGE_SECONDS
            )
        )

    def update(self, file_manager: Any) -> None:
        """Refresh cache from file_manager.compute_storage_stats(). Skip if cache is still fresh."""
        now = time.time()
        if (
            self._cached_time is not None
            and (now - self._cached_time) < self._max_age_seconds
        ):
            logger.debug("Skipping storage stats update (cache still fresh)")
            return
        logger.debug("Updating storage stats...")
        try:
            self._cached = file_manager.compute_storage_stats()
            self._cached_time = now
            logger.debug("Storage stats updated")
        except Exception as e:
            logger.error("Failed to update storage stats: %s", e)

    def get(self) -> dict[str, Any]:
        """Return the cached storage stats dict (clips, snapshots, descriptions, total, by_camera)."""
        return self._cached

    def fetch_ha_state(self, ha_url: str, ha_token: str, entity_id: str) -> str | None:
        """Fetch entity state from Home Assistant. Delegates to module-level fetch_ha_state."""
        return fetch_ha_state(ha_url, ha_token, entity_id)
