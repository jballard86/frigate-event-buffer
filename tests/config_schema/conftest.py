"""Pytest fixtures scoped to ``tests/config_schema/`` only.

Why not root ``tests/conftest.py``: keep optional dict factories out of global
collection so unrelated tests do not pay import or fixture setup cost.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def minimal_network_yaml() -> dict[str, str]:
    """Shared ``network`` block for synthetic YAML in ``load_config`` tests."""
    return {
        "mqtt_broker": "localhost",
        "frigate_url": "http://frigate",
        "buffer_ip": "localhost",
        "storage_path": "/tmp",
    }


@pytest.fixture
def minimal_cameras_yaml() -> list[dict[str, str]]:
    """Minimal valid ``cameras`` list for schema tests."""
    return [{"name": "cam1"}]
