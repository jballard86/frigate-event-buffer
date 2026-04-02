"""typing.Protocol for callers that only need the query service public API."""

from __future__ import annotations

from typing import Any, Protocol


class EventQueryServiceProtocol(Protocol):
    """Event listing surface used by routes, preferences, and orchestrator."""

    storage_path: str

    def evict_cache(self, key: str) -> None: ...

    def get_events(self, camera_name: str) -> list[dict[str, Any]]: ...

    def get_all_events(self) -> tuple[list[dict[str, Any]], list[str]]: ...

    def get_saved_events(self, camera: str | None = None) -> list[dict[str, Any]]: ...

    def get_test_events(self) -> list[dict[str, Any]]: ...

    def get_cameras(self) -> list[str]: ...

    def get_unread_count(self) -> int: ...
