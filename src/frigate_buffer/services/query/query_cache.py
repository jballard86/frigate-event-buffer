"""TTL list-result cache and mtime-validated LRU cache for parsed event folders."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

ParseFolderFn = Callable[[str], dict[str, Any]]


class TtlListCache:
    """Short-lived cache for list endpoints (events, cameras, unread count)."""

    def __init__(self, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds
        self._cache: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> Any | None:
        if key in self._cache:
            entry = self._cache[key]
            if time.monotonic() - entry["timestamp"] < self._ttl:
                return entry["data"]
        return None

    def set(self, key: str, data: Any) -> None:
        self._cache[key] = {"timestamp": time.monotonic(), "data": data}

    def pop(self, key: str, default: Any = None) -> Any:
        return self._cache.pop(key, default)


class FolderEventParseCache:
    """LRU of folder_path → parsed event dict, invalidated when folder mtime changes."""

    def __init__(self, max_entries: int, parse_folder: ParseFolderFn) -> None:
        self._max = max_entries
        self._parse_folder = parse_folder
        self._event_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, folder_path: str, mtime: float) -> dict[str, Any]:
        if folder_path in self._event_cache:
            self._event_cache.move_to_end(folder_path)
            entry = self._event_cache[folder_path]
            if entry["mtime"] == mtime:
                return entry["data"]

        data = self._parse_folder(folder_path)
        self._event_cache[folder_path] = {"mtime": mtime, "data": data}
        if len(self._event_cache) > self._max:
            self._event_cache.popitem(last=False)
        return data
