"""
Path safety helpers for the web layer.

Centralizes "path under storage root" checks so route handlers do not duplicate
realpath/startswith logic. Matches the rule used by server and FileManager:
resolved path must be under the real storage root and not equal to the root.
"""

import os


def resolve_under_storage(storage_path: str, *path_parts: str) -> str | None:
    """
    Resolve a path under the storage root and return it if safe, else None.

    The result is the normalized absolute path of join(storage_path, *path_parts)
    only when it lies strictly under the real storage root (no path traversal).
    Returns None if the path would escape storage or equals the storage root.

    Does not require the resolved path to exist (e.g. for delete 404 checks).
    """
    if not storage_path:
        return None
    base = os.path.realpath(storage_path)
    if not path_parts:
        return None
    candidate = os.path.normpath(os.path.abspath(os.path.join(storage_path, *path_parts)))
    if not candidate.startswith(base) or candidate == base:
        return None
    return candidate
