"""Thread-safe preferences storage for mobile/FCM (e.g. device token).

The file lives under the application storage path so it persists across
process restarts and is co-located with other app data. The lock ensures
safe concurrent access from multiple Flask request threads and any
background code that may read or write preferences.
"""

import json
import logging
import os
import tempfile
import threading

logger = logging.getLogger("frigate-buffer")

PREFERENCES_FILENAME = "mobile_preferences.json"


class PreferencesManager:
    """Thread-safe read/write of mobile_preferences.json under storage_path.

    Used to store FCM device token and other mobile preferences without
    requiring a server restart. All file I/O is protected by a single lock
    so concurrent Flask requests and background tasks do not corrupt the file.
    """

    def __init__(self, storage_path: str) -> None:
        """Initialize with the application storage root.

        Args:
            storage_path: Directory under which mobile_preferences.json
                will be created (e.g. config STORAGE_PATH).
        """
        self._storage_path = os.path.realpath(os.path.abspath(storage_path))
        self._file_path = os.path.join(self._storage_path, PREFERENCES_FILENAME)
        self._lock = threading.Lock()

    def get_fcm_token(self) -> str | None:
        """Return the stored FCM device token, or None if not set or file missing.

        Returns:
            The token string, or None if the file does not exist, is invalid
            JSON, or has no non-empty "token" key.
        """
        with self._lock:
            data = self._read()
            if not data:
                return None
            token = data.get("token")
            if not isinstance(token, str) or not token.strip():
                return None
            return token.strip()

    def set_fcm_token(self, token: str) -> None:
        """Store the FCM device token; creates the file on first write.

        Args:
            token: Non-empty token string to persist.
        """
        with self._lock:
            data = self._read() or {}
            data["token"] = token
            self._write(data)

    def _read(self) -> dict | None:
        """Read and parse the JSON file; return None if missing or invalid."""
        if not os.path.isfile(self._file_path):
            return None
        try:
            with open(self._file_path, encoding="utf-8") as f:
                out = json.load(f)
            if not isinstance(out, dict):
                return None
            return out
        except (OSError, json.JSONDecodeError) as e:
            logger.debug("Could not read preferences file %s: %s", self._file_path, e)
            return None

    def _write(self, data: dict) -> None:
        """Write data as JSON atomically; creates parent dir and file if needed."""
        os.makedirs(self._storage_path, exist_ok=True)
        tmp_fd = tempfile.NamedTemporaryFile(
            mode="w",
            dir=self._storage_path,
            delete=False,
            suffix=".tmp",
            encoding="utf-8",
        )
        tmp_path = tmp_fd.name
        try:
            json.dump(data, tmp_fd, indent=2)
            tmp_fd.close()
            os.replace(tmp_path, self._file_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
