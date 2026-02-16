"""
Tests for web server path safety: /files/<path> and /delete/<path> must not allow path traversal.
"""

import os
import shutil
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock


class TestWebServerPathSafety(unittest.TestCase):
    """Verify path traversal attempts return 400/404 and do not expose files outside storage.
    Run this file alone (pytest tests/test_web_server_path_safety.py) to avoid create_app
    being mocked by other tests in the suite.
    """

    def setUp(self):
        from frigate_buffer.web.server import create_app
        self.storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.storage, ignore_errors=True))
        self.orchestrator = SimpleNamespace(
            config={"STORAGE_PATH": self.storage, "ALLOWED_CAMERAS": [], "STATS_REFRESH_SECONDS": 60},
            _request_count_lock=threading.Lock(),
            _request_count=0,
            state_manager=SimpleNamespace(),
            file_manager=SimpleNamespace(),
        )
        self.app = create_app(self.orchestrator)
        self.client = self.app.test_client()
        self._app_is_mock = isinstance(self.app, MagicMock)

    def _skip_if_app_mocked(self):
        if self._app_is_mock:
            self.skipTest("create_app was mocked (run this file alone: pytest tests/test_web_server_path_safety.py)")

    def test_files_path_traversal_returns_404(self):
        """Requesting /files/../../../etc/passwd must not serve a file outside storage."""
        self._skip_if_app_mocked()
        r = self.client.get("/files/../../../etc/passwd")
        self.assertIn(r.status_code, (404, 400), "Path traversal should be rejected with 404 or 400")

    def test_files_double_dot_in_path_returns_404_or_400(self):
        """Requesting /files/cam/..%2f..%2fetc/passwd (URL-encoded) should be rejected."""
        self._skip_if_app_mocked()
        r = self.client.get("/files/cam/..%2f..%2fetc%2fpasswd")
        self.assertIn(r.status_code, (404, 400))

    def test_delete_path_traversal_returns_400(self):
        """POST /delete/../../../evil must not delete outside storage."""
        self._skip_if_app_mocked()
        r = self.client.post("/delete/../../../evil")
        self.assertEqual(r.status_code, 400)
        data = r.get_json() or {}
        self.assertIn("error", str(data).lower() or "message" in data)

    def test_delete_valid_subdir_under_storage_succeeds_or_404(self):
        """POST /delete/camera/123_evt is valid; 404 if folder does not exist."""
        self._skip_if_app_mocked()
        r = self.client.post("/delete/camera/123_evt")
        # 404 if folder not found, 200 if deleted
        self.assertIn(r.status_code, (200, 404))

    def test_viewed_path_traversal_rejected(self):
        """POST /viewed with path traversal must be rejected (400 invalid path or 404 not found)."""
        self._skip_if_app_mocked()
        r = self.client.post("/viewed/cam/..%2f..%2fevil")
        self.assertIn(r.status_code, (400, 404))
