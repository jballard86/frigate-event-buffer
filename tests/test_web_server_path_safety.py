"""
Tests for web server path safety: /files/<path> and /delete/<path> must not
allow path traversal. Also tests timeline download route:
/events/<camera>/<subdir>/timeline/download.
"""

import json
import os
import shutil
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from frigate_buffer.services.query import EventQueryService


class TestWebServerPathSafety(unittest.TestCase):
    """Verify path traversal attempts return 400/404 and do not expose files
    outside storage. Run this file alone
    (pytest tests/test_web_server_path_safety.py) to avoid create_app
    being mocked by other tests in the suite.
    """

    def setUp(self):
        from frigate_buffer.web.server import create_app

        self.storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.storage, ignore_errors=True))
        self.orchestrator = SimpleNamespace(
            config={
                "STORAGE_PATH": self.storage,
                "ALLOWED_CAMERAS": [],
                "STATS_REFRESH_SECONDS": 60,
            },
            _request_count_lock=threading.Lock(),
            _request_count=0,
            state_manager=SimpleNamespace(),
            file_manager=SimpleNamespace(),
            query_service=EventQueryService(self.storage),
            download_service=MagicMock(),
        )
        self.app = create_app(self.orchestrator)
        self.client = self.app.test_client()
        self._app_is_mock = isinstance(self.app, MagicMock)

    def _skip_if_app_mocked(self):
        if self._app_is_mock:
            self.skipTest(
                "create_app was mocked (run this file alone: "
                "pytest tests/test_web_server_path_safety.py)"
            )

    def test_files_path_traversal_returns_404(self):
        """Requesting /files/../../../etc/passwd must not serve a file
        outside storage."""
        self._skip_if_app_mocked()
        r = self.client.get("/files/../../../etc/passwd")
        assert r.status_code in (404, 400), (
            "Path traversal should be rejected with 404 or 400"
        )

    def test_files_double_dot_in_path_returns_404_or_400(self):
        """Requesting /files/cam/..%2f..%2fetc/passwd (URL-encoded)
        should be rejected."""
        self._skip_if_app_mocked()
        r = self.client.get("/files/cam/..%2f..%2fetc%2fpasswd")
        assert r.status_code in (404, 400)

    def test_delete_path_traversal_returns_400(self):
        """POST /delete/../../../evil must not delete outside storage."""
        self._skip_if_app_mocked()
        r = self.client.post("/delete/../../../evil")
        assert r.status_code == 400
        data = r.get_json() or {}
        assert "error" in (str(data).lower() or "message" in data)

    def test_delete_valid_subdir_under_storage_succeeds_or_404(self):
        """POST /delete/camera/123_evt is valid; 404 if folder does not exist."""
        self._skip_if_app_mocked()
        r = self.client.post("/delete/camera/123_evt")
        # 404 if folder not found, 200 if deleted
        assert r.status_code in (200, 404)

    def test_viewed_path_traversal_rejected(self):
        """POST /viewed with path traversal must be rejected
        (400 invalid path or 404 not found)."""
        self._skip_if_app_mocked()
        r = self.client.post("/viewed/cam/..%2f..%2fevil")
        assert r.status_code in (400, 404)

    def test_timeline_download_returns_merged_json_when_only_append_exists(self):
        """GET timeline/download returns 200 and merged timeline when only
        append file exists."""
        self._skip_if_app_mocked()
        camera = "Doorbell"
        subdir = "1771359165_3aa0dabd"
        event_dir = os.path.join(self.storage, camera, subdir)
        os.makedirs(event_dir, exist_ok=True)
        append_path = os.path.join(event_dir, "notification_timeline_append.jsonl")
        with open(append_path, "w") as f:
            f.write(
                '{"source": "frigate_mqtt", "label": "Event update", '
                '"ts": "2026-02-17T15:12:49"}\n'
            )
        r = self.client.get(f"/events/{camera}/{subdir}/timeline/download")
        assert r.status_code == 200
        assert r.content_type == "application/json"
        data = r.get_json()
        assert isinstance(data, dict)
        assert "entries" in data
        assert len(data["entries"]) == 1
        assert data["entries"][0].get("label") == "Event update"
        assert "Content-Disposition" in r.headers
        assert "attachment" in r.headers["Content-Disposition"]
        assert "notification_timeline.json" in r.headers["Content-Disposition"]

    def test_timeline_download_returns_merged_json_when_both_files_exist(self):
        """GET timeline/download returns merged base + append when both
        timeline files exist."""
        self._skip_if_app_mocked()
        camera = "front_door"
        subdir = "123_evt1"
        event_dir = os.path.join(self.storage, camera, subdir)
        os.makedirs(event_dir, exist_ok=True)
        base_path = os.path.join(event_dir, "notification_timeline.json")
        with open(base_path, "w") as f:
            json.dump({"event_id": "evt1", "entries": [{"label": "base"}]}, f)
        append_path = os.path.join(event_dir, "notification_timeline_append.jsonl")
        with open(append_path, "w") as f:
            f.write('{"label": "append_entry"}\n')
        r = self.client.get(f"/events/{camera}/{subdir}/timeline/download")
        assert r.status_code == 200
        data = r.get_json()
        assert data.get("event_id") == "evt1"
        assert len(data["entries"]) == 2
        assert data["entries"][0]["label"] == "base"
        assert data["entries"][1]["label"] == "append_entry"

    def test_timeline_download_404_when_event_folder_missing(self):
        """GET timeline/download returns 404 when camera/subdir folder
        does not exist."""
        self._skip_if_app_mocked()
        r = self.client.get("/events/Doorbell/nonexistent_subdir/timeline/download")
        assert r.status_code == 404

    def test_timeline_download_invalid_path_rejected(self):
        """GET timeline/download with path traversal is rejected (400 or 404)."""
        self._skip_if_app_mocked()
        r = self.client.get("/events/cam/..%2f..%2fetc/timeline/download")
        assert r.status_code in (400, 404)
