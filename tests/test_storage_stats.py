"""Tests for storage stats: compute_storage_stats (legacy, consolidated, daily_reports/daily_reviews)
and /stats API storage format (total_display, breakdown with value+unit).
"""

import os
import tempfile
import threading
import unittest
from types import SimpleNamespace

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frigate_buffer.managers.file import FileManager


class TestStorageStats(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.fm = FileManager(self.tmp, retention_days=7)

    def tearDown(self):
        import shutil
        if os.path.exists(self.tmp):
            try:
                shutil.rmtree(self.tmp)
            except OSError:
                pass

    def test_empty_storage_returns_zero_totals(self):
        stats = self.fm.compute_storage_stats()
        self.assertEqual(stats['total'], 0)
        self.assertEqual(stats['clips'], 0)
        self.assertEqual(stats['snapshots'], 0)
        self.assertEqual(stats['descriptions'], 0)
        self.assertEqual(stats['by_camera'], {})

    def test_legacy_camera_event_folder_counted(self):
        """Legacy layout: camera/ts_eventid/ with clip, snapshot, summary."""
        cam = os.path.join(self.tmp, "carport")
        ev = os.path.join(cam, "1700000000_evt1")
        os.makedirs(ev, exist_ok=True)
        with open(os.path.join(ev, "clip.mp4"), "wb") as f:
            f.write(b"x" * 1000)
        with open(os.path.join(ev, "snapshot.jpg"), "wb") as f:
            f.write(b"y" * 200)
        with open(os.path.join(ev, "summary.txt"), "w") as f:
            f.write("Event")
        stats = self.fm.compute_storage_stats()
        self.assertEqual(stats['clips'], 1000)
        self.assertEqual(stats['snapshots'], 200)
        self.assertGreaterEqual(stats['descriptions'], 5)
        self.assertIn("carport", stats['by_camera'])
        self.assertEqual(stats['by_camera']["carport"]['clips'], 1000)
        self.assertEqual(stats['by_camera']["carport"]['snapshots'], 200)
        self.assertEqual(stats['total'], stats['clips'] + stats['snapshots'] + stats['descriptions'])

    def test_consolidated_events_counted(self):
        """Consolidated layout: events/ce_id/camera/ with clip, snapshot, descriptions."""
        events_dir = os.path.join(self.tmp, "events")
        ce_dir = os.path.join(events_dir, "1700000001_ce1")
        cam_dir = os.path.join(ce_dir, "carport")
        os.makedirs(cam_dir, exist_ok=True)
        with open(os.path.join(cam_dir, "clip.mp4"), "wb") as f:
            f.write(b"a" * 5000)
        with open(os.path.join(cam_dir, "snapshot.jpg"), "wb") as f:
            f.write(b"b" * 500)
        with open(os.path.join(cam_dir, "metadata.json"), "w") as f:
            f.write('{}')
        # CE root file
        with open(os.path.join(ce_dir, "review_summary.md"), "w") as f:
            f.write("# Summary")
        stats = self.fm.compute_storage_stats()
        self.assertIn("events", stats['by_camera'])
        self.assertGreaterEqual(stats['clips'], 5000)
        self.assertGreaterEqual(stats['snapshots'], 500)
        self.assertGreater(stats['descriptions'], 0)
        self.assertEqual(stats['by_camera']["events"]['clips'], 5000)
        self.assertEqual(stats['by_camera']["events"]['snapshots'], 500)

    def test_daily_reports_and_daily_reviews_included_in_total(self):
        """daily_reports/ and daily_reviews/ bytes are included in total and descriptions."""
        for sub in ("daily_reports", "daily_reviews"):
            path = os.path.join(self.tmp, sub)
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "report.md"), "w") as f:
                f.write("x" * 300)
        stats = self.fm.compute_storage_stats()
        # 300 * 2 = 600 bytes in descriptions (and total, since no clips/snapshots)
        self.assertGreaterEqual(stats['descriptions'], 600)
        self.assertGreaterEqual(stats['total'], 600)

    def test_legacy_and_consolidated_both_counted(self):
        """Both legacy camera folder and events/ contribute to totals."""
        # Legacy
        cam = os.path.join(self.tmp, "doorbell")
        ev = os.path.join(cam, "1700000002_evt2")
        os.makedirs(ev, exist_ok=True)
        with open(os.path.join(ev, "clip.mp4"), "wb") as f:
            f.write(b"1" * 100)
        # Consolidated
        events_dir = os.path.join(self.tmp, "events")
        ce_dir = os.path.join(events_dir, "1700000003_ce2")
        cam_dir = os.path.join(ce_dir, "doorbell")
        os.makedirs(cam_dir, exist_ok=True)
        with open(os.path.join(cam_dir, "clip.mp4"), "wb") as f:
            f.write(b"2" * 200)
        stats = self.fm.compute_storage_stats()
        self.assertEqual(stats['clips'], 100 + 200)
        self.assertIn("doorbell", stats['by_camera'])
        self.assertIn("events", stats['by_camera'])
        self.assertEqual(stats['by_camera']["doorbell"]['clips'], 100)
        self.assertEqual(stats['by_camera']["events"]['clips'], 200)

    def test_stats_api_returns_storage_with_value_unit_format(self):
        """GET /stats returns storage.total_display and breakdown with value+unit (KB/MB/GB)."""
        from unittest.mock import MagicMock
        from frigate_buffer.web.server import create_app
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__('shutil').rmtree(storage, ignore_errors=True))
        storage_stats = {
            "clips": 500 * 1024,
            "snapshots": 100 * 1024,
            "descriptions": 50 * 1024,
            "total": 650 * 1024,
            "by_camera": {"carport": {"clips": 500 * 1024, "snapshots": 100 * 1024, "descriptions": 50 * 1024, "total": 650 * 1024}},
        }
        orch = SimpleNamespace(
            config={
                "STORAGE_PATH": storage,
                "STATS_REFRESH_SECONDS": 60,
                "RETENTION_DAYS": 7,
                "CLEANUP_INTERVAL_HOURS": 1,
            },
            get_storage_stats=lambda: storage_stats,
            fetch_ha_state=lambda *args, **kwargs: None,
            _last_cleanup_time=None,
            _last_cleanup_deleted=0,
            _start_time=0,
            _request_count_lock=threading.Lock(),
            _request_count=0,
            state_manager=MagicMock(get_active_event_ids=MagicMock(return_value=[])),
            consolidated_manager=MagicMock(get_all=MagicMock(return_value=[])),
            file_manager=MagicMock(),
            mqtt_wrapper=MagicMock(mqtt_connected=False),
        )
        app = create_app(orch)
        if isinstance(app, MagicMock):
            self.skipTest("create_app was mocked")
        client = app.test_client()
        r = client.get("/stats")
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        data = r.get_json()
        self.assertIn("storage", data)
        s = data["storage"]
        self.assertIn("total_display", s)
        self.assertIsNotNone(s["total_display"].get("value"))
        self.assertIn(s["total_display"].get("unit"), ("KB", "MB", "GB"))
        self.assertIn("breakdown", s)
        for key in ("clips", "snapshots", "descriptions"):
            self.assertIn(key, s["breakdown"])
            self.assertIsNotNone(s["breakdown"][key].get("value"))
            self.assertIn(s["breakdown"][key].get("unit"), ("KB", "MB", "GB"))
        self.assertIn("by_camera", s)
        for cam, val in s["by_camera"].items():
            self.assertIn("value", val)
            self.assertIn("unit", val)


if __name__ == '__main__':
    unittest.main()
