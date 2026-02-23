"""
Tests for FileManager: path validation, cleanup_old_events, max event length (rename/canceled),
and compute_storage_stats (legacy/consolidated/daily_reports and /stats API format).
"""

import os
import shutil
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from frigate_buffer.managers.file import FileManager


class TestFileManagerPathValidation(unittest.TestCase):
    """Path traversal safety and sanitize_camera_name."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        self.fm = FileManager(self.tmp, 3)

    def test_sanitize_camera_name_valid(self):
        self.assertEqual(self.fm.sanitize_camera_name("Front Door"), "front_door")
        self.assertEqual(self.fm.sanitize_camera_name("cam1"), "cam1")

    def test_sanitize_camera_name_strips_special_chars(self):
        out = self.fm.sanitize_camera_name("Cam/with\\path")
        self.assertNotIn("/", out)
        self.assertNotIn("\\", out)

    def test_sanitize_camera_name_empty_or_all_special_returns_unknown(self):
        self.assertEqual(self.fm.sanitize_camera_name(""), "unknown")
        self.assertEqual(self.fm.sanitize_camera_name("!!!@@@###"), "unknown")

    def test_create_event_folder_path_traversal_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.fm.create_event_folder(
                event_id="../../../../evil",
                camera="cam1",
                timestamp=1000.0
            )
        self.assertIn("Invalid event path", str(ctx.exception))

    def test_create_consolidated_event_folder_path_traversal_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.fm.create_consolidated_event_folder("../../../evil")
        self.assertIn("Invalid consolidated event path", str(ctx.exception))

    def test_create_consolidated_event_folder_dotdot_escape_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.fm.create_consolidated_event_folder("../../outside")
        self.assertIn("Invalid consolidated event path", str(ctx.exception))

    def test_create_event_folder_valid_succeeds(self):
        path = self.fm.create_event_folder("evt123", "cam1", 1000.0)
        self.assertTrue(path.startswith(os.path.realpath(self.tmp)))
        self.assertTrue(os.path.isdir(path))

    def test_delete_event_folder_under_storage_succeeds(self):
        sub = os.path.join(self.tmp, "cam1", "1000_evt1")
        os.makedirs(sub, exist_ok=True)
        self.assertTrue(os.path.isdir(sub))
        result = self.fm.delete_event_folder(sub)
        self.assertTrue(result)
        self.assertFalse(os.path.exists(sub))

    def test_delete_event_folder_outside_storage_returns_false(self):
        result = self.fm.delete_event_folder("/tmp/outside")
        self.assertFalse(result)

    def test_delete_event_folder_nonexistent_returns_false(self):
        result = self.fm.delete_event_folder(os.path.join(self.tmp, "nonexistent"))
        self.assertFalse(result)


class TestFileManagerCleanup(unittest.TestCase):
    """cleanup_old_events: testN folders, CE folders, canceled folders by mtime and active sets."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def test_cleanup_deletes_old_test_folder(self):
        fm = FileManager(storage_path=self.tmp, retention_days=1)
        events_dir = os.path.join(self.tmp, "events")
        os.makedirs(events_dir, exist_ok=True)
        test1 = os.path.join(events_dir, "test1")
        os.makedirs(test1, exist_ok=True)
        old = time.time() - (2 * 86400)
        os.utime(test1, (old, old))
        deleted = fm.cleanup_old_events(active_event_ids=[], active_ce_folder_names=[])
        self.assertEqual(deleted, 1)
        self.assertFalse(os.path.isdir(test1))

    def test_cleanup_keeps_recent_test_folder(self):
        fm = FileManager(storage_path=self.tmp, retention_days=1)
        events_dir = os.path.join(self.tmp, "events")
        os.makedirs(events_dir, exist_ok=True)
        test1 = os.path.join(events_dir, "test1")
        os.makedirs(test1, exist_ok=True)
        deleted = fm.cleanup_old_events(active_event_ids=[], active_ce_folder_names=[])
        self.assertEqual(deleted, 0)
        self.assertTrue(os.path.isdir(test1))

    def test_cleanup_ignores_non_test_events_folders(self):
        fm = FileManager(storage_path=self.tmp, retention_days=1)
        events_dir = os.path.join(self.tmp, "events")
        os.makedirs(events_dir, exist_ok=True)
        old_ce = os.path.join(events_dir, "1730000000_myce")
        os.makedirs(old_ce, exist_ok=True)
        os.utime(old_ce, (0, 0))
        deleted = fm.cleanup_old_events(active_event_ids=[], active_ce_folder_names=[])
        self.assertEqual(deleted, 1)
        self.assertFalse(os.path.isdir(old_ce))

    def test_cleanup_keeps_canceled_folder_when_base_id_active(self):
        fm = FileManager(storage_path=self.tmp, retention_days=1)
        events_dir = os.path.join(self.tmp, "events")
        os.makedirs(events_dir, exist_ok=True)
        canceled = os.path.join(events_dir, "1700000000_ev123-canceled")
        os.makedirs(canceled, exist_ok=True)
        deleted = fm.cleanup_old_events(active_event_ids=["ev123"], active_ce_folder_names=[])
        self.assertEqual(deleted, 0)
        self.assertTrue(os.path.isdir(canceled))

    def test_cleanup_deletes_canceled_folder_when_past_retention(self):
        fm = FileManager(storage_path=self.tmp, retention_days=1)
        events_dir = os.path.join(self.tmp, "events")
        os.makedirs(events_dir, exist_ok=True)
        canceled = os.path.join(events_dir, "1700000000_ev123-canceled")
        os.makedirs(canceled, exist_ok=True)
        deleted = fm.cleanup_old_events(active_event_ids=[], active_ce_folder_names=[])
        self.assertEqual(deleted, 1)
        self.assertFalse(os.path.isdir(canceled))


class TestFileManagerMaxEventLength(unittest.TestCase):
    """rename_event_folder and write_canceled_summary."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def test_rename_event_folder_appends_suffix(self):
        fm = FileManager(storage_path=self.tmp, retention_days=1)
        folder = os.path.join(self.tmp, "cam1", "1700000000_ev123")
        os.makedirs(folder, exist_ok=True)
        new_path = fm.rename_event_folder(folder)
        self.assertEqual(os.path.basename(new_path), "1700000000_ev123-canceled")
        self.assertFalse(os.path.isdir(folder))
        self.assertTrue(os.path.isdir(new_path))

    def test_rename_event_folder_idempotent_if_already_suffixed(self):
        fm = FileManager(storage_path=self.tmp, retention_days=1)
        folder = os.path.join(self.tmp, "events", "1700000000_ev123-canceled")
        os.makedirs(folder, exist_ok=True)
        new_path = fm.rename_event_folder(folder)
        self.assertEqual(new_path, folder)
        self.assertTrue(os.path.isdir(folder))

    def test_write_canceled_summary_writes_title_and_description(self):
        fm = FileManager(storage_path=self.tmp, retention_days=1)
        folder = os.path.join(self.tmp, "events", "1700000000_ce1")
        os.makedirs(folder, exist_ok=True)
        result = fm.write_canceled_summary(folder)
        self.assertTrue(result)
        summary_path = os.path.join(folder, "summary.txt")
        self.assertTrue(os.path.isfile(summary_path))
        with open(summary_path) as f:
            content = f.read()
        self.assertIn("Title: Canceled event: max event length exceeded", content)
        self.assertIn("Event exceeded max_event_length_seconds", content)


class TestFileManagerTimeline(unittest.TestCase):
    """append_timeline_entry writes to JSONL only (not full notification_timeline.json)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def test_append_timeline_entry_creates_jsonl_not_full_json(self):
        fm = FileManager(self.tmp, 3)
        folder = os.path.join(self.tmp, "cam1", "123_ev1")
        os.makedirs(folder, exist_ok=True)
        fm.append_timeline_entry(folder, {"label": "HA", "data": {}})
        append_path = os.path.join(folder, "notification_timeline_append.jsonl")
        base_path = os.path.join(folder, "notification_timeline.json")
        self.assertTrue(os.path.isfile(append_path), "append should create .jsonl file")
        self.assertFalse(os.path.isfile(base_path), "append should NOT create/overwrite full .json")


class TestFileManagerStorageStats(unittest.TestCase):
    """compute_storage_stats (legacy, consolidated, daily_reports) and /stats API format."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        self.fm = FileManager(self.tmp, retention_days=7)

    def test_empty_storage_returns_zero_totals(self):
        stats = self.fm.compute_storage_stats()
        self.assertEqual(stats['total'], 0)
        self.assertEqual(stats['clips'], 0)
        self.assertEqual(stats['snapshots'], 0)
        self.assertEqual(stats['descriptions'], 0)
        self.assertEqual(stats['by_camera'], {})

    def test_legacy_camera_event_folder_counted(self):
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
        for sub in ("daily_reports", "daily_reviews"):
            path = os.path.join(self.tmp, sub)
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "report.md"), "w") as f:
                f.write("x" * 300)
        stats = self.fm.compute_storage_stats()
        self.assertGreaterEqual(stats['descriptions'], 600)
        self.assertGreaterEqual(stats['total'], 600)

    def test_legacy_and_consolidated_both_counted(self):
        cam = os.path.join(self.tmp, "doorbell")
        ev = os.path.join(cam, "1700000002_evt2")
        os.makedirs(ev, exist_ok=True)
        with open(os.path.join(ev, "clip.mp4"), "wb") as f:
            f.write(b"1" * 100)
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
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
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
        from frigate_buffer.web.server import create_app
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


if __name__ == "__main__":
    unittest.main()
