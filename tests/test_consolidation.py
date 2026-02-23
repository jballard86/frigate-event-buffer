"""Tests for ConsolidatedEventManager: mark_closing, schedule_close_timer, get_or_create with closing state."""
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frigate_buffer.managers.consolidation import ConsolidatedEventManager
from frigate_buffer.models import ConsolidatedEvent


def _make_file_manager_mock():
    fm = MagicMock()
    tmp = tempfile.mkdtemp()
    def _create_folder(name):
        p = os.path.join(tmp, "events", name)
        os.makedirs(p, exist_ok=True)
        return p
    def _ensure_camera(ce_path, camera):
        p = os.path.join(ce_path, camera.replace(" ", "_").lower())
        os.makedirs(p, exist_ok=True)
        return p
    fm.create_consolidated_event_folder.side_effect = _create_folder
    fm.ensure_consolidated_camera_folder.side_effect = _ensure_camera
    return fm, tmp


class TestConsolidationClosingState(unittest.TestCase):

    def setUp(self):
        self.fm, self.tmp = _make_file_manager_mock()
        self.mgr = ConsolidatedEventManager(self.fm, event_gap_seconds=120)

    def tearDown(self):
        import shutil
        if os.path.exists(self.tmp):
            try:
                shutil.rmtree(self.tmp)
            except OSError:
                pass

    def test_mark_closing_returns_true_then_false(self):
        now = 1000.0
        ce, is_new, _ = self.mgr.get_or_create("e1", "cam1", "person", now)
        self.assertTrue(is_new)
        ce_id = ce.consolidated_id

        self.assertTrue(self.mgr.mark_closing(ce_id))
        self.assertTrue(ce.closing)
        self.assertFalse(ce.closed)

        self.assertFalse(self.mgr.mark_closing(ce_id))

    def test_mark_closing_unknown_ce_returns_false(self):
        self.assertFalse(self.mgr.mark_closing("nonexistent"))

    def test_get_active_ce_folders_returns_tuple_of_folder_names(self):
        """get_active_ce_folders() returns tuple of folder names (no full CE list)."""
        ce1, _, _ = self.mgr.get_or_create("e1", "cam1", "person", 1000.0)
        folders = self.mgr.get_active_ce_folders()
        self.assertIsInstance(folders, tuple, "get_active_ce_folders must return a tuple")
        self.assertGreaterEqual(len(folders), 1)
        self.assertIn(ce1.folder_name, folders)
        for f in folders:
            self.assertIsInstance(f, str, "each folder must be a string (folder_name), not a CE object")

    def test_schedule_close_timer_noop_when_closing(self):
        now = 1000.0
        ce, _, _ = self.mgr.get_or_create("e1", "cam1", "person", now)
        ce_id = ce.consolidated_id
        self.mgr.mark_closing(ce_id)

        self.mgr.schedule_close_timer(ce_id)
        self.assertEqual(len(self.mgr._close_timers), 0)

    def test_schedule_close_timer_noop_when_closed(self):
        now = 1000.0
        ce, _, _ = self.mgr.get_or_create("e1", "cam1", "person", now)
        ce_id = ce.consolidated_id
        ce.closed = True

        self.mgr.schedule_close_timer(ce_id)
        self.assertEqual(len(self.mgr._close_timers), 0)

    def test_get_or_create_does_not_add_to_closing_ce(self):
        now = 1000.0
        ce1, is_new, _ = self.mgr.get_or_create("e1", "cam1", "person", now)
        self.assertTrue(is_new)
        ce_id = ce1.consolidated_id
        self.mgr.mark_closing(ce_id)

        ce2, is_new2, _ = self.mgr.get_or_create(
            "e2", "cam2", "car", now + 10
        )
        self.assertTrue(is_new2)
        self.assertIsNot(ce2.consolidated_id, ce_id)
        self.assertEqual(len(ce1.frigate_event_ids), 1)
        self.assertNotIn("e2", ce1.frigate_event_ids)

    def test_remove_event_from_ce_returns_none_when_ce_has_other_events(self):
        now = 1000.0
        ce1, _, _ = self.mgr.get_or_create("e1", "cam1", "person", now)
        self.mgr.get_or_create("e2", "cam2", "car", now + 5)
        self.assertEqual(len(ce1.frigate_event_ids), 2)
        result = self.mgr.remove_event_from_ce("e1")
        self.assertIsNone(result)
        self.assertEqual(len(ce1.frigate_event_ids), 1)
        self.assertIn("e2", ce1.frigate_event_ids)

    def test_remove_event_from_ce_returns_folder_path_when_ce_becomes_empty(self):
        now = 1000.0
        ce1, _, _ = self.mgr.get_or_create("e1", "cam1", "person", now)
        ce_id = ce1.consolidated_id
        ce_folder = ce1.folder_path
        result = self.mgr.remove_event_from_ce("e1")
        self.assertEqual(result, ce_folder)
        self.assertNotIn(ce_id, self.mgr._events)

    def test_remove_event_from_ce_unknown_event_returns_none(self):
        result = self.mgr.remove_event_from_ce("nonexistent")
        self.assertIsNone(result)

    def test_get_or_create_does_not_add_to_closed_ce(self):
        now = 1000.0
        ce1, is_new, _ = self.mgr.get_or_create("e1", "cam1", "person", now)
        ce1.closed = True
        ce_id = ce1.consolidated_id

        ce2, is_new2, _ = self.mgr.get_or_create(
            "e2", "cam2", "car", now + 10
        )
        self.assertTrue(is_new2)
        self.assertIsNot(ce2.consolidated_id, ce_id)


if __name__ == '__main__':
    unittest.main()
