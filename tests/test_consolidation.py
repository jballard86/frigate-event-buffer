"""Tests for ConsolidatedEventManager: mark_closing, schedule_close_timer,
get_or_create with closing state."""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frigate_buffer.managers.consolidation import ConsolidatedEventManager


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
        assert is_new
        ce_id = ce.consolidated_id

        assert self.mgr.mark_closing(ce_id)
        assert ce.closing
        assert not ce.closed

        assert not self.mgr.mark_closing(ce_id)

    def test_mark_closing_unknown_ce_returns_false(self):
        assert not self.mgr.mark_closing("nonexistent")

    def test_get_active_ce_folders_returns_tuple_of_folder_names(self):
        """get_active_ce_folders() returns tuple of folder names (no full CE list)."""
        ce1, _, _ = self.mgr.get_or_create("e1", "cam1", "person", 1000.0)
        folders = self.mgr.get_active_ce_folders()
        assert isinstance(folders, tuple), "get_active_ce_folders must return a tuple"
        assert len(folders) >= 1
        assert ce1.folder_name in folders
        for f in folders:
            assert isinstance(f, str), (
                "each folder must be a string (folder_name), not a CE object"
            )

    def test_schedule_close_timer_noop_when_closing(self):
        now = 1000.0
        ce, _, _ = self.mgr.get_or_create("e1", "cam1", "person", now)
        ce_id = ce.consolidated_id
        self.mgr.mark_closing(ce_id)

        self.mgr.schedule_close_timer(ce_id)
        assert len(self.mgr._close_timers) == 0

    def test_schedule_close_timer_noop_when_closed(self):
        now = 1000.0
        ce, _, _ = self.mgr.get_or_create("e1", "cam1", "person", now)
        ce_id = ce.consolidated_id
        ce.closed = True

        self.mgr.schedule_close_timer(ce_id)
        assert len(self.mgr._close_timers) == 0

    def test_get_or_create_does_not_add_to_closing_ce(self):
        now = 1000.0
        ce1, is_new, _ = self.mgr.get_or_create("e1", "cam1", "person", now)
        assert is_new
        ce_id = ce1.consolidated_id
        self.mgr.mark_closing(ce_id)

        ce2, is_new2, _ = self.mgr.get_or_create("e2", "cam2", "car", now + 10)
        assert is_new2
        assert ce2.consolidated_id is not ce_id
        assert len(ce1.frigate_event_ids) == 1
        assert "e2" not in ce1.frigate_event_ids

    def test_remove_event_from_ce_returns_none_when_ce_has_other_events(self):
        now = 1000.0
        ce1, _, _ = self.mgr.get_or_create("e1", "cam1", "person", now)
        self.mgr.get_or_create("e2", "cam2", "car", now + 5)
        assert len(ce1.frigate_event_ids) == 2
        result = self.mgr.remove_event_from_ce("e1")
        assert result is None
        assert len(ce1.frigate_event_ids) == 1
        assert "e2" in ce1.frigate_event_ids

    def test_remove_event_from_ce_returns_folder_path_when_ce_becomes_empty(self):
        now = 1000.0
        ce1, _, _ = self.mgr.get_or_create("e1", "cam1", "person", now)
        ce_id = ce1.consolidated_id
        ce_folder = ce1.folder_path
        result = self.mgr.remove_event_from_ce("e1")
        assert result == ce_folder
        assert ce_id not in self.mgr._events

    def test_remove_event_from_ce_unknown_event_returns_none(self):
        result = self.mgr.remove_event_from_ce("nonexistent")
        assert result is None

    def test_set_final_from_frigate_sets_ce_final_fields(self):
        """set_final_from_frigate sets final_title, description, threat_level on CE."""
        now = 1000.0
        ce, _, _ = self.mgr.get_or_create("e1", "cam1", "person", now)
        assert ce.final_title is None
        assert ce.final_description is None
        assert ce.final_threat_level == 0
        self.mgr.set_final_from_frigate(
            "e1", title="Test Title", description="Test desc", threat_level=2
        )
        assert ce.final_title == "Test Title"
        assert ce.final_description == "Test desc"
        assert ce.final_threat_level == 2

    def test_set_final_from_ce_analysis_sets_ce_final_fields(self):
        """set_final_from_ce_analysis sets final_* on the CE by ce_id."""
        now = 1000.0
        ce, _, _ = self.mgr.get_or_create("e1", "cam1", "person", now)
        ce_id = ce.consolidated_id
        self.mgr.set_final_from_ce_analysis(
            ce_id, title="CE Title", description="CE desc", threat_level=1
        )
        assert ce.final_title == "CE Title"
        assert ce.final_description == "CE desc"
        assert ce.final_threat_level == 1

    def test_get_or_create_does_not_add_to_closed_ce(self):
        now = 1000.0
        ce1, is_new, _ = self.mgr.get_or_create("e1", "cam1", "person", now)
        ce1.closed = True
        ce_id = ce1.consolidated_id

        ce2, is_new2, _ = self.mgr.get_or_create("e2", "cam2", "car", now + 10)
        assert is_new2
        assert ce2.consolidated_id is not ce_id


if __name__ == "__main__":
    unittest.main()
