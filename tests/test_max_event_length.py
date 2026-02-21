"""
Tests for max event length (cancel long events): config, FileManager rename/canceled summary,
and lifecycle ensuring API is never sent for over-max events (see test_lifecycle_service).
"""

import os
import shutil
import tempfile
import unittest

from frigate_buffer.managers.file import FileManager


class TestFileManagerMaxEventLength(unittest.TestCase):
    """Test rename_event_folder and write_canceled_summary."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def test_rename_event_folder_appends_suffix(self):
        """rename_event_folder renames to basename + suffix and returns new path."""
        fm = FileManager(storage_path=self.tmp, retention_days=1)
        folder = os.path.join(self.tmp, "cam1", "1700000000_ev123")
        os.makedirs(folder, exist_ok=True)
        new_path = fm.rename_event_folder(folder)
        self.assertEqual(os.path.basename(new_path), "1700000000_ev123-canceled")
        self.assertFalse(os.path.isdir(folder))
        self.assertTrue(os.path.isdir(new_path))

    def test_rename_event_folder_idempotent_if_already_suffixed(self):
        """If folder already ends with suffix, rename returns same path."""
        fm = FileManager(storage_path=self.tmp, retention_days=1)
        folder = os.path.join(self.tmp, "events", "1700000000_ev123-canceled")
        os.makedirs(folder, exist_ok=True)
        new_path = fm.rename_event_folder(folder)
        self.assertEqual(new_path, folder)
        self.assertTrue(os.path.isdir(folder))

    def test_write_canceled_summary_writes_title_and_description(self):
        """write_canceled_summary writes summary.txt with cancel title for event view."""
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
