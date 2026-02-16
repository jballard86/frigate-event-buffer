"""Tests for FileManager path validation (path traversal safety)."""
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frigate_buffer.managers.file import FileManager


class TestFileManagerPathValidation(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.fm = FileManager(
            self.tmp,
            3
        )

    def tearDown(self):
        import shutil
        if os.path.exists(self.tmp):
            try:
                shutil.rmtree(self.tmp)
            except OSError:
                pass

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
        # folder_name that resolves outside storage (path traversal)
        with self.assertRaises(ValueError) as ctx:
            self.fm.create_consolidated_event_folder("../../outside")
        self.assertIn("Invalid consolidated event path", str(ctx.exception))

    def test_create_event_folder_valid_succeeds(self):
        path = self.fm.create_event_folder("evt123", "cam1", 1000.0)
        self.assertTrue(path.startswith(os.path.realpath(self.tmp)))
        self.assertTrue(os.path.isdir(path))


if __name__ == '__main__':
    unittest.main()
