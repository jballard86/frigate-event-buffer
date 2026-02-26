"""Tests for main module version loading."""

import unittest
from pathlib import Path
from unittest.mock import patch

from frigate_buffer.main import _load_version


class TestLoadVersion(unittest.TestCase):
    """Tests for _load_version startup behavior."""

    def test_load_version_returns_string_from_version_txt(self) -> None:
        """When version.txt exists (root or package), return its stripped contents."""
        result = _load_version()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertNotEqual(
            result,
            "unknown",
            "version.txt should be found from project root or package",
        )

    def test_load_version_returns_unknown_when_file_missing(self) -> None:
        """When version.txt does not exist, return 'unknown'."""
        real_exists = Path.exists

        def mock_exists(self: object) -> bool:
            if getattr(self, "name", "") == "version.txt":
                return False
            return real_exists(self)

        with patch.object(Path, "exists", mock_exists):
            result = _load_version()

        self.assertEqual(result, "unknown")
