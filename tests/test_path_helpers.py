"""
Tests for web path_helpers: resolve_under_storage must reject path traversal
and return normalized path only when the result is strictly under storage root.
"""

import os
import tempfile
import unittest

from frigate_buffer.web.path_helpers import resolve_under_storage


class TestResolveUnderStorage(unittest.TestCase):
    """Verify resolve_under_storage rejects escapes and returns path when safe."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(
            lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True)
        )

    def test_valid_subpath_returns_resolved_path(self):
        """A path under storage returns the normalized absolute path."""
        subdir = os.path.join("camera", "123_evt")
        result = resolve_under_storage(self.tmp, subdir)
        assert result is not None
        base = os.path.realpath(self.tmp)
        assert os.path.normpath(result).startswith(base)
        assert "camera" in result
        assert "123_evt" in result

    def test_path_traversal_returns_none(self):
        """Path with .. that would escape storage returns None."""
        result = resolve_under_storage(self.tmp, "..", "etc", "passwd")
        assert result is None

    def test_path_traversal_within_join_returns_none(self):
        """Path like camera/../../../etc returns None."""
        result = resolve_under_storage(self.tmp, "camera", "..", "..", "..", "etc")
        assert result is None

    def test_empty_path_parts_returns_none(self):
        """No path_parts returns None (storage root is not allowed)."""
        result = resolve_under_storage(self.tmp)
        assert result is None

    def test_single_part_under_storage_returns_path(self):
        """Single segment under storage returns resolved path."""
        result = resolve_under_storage(self.tmp, "daily_reports")
        assert result is not None
        assert result.startswith(os.path.realpath(self.tmp))
        assert result != os.path.realpath(self.tmp)

    def test_storage_root_only_returns_none(self):
        """Only segment "." resolves to base; must not equal base (returns None)."""
        # resolve_under_storage(tmp, ".") normalizes to tmp, equals base -> None
        result = resolve_under_storage(self.tmp, ".")
        assert result is None
