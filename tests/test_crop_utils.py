"""Tests for crop_utils (motion crop, timestamp overlay)."""

import unittest

import numpy as np

from frigate_buffer.services import crop_utils


class TestDrawTimestampOverlay(unittest.TestCase):
    """Tests for draw_timestamp_overlay return value and read-only frame handling."""

    def test_draw_timestamp_overlay_readonly_returns_writable_copy(self):
        """When frame is read-only, overlay copies and returns a writable array."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame.setflags(write=False)
        self.assertFalse(frame.flags.writeable)

        result = crop_utils.draw_timestamp_overlay(
            frame,
            "2026-02-17 15:42:55",
            "test_cam",
            1,
            3,
        )

        self.assertIsNotNone(result)
        self.assertTrue(result.flags.writeable)
        self.assertIsNot(result, frame)

    def test_draw_timestamp_overlay_writable_returns_same_object(self):
        """When frame is writable, overlay draws in-place and returns same array."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertTrue(frame.flags.writeable)

        result = crop_utils.draw_timestamp_overlay(
            frame,
            "2026-02-17 15:42:55",
            "test_cam",
            1,
            3,
        )

        self.assertIs(result, frame)
        self.assertTrue(result.flags.writeable)
