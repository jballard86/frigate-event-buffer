"""Tests for shared constants and is_tensor helper."""

import unittest

import numpy as np

from frigate_buffer.constants import (
    DEFAULT_STORAGE_STATS_MAX_AGE_SECONDS,
    ERROR_BUFFER_MAX_SIZE,
    FRAME_MAX_WIDTH,
    HTTP_DOWNLOAD_CHUNK_SIZE,
    HTTP_STREAM_CHUNK_SIZE,
    LOG_MAX_RESPONSE_BODY,
    NON_CAMERA_DIRS,
    is_tensor,
)


class TestIsTensor(unittest.TestCase):
    """Tests for is_tensor(): name-based check without importing torch."""

    def test_numpy_array_returns_false(self):
        """Non-tensor types (e.g. numpy array) return False."""
        arr = np.zeros((1, 3, 100, 100), dtype=np.uint8)
        self.assertFalse(is_tensor(arr))

    def test_none_returns_false(self):
        """None returns False."""
        self.assertFalse(is_tensor(None))

    def test_int_returns_false(self):
        """Plain int returns False."""
        self.assertFalse(is_tensor(42))

    def test_tensor_like_by_name_returns_true(self):
        """Object whose type __name__ is 'Tensor' returns True (same check as real torch.Tensor)."""
        class Tensor:
            pass
        obj = Tensor()
        self.assertTrue(is_tensor(obj))

    def test_tensor_instance_returns_true_when_torch_available(self):
        """When torch is available, a real torch.Tensor returns True."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        t = torch.zeros(1, 3, 10, 10)
        self.assertTrue(is_tensor(t))


class TestNonCameraDirs(unittest.TestCase):
    """Sanity checks for NON_CAMERA_DIRS constant."""

    def test_contains_expected_dirs(self):
        """NON_CAMERA_DIRS includes ultralytics, yolo_models, daily_reports, daily_reviews."""
        expected = {"ultralytics", "yolo_models", "daily_reports", "daily_reviews"}
        self.assertEqual(NON_CAMERA_DIRS, frozenset(expected))

    def test_is_frozenset(self):
        """NON_CAMERA_DIRS is immutable."""
        self.assertIsInstance(NON_CAMERA_DIRS, frozenset)


class TestNumericConstants(unittest.TestCase):
    """Sanity checks for shared numeric constants (avoid magic number drift)."""

    def test_http_chunk_sizes(self):
        """Stream and download chunk sizes are positive and distinct."""
        self.assertEqual(HTTP_STREAM_CHUNK_SIZE, 8192)
        self.assertEqual(HTTP_DOWNLOAD_CHUNK_SIZE, 65536)
        self.assertLess(HTTP_STREAM_CHUNK_SIZE, HTTP_DOWNLOAD_CHUNK_SIZE)

    def test_display_and_log_constants(self):
        """LOG_MAX_RESPONSE_BODY and FRAME_MAX_WIDTH are positive."""
        self.assertEqual(LOG_MAX_RESPONSE_BODY, 2000)
        self.assertEqual(FRAME_MAX_WIDTH, 1280)

    def test_storage_stats_and_error_buffer(self):
        """Storage stats TTL and error buffer size are positive."""
        self.assertEqual(DEFAULT_STORAGE_STATS_MAX_AGE_SECONDS, 30 * 60)
        self.assertEqual(ERROR_BUFFER_MAX_SIZE, 10)
