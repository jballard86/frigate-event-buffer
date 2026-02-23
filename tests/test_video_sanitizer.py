"""
Tests for video_sanitizer.sanitize_for_nelux.

Context manager yields temp path on FFmpeg success, original path on failure;
temp file is always removed in finally.
"""

import os
import unittest
from unittest.mock import MagicMock, patch


class TestSanitizeForNelux(unittest.TestCase):
    """Tests for sanitize_for_nelux context manager."""

    @patch("frigate_buffer.services.video_sanitizer.os.path.exists")
    @patch("frigate_buffer.services.video_sanitizer.subprocess.run")
    @patch("frigate_buffer.services.video_sanitizer.os.remove")
    @patch("frigate_buffer.services.video_sanitizer.os.close")
    @patch("frigate_buffer.services.video_sanitizer.tempfile.mkstemp")
    def test_sanitize_for_nelux_success_yields_temp_path_and_removes_on_exit(
        self, mock_mkstemp, mock_close, mock_remove, mock_run, mock_exists
    ):
        """On FFmpeg success, yield temp path; in finally remove temp file."""
        mock_mkstemp.return_value = (99, "/dev/shm/sanitized_abc123.mp4")
        mock_exists.return_value = True
        mock_run.return_value = MagicMock(returncode=0)
        from frigate_buffer.services.video_sanitizer import sanitize_for_nelux

        clip_path = "/storage/events/cam1/clip.mp4"
        with sanitize_for_nelux(clip_path) as safe_path:
            self.assertNotEqual(safe_path, clip_path)
            self.assertIn("sanitized_", safe_path)
            self.assertTrue(safe_path.endswith(".mp4"))
        mock_remove.assert_called_once()
        self.assertEqual(mock_remove.call_args[0][0], "/dev/shm/sanitized_abc123.mp4")
        mock_close.assert_called_once_with(99)

    @patch("frigate_buffer.services.video_sanitizer.os.path.exists")
    @patch("frigate_buffer.services.video_sanitizer.logger")
    @patch("frigate_buffer.services.video_sanitizer.subprocess.run")
    @patch("frigate_buffer.services.video_sanitizer.os.remove")
    @patch("frigate_buffer.services.video_sanitizer.os.close")
    @patch("frigate_buffer.services.video_sanitizer.tempfile.mkstemp")
    def test_sanitize_for_nelux_ffmpeg_failure_yields_original_path_and_logs(
        self, mock_mkstemp, mock_close, mock_remove, mock_run, mock_logger, mock_exists
    ):
        """On CalledProcessError, log stderr and yield clip_path as fallback."""
        import subprocess

        mock_mkstemp.return_value = (99, "/dev/shm/sanitized_xyz.mp4")
        mock_exists.return_value = True
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg", stderr="Invalid data")
        from frigate_buffer.services.video_sanitizer import sanitize_for_nelux

        clip_path = "/storage/events/cam1/bad.mp4"
        with sanitize_for_nelux(clip_path) as safe_path:
            self.assertEqual(safe_path, clip_path)
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0]
        self.assertIn(clip_path, call_args)
        mock_remove.assert_called_once()

    @patch("frigate_buffer.services.video_sanitizer.subprocess.run")
    @patch("frigate_buffer.services.video_sanitizer.os.path.exists")
    @patch("frigate_buffer.services.video_sanitizer.os.remove")
    @patch("frigate_buffer.services.video_sanitizer.os.close")
    @patch("frigate_buffer.services.video_sanitizer.tempfile.mkstemp")
    def test_sanitize_for_nelux_removes_temp_even_when_path_exists(
        self, mock_mkstemp, mock_close, mock_remove, mock_exists, mock_run
    ):
        """Finally block calls os.remove(temp_path) when file exists."""
        mock_mkstemp.return_value = (99, "/tmp/sanitized_foo.mp4")
        mock_exists.return_value = True
        mock_run.return_value = MagicMock(returncode=0)
        from frigate_buffer.services.video_sanitizer import sanitize_for_nelux

        with sanitize_for_nelux("/some/clip.mp4") as safe_path:
            self.assertEqual(safe_path, "/tmp/sanitized_foo.mp4")
        mock_remove.assert_called_once_with("/tmp/sanitized_foo.mp4")
