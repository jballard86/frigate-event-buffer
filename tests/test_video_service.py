import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frigate_buffer.services.video import VideoService

class TestVideoService(unittest.TestCase):

    def setUp(self):
        self.video_service = VideoService(ffmpeg_timeout=1)

    @patch('frigate_buffer.services.video.subprocess.Popen')
    @patch('frigate_buffer.services.video.os.remove')
    @patch('frigate_buffer.services.video.os.path.exists')
    def test_transcode_success(self, mock_exists, mock_remove, mock_popen):
        # Setup mock process
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        mock_exists.return_value = False # Temp file removed

        result = self.video_service.transcode_clip_to_h264("evt1", "temp.mp4", "final.mp4")

        self.assertTrue(result)
        mock_remove.assert_called_with("temp.mp4")
        mock_popen.assert_called_once()

    @patch('frigate_buffer.services.video.subprocess.Popen')
    @patch('frigate_buffer.services.video.os.rename')
    @patch('frigate_buffer.services.video.os.path.exists')
    def test_transcode_failure(self, mock_exists, mock_rename, mock_popen):
        # Setup mock process
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"", b"Error")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        mock_exists.return_value = True # Temp file exists

        result = self.video_service.transcode_clip_to_h264("evt1", "temp.mp4", "final.mp4")

        self.assertTrue(result) # It returns True even on error in current impl (logs error and renames)
        mock_rename.assert_called_with("temp.mp4", "final.mp4")

    @patch('frigate_buffer.services.video.subprocess.run')
    @patch('frigate_buffer.services.video.os.path.exists')
    def test_generate_gif_success(self, mock_exists, mock_run):
        mock_exists.return_value = True
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc

        result = self.video_service.generate_gif_from_clip("clip.mp4", "out.gif")
        self.assertTrue(result)

    @patch('frigate_buffer.services.video.subprocess.run')
    def test_generate_gif_failure(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_run.return_value = mock_proc

        result = self.video_service.generate_gif_from_clip("clip.mp4", "out.gif")
        self.assertFalse(result)

if __name__ == "__main__":
    unittest.main()
