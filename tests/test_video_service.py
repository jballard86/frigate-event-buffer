import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frigate_buffer.services.video import VideoService, ensure_detection_model_ready

class TestVideoService(unittest.TestCase):

    def setUp(self):
        self.video_service = VideoService(ffmpeg_timeout=1)

    @patch('frigate_buffer.services.video.subprocess.Popen')
    @patch('frigate_buffer.services.video.os.remove')
    @patch('frigate_buffer.services.video.os.path.exists')
    @patch('frigate_buffer.services.video.ffmpegcv.VideoCaptureNV')
    def test_transcode_success(self, mock_video_capture_nv, mock_exists, mock_remove, mock_popen):
        # GPU path fails (e.g. no GPU in CI), so fallback to libx264 is used
        mock_video_capture_nv.side_effect = RuntimeError("No GPU")
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        mock_exists.return_value = False

        result = self.video_service.transcode_clip_to_h264("evt1", "temp.mp4", "final.mp4")

        self.assertTrue(result)
        mock_remove.assert_called_with("temp.mp4")
        mock_popen.assert_called_once()

    @patch('frigate_buffer.services.video.subprocess.Popen')
    @patch('frigate_buffer.services.video.os.rename')
    @patch('frigate_buffer.services.video.os.path.exists')
    @patch('frigate_buffer.services.video.ffmpegcv.VideoCaptureNV')
    def test_transcode_failure(self, mock_video_capture_nv, mock_exists, mock_rename, mock_popen):
        mock_video_capture_nv.side_effect = RuntimeError("No GPU")
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"", b"Error")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        mock_exists.return_value = True

        result = self.video_service.transcode_clip_to_h264("evt1", "temp.mp4", "final.mp4")

        self.assertTrue(result)
        mock_rename.assert_called_with("temp.mp4", "final.mp4")

    @patch('frigate_buffer.services.video.subprocess.run')
    @patch('frigate_buffer.services.video.os.remove')
    @patch('frigate_buffer.services.video.os.path.exists')
    @patch('frigate_buffer.services.video.ffmpegcv.VideoWriterNV')
    @patch('frigate_buffer.services.video.ffmpegcv.VideoCaptureNV')
    def test_transcode_nvenc_success(self, mock_capture_nv, mock_writer_nv, mock_exists, mock_remove, mock_run):
        """When GPU is available, transcode uses VideoCaptureNV and VideoWriterNV then muxes audio."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.fps = 30.0
        import numpy as np
        mock_cap.read.side_effect = [(True, np.zeros((480, 640, 3), dtype=np.uint8)), (False, None)]
        mock_cap.release = MagicMock()
        mock_capture_nv.return_value = mock_cap

        mock_writer = MagicMock()
        mock_writer_nv.return_value = mock_writer

        mock_run.return_value = MagicMock(returncode=0)
        mock_exists.side_effect = lambda p: True

        result = self.video_service.transcode_clip_to_h264("evt1", "/tmp/temp.mp4", "/tmp/final.mp4")

        self.assertTrue(result)
        mock_capture_nv.assert_called_once()
        mock_writer_nv.assert_called_once()
        mock_writer.write.assert_called()
        mock_run.assert_called_once()
        mock_remove.assert_any_call("/tmp/temp.mp4")

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


class TestEnsureDetectionModelReady(unittest.TestCase):
    """Tests for startup detection model check."""

    def test_ensure_detection_model_ready_skips_when_not_configured(self):
        """When DETECTION_MODEL is empty or missing, returns False and skips load."""
        self.assertFalse(ensure_detection_model_ready({}))
        self.assertFalse(ensure_detection_model_ready({"DETECTION_MODEL": ""}))

    @patch("ultralytics.YOLO")
    @patch("frigate_buffer.services.video.os.path.isfile")
    def test_ensure_detection_model_ready_loads_and_reports(self, mock_isfile, mock_yolo_cls):
        """When DETECTION_MODEL is set, loads model (download if needed) and returns True."""
        mock_isfile.return_value = False
        mock_model = MagicMock()
        mock_model.ckpt_path = "/path/to/yolov8n.pt"
        mock_yolo_cls.return_value = mock_model
        with patch("frigate_buffer.services.video.os.path.isfile", side_effect=lambda p: p == "/path/to/yolov8n.pt"):
            result = ensure_detection_model_ready({"DETECTION_MODEL": "yolov8n.pt"})
        self.assertTrue(result)
        mock_yolo_cls.assert_called_once_with("yolov8n.pt")

if __name__ == "__main__":
    unittest.main()
