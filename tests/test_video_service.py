import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from frigate_buffer.services.video import (
    VideoService,
    ensure_detection_model_ready,
)


class TestVideoService(unittest.TestCase):
    def setUp(self):
        self.video_service = VideoService(ffmpeg_timeout=1)

    @patch("frigate_buffer.services.video.subprocess.run")
    @patch("frigate_buffer.services.video.os.path.exists")
    def test_generate_gif_success(self, mock_exists, mock_run):
        mock_exists.return_value = True
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc

        result = self.video_service.generate_gif_from_clip("clip.mp4", "out.gif")
        self.assertTrue(result)

    @patch("frigate_buffer.services.video.subprocess.run")
    def test_generate_gif_failure(self, mock_run):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_run.return_value = mock_proc

        result = self.video_service.generate_gif_from_clip("clip.mp4", "out.gif")
        self.assertFalse(result)

    @patch("frigate_buffer.services.video.ffmpegcv.VideoCapture")
    @patch("frigate_buffer.services.video.ffmpegcv.VideoCaptureNV")
    def test_generate_detection_sidecar_opens_clip_and_writes_json_schema(
        self, mock_capture_nv, mock_capture_cpu
    ):
        """generate_detection_sidecar reads frames and writes detection.json with expected schema."""
        import numpy as np
        mock_capture_nv.side_effect = RuntimeError("No GPU")
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.fps = 30.0
        # Return 3 frames then EOF
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),
            (False, None),
        ]
        mock_cap.release = MagicMock()
        mock_capture_cpu.return_value = mock_cap

        config = {
            "DETECTION_MODEL": "",
            "DETECTION_DEVICE": "",
            "DETECTION_FRAME_INTERVAL": 2,
        }
        tmp = os.path.join(os.path.dirname(__file__), "tmp_video_sidecar")
        os.makedirs(tmp, exist_ok=True)
        try:
            clip_path = os.path.join(tmp, "clip.mp4")
            sidecar_path = os.path.join(tmp, "detection.json")
            with open(clip_path, "wb"):
                pass
            result = self.video_service.generate_detection_sidecar(
                clip_path, sidecar_path, config
            )
            self.assertTrue(result)
            self.assertTrue(os.path.isfile(sidecar_path))
            with open(sidecar_path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsInstance(data, dict)
            self.assertIn("entries", data)
            self.assertIn("native_width", data)
            self.assertIn("native_height", data)
            entries = data["entries"]
            self.assertIsInstance(entries, list)
            for entry in entries:
                self.assertIn("frame_number", entry)
                self.assertIn("timestamp_sec", entry)
                self.assertIn("detections", entry)
                self.assertIsInstance(entry["detections"], list)
        finally:
            for f in (sidecar_path, clip_path):
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
            if os.path.isdir(tmp):
                try:
                    os.rmdir(tmp)
                except OSError:
                    pass

    @patch("frigate_buffer.services.video.ffmpegcv.VideoCapture")
    @patch("frigate_buffer.services.video.ffmpegcv.VideoCaptureNV")
    def test_generate_detection_sidecar_returns_false_when_clip_wont_open(
        self, mock_capture_nv, mock_capture_cpu
    ):
        mock_capture_nv.side_effect = RuntimeError("No GPU")
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_capture_cpu.return_value = mock_cap

        tmp = os.path.join(os.path.dirname(__file__), "tmp_video_sidecar_noclip")
        os.makedirs(tmp, exist_ok=True)
        try:
            sidecar_path = os.path.join(tmp, "detection.json")
            result = self.video_service.generate_detection_sidecar(
                "/nonexistent/clip.mp4", sidecar_path, {}
            )
            self.assertFalse(result)
            self.assertFalse(os.path.isfile(sidecar_path))
        finally:
            if os.path.isdir(tmp):
                try:
                    for f in os.listdir(tmp):
                        os.remove(os.path.join(tmp, f))
                    os.rmdir(tmp)
                except OSError:
                    pass

    def test_generate_detection_sidecars_for_cameras_acquires_and_releases_app_lock(self):
        """When set_sidecar_app_lock was called, generate_detection_sidecars_for_cameras acquires and releases the lock."""
        mock_lock = MagicMock()
        self.video_service.set_sidecar_app_lock(mock_lock)
        with patch.object(
            self.video_service,
            "generate_detection_sidecar",
            return_value=True,
        ):
            result = self.video_service.generate_detection_sidecars_for_cameras(
                [("cam1", "/fake/clip.mp4", "/fake/detection.json")], {}
            )
        self.assertEqual(result, [("cam1", True)])
        mock_lock.acquire.assert_called_once()
        mock_lock.release.assert_called_once()

    def test_generate_detection_sidecars_for_cameras_no_lock_when_not_set(self):
        """When set_sidecar_app_lock was never called, no lock is used (e.g. in tests)."""
        self.assertIsNone(self.video_service._sidecar_app_lock)
        result = self.video_service.generate_detection_sidecars_for_cameras([], {})
        self.assertEqual(result, [])


class TestEnsureDetectionModelReady(unittest.TestCase):
    def test_ensure_detection_model_ready_skips_when_not_configured(self):
        self.assertFalse(ensure_detection_model_ready({}))
        self.assertFalse(ensure_detection_model_ready({"DETECTION_MODEL": ""}))

    @patch("ultralytics.YOLO")
    def test_ensure_detection_model_ready_loads_and_reports(self, mock_yolo_cls):
        mock_model = MagicMock()
        mock_model.ckpt_path = "/path/to/yolov8n.pt"
        mock_yolo_cls.return_value = mock_model
        with patch(
            "frigate_buffer.services.video.os.path.isfile",
            side_effect=lambda p: p == "/path/to/yolov8n.pt" or p == "yolov8n.pt",
        ):
            result = ensure_detection_model_ready({"DETECTION_MODEL": "yolov8n.pt"})
        self.assertTrue(result)
        mock_yolo_cls.assert_called_once_with("yolov8n.pt")


if __name__ == "__main__":
    unittest.main()
