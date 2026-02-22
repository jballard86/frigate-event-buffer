import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Fake nelux module so we can patch VideoReader when the real wheel is not installed (e.g. on Windows).
if "nelux" not in sys.modules:
    sys.modules["nelux"] = MagicMock()

from frigate_buffer.services.video import (
    VideoService,
    BATCH_SIZE,
    ensure_detection_model_ready,
    get_detection_model_path,
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

    @patch("frigate_buffer.services.video._run_detection_on_batch")
    @patch("frigate_buffer.services.video._get_video_metadata")
    def test_generate_detection_sidecar_opens_clip_and_writes_json_schema(
        self, mock_get_metadata, mock_run_batch
    ):
        """generate_detection_sidecar uses NeLux VideoReader, batches frames, writes detection.json with expected schema."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_get_metadata.return_value = (640, 480, 30.0, 10.0)
        mock_reader = MagicMock()
        mock_reader.fps = 30.0
        mock_reader.__len__ = lambda self: 90

        def get_batch(indices):
            return torch.zeros((len(indices), 3, 480, 640), dtype=torch.uint8)

        mock_reader.get_batch.side_effect = get_batch

        def run_batch_side_effect(model, batch, device, imgsz=640):
            return [[] for _ in range(batch.shape[0])]

        mock_run_batch.side_effect = run_batch_side_effect

        config = {
            "DETECTION_MODEL": "yolov8n.pt",
            "DETECTION_DEVICE": "",
            "DETECTION_FRAME_INTERVAL": 2,
            "STORAGE_PATH": "/tmp",
        }
        tmp = tempfile.mkdtemp(prefix="tmp_video_sidecar_")
        clip_path = os.path.join(tmp, "clip.mp4")
        sidecar_path = os.path.join(tmp, "detection.json")
        try:
            with open(clip_path, "wb"):
                pass
            with patch.object(sys.modules["nelux"], "VideoReader", return_value=mock_reader):
                with patch("frigate_buffer.services.video.get_detection_model_path", return_value=os.path.join("/tmp", "yolo_models", "yolov8n.pt")):
                    with patch("ultralytics.YOLO") as mock_yolo:
                        mock_yolo.return_value = MagicMock()
                        result = self.video_service.generate_detection_sidecar(
                            clip_path, sidecar_path, config
                        )
            self.assertTrue(result, "generate_detection_sidecar should return True")
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

    @patch("frigate_buffer.services.video._get_video_metadata")
    def test_generate_detection_sidecar_returns_false_when_clip_wont_open(
        self, mock_get_metadata
    ):
        mock_get_metadata.return_value = (640, 480, 30.0, 10.0)
        with patch.object(sys.modules["nelux"], "VideoReader", side_effect=RuntimeError("NeLux open failed")):

            tmp = tempfile.mkdtemp(prefix="tmp_video_sidecar_noclip_")
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
        storage = "/tmp/test_storage"
        model_path = os.path.join(storage, "yolo_models", "yolov8n.pt")
        mock_model.ckpt_path = model_path
        mock_yolo_cls.return_value = mock_model
        config = {"DETECTION_MODEL": "yolov8n.pt", "STORAGE_PATH": storage}
        with patch(
            "frigate_buffer.services.video.os.path.isfile",
            side_effect=lambda p: p == model_path,
        ):
            result = ensure_detection_model_ready(config)
        self.assertTrue(result)
        mock_yolo_cls.assert_called_once_with(model_path)

    def test_get_detection_model_path_under_storage(self):
        config = {"STORAGE_PATH": "/app/storage", "DETECTION_MODEL": "yolo26m.pt"}
        path = get_detection_model_path(config)
        self.assertEqual(path, os.path.join("/app/storage", "yolo_models", "yolo26m.pt"))

    def test_get_detection_model_path_default_model_when_empty(self):
        config = {"STORAGE_PATH": "/data"}
        path = get_detection_model_path(config)
        self.assertEqual(path, os.path.join("/data", "yolo_models", "yolov8n.pt"))


if __name__ == "__main__":
    unittest.main()
