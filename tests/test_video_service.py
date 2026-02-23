import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from frigate_buffer.services.video import (
    VideoService,
    BATCH_SIZE,
    _decoder_frame_count,
    _decoder_reader_ready,
    _run_detection_on_batch,
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
        """generate_detection_sidecar uses gpu_decoder create_decoder, batches frames, writes detection.json with expected schema."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_get_metadata.return_value = (640, 480, 30.0, 10.0)
        mock_ctx = MagicMock()
        mock_ctx.frame_count = 90
        mock_ctx.__len__ = lambda self: 90

        def get_frames(indices):
            return torch.zeros((len(indices), 3, 480, 640), dtype=torch.uint8)

        mock_ctx.get_frames.side_effect = get_frames

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

            @contextmanager
            def fake_create_decoder(path, gpu_id=0):
                yield mock_ctx

            with patch("frigate_buffer.services.video.create_decoder", side_effect=fake_create_decoder):
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

    def test_decoder_reader_ready(self):
        """_decoder_reader_ready returns True for DecoderContext (frame_count) or reader with _decoder."""
        reader_no_decoder = MagicMock(spec=["fps", "get_frames"])
        reader_no_decoder.fps = 30.0
        self.assertFalse(_decoder_reader_ready(reader_no_decoder))
        reader_with_decoder = MagicMock()
        reader_with_decoder._decoder = MagicMock()
        self.assertTrue(_decoder_reader_ready(reader_with_decoder))
        ctx_with_frame_count = MagicMock(spec=["frame_count"])
        ctx_with_frame_count.frame_count = 100
        self.assertTrue(_decoder_reader_ready(ctx_with_frame_count))

    @patch("frigate_buffer.services.video._run_detection_on_batch")
    @patch("frigate_buffer.services.video._get_video_metadata")
    def test_generate_detection_sidecar_calls_get_frames(self, mock_get_metadata, mock_run_batch):
        """generate_detection_sidecar uses create_decoder and calls get_frames on context."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_get_metadata.return_value = (640, 480, 30.0, 10.0)
        mock_run_batch.return_value = [[]]
        mock_ctx = MagicMock()
        mock_ctx.frame_count = 20
        mock_ctx.__len__ = lambda self: 20
        mock_ctx.get_frames.side_effect = lambda indices: torch.zeros((len(indices), 3, 480, 640), dtype=torch.uint8)
        tmp = tempfile.mkdtemp(prefix="tmp_video_sidecar_")
        clip_path = os.path.join(tmp, "clip.mp4")
        sidecar_path = os.path.join(tmp, "detection.json")
        try:
            with open(clip_path, "wb"):
                pass

            @contextmanager
            def fake_create_decoder(path, gpu_id=0):
                yield mock_ctx

            with patch("frigate_buffer.services.video.create_decoder", side_effect=fake_create_decoder):
                result = self.video_service.generate_detection_sidecar(clip_path, sidecar_path, {"DETECTION_FRAME_INTERVAL": 5})
            self.assertTrue(result)
            self.assertTrue(mock_ctx.get_frames.called)
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

        @contextmanager
        def fake_create_decoder_raises(path, gpu_id=0):
            raise RuntimeError("decoder open failed")

        with patch("frigate_buffer.services.video.create_decoder", side_effect=fake_create_decoder_raises):
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

    @patch("frigate_buffer.services.video._run_detection_on_batch")
    @patch("frigate_buffer.services.video._get_video_metadata")
    def test_generate_detection_sidecar_uses_metadata_fallback_when_len_reader_raises(
        self, mock_get_metadata, mock_run_batch
    ):
        """When decoder reports 0 frames, frame count comes from ffprobe metadata (duration*fps) and sidecar still succeeds."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_get_metadata.return_value = (640, 480, 30.0, 10.0)
        mock_ctx = MagicMock()
        mock_ctx.frame_count = 0
        mock_ctx.__len__ = lambda self: 0

        def get_frames(indices):
            return torch.zeros((len(indices), 3, 480, 640), dtype=torch.uint8)

        mock_ctx.get_frames.side_effect = get_frames
        mock_run_batch.side_effect = lambda model, batch, device, imgsz=640: [[] for _ in range(batch.shape[0])]

        config = {
            "DETECTION_MODEL": "yolov8n.pt",
            "DETECTION_DEVICE": "",
            "DETECTION_FRAME_INTERVAL": 5,
            "STORAGE_PATH": "/tmp",
        }
        tmp = tempfile.mkdtemp(prefix="tmp_video_sidecar_fallback_")
        clip_path = os.path.join(tmp, "clip.mp4")
        sidecar_path = os.path.join(tmp, "detection.json")
        try:
            with open(clip_path, "wb"):
                pass

            @contextmanager
            def fake_create_decoder(path, gpu_id=0):
                yield mock_ctx

            with patch("frigate_buffer.services.video.create_decoder", side_effect=fake_create_decoder):
                with patch("frigate_buffer.services.video.get_detection_model_path", return_value=os.path.join("/tmp", "yolo_models", "yolov8n.pt")):
                    with patch("ultralytics.YOLO") as mock_yolo:
                        mock_yolo.return_value = MagicMock()
                        result = self.video_service.generate_detection_sidecar(
                            clip_path, sidecar_path, config
                        )
            self.assertTrue(result, "generate_detection_sidecar should succeed using metadata fallback")
            self.assertTrue(os.path.isfile(sidecar_path))
            with open(sidecar_path, encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("entries", data)
            entries = data["entries"]
            self.assertGreater(len(entries), 0)
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


class TestDecoderFrameCount(unittest.TestCase):
    """Tests for _decoder_frame_count."""

    def test_decoder_frame_count_uses_frame_count_when_available(self):
        """When reader has frame_count (e.g. DecoderContext), _decoder_frame_count returns it."""
        reader = MagicMock(spec=["frame_count"])
        reader.frame_count = 42
        result = _decoder_frame_count(reader, 30.0, 10.0)
        self.assertEqual(result, 42)

    def test_decoder_frame_count_uses_len_when_available(self):
        """When len(reader) works and no frame_count, _decoder_frame_count returns it."""
        class ReaderWithLen:
            def __len__(self):
                return 42
        result = _decoder_frame_count(ReaderWithLen(), 30.0, 10.0)
        self.assertEqual(result, 42)

    def test_decoder_frame_count_fallback_when_len_raises(self):
        """When len(reader) raises (e.g. missing _decoder), _decoder_frame_count returns duration * fps."""
        class ReaderNoLen:
            pass
        result = _decoder_frame_count(ReaderNoLen(), 30.0, 10.0)
        self.assertEqual(result, 300)

    def test_decoder_frame_count_uses_shape_when_len_raises(self):
        """When len(reader) raises but reader.shape is (N, ...), _decoder_frame_count returns N."""
        class ReaderShapeOnly:
            shape = (100, 3, 480, 640)
        result = _decoder_frame_count(ReaderShapeOnly(), 30.0, 10.0)
        self.assertEqual(result, 100)

    def test_decoder_frame_count_fallback_zero_duration_returns_zero(self):
        """When both len and shape fail and duration is 0, _decoder_frame_count returns 0."""
        class ReaderNoLen:
            shape = ()
        result = _decoder_frame_count(ReaderNoLen(), 30.0, 0.0)
        self.assertEqual(result, 0)


class TestRunDetectionOnBatch(unittest.TestCase):
    """Tests for _run_detection_on_batch resize and bbox scale-back."""

    def test_run_detection_on_batch_scales_bboxes_back_to_read_space(self):
        """YOLO returns xyxy in resized space; detections are scaled back to decoder (read) space."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch/numpy not available")
        # Batch: 1 frame, 3 channels, read_h=1920, read_w=2560 -> target 1280x960, scale_x=2, scale_y=2
        batch = torch.zeros((1, 3, 1920, 2560), dtype=torch.float32) * 0.5
        mock_result = MagicMock()
        # One box in resized space (1280x960): x1=100, y1=200, x2=300, y2=400
        mock_result.boxes.xyxy = torch.tensor([[100.0, 200.0, 300.0, 400.0]])
        mock_model = MagicMock(return_value=[mock_result])
        out = _run_detection_on_batch(mock_model, batch, None, imgsz=1280)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), 1)
        det = out[0][0]
        self.assertEqual(det["label"], "person")
        # Scaled to read space: x*2, y*2 -> [200, 400, 600, 800]
        self.assertEqual(det["bbox"], [200.0, 400.0, 600.0, 800.0])
        self.assertEqual(det["centerpoint"], [400.0, 600.0])
        self.assertEqual(det["area"], 160000)

    def test_run_detection_on_batch_small_batch_no_crash(self):
        """Batch already 640x480 with imgsz=640 returns detections in read space without error."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        batch = torch.zeros((2, 3, 480, 640), dtype=torch.float32)
        mock_model = MagicMock(return_value=[MagicMock(boxes=MagicMock(xyxy=None)), MagicMock(boxes=MagicMock(xyxy=None))])
        out = _run_detection_on_batch(mock_model, batch, None, imgsz=640)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], [])
        self.assertEqual(out[1], [])


if __name__ == "__main__":
    unittest.main()
