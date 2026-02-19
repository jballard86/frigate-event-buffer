import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frigate_buffer.services.video import (
    VideoService,
    decode_nvenc_returncode,
    ensure_detection_model_ready,
    _nvenc_probe_cmd,
    run_nvenc_preflight_probe,
)


class TestDecodeNvencReturncode(unittest.TestCase):
    """Tests for returncode-to-signal/errno mapping utility."""

    def test_returncode_0_returns_empty(self):
        self.assertEqual(decode_nvenc_returncode(0), [])

    def test_returncode_234_includes_interpretations(self):
        interp = decode_nvenc_returncode(234)
        self.assertGreater(len(interp), 0)
        # 256-22=234 (SIGPIPE)
        self.assertTrue(
            any("256" in s and "22" in s for s in interp),
            "expected 256-22 or similar in %s" % interp,
        )
        self.assertTrue(
            any("234" in s and "AVERROR" in s for s in interp),
            "expected 234/AVERROR hint in %s" % interp,
        )

    def test_returncode_132_includes_sigill(self):
        interp = decode_nvenc_returncode(132)
        self.assertTrue(
            any("SIGILL" in s or "128+4" in s for s in interp),
            "expected SIGILL/128+4 in %s" % interp,
        )

    def test_returncode_139_includes_sigsegv(self):
        interp = decode_nvenc_returncode(139)
        self.assertTrue(
            any("SIGSEGV" in s or "128+11" in s for s in interp),
            "expected SIGSEGV/128+11 in %s" % interp,
        )


class TestNvencProbeCmd(unittest.TestCase):
    """Tests for NVENC probe command building (configurable resolution)."""

    @patch("frigate_buffer.services.video._nvenc_probe_width", 1280)
    @patch("frigate_buffer.services.video._nvenc_probe_height", 720)
    def test_nvenc_probe_cmd_uses_default_1280x720(self, *_):
        """Probe command uses width x height from module state (default 1280x720)."""
        cmd = _nvenc_probe_cmd()
        cmd_str = " ".join(cmd)
        self.assertIn("1280x720", cmd_str, "probe command should use 1280x720 by default")

    @patch("frigate_buffer.services.video._nvenc_preflight_result", None)
    @patch("frigate_buffer.services.video.subprocess.run")
    def test_run_nvenc_preflight_probe_sets_probe_dimensions_from_config(self, mock_run):
        """run_nvenc_preflight_probe(config) sets probe size so _nvenc_probe_cmd uses it."""
        mock_run.return_value = MagicMock(returncode=0, stderr=b"")
        config = {"NVENC_PROBE_WIDTH": 1920, "NVENC_PROBE_HEIGHT": 1080}
        run_nvenc_preflight_probe(config)
        cmd = _nvenc_probe_cmd()
        cmd_str = " ".join(cmd)
        self.assertIn("1920x1080", cmd_str, "probe command should use config dimensions")


class TestProbeNvenc(unittest.TestCase):
    """Tests for NVENC probe behavior and logging."""

    def setUp(self):
        self.video_service = VideoService(ffmpeg_timeout=1)
        # Reset probe cache so each test gets a fresh probe
        self.video_service._nvenc_available = None

    @patch("frigate_buffer.services.video.time.sleep")
    @patch("frigate_buffer.services.video.subprocess.run")
    @patch("frigate_buffer.services.video.logger")
    def test_probe_nvenc_logs_warning_with_stderr_when_no_capable_devices(
        self, mock_logger, mock_run, mock_sleep
    ):
        """When probe fails with 'No capable devices found', WARNING is logged with stderr snippet."""
        proc = MagicMock()
        proc.returncode = 1
        proc.stderr = b"[h264_nvenc] No capable devices found\nError opening encoder\n"
        mock_run.return_value = proc

        result = self.video_service._probe_nvenc()

        self.assertFalse(result)
        mock_logger.warning.assert_called()
        call_args = mock_logger.warning.call_args[0]
        self.assertIn("No capable devices found", str(call_args))


class TestVideoService(unittest.TestCase):

    def setUp(self):
        self.video_service = VideoService(ffmpeg_timeout=1)

    @patch('frigate_buffer.services.video.VideoService._probe_nvenc')
    @patch('frigate_buffer.services.video.subprocess.Popen')
    @patch('frigate_buffer.services.video.os.remove')
    @patch('frigate_buffer.services.video.os.path.exists')
    @patch('frigate_buffer.services.video.ffmpegcv.VideoCaptureNV')
    def test_transcode_success(self, mock_video_capture_nv, mock_exists, mock_remove, mock_popen, mock_probe):
        # Probe says NVENC unavailable, so libx264 is used
        mock_probe.return_value = False
        mock_video_capture_nv.side_effect = RuntimeError("No GPU")
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        mock_exists.return_value = False

        ok, _ = self.video_service.transcode_clip_to_h264("evt1", "temp.mp4", "final.mp4")

        self.assertTrue(ok)
        mock_remove.assert_called_with("temp.mp4")
        mock_popen.assert_called_once()
        mock_probe.assert_called_once()

    @patch('frigate_buffer.services.video._nvenc_preflight_result', True)
    @patch('frigate_buffer.services.video.subprocess.run')
    def test_probe_nvenc_uses_preflight_cache_and_skips_subprocess(self, mock_run):
        """When preflight cache is True, _probe_nvenc returns True without calling subprocess."""
        self.video_service._nvenc_available = None
        result = self.video_service._probe_nvenc()
        self.assertTrue(result)
        mock_run.assert_not_called()

    @patch('frigate_buffer.services.video.VideoService._transcode_clip_nvenc')
    @patch('frigate_buffer.services.video.VideoService._transcode_clip_libx264')
    @patch('frigate_buffer.services.video.VideoService._probe_nvenc')
    def test_transcode_skips_nvenc_when_probe_unavailable(self, mock_probe, mock_libx264, mock_nvenc):
        """When NVENC probe returns False, transcode uses libx264 only and never calls _transcode_clip_nvenc."""
        mock_probe.return_value = False
        mock_libx264.return_value = (True, "CPU: NVENC unavailable")

        ok, backend = self.video_service.transcode_clip_to_h264("evt1", "temp.mp4", "final.mp4")

        self.assertTrue(ok)
        self.assertEqual(backend, "CPU: NVENC unavailable")
        mock_probe.assert_called_once()
        mock_libx264.assert_called_once_with(
            "evt1", "temp.mp4", "final.mp4",
            detection_sidecar_path=None, detection_model=None, detection_device=None,
            cpu_reason="NVENC unavailable",
        )
        mock_nvenc.assert_not_called()

    @patch('frigate_buffer.services.video.VideoService._transcode_clip_nvenc')
    @patch('frigate_buffer.services.video.VideoService._transcode_clip_libx264')
    @patch('frigate_buffer.services.video.VideoService._probe_nvenc')
    def test_transcode_fallback_passes_sidecar_args_to_libx264(self, mock_probe, mock_libx264, mock_nvenc):
        """When NVENC probe returns False, transcode passes detection_sidecar_path/model/device to libx264."""
        mock_probe.return_value = False
        mock_libx264.return_value = (True, "CPU: NVENC unavailable")

        ok, _ = self.video_service.transcode_clip_to_h264(
            "evt1", "temp.mp4", "final.mp4",
            detection_sidecar_path="/ce/cam/detection.json",
            detection_model="yolov8n.pt",
            detection_device="0",
        )

        self.assertTrue(ok)
        mock_libx264.assert_called_once_with(
            "evt1", "temp.mp4", "final.mp4",
            detection_sidecar_path="/ce/cam/detection.json",
            detection_model="yolov8n.pt",
            detection_device="0",
            cpu_reason="NVENC unavailable",
        )

    @patch('frigate_buffer.services.video.VideoService._probe_nvenc')
    @patch('frigate_buffer.services.video.subprocess.Popen')
    @patch('frigate_buffer.services.video.os.rename')
    @patch('frigate_buffer.services.video.os.path.exists')
    @patch('frigate_buffer.services.video.ffmpegcv.VideoCaptureNV')
    def test_transcode_failure(self, mock_video_capture_nv, mock_exists, mock_rename, mock_popen, mock_probe):
        mock_probe.return_value = False
        mock_video_capture_nv.side_effect = RuntimeError("No GPU")
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"", b"Error")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process

        mock_exists.return_value = True

        ok, _ = self.video_service.transcode_clip_to_h264("evt1", "temp.mp4", "final.mp4")

        self.assertTrue(ok)
        mock_rename.assert_called_with("temp.mp4", "final.mp4")

    @patch('frigate_buffer.services.video.VideoService._probe_nvenc')
    @patch('frigate_buffer.services.video.subprocess.run')
    @patch('frigate_buffer.services.video.os.remove')
    @patch('frigate_buffer.services.video.os.path.exists')
    @patch('frigate_buffer.services.video.ffmpegcv.VideoWriterNV')
    @patch('frigate_buffer.services.video.ffmpegcv.VideoCaptureNV')
    def test_transcode_nvenc_success(self, mock_capture_nv, mock_writer_nv, mock_exists, mock_remove, mock_run, mock_probe):
        """When GPU is available, transcode uses VideoCaptureNV and VideoWriterNV then muxes audio."""
        mock_probe.return_value = True
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

        ok, backend = self.video_service.transcode_clip_to_h264("evt1", "/tmp/temp.mp4", "/tmp/final.mp4")

        self.assertTrue(ok)
        self.assertEqual(backend, "GPU")
        mock_capture_nv.assert_called_once()
        mock_writer_nv.assert_called_once()
        mock_writer.write.assert_called()
        mock_run.assert_called_once()
        mock_remove.assert_any_call("/tmp/temp.mp4")

    @patch('frigate_buffer.services.video.os.remove')
    @patch('frigate_buffer.services.video.VideoService._write_detection_sidecar_cpu')
    @patch('frigate_buffer.services.video.subprocess.Popen')
    def test_transcode_libx264_with_sidecar_writes_sidecar_then_runs_ffmpeg(
        self, mock_popen, mock_write_sidecar, mock_remove
    ):
        """When _transcode_clip_libx264 is called with detection_sidecar_path and model, writes sidecar then runs ffmpeg."""
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        ok, backend = self.video_service._transcode_clip_libx264(
            "evt1", "/tmp/temp.mp4", "/tmp/final.mp4",
            detection_sidecar_path="/ce/cam/detection.json",
            detection_model="yolov8n.pt",
            detection_device=None,
        )

        self.assertTrue(ok)
        self.assertIn("CPU", backend)
        mock_write_sidecar.assert_called_once_with(
            "/tmp/temp.mp4",
            "/ce/cam/detection.json",
            "yolov8n.pt",
            None,
        )
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        self.assertIn("ffmpeg", call_args)
        self.assertIn("libx264", call_args)

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
