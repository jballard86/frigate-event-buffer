"""generate_compilation_video orchestration tests."""

import json
import unittest
from unittest.mock import MagicMock, mock_open, patch

import pytest

from frigate_buffer.services.video_compilation import (
    _run_pynv_compilation,
    generate_compilation_video,
)
from helpers.video_compilation import (
    fake_create_decoder_context,
    gpu_backend_for_compilation_tests,
)


class TestGenerateCompilationVideo(unittest.TestCase):
    """Tests for generate_compilation_video function (PyNvVideoCodec GPU pipeline)."""

    def _sidecar_mock(self):
        return {
            "native_width": 2560,
            "native_height": 1920,
            "entries": [],
        }

    @patch("frigate_buffer.services.video_compilation._run_pynv_compilation")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    @patch("frigate_buffer.services.video_compilation.os.rename")
    @patch("frigate_buffer.services.video_compilation.os.path.getsize")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    @patch("builtins.open", new_callable=mock_open)
    def test_pynv_pipeline_writes_to_tmp_then_renames(
        self,
        mock_file,
        mock_resolve,
        mock_getsize,
        mock_rename,
        mock_isfile,
        mock_run_pynv,
    ):
        """PyNv pipeline uses tmp path; on success temp file renamed to final output."""
        mock_resolve.return_value = "doorbell-123.mp4"
        mock_isfile.return_value = True
        mock_getsize.return_value = 1000
        mock_file.return_value.read.return_value = json.dumps(self._sidecar_mock())

        segments = [{"camera": "doorbell", "start_sec": 0.0, "end_sec": 10.0}]
        ce_dir = "/app/storage/events/test36"
        output_path = "/app/storage/events/test36/test36_summary.mp4"

        generate_compilation_video(segments, ce_dir, output_path)

        mock_run_pynv.assert_called_once()
        call_kw = mock_run_pynv.call_args[1]
        temp_path = output_path.replace(".mp4", "_temp.mp4")
        assert call_kw["tmp_output_path"] == temp_path
        assert call_kw["target_w"] == 1440
        assert call_kw["target_h"] == 1080
        mock_rename.assert_called_once_with(temp_path, output_path)

    @patch("frigate_buffer.services.video_compilation._run_pynv_compilation")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    @patch("frigate_buffer.services.video_compilation.os.rename")
    @patch("frigate_buffer.services.video_compilation.os.path.getsize")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    @patch("builtins.open", new_callable=mock_open)
    def test_pynv_pipeline_receives_slices_and_target_resolution(
        self,
        mock_file,
        mock_resolve,
        mock_getsize,
        mock_rename,
        mock_isfile,
        mock_run_pynv,
    ):
        """_run_pynv_compilation called with correct slices and target dimensions."""
        mock_resolve.return_value = "doorbell-123.mp4"
        mock_isfile.return_value = True
        mock_getsize.return_value = 1000
        mock_file.return_value.read.return_value = json.dumps(self._sidecar_mock())

        segments = [{"camera": "doorbell", "start_sec": 0.0, "end_sec": 10.0}]
        ce_dir = "/app/storage/events/test36"
        output_path = "/app/storage/events/test36/test36_summary.mp4"

        generate_compilation_video(
            segments, ce_dir, output_path, target_w=1280, target_h=720
        )

        call_kw = mock_run_pynv.call_args[1]
        assert call_kw["target_w"] == 1280
        assert call_kw["target_h"] == 720
        assert len(call_kw["slices"]) == 1
        assert "crop_start" in call_kw["slices"][0]
        assert "crop_end" in call_kw["slices"][0]

    @patch("frigate_buffer.services.video_compilation._run_pynv_compilation")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    @patch("frigate_buffer.services.video_compilation.os.rename")
    @patch("frigate_buffer.services.video_compilation.os.path.getsize")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    @patch("builtins.open", new_callable=mock_open)
    def test_successful_completion_renames_temp_file(
        self,
        mock_file,
        mock_resolve,
        mock_getsize,
        mock_rename,
        mock_isfile,
        mock_run_pynv,
    ):
        """On success, temp file is renamed to final output path."""
        mock_resolve.return_value = "doorbell-123.mp4"
        mock_isfile.return_value = True
        mock_getsize.return_value = 1000
        mock_file.return_value.read.return_value = json.dumps(self._sidecar_mock())

        segments = [{"camera": "doorbell", "start_sec": 0.0, "end_sec": 10.0}]
        ce_dir = "/app/storage/events/test36"
        output_path = "/app/storage/events/test36/test36_summary.mp4"

        generate_compilation_video(segments, ce_dir, output_path)

        temp_path = output_path.replace(".mp4", "_temp.mp4")
        mock_rename.assert_called_once_with(temp_path, output_path)

    @patch("frigate_buffer.services.video_compilation._run_pynv_compilation")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    @patch("builtins.open", new_callable=mock_open)
    def test_compilation_failure_raises(
        self, mock_file, mock_resolve, mock_isfile, mock_run_pynv
    ):
        """When PyNv pipeline raises, generate_compilation_video re-raises."""
        mock_resolve.return_value = "doorbell-123.mp4"
        mock_isfile.return_value = True
        mock_file.return_value.read.return_value = json.dumps(self._sidecar_mock())

        segments = [{"camera": "doorbell", "start_sec": 0.0, "end_sec": 10.0}]
        ce_dir = "/app/storage/events/test36"
        output_path = "/app/storage/events/test36/test36_summary.mp4"

        mock_run_pynv.side_effect = RuntimeError("PyNv encode failed")

        with pytest.raises(RuntimeError):
            generate_compilation_video(segments, ce_dir, output_path)

    @patch("frigate_buffer.services.video_compilation._run_pynv_compilation")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    @patch("frigate_buffer.services.video_compilation.os.rename")
    @patch("frigate_buffer.services.video_compilation.os.path.getsize")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    @patch("builtins.open", new_callable=mock_open)
    def test_multiple_slices_passed_to_pynv(
        self,
        mock_file,
        mock_resolve,
        mock_getsize,
        mock_rename,
        mock_isfile,
        mock_run_pynv,
    ):
        """Multiple slices passed to _run_pynv_compilation (one segment per slice)."""
        mock_resolve.return_value = "doorbell.mp4"
        mock_isfile.return_value = True
        mock_getsize.return_value = 1000
        sidecar_data = {
            "native_width": 1920,
            "native_height": 1080,
            "entries": [],
        }
        mock_file.return_value.read.return_value = json.dumps(sidecar_data)
        slices = [
            {"camera": "doorbell", "start_sec": 0.0, "end_sec": 5.0},
            {"camera": "doorbell", "start_sec": 5.0, "end_sec": 10.0},
        ]
        ce_dir = "/app/storage/events/test"
        output_path = "/app/storage/events/test/test_summary.mp4"

        generate_compilation_video(slices, ce_dir, output_path)

        call_kw = mock_run_pynv.call_args[1]
        assert len(call_kw["slices"]) == 2
        assert call_kw["slices"][0]["camera"] == "doorbell"
        assert call_kw["slices"][0]["start_sec"] == 0.0
        assert call_kw["slices"][0]["end_sec"] == 5.0
        assert call_kw["slices"][1]["start_sec"] == 5.0
        assert call_kw["slices"][1]["end_sec"] == 10.0
        assert "native_width" in call_kw["slices"][0]
        assert "native_height" in call_kw["slices"][0]
        assert call_kw["slices"][0]["native_width"] == 1920
        assert call_kw["slices"][0]["native_height"] == 1080

    @patch("frigate_buffer.services.video_compilation._run_pynv_compilation")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    @patch("frigate_buffer.services.video_compilation.os.rename")
    @patch("frigate_buffer.services.video_compilation.os.path.getsize")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    @patch("builtins.open", new_callable=mock_open)
    def test_last_slice_of_camera_run_holds_crop(
        self,
        mock_file,
        mock_resolve,
        mock_getsize,
        mock_rename,
        mock_isfile,
        mock_run_pynv,
    ):
        """Last slice before camera switch: crop_end == crop_start (smooth hold)."""
        mock_resolve.return_value = "cam.mp4"
        mock_isfile.return_value = True
        mock_getsize.return_value = 1000
        sidecar_data = {
            "native_width": 1920,
            "native_height": 1080,
            "entries": [],
        }
        mock_file.return_value.read.return_value = json.dumps(sidecar_data)
        slices = [
            {"camera": "doorbell", "start_sec": 0.0, "end_sec": 5.0},
            {"camera": "carport", "start_sec": 5.0, "end_sec": 10.0},
        ]
        ce_dir = "/app/storage/events/test"
        output_path = "/app/storage/events/test/test_summary.mp4"
        generate_compilation_video(slices, ce_dir, output_path)
        assert slices[0]["crop_start"] == slices[0]["crop_end"]

    @patch("frigate_buffer.services.video_compilation.logger")
    @patch("frigate_buffer.services.video_compilation._run_pynv_compilation")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    @patch("frigate_buffer.services.video_compilation.os.rename")
    @patch("frigate_buffer.services.video_compilation.os.path.getsize")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    def test_fallback_log_once_per_camera_no_detections(
        self,
        mock_resolve,
        mock_getsize,
        mock_rename,
        mock_isfile,
        mock_run_pynv,
        mock_logger,
    ):
        """No detections at start/end → one ERROR per camera."""
        mock_resolve.return_value = "doorbell.mp4"
        mock_isfile.return_value = True
        mock_getsize.return_value = 1000
        # Sidecar: empty detections at boundaries (0, 0.75, 1.5 have no dets).
        sidecars = {
            "doorbell": {
                "native_width": 1920,
                "native_height": 1080,
                "entries": [
                    {"timestamp_sec": 0.0, "detections": []},
                    {"timestamp_sec": 0.75, "detections": []},
                    {"timestamp_sec": 1.5, "detections": []},
                    {"timestamp_sec": 2.25, "detections": []},
                ],
            },
        }
        slices = [
            {"camera": "doorbell", "start_sec": 0.0, "end_sec": 0.75},
            {"camera": "doorbell", "start_sec": 0.75, "end_sec": 1.5},
            {"camera": "doorbell", "start_sec": 1.5, "end_sec": 2.25},
        ]
        ce_dir = "/app/storage/events/test"
        output_path = "/app/storage/events/test/test_summary.mp4"

        generate_compilation_video(slices, ce_dir, output_path, sidecars=sidecars)

        no_detection_calls = [
            c
            for c in mock_logger.error.call_args_list
            if c[0] and "no detections at slice start/end" in str(c[0][0])
        ]
        assert len(no_detection_calls) == 1, (
            "Expected exactly one ERROR per camera for no detections"
        )
        msg = no_detection_calls[0][0]
        assert "doorbell" in str(msg)
        assert "in %s slices" in msg[0]
        assert msg[2] == 3, "Expected slice count 3"
        assert "fallback crop (center or nearby detection within 5s)" in msg[0]

    @patch("frigate_buffer.services.video_compilation.logger")
    @patch("frigate_buffer.services.video_compilation._run_pynv_compilation")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    @patch("frigate_buffer.services.video_compilation.os.rename")
    @patch("frigate_buffer.services.video_compilation.os.path.getsize")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    def test_fallback_log_once_per_camera_no_entries(
        self,
        mock_resolve,
        mock_getsize,
        mock_rename,
        mock_isfile,
        mock_run_pynv,
        mock_logger,
    ):
        """No sidecar entries for camera → one ERROR logged per camera."""
        mock_resolve.return_value = "doorbell.mp4"
        mock_isfile.return_value = True
        mock_getsize.return_value = 1000
        sidecars = {
            "doorbell": {
                "native_width": 1920,
                "native_height": 1080,
                "entries": [],
            },
        }
        slices = [
            {"camera": "doorbell", "start_sec": 0.0, "end_sec": 2.0},
            {"camera": "doorbell", "start_sec": 2.0, "end_sec": 4.0},
            {"camera": "doorbell", "start_sec": 4.0, "end_sec": 6.0},
        ]
        ce_dir = "/app/storage/events/test"
        output_path = "/app/storage/events/test/test_summary.mp4"

        generate_compilation_video(slices, ce_dir, output_path, sidecars=sidecars)

        no_entries_calls = [
            c
            for c in mock_logger.error.call_args_list
            if c[0] and "no sidecar entries" in str(c[0][0])
        ]
        assert len(no_entries_calls) == 1, (
            "Expected exactly one ERROR per camera for no sidecar entries"
        )
        msg = no_entries_calls[0][0]
        assert "doorbell" in str(msg)
        assert "in %s slices" in msg[0]
        assert msg[2] == 3, "Expected slice count 3"
        assert "using fallback crop" in msg[0]

    @patch("frigate_buffer.services.video_compilation.GPU_LOCK", MagicMock())
    @patch("frigate_buffer.services.video_compilation.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    @patch("frigate_buffer.services.video_compilation._get_video_metadata")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    def test_run_pynv_compilation_uses_metadata_fallback_when_len_zero(
        self,
        mock_isfile,
        mock_get_metadata,
        mock_open_fn,
        mock_popen,
    ):
        """Decoder len 0 → _get_video_metadata; _run_pynv completes; FFmpeg encode."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_isfile.return_value = True
        mock_get_metadata.return_value = (640, 480, 20.0, 10.0)
        mock_ctx = fake_create_decoder_context(frame_count=0, height=480, width=640)
        if mock_ctx is None:
            self.skipTest("torch not available")
        mock_ctx.get_index_from_time_in_seconds = lambda t: min(int(t * 20), 39)
        mock_ctx.get_frames = lambda indices: torch.zeros(
            (len(indices), 3, 480, 640), dtype=torch.uint8
        )
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_ctx)
        mock_cm.__exit__ = MagicMock(return_value=None)
        gpu_backend = gpu_backend_for_compilation_tests(return_value=mock_cm)
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.returncode = 0
        mock_popen.return_value = proc

        slices = [
            {
                "camera": "doorbell",
                "start_sec": 0.0,
                "end_sec": 2.0,
                "crop_start": (0, 0, 320, 240),
                "crop_end": (0, 0, 320, 240),
            },
        ]
        resolve_clip = MagicMock(return_value="doorbell-1.mp4")

        _run_pynv_compilation(
            slices=slices,
            ce_dir="/ce",
            tmp_output_path="/out.mp4.tmp",
            target_w=1440,
            target_h=1080,
            resolve_clip_in_folder=resolve_clip,
            gpu_backend=gpu_backend,
            gpu_device_index=0,
        )

        mock_get_metadata.assert_called()
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert "h264_nvenc" in call_args
        assert call_args[call_args.index("-s") + 1] == "1440x1080"
        assert call_args[-1] == "/out.mp4.tmp"
        # 2 s at 20 fps = 40 frames
        assert proc.stdin.write.call_count == 40
        proc.stdin.flush.assert_called()
        proc.stdin.close.assert_called_once()
        proc.wait.assert_called()

    @patch("frigate_buffer.services.video_compilation.GPU_LOCK", MagicMock())
    @patch("frigate_buffer.services.video_compilation.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    @patch("frigate_buffer.services.video_compilation._get_video_metadata")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    def test_run_pynv_compilation_clamps_time_strictly_less_than_duration(
        self,
        mock_isfile,
        mock_get_metadata,
        mock_open_fn,
        mock_popen,
    ):
        """When output_times includes a time at slice end (e.g. 2.0), decoder
        is called with t_safe <= max_valid_t so NVDEC never sees t_sec >= duration."""
        from frigate_buffer.constants import DECODER_TIME_EPSILON_SEC

        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_isfile.return_value = True
        frame_count = 40
        fallback_fps = 20.0
        mock_get_metadata.return_value = (640, 480, fallback_fps, 2.05)
        times_passed: list[float] = []

        def record_and_return(t_sec):
            times_passed.append(t_sec)
            return min(max(0, int(t_sec * fallback_fps)), frame_count - 1)

        mock_ctx = MagicMock()
        mock_ctx.__len__ = lambda self: frame_count
        mock_ctx.get_index_from_time_in_seconds = record_and_return
        mock_ctx.get_frames = lambda indices: torch.zeros(
            (len(indices), 3, 480, 640), dtype=torch.uint8
        )
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_ctx)
        mock_cm.__exit__ = MagicMock(return_value=None)
        gpu_backend = gpu_backend_for_compilation_tests(return_value=mock_cm)
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.returncode = 0
        mock_popen.return_value = proc

        slices = [
            {
                "camera": "doorbell",
                "start_sec": 0.0,
                "end_sec": 2.05,
                "crop_start": (0, 0, 320, 240),
                "crop_end": (0, 0, 320, 240),
            },
        ]
        resolve_clip = MagicMock(return_value="doorbell-1.mp4")

        _run_pynv_compilation(
            slices=slices,
            ce_dir="/ce",
            tmp_output_path="/out.mp4.tmp",
            target_w=1440,
            target_h=1080,
            resolve_clip_in_folder=resolve_clip,
            gpu_backend=gpu_backend,
            gpu_device_index=0,
        )

        max_valid_t = (frame_count - 1) / fallback_fps - DECODER_TIME_EPSILON_SEC
        assert all(t <= max_valid_t for t in times_passed), (
            f"Decoder must not be called with t > max_valid_t: "
            f"got max(times)={max(times_passed)!r} max_valid_t={max_valid_t}"
        )

    @patch("frigate_buffer.services.video_compilation.GPU_LOCK", MagicMock())
    @patch("frigate_buffer.services.video_compilation.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    @patch("frigate_buffer.services.video_compilation._get_video_metadata")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    def test_run_pynv_compilation_calls_ffmpeg_encode_with_h264_nvenc(
        self,
        mock_isfile,
        mock_get_metadata,
        mock_open_fn,
        mock_popen,
    ):
        """Compilation encode via FFmpeg (h264_nvenc; no CPU fallback)."""
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch not available")
        mock_isfile.return_value = True
        mock_get_metadata.return_value = (640, 480, 20.0, 2.0)
        mock_ctx = fake_create_decoder_context(200, 480, 640)
        if mock_ctx is None:
            self.skipTest("torch not available")
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_ctx)
        mock_cm.__exit__ = MagicMock(return_value=None)
        gpu_backend = gpu_backend_for_compilation_tests(return_value=mock_cm)
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.returncode = 0
        mock_popen.return_value = proc

        slices = [
            {
                "camera": "doorbell",
                "start_sec": 0.0,
                "end_sec": 2.0,
                "crop_start": (0, 0, 320, 240),
                "crop_end": (0, 0, 320, 240),
            },
        ]
        resolve_clip = MagicMock(return_value="doorbell-1.mp4")

        _run_pynv_compilation(
            slices=slices,
            ce_dir="/ce",
            tmp_output_path="/out.mp4.tmp",
            target_w=1440,
            target_h=1080,
            resolve_clip_in_folder=resolve_clip,
            gpu_backend=gpu_backend,
            gpu_device_index=0,
        )

        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert "h264_nvenc" in call_args
        assert "-thread_queue_size" in call_args
        assert "512" in call_args
        assert "p1" in call_args
        assert "hq" in call_args
        assert call_args[call_args.index("-s") + 1] == "1440x1080"
        assert call_args[-1] == "/out.mp4.tmp"
        # 2 s at 20 fps = 40 frames
        assert proc.stdin.write.call_count == 40
        proc.stdin.flush.assert_called()
        proc.stdin.close.assert_called_once()
        proc.wait.assert_called()

    @patch("frigate_buffer.services.video_compilation.GPU_LOCK", MagicMock())
    @patch("frigate_buffer.services.video_compilation.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    @patch("frigate_buffer.services.video_compilation._get_video_metadata")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    def test_run_pynv_compilation_skips_slice_on_decoder_failure(
        self,
        mock_isfile,
        mock_get_metadata,
        mock_open_fn,
        mock_popen,
    ):
        """create_decoder raises → slice skipped; FFmpeg opened, no frames written."""
        mock_isfile.return_value = True
        mock_get_metadata.return_value = (640, 480, 20.0, 2.0)
        gpu_backend = gpu_backend_for_compilation_tests(
            side_effect=RuntimeError("NVDEC init failed")
        )
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.returncode = 0
        mock_popen.return_value = proc

        slices = [
            {
                "camera": "doorbell",
                "start_sec": 0.0,
                "end_sec": 2.0,
                "crop_start": (0, 0, 320, 240),
                "crop_end": (0, 0, 320, 240),
            },
        ]
        resolve_clip = MagicMock(return_value="doorbell-1.mp4")

        _run_pynv_compilation(
            slices=slices,
            ce_dir="/ce",
            tmp_output_path="/out.mp4.tmp",
            target_w=1440,
            target_h=1080,
            resolve_clip_in_folder=resolve_clip,
            gpu_backend=gpu_backend,
            gpu_device_index=0,
        )

        mock_popen.assert_called_once()
        assert proc.stdin.write.call_count == 0
        proc.stdin.close.assert_called_once()
        proc.wait.assert_called()

    @patch("frigate_buffer.services.video_compilation.GPU_LOCK", MagicMock())
    @patch("frigate_buffer.services.video_compilation.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    @patch("frigate_buffer.services.video_compilation.logger")
    @patch("frigate_buffer.services.video_compilation._get_video_metadata")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    def test_run_pynv_compilation_skips_slice_when_decoder_returns_zero_frames(
        self,
        mock_isfile,
        mock_get_metadata,
        mock_logger,
        mock_open_fn,
        mock_popen,
    ):
        """get_frames returns 0 → slice skipped; log clarifies not hardware init."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_isfile.return_value = True
        mock_get_metadata.return_value = (640, 480, 20.0, 2.0)
        mock_ctx = fake_create_decoder_context(200, height=480, width=640)
        if mock_ctx is None:
            self.skipTest("torch not available")
        mock_ctx.get_frames = lambda indices: torch.empty(
            0, 3, 480, 640, dtype=torch.uint8
        )
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_ctx)
        mock_cm.__exit__ = MagicMock(return_value=None)
        gpu_backend = gpu_backend_for_compilation_tests(return_value=mock_cm)
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.returncode = 0
        mock_popen.return_value = proc

        slices = [
            {
                "camera": "doorbell",
                "start_sec": 0.0,
                "end_sec": 2.0,
                "crop_start": (0, 0, 320, 240),
                "crop_end": (0, 0, 320, 240),
            },
        ]
        resolve_clip = MagicMock(return_value="doorbell-1.mp4")

        _run_pynv_compilation(
            slices=slices,
            ce_dir="/ce",
            tmp_output_path="/out.mp4.tmp",
            target_w=1440,
            target_h=1080,
            resolve_clip_in_folder=resolve_clip,
            gpu_backend=gpu_backend,
            gpu_device_index=0,
        )

        assert proc.stdin.write.call_count == 0
        error_calls = [c for c in mock_logger.error.call_args_list if c[0]]
        assert len(error_calls) > 0
        msg = str(error_calls[0][0][0])
        assert "0 frames" in msg
        assert "Skipping chunk" in msg

    @patch("frigate_buffer.services.video_compilation.GPU_LOCK", MagicMock())
    @patch("frigate_buffer.services.video_compilation.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    @patch("frigate_buffer.services.video_compilation._get_video_metadata")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    def test_run_pynv_compilation_chunks_decode_and_calls_empty_cache_per_chunk(
        self,
        mock_isfile,
        mock_get_metadata,
        mock_open_fn,
        mock_popen,
    ):
        """get_frames in BATCH_SIZE chunks; empty_cache after each and in finally."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_isfile.return_value = True
        # 1.0 s at 20 fps = 20 frames → 5 chunks of 4 (BATCH_SIZE)
        mock_get_metadata.return_value = (640, 480, 20.0, 1.0)
        get_frames_calls = []

        def record_get_frames(indices):
            get_frames_calls.append(list(indices))
            return torch.zeros((len(indices), 3, 480, 640), dtype=torch.uint8)

        mock_ctx = fake_create_decoder_context(200, height=480, width=640)
        if mock_ctx is None:
            self.skipTest("torch not available")
        mock_ctx.get_frames = record_get_frames
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_ctx)
        mock_cm.__exit__ = MagicMock(return_value=None)
        gpu_backend = gpu_backend_for_compilation_tests(return_value=mock_cm)
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.returncode = 0
        mock_popen.return_value = proc

        slices = [
            {
                "camera": "doorbell",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "crop_start": (0, 0, 320, 240),
                "crop_end": (0, 0, 320, 240),
            },
        ]
        resolve_clip = MagicMock(return_value="doorbell-1.mp4")

        with patch(
            "frigate_buffer.services.gpu_backends.nvidia.runtime.empty_cache"
        ) as mock_empty_cache:
            _run_pynv_compilation(
                slices=slices,
                ce_dir="/ce",
                tmp_output_path="/out.mp4.tmp",
                target_w=1440,
                target_h=1080,
                resolve_clip_in_folder=resolve_clip,
                gpu_backend=gpu_backend,
                gpu_device_index=0,
            )

        # 20 frames in chunks of BATCH_SIZE (4) → 5 chunk calls
        assert len(get_frames_calls) > 1, (
            "get_frames should be called multiple times per slice"
        )
        assert len(get_frames_calls) == 5, "20 frames / BATCH_SIZE=4 → 5 chunks"
        assert get_frames_calls[0] == [0, 1, 2, 3]
        assert get_frames_calls[1] == [4, 5, 6, 7]
        assert get_frames_calls[2] == [8, 9, 10, 11]
        assert get_frames_calls[3] == [12, 13, 14, 15]
        assert get_frames_calls[4] == [16, 17, 18, 19]
        # empty_cache: once after each of 5 chunks + once in slice finally = 6
        assert mock_empty_cache.call_count == 6
        assert proc.stdin.write.call_count == 20

    @patch("frigate_buffer.services.video_compilation.GPU_LOCK", MagicMock())
    @patch("frigate_buffer.services.video_compilation.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    @patch(
        "frigate_buffer.services.video_compilation.crop_utils.crop_around_center_to_size"
    )
    @patch("frigate_buffer.services.video_compilation._get_video_metadata")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    def test_run_pynv_compilation_repeats_last_frame_when_decoder_returns_fewer_frames(
        self,
        mock_isfile,
        mock_get_metadata,
        mock_crop_to_size,
        mock_open_fn,
        mock_popen,
    ):
        """get_frames fewer than requested → repeat last frame; no IndexError."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_isfile.return_value = True
        mock_get_metadata.return_value = (640, 480, 20.0, 2.0)
        mock_ctx = fake_create_decoder_context(200, height=480, width=640)
        if mock_ctx is None:
            self.skipTest("torch not available")
        mock_ctx.get_frames = lambda indices: torch.zeros(
            (1, 3, 480, 640), dtype=torch.uint8
        )
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_ctx)
        mock_cm.__exit__ = MagicMock(return_value=None)
        gpu_backend = gpu_backend_for_compilation_tests(return_value=mock_cm)
        mock_crop_to_size.side_effect = lambda frame, cx, cy, cw, ch, out_w, out_h: (
            torch.zeros((1, 3, out_h, out_w), dtype=torch.uint8)
        )
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.returncode = 0
        mock_popen.return_value = proc

        slices = [
            {
                "camera": "doorbell",
                "start_sec": 0.0,
                "end_sec": 2.0,
                "crop_start": (0, 0, 320, 240),
                "crop_end": (0, 0, 320, 240),
                "native_width": 640,
                "native_height": 480,
            },
        ]
        resolve_clip = MagicMock(return_value="doorbell-1.mp4")

        _run_pynv_compilation(
            slices=slices,
            ce_dir="/ce",
            tmp_output_path="/out.mp4.tmp",
            target_w=1440,
            target_h=1080,
            resolve_clip_in_folder=resolve_clip,
            gpu_backend=gpu_backend,
            gpu_device_index=0,
        )

        n_frames_expected = 40
        assert proc.stdin.write.call_count == n_frames_expected

    @patch("frigate_buffer.services.video_compilation.GPU_LOCK", MagicMock())
    @patch("frigate_buffer.services.video_compilation.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    @patch("frigate_buffer.services.video_compilation.logger")
    @patch(
        "frigate_buffer.services.video_compilation.crop_utils.crop_around_center_to_size"
    )
    @patch("frigate_buffer.services.video_compilation._get_video_metadata")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    def test_run_pynv_compilation_logs_info_once_per_camera_when_fewer_frames(
        self,
        mock_isfile,
        mock_get_metadata,
        mock_crop_to_size,
        mock_logger,
        mock_open_fn,
        mock_popen,
    ):
        """Decoder fewer than requested → DEBUG + one INFO per camera per event."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_isfile.return_value = True
        mock_get_metadata.return_value = (640, 480, 20.0, 2.0)
        mock_ctx = fake_create_decoder_context(200, height=480, width=640)
        if mock_ctx is None:
            self.skipTest("torch not available")
        mock_ctx.get_frames = lambda indices: torch.zeros(
            (1, 3, 480, 640), dtype=torch.uint8
        )
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_ctx)
        mock_cm.__exit__ = MagicMock(return_value=None)
        gpu_backend = gpu_backend_for_compilation_tests(return_value=mock_cm)
        mock_crop_to_size.side_effect = lambda frame, cx, cy, cw, ch, out_w, out_h: (
            torch.zeros((1, 3, out_h, out_w), dtype=torch.uint8)
        )
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.returncode = 0
        mock_popen.return_value = proc
        resolve_clip = MagicMock(return_value="/path/doorbell-1.mp4")
        slices = [
            {
                "camera": "doorbell",
                "start_sec": 0.0,
                "end_sec": 2.0,
                "crop_start": (0, 0, 320, 240),
                "crop_end": (0, 0, 320, 240),
                "native_width": 640,
                "native_height": 480,
            },
        ]

        _run_pynv_compilation(
            slices=slices,
            ce_dir="/ce",
            tmp_output_path="/out.mp4.tmp",
            target_w=1440,
            target_h=1080,
            resolve_clip_in_folder=resolve_clip,
            gpu_backend=gpu_backend,
            gpu_device_index=0,
        )

        info_calls = [
            c
            for c in mock_logger.info.call_args_list
            if c[0]
            and "Possible stutter or missing frames" in (str(c[0][0]) if c[0] else "")
        ]
        assert len(info_calls) == 1, "Expected exactly one INFO per camera per event"
        assert "doorbell" in str(info_calls[0])
        assert "/path/doorbell-1.mp4" in str(info_calls[0])

    @patch("frigate_buffer.services.video_compilation.GPU_LOCK", MagicMock())
    @patch("frigate_buffer.services.video_compilation.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    @patch(
        "frigate_buffer.services.video_compilation.crop_utils.crop_around_center_to_size"
    )
    @patch("frigate_buffer.services.video_compilation._get_video_metadata")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    def test_run_pynv_compilation_uses_crop_utils_and_outputs_target_resolution(
        self,
        mock_isfile,
        mock_get_metadata,
        mock_crop_to_size,
        mock_open_fn,
        mock_popen,
    ):
        """_run_pynv uses crop_around_center_to_size; frames to FFmpeg at target res."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_isfile.return_value = True
        mock_get_metadata.return_value = (640, 480, 20.0, 2.0)
        # Decoder returns 640x480; slice has native 1920x1080 so scaling is applied.
        mock_ctx = fake_create_decoder_context(200, height=480, width=640)
        if mock_ctx is None:
            self.skipTest("torch not available")
        mock_ctx.get_frames = lambda indices: torch.zeros(
            (len(indices), 3, 480, 640), dtype=torch.uint8
        )
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_ctx)
        mock_cm.__exit__ = MagicMock(return_value=None)
        gpu_backend = gpu_backend_for_compilation_tests(return_value=mock_cm)
        # crop_around_center_to_size(frame, cx, cy, crop_w, crop_h, output_w, output_h)
        mock_crop_to_size.side_effect = lambda frame, cx, cy, cw, ch, out_w, out_h: (
            torch.zeros((1, 3, out_h, out_w), dtype=torch.uint8)
        )
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.returncode = 0
        mock_popen.return_value = proc

        slices = [
            {
                "camera": "doorbell",
                "start_sec": 0.0,
                "end_sec": 2.0,
                "crop_start": (240, 0, 1440, 1080),
                "crop_end": (300, 0, 1440, 1080),
                "native_width": 1920,
                "native_height": 1080,
            },
        ]
        resolve_clip = MagicMock(return_value="doorbell-1.mp4")
        target_w, target_h = 1440, 1080

        _run_pynv_compilation(
            slices=slices,
            ce_dir="/ce",
            tmp_output_path="/out.mp4.tmp",
            target_w=target_w,
            target_h=target_h,
            resolve_clip_in_folder=resolve_clip,
            gpu_backend=gpu_backend,
            gpu_device_index=0,
        )

        mock_crop_to_size.assert_called()
        args = mock_crop_to_size.call_args[0]
        assert args[5] == target_w, "output_w should be target_w"
        assert args[6] == target_h, "output_h should be target_h"
        mock_popen.assert_called_once()
        # 2s at 20fps = 40 frames; each write one frame (target_h*target_w*3 bytes)
        assert proc.stdin.write.call_count == 40
        expected_frame_bytes = target_h * target_w * 3
        for call in proc.stdin.write.call_args_list:
            assert len(call[0][0]) == expected_frame_bytes
        proc.stdin.flush.assert_called()
        proc.stdin.close.assert_called_once()
        proc.wait.assert_called()

    @patch("frigate_buffer.services.video_compilation.GPU_LOCK", MagicMock())
    @patch("frigate_buffer.services.video_compilation.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    @patch("frigate_buffer.services.video_compilation.logger")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    def test_run_pynv_compilation_logs_error_when_all_src_indices_identical(
        self, mock_isfile, mock_logger, mock_open_fn, mock_popen
    ):
        """Same frame index for all → DEBUG and INFO once per camera (static)."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_isfile.return_value = True
        mock_ctx = fake_create_decoder_context(200, height=480, width=640)
        if mock_ctx is None:
            self.skipTest("torch not available")
        mock_ctx.get_index_from_time_in_seconds = lambda t: 0
        mock_ctx.get_frames = lambda indices: torch.zeros(
            (len(indices), 3, 480, 640), dtype=torch.uint8
        )
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_ctx)
        mock_cm.__exit__ = MagicMock(return_value=None)
        gpu_backend = gpu_backend_for_compilation_tests(return_value=mock_cm)
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.returncode = 0
        mock_popen.return_value = proc

        slices = [
            {
                "camera": "doorbell",
                "start_sec": 0.0,
                "end_sec": 2.0,
                "crop_start": (0, 0, 320, 240),
                "crop_end": (0, 0, 320, 240),
                "native_width": 640,
                "native_height": 480,
            },
        ]
        resolve_clip = MagicMock(return_value="doorbell-1.mp4")

        _run_pynv_compilation(
            slices=slices,
            ce_dir="/ce",
            tmp_output_path="/out.mp4.tmp",
            target_w=1440,
            target_h=1080,
            resolve_clip_in_folder=resolve_clip,
            gpu_backend=gpu_backend,
            gpu_device_index=0,
        )

        debug_calls = [
            c
            for c in mock_logger.debug.call_args_list
            if c[0] and "static frame" in (str(c[0][0]) if c[0] else "")
        ]
        assert len(debug_calls) > 0, (
            "Expected at least one DEBUG for static frame (same index for all)"
        )
        info_calls = [
            c
            for c in mock_logger.info.call_args_list
            if c[0]
            and "Possible stutter or missing frames" in (str(c[0][0]) if c[0] else "")
        ]
        assert len(info_calls) > 0, (
            "Expected one INFO for stutter/missing (once per camera per event)"
        )
        assert "doorbell" in str(info_calls[0])
        assert "doorbell-1.mp4" in str(info_calls[0])

    @patch("frigate_buffer.services.video_compilation.GPU_LOCK", MagicMock())
    @patch("frigate_buffer.services.video_compilation.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    @patch("frigate_buffer.services.video_compilation.logger")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    def test_run_pynv_compilation_logs_once_per_camera_not_per_slice(
        self, mock_isfile, mock_logger, mock_open_fn, mock_popen
    ):
        """DEBUG compilation log emitted once per camera, not per slice."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_isfile.return_value = True
        mock_ctx = fake_create_decoder_context(200, height=480, width=640)
        if mock_ctx is None:
            self.skipTest("torch not available")
        mock_ctx.get_frames = lambda indices: torch.zeros(
            (len(indices), 3, 480, 640), dtype=torch.uint8
        )
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_ctx)
        mock_cm.__exit__ = MagicMock(return_value=None)
        gpu_backend = gpu_backend_for_compilation_tests(return_value=mock_cm)
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.returncode = 0
        mock_popen.return_value = proc

        # 3 slices for doorbell, 2 for front_door (5 slices, 2 distinct cameras).
        slices = [
            {
                "camera": "doorbell",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "crop_start": (0, 0, 320, 240),
                "crop_end": (0, 0, 320, 240),
            },
            {
                "camera": "doorbell",
                "start_sec": 1.0,
                "end_sec": 2.0,
                "crop_start": (0, 0, 320, 240),
                "crop_end": (0, 0, 320, 240),
            },
            {
                "camera": "doorbell",
                "start_sec": 2.0,
                "end_sec": 3.0,
                "crop_start": (0, 0, 320, 240),
                "crop_end": (0, 0, 320, 240),
            },
            {
                "camera": "front_door",
                "start_sec": 3.0,
                "end_sec": 4.0,
                "crop_start": (0, 0, 320, 240),
                "crop_end": (0, 0, 320, 240),
            },
            {
                "camera": "front_door",
                "start_sec": 4.0,
                "end_sec": 5.0,
                "crop_start": (0, 0, 320, 240),
                "crop_end": (0, 0, 320, 240),
            },
        ]
        resolve_clip = MagicMock(return_value="clip.mp4")

        _run_pynv_compilation(
            slices=slices,
            ce_dir="/ce",
            tmp_output_path="/out.mp4.tmp",
            target_w=1440,
            target_h=1080,
            resolve_clip_in_folder=resolve_clip,
            gpu_backend=gpu_backend,
            gpu_device_index=0,
        )

        compilation_debug_calls = [
            c
            for c in mock_logger.debug.call_args_list
            if c[0] and "Compilation camera=" in (str(c[0][0]) if c[0] else "")
        ]
        assert len(compilation_debug_calls) == 2, (
            "Expected 2 DEBUG (one per camera), not per slice. "
            f"Got {len(compilation_debug_calls)} calls."
        )
