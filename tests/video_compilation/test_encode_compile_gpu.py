"""FFmpeg encode, compile_ce_video config, and GpuBackend surface tests."""

import json
import unittest
from unittest.mock import MagicMock, mock_open, patch

import pytest

from frigate_buffer.services import video_compilation
from frigate_buffer.services.video_compilation import (
    _encode_frames_via_ffmpeg,
    _run_pynv_compilation,
)
from helpers.video_compilation import (
    fake_create_decoder_context,
    gpu_backend_for_compilation_tests,
    gpu_backend_for_compilation_tests_amd,
    gpu_backend_for_compilation_tests_intel,
)


class TestEncodeFramesViaFfmpeg(unittest.TestCase):
    """Tests for _encode_frames_via_ffmpeg: h264_nvenc, descriptive error on failure."""

    def test_encode_uses_h264_nvenc_in_command(self):
        """FFmpeg -c:v h264_nvenc and -pix_fmt yuv420p (no libx264)."""
        import numpy as np

        frames = [np.zeros((1080, 1440, 3), dtype=np.uint8)]
        with patch("builtins.open", mock_open()):
            with patch(
                "frigate_buffer.services.video_compilation.subprocess.Popen"
            ) as mock_popen:
                proc = MagicMock()
                proc.stdin = MagicMock()
                proc.returncode = 0
                mock_popen.return_value = proc
                be = gpu_backend_for_compilation_tests()
                _encode_frames_via_ffmpeg(
                    frames, 1440, 1080, "/tmp/out.mp4", gpu_backend=be
                )
        call_args = mock_popen.call_args[0][0]
        assert "h264_nvenc" in call_args
        assert "libx264" not in call_args
        assert "yuv420p" in call_args

    def test_encode_uses_thread_queue_size_and_nvenc_tune(self):
        """FFmpeg: -thread_queue_size 512, h264_nvenc preset/tune/rc/cq."""
        import numpy as np

        frames = [np.zeros((1080, 1440, 3), dtype=np.uint8)]
        with patch("builtins.open", mock_open()):
            with patch(
                "frigate_buffer.services.video_compilation.subprocess.Popen"
            ) as mock_popen:
                proc = MagicMock()
                proc.stdin = MagicMock()
                proc.returncode = 0
                mock_popen.return_value = proc
                be = gpu_backend_for_compilation_tests()
                _encode_frames_via_ffmpeg(
                    frames, 1440, 1080, "/tmp/out.mp4", gpu_backend=be
                )
        call_args = mock_popen.call_args[0][0]
        cmd_str = " ".join(call_args) if isinstance(call_args, list) else str(call_args)
        assert "thread_queue_size" in cmd_str
        assert "512" in cmd_str
        assert "p1" in cmd_str
        assert "hq" in cmd_str
        assert "vbr" in cmd_str
        assert "24" in cmd_str

    def test_encode_failure_raises_with_descriptive_message(self):
        """On non-zero exit, RuntimeError advises checking ffmpeg_compile.log."""
        import numpy as np

        frames = [np.zeros((1080, 1440, 3), dtype=np.uint8)]
        with patch("builtins.open", mock_open()):
            with patch(
                "frigate_buffer.services.video_compilation.subprocess.Popen"
            ) as mock_popen:
                proc = MagicMock()
                proc.stdin = MagicMock()
                proc.returncode = 1
                mock_popen.return_value = proc
                be = gpu_backend_for_compilation_tests()
                with pytest.raises(RuntimeError) as ctx:
                    _encode_frames_via_ffmpeg(
                        frames, 1440, 1080, "/tmp/out.mp4", gpu_backend=be
                    )
        assert "h264_nvenc" in str(ctx.value)
        assert "ffmpeg_compile.log" in str(ctx.value)

    def test_broken_pipe_logs_and_raises_with_log_path(self):
        """FFmpeg closes stdin → BrokenPipeError; RuntimeError re ffmpeg_compile.log."""
        import numpy as np

        frames = [np.zeros((1080, 1440, 3), dtype=np.uint8)]
        with patch("builtins.open", mock_open()):
            with patch(
                "frigate_buffer.services.video_compilation.subprocess.Popen"
            ) as mock_popen:
                proc = MagicMock()
                proc.stdin = MagicMock()
                proc.stdin.write.side_effect = BrokenPipeError
                mock_popen.return_value = proc
                be = gpu_backend_for_compilation_tests()
                with pytest.raises(RuntimeError) as ctx:
                    _encode_frames_via_ffmpeg(
                        frames, 1440, 1080, "/tmp/out.mp4", gpu_backend=be
                    )
        assert "broke pipe" in str(ctx.value)
        assert "ffmpeg_compile.log" in str(ctx.value)
        proc.wait.assert_called_once()


class TestCompileCeVideoConfig(unittest.TestCase):
    """Tests that compile_ce_video uses same timeline config as frame timeline."""

    @patch("frigate_buffer.services.video_compilation.generate_compilation_video")
    @patch(
        "frigate_buffer.services.video_compilation.timeline_ema.build_phase1_assignments"
    )
    @patch("frigate_buffer.services.video_compilation.timeline_ema.build_dense_times")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    def test_uses_max_multi_cam_frames_config_keys(
        self, mock_resolve, mock_build_dense, mock_build_assignments, mock_gen
    ):
        """Uses MAX_MULTI_CAM_FRAMES_SEC and MAX_MULTI_CAM_FRAMES_MIN."""
        mock_resolve.return_value = "cam1.mp4"
        mock_build_dense.return_value = [0.0, 1.0, 2.0]
        mock_build_assignments.return_value = [
            (0.0, "cam1"),
            (1.0, "cam1"),
            (2.0, "cam1"),
        ]
        from frigate_buffer.services.video_compilation import compile_ce_video

        config = {
            "MAX_MULTI_CAM_FRAMES_SEC": 2.5,
            "MAX_MULTI_CAM_FRAMES_MIN": 50,
            "CAMERA_TIMELINE_ANALYSIS_MULTIPLIER": 2.0,
            "CAMERA_TIMELINE_EMA_ALPHA": 0.4,
            "CAMERA_TIMELINE_PRIMARY_BIAS_MULTIPLIER": 1.2,
            "CAMERA_SWITCH_HYSTERESIS_MARGIN": 1.15,
            "CAMERA_SWITCH_MIN_SEGMENT_FRAMES": 5,
            "SUMMARY_TARGET_WIDTH": 1440,
            "SUMMARY_TARGET_HEIGHT": 1080,
        }
        with patch(
            "frigate_buffer.services.video_compilation.os.scandir"
        ) as mock_scandir:
            with patch(
                "frigate_buffer.services.video_compilation.os.path.isfile",
                return_value=True,
            ):
                with patch(
                    "builtins.open",
                    mock_open(
                        read_data=json.dumps(
                            {"entries": [], "native_width": 1920, "native_height": 1080}
                        )
                    ),
                ):
                    mock_ent = MagicMock()
                    mock_ent.is_dir.return_value = True
                    mock_ent.name = "cam1"
                    mock_ent.path = "/ce/cam1"
                    mock_scandir.return_value.__enter__.return_value = [mock_ent]
                    compile_ce_video("/ce", 3.0, config, None)
        call_args = mock_build_dense.call_args[0]
        assert call_args[0] == 2.5
        assert call_args[1] == 50

    @patch("frigate_buffer.services.video_compilation.generate_compilation_video")
    @patch("frigate_buffer.services.video_compilation._trim_slices_to_action_window")
    @patch(
        "frigate_buffer.services.video_compilation.timeline_ema.build_phase1_assignments"
    )
    @patch("frigate_buffer.services.video_compilation.timeline_ema.build_dense_times")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    def test_returns_none_when_trim_leaves_no_slices(
        self,
        mock_resolve,
        mock_build_dense,
        mock_build_assignments,
        mock_trim,
        mock_gen,
    ):
        """_trim_slices empty → returns None, no generate_compilation_video."""
        mock_resolve.return_value = "cam1.mp4"
        mock_build_dense.return_value = [0.0, 1.0, 2.0]
        mock_build_assignments.return_value = [
            (0.0, "cam1"),
            (1.0, "cam1"),
            (2.0, "cam1"),
        ]
        mock_trim.return_value = []
        from frigate_buffer.services.video_compilation import compile_ce_video

        config = {
            "MAX_MULTI_CAM_FRAMES_SEC": 2,
            "MAX_MULTI_CAM_FRAMES_MIN": 45,
            "CAMERA_TIMELINE_ANALYSIS_MULTIPLIER": 2.0,
            "CAMERA_TIMELINE_EMA_ALPHA": 0.4,
            "CAMERA_TIMELINE_PRIMARY_BIAS_MULTIPLIER": 1.2,
            "CAMERA_SWITCH_HYSTERESIS_MARGIN": 1.15,
            "CAMERA_SWITCH_MIN_SEGMENT_FRAMES": 5,
            "SUMMARY_TARGET_WIDTH": 1440,
            "SUMMARY_TARGET_HEIGHT": 1080,
        }
        with patch(
            "frigate_buffer.services.video_compilation.os.scandir"
        ) as mock_scandir:
            with patch(
                "frigate_buffer.services.video_compilation.os.path.isfile",
                return_value=True,
            ):
                with patch(
                    "builtins.open",
                    mock_open(
                        read_data=json.dumps(
                            {"entries": [], "native_width": 1920, "native_height": 1080}
                        )
                    ),
                ):
                    mock_ent = MagicMock()
                    mock_ent.is_dir.return_value = True
                    mock_ent.name = "cam1"
                    mock_ent.path = "/ce/cam1"
                    mock_scandir.return_value.__enter__.return_value = [mock_ent]
                    result = compile_ce_video("/ce", 3.0, config, None)
        assert result is None
        mock_gen.assert_not_called()

    @patch("frigate_buffer.services.video_compilation.generate_compilation_video")
    @patch(
        "frigate_buffer.services.video_compilation.timeline_ema.build_phase1_assignments"
    )
    @patch("frigate_buffer.services.video_compilation.timeline_ema.build_dense_times")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    def test_extends_global_end_when_sidecar_longer_than_requested_window(
        self, mock_resolve, mock_build_dense, mock_build_assignments, mock_gen
    ):
        """Sidecar longer than global_end → extends timeline to sidecar duration."""
        mock_resolve.return_value = "cam1.mp4"
        mock_build_dense.return_value = [0.0, 2.0, 4.0]
        mock_build_assignments.return_value = [
            (0.0, "cam1"),
            (2.0, "cam1"),
            (4.0, "cam1"),
        ]
        # Sidecar with last entry at 87s (longer than requested 50s)
        sidecar_data = {
            "entries": [
                {"timestamp_sec": 0, "detections": []},
                {
                    "timestamp_sec": 87.0,
                    "detections": [{"label": "person", "area": 1000}],
                },
            ],
            "native_width": 1920,
            "native_height": 1080,
        }
        from frigate_buffer.services.video_compilation import compile_ce_video

        config = {
            "MAX_MULTI_CAM_FRAMES_SEC": 2,
            "MAX_MULTI_CAM_FRAMES_MIN": 45,
            "CAMERA_TIMELINE_ANALYSIS_MULTIPLIER": 2.0,
            "CAMERA_TIMELINE_EMA_ALPHA": 0.4,
            "CAMERA_TIMELINE_PRIMARY_BIAS_MULTIPLIER": 1.2,
            "CAMERA_SWITCH_HYSTERESIS_MARGIN": 1.15,
            "CAMERA_SWITCH_MIN_SEGMENT_FRAMES": 5,
            "SUMMARY_TARGET_WIDTH": 1440,
            "SUMMARY_TARGET_HEIGHT": 1080,
        }
        with patch(
            "frigate_buffer.services.video_compilation.os.scandir"
        ) as mock_scandir:
            with patch(
                "frigate_buffer.services.video_compilation.os.path.isfile",
                return_value=True,
            ):
                with patch(
                    "builtins.open", mock_open(read_data=json.dumps(sidecar_data))
                ):
                    mock_ent = MagicMock()
                    mock_ent.is_dir.return_value = True
                    mock_ent.name = "cam1"
                    mock_ent.path = "/ce/cam1"
                    mock_scandir.return_value.__enter__.return_value = [mock_ent]
                    compile_ce_video("/ce", 50.0, config, None)
        # build_dense_times(step_sec, max_frames_min, mult, global_end) -> global_end=87
        call_args = mock_build_dense.call_args[0]
        assert call_args[3] == 87.0, (
            "global_end should extend to sidecar last timestamp"
        )

    @patch("frigate_buffer.services.video_compilation.generate_compilation_video")
    @patch(
        "frigate_buffer.services.video_compilation.timeline_ema.build_phase1_assignments"
    )
    @patch("frigate_buffer.services.video_compilation.timeline_ema.build_dense_times")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    def test_keeps_global_end_when_sidecar_shorter_than_requested_window(
        self, mock_resolve, mock_build_dense, mock_build_assignments, mock_gen
    ):
        """Sidecar shorter than global_end → keeps caller global_end."""
        mock_resolve.return_value = "cam1.mp4"
        mock_build_dense.return_value = [0.0, 2.0, 4.0]
        mock_build_assignments.return_value = [
            (0.0, "cam1"),
            (2.0, "cam1"),
            (4.0, "cam1"),
        ]
        # Sidecar with last entry at 20s (shorter than requested 50s)
        sidecar_data = {
            "entries": [
                {"timestamp_sec": 0, "detections": []},
                {
                    "timestamp_sec": 20.0,
                    "detections": [{"label": "person", "area": 1000}],
                },
            ],
            "native_width": 1920,
            "native_height": 1080,
        }
        from frigate_buffer.services.video_compilation import compile_ce_video

        config = {
            "MAX_MULTI_CAM_FRAMES_SEC": 2,
            "MAX_MULTI_CAM_FRAMES_MIN": 45,
            "CAMERA_TIMELINE_ANALYSIS_MULTIPLIER": 2.0,
            "CAMERA_TIMELINE_EMA_ALPHA": 0.4,
            "CAMERA_TIMELINE_PRIMARY_BIAS_MULTIPLIER": 1.2,
            "CAMERA_SWITCH_HYSTERESIS_MARGIN": 1.15,
            "CAMERA_SWITCH_MIN_SEGMENT_FRAMES": 5,
            "SUMMARY_TARGET_WIDTH": 1440,
            "SUMMARY_TARGET_HEIGHT": 1080,
        }
        with patch(
            "frigate_buffer.services.video_compilation.os.scandir"
        ) as mock_scandir:
            with patch(
                "frigate_buffer.services.video_compilation.os.path.isfile",
                return_value=True,
            ):
                with patch(
                    "builtins.open", mock_open(read_data=json.dumps(sidecar_data))
                ):
                    mock_ent = MagicMock()
                    mock_ent.is_dir.return_value = True
                    mock_ent.name = "cam1"
                    mock_ent.path = "/ce/cam1"
                    mock_scandir.return_value.__enter__.return_value = [mock_ent]
                    compile_ce_video("/ce", 50.0, config, None)
        # global_end should remain 50 (max(50, 20) = 50)
        call_args = mock_build_dense.call_args[0]
        assert call_args[3] == 50.0, (
            "global_end should remain caller value when sidecar is shorter"
        )


class TestCompilationGpuBackendSurface(unittest.TestCase):
    """Compilation uses GpuBackend for decode argv (NVENC / QSV / AMF).

    Does not use gpu_decoder shim.
    """

    def test_generate_compilation_video_has_gpu_backend_parameter(self):
        from inspect import signature

        sig = signature(video_compilation.generate_compilation_video)
        assert "gpu_backend" in sig.parameters

    def test_run_pynv_compilation_requires_gpu_backend(self):
        from inspect import signature

        sig = signature(_run_pynv_compilation)
        assert "gpu_backend" in sig.parameters

    @patch("frigate_buffer.services.video_compilation.GPU_LOCK", MagicMock())
    @patch("frigate_buffer.services.video_compilation.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    @patch("frigate_buffer.services.video_compilation._get_video_metadata")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    def test_run_pynv_compilation_intel_backend_calls_h264_qsv(
        self,
        mock_isfile,
        mock_get_metadata,
        mock_open_fn,
        mock_popen,
    ):
        """Same compilation path as NVENC test; Intel backend → h264_qsv argv."""
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
        gpu_backend = gpu_backend_for_compilation_tests_intel(return_value=mock_cm)
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
        assert "h264_qsv" in call_args
        assert "h264_nvenc" not in call_args
        assert "-thread_queue_size" in call_args
        assert "512" in call_args
        assert "medium" in call_args
        gq = call_args.index("-global_quality")
        assert call_args[gq + 1] == "24"
        assert call_args[call_args.index("-s") + 1] == "1440x1080"
        assert call_args[-1] == "/out.mp4.tmp"
        assert proc.stdin.write.call_count == 40
        proc.stdin.flush.assert_called()
        proc.stdin.close.assert_called_once()
        proc.wait.assert_called()

    @patch("frigate_buffer.services.video_compilation.GPU_LOCK", MagicMock())
    @patch("frigate_buffer.services.video_compilation.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    @patch("frigate_buffer.services.video_compilation._get_video_metadata")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    def test_run_pynv_compilation_intel_backend_uses_config_qsv_options(
        self,
        mock_isfile,
        mock_get_metadata,
        mock_open_fn,
        mock_popen,
    ):
        """INTEL_QSV_* from merged config flow into h264_qsv argv."""
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
        gpu_backend = gpu_backend_for_compilation_tests_intel(return_value=mock_cm)
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
        cfg = {
            "INTEL_QSV_ENCODE_PRESET": "faster",
            "INTEL_QSV_ENCODE_GLOBAL_QUALITY": 30,
        }

        _run_pynv_compilation(
            slices=slices,
            ce_dir="/ce",
            tmp_output_path="/out.mp4.tmp",
            target_w=1440,
            target_h=1080,
            resolve_clip_in_folder=resolve_clip,
            gpu_backend=gpu_backend,
            gpu_device_index=0,
            config=cfg,
        )

        mock_popen.assert_called_once()
        call_args = mock_popen.call_args[0][0]
        assert call_args[call_args.index("-preset") + 1] == "faster"
        assert call_args[call_args.index("-global_quality") + 1] == "30"

    @patch("frigate_buffer.services.video_compilation.GPU_LOCK", MagicMock())
    @patch("frigate_buffer.services.video_compilation.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    @patch("frigate_buffer.services.video_compilation._get_video_metadata")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    def test_run_pynv_compilation_amd_backend_calls_h264_amf(
        self,
        mock_isfile,
        mock_get_metadata,
        mock_open_fn,
        mock_popen,
    ):
        """Same compilation path as QSV test; AMD backend → h264_amf argv."""
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
        gpu_backend = gpu_backend_for_compilation_tests_amd(return_value=mock_cm)
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
        assert "h264_amf" in call_args
        assert "h264_nvenc" not in call_args
        assert "h264_qsv" not in call_args
        assert "-thread_queue_size" in call_args
        assert "512" in call_args
        q = call_args.index("-quality")
        assert call_args[q + 1] == "balanced"
        assert call_args[call_args.index("-s") + 1] == "1440x1080"
        assert call_args[-1] == "/out.mp4.tmp"
        assert proc.stdin.write.call_count == 40
        proc.stdin.flush.assert_called()
        proc.stdin.close.assert_called_once()
        proc.wait.assert_called()
