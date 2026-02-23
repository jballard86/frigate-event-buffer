"""Tests for video_compilation service."""

import json
import os
import unittest
from unittest.mock import MagicMock, patch, mock_open


from frigate_buffer.services.video_compilation import (
    _encode_frames_via_ffmpeg,
    _run_pynv_compilation,
    assignments_to_slices,
    calculate_crop_at_time,
    calculate_segment_crop,
    convert_timeline_to_segments,
    generate_compilation_video,
    smooth_crop_centers_ema,
)


def _fake_create_decoder_context(frame_count: int = 200, height: int = 480, width: int = 640):
    """Build a mock DecoderContext for _run_pynv_compilation tests."""
    try:
        import torch
    except ImportError:
        return None
    mock_ctx = MagicMock()
    mock_ctx.__len__ = lambda self: frame_count
    # Called as ctx.get_index_from_time_in_seconds(t) -> one arg t
    mock_ctx.get_index_from_time_in_seconds = lambda t: min(max(0, int(t * 20)), max(0, frame_count - 1))
    # Called as ctx.get_frames(indices) -> one arg indices
    mock_ctx.get_frames = lambda indices: torch.zeros(
        (len(indices), 3, height, width), dtype=torch.uint8
    )
    return mock_ctx


class TestConvertTimelineToSegments(unittest.TestCase):
    """Tests for convert_timeline_to_segments function."""

    def test_empty_timeline_returns_empty_list(self):
        """Empty timeline should return empty segments list."""
        result = convert_timeline_to_segments([], 60.0)
        self.assertEqual(result, [])

    def test_single_camera_continuous_timeline(self):
        """Single camera throughout creates one segment."""
        timeline = [
            (0.0, "doorbell"),
            (1.0, "doorbell"),
            (2.0, "doorbell"),
        ]
        result = convert_timeline_to_segments(timeline, 10.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["camera"], "doorbell")
        self.assertEqual(result[0]["start_sec"], 0.0)
        self.assertEqual(result[0]["end_sec"], 10.0)

    def test_camera_switch_creates_multiple_segments(self):
        """Camera switches create multiple segments."""
        timeline = [
            (0.0, "doorbell"),
            (5.0, "doorbell"),
            (10.0, "carport"),
            (15.0, "carport"),
        ]
        result = convert_timeline_to_segments(timeline, 20.0)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["camera"], "doorbell")
        self.assertEqual(result[0]["start_sec"], 0.0)
        self.assertEqual(result[0]["end_sec"], 10.0)
        self.assertEqual(result[1]["camera"], "carport")
        self.assertEqual(result[1]["start_sec"], 10.0)
        self.assertEqual(result[1]["end_sec"], 20.0)

    def test_multiple_switches_create_correct_segments(self):
        """Multiple camera switches create correct segment boundaries."""
        timeline = [
            (0.0, "doorbell"),
            (5.0, "carport"),
            (10.0, "doorbell"),
            (15.0, "carport"),
        ]
        result = convert_timeline_to_segments(timeline, 20.0)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0]["end_sec"], 5.0)
        self.assertEqual(result[1]["start_sec"], 5.0)
        self.assertEqual(result[1]["end_sec"], 10.0)

    def test_global_end_fallback_when_zero(self):
        """When global_end is 0, fallback to last point + step."""
        timeline = [
            (0.0, "doorbell"),
            (1.0, "doorbell"),
            (2.0, "doorbell"),
        ]
        result = convert_timeline_to_segments(timeline, 0.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["end_sec"], 3.0)  # 2.0 + 1.0 step


class TestAssignmentsToSlices(unittest.TestCase):
    """Tests for assignments_to_slices function."""

    def test_empty_assignments_returns_empty_list(self):
        result = assignments_to_slices([], 60.0)
        self.assertEqual(result, [])

    def test_one_assignment_creates_one_slice_with_global_end(self):
        assignments = [(0.0, "cam1")]
        result = assignments_to_slices(assignments, 10.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["camera"], "cam1")
        self.assertEqual(result[0]["start_sec"], 0.0)
        self.assertEqual(result[0]["end_sec"], 10.0)

    def test_multiple_assignments_creates_slices_with_correct_boundaries(self):
        assignments = [(0.0, "cam1"), (2.0, "cam1"), (4.0, "cam2")]
        result = assignments_to_slices(assignments, 6.0)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["start_sec"], 0.0)
        self.assertEqual(result[0]["end_sec"], 2.0)
        self.assertEqual(result[1]["start_sec"], 2.0)
        self.assertEqual(result[1]["end_sec"], 4.0)
        self.assertEqual(result[2]["start_sec"], 4.0)
        self.assertEqual(result[2]["end_sec"], 6.0)


class TestCalculateCropAtTime(unittest.TestCase):
    """Tests for calculate_crop_at_time function."""

    def test_no_entries_uses_center_crop(self):
        sidecar = {"entries": []}
        x, y, w, h = calculate_crop_at_time(sidecar, 5.0, 1920, 1080, 1440, 1080)
        self.assertEqual(w, 1440)
        self.assertEqual(h, 1080)
        self.assertEqual(x, 240)
        self.assertEqual(y, 0)

    def test_single_entry_detection_uses_weighted_center(self):
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 5.0,
                    "detections": [
                        {"centerpoint": [1000, 500], "area": 1000.0}
                    ],
                }
            ]
        }
        x, y, w, h = calculate_crop_at_time(sidecar, 5.0, 1920, 1080, 1440, 1080)
        expected_x = max(0, min(1920 - 1440, int(1000 - 1440 / 2.0)))
        expected_y = max(0, min(1080 - 1080, int(500 - 1080 / 2.0)))
        self.assertEqual(x, expected_x)
        self.assertEqual(y, expected_y)

    def test_with_timestamps_sorted_uses_bisect(self):
        sidecar = {
            "entries": [
                {"timestamp_sec": 1.0, "detections": [{"centerpoint": [100, 100], "area": 100.0}]},
                {"timestamp_sec": 3.0, "detections": [{"centerpoint": [500, 500], "area": 100.0}]},
            ]
        }
        ts_sorted = [1.0, 3.0]
        x, y, w, h = calculate_crop_at_time(
            sidecar, 2.0, 1920, 1080, 1440, 1080, timestamps_sorted=ts_sorted
        )
        self.assertEqual(w, 1440)
        self.assertEqual(h, 1080)


class TestSmoothCropCentersEma(unittest.TestCase):
    """Tests for smooth_crop_centers_ema function."""

    def test_alpha_zero_or_one_does_not_change_slices(self):
        slices = [
            {"crop_start": (100, 50, 1440, 1080), "crop_end": (200, 60, 1440, 1080)},
        ]
        orig = [dict(s) for s in slices]
        smooth_crop_centers_ema(slices, 0.0)
        self.assertEqual(slices[0]["crop_start"], orig[0]["crop_start"])
        smooth_crop_centers_ema(slices, 1.0)
        self.assertEqual(slices[0]["crop_start"], orig[0]["crop_start"])

    def test_ema_smooths_center_trajectory(self):
        slices = [
            {"crop_start": (0, 0, 100, 100), "crop_end": (10, 10, 100, 100)},
            {"crop_start": (100, 100, 100, 100), "crop_end": (110, 110, 100, 100)},
        ]
        smooth_crop_centers_ema(slices, 0.5)
        self.assertIn("crop_start", slices[0])
        self.assertIn("crop_end", slices[0])
        xs0, ys0, _, _ = slices[0]["crop_start"]
        xs1, ys1, _, _ = slices[1]["crop_start"]
        self.assertLess(xs1, 100)


class TestCalculateSegmentCrop(unittest.TestCase):
    """Tests for calculate_segment_crop function."""

    def test_no_detections_uses_center_crop(self):
        """When no detections, crop is centered on source frame."""
        segment = {"start_sec": 0.0, "end_sec": 10.0}
        sidecar = {"entries": []}
        x, y, w, h = calculate_segment_crop(segment, sidecar, 1920, 1080, 1440, 1080)
        self.assertEqual(w, 1440)
        self.assertEqual(h, 1080)
        # Should be centered
        self.assertEqual(x, 240)  # (1920 - 1440) / 2
        self.assertEqual(y, 0)    # (1080 - 1080) / 2

    def test_detections_use_weighted_center(self):
        """Detections are used for area-weighted center calculation."""
        segment = {"start_sec": 0.0, "end_sec": 10.0}
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 5.0,
                    "detections": [
                        {"centerpoint": [1000, 500], "area": 1000.0}
                    ]
                }
            ]
        }
        x, y, w, h = calculate_segment_crop(segment, sidecar, 1920, 1080, 1440, 1080)
        self.assertEqual(w, 1440)
        self.assertEqual(h, 1080)
        # Center should be near the detection centerpoint
        expected_x = int(1000 - 1440 / 2.0)
        expected_y = int(500 - 1080 / 2.0)
        self.assertEqual(x, max(0, min(1920 - 1440, expected_x)))
        self.assertEqual(y, max(0, min(1080 - 1080, expected_y)))

    def test_bbox_fallback_when_no_centerpoint(self):
        """Uses bbox when centerpoint is not available."""
        segment = {"start_sec": 0.0, "end_sec": 10.0}
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 5.0,
                    "detections": [
                        {"box": [800, 400, 1200, 600], "area": 40000.0}
                    ]
                }
            ]
        }
        x, y, w, h = calculate_segment_crop(segment, sidecar, 1920, 1080, 1440, 1080)
        # Center from bbox: (800 + 1200) / 2 = 1000, (400 + 600) / 2 = 500
        expected_x = int(1000 - 1440 / 2.0)
        expected_y = int(500 - 1080 / 2.0)
        self.assertEqual(x, max(0, min(1920 - 1440, expected_x)))
        self.assertEqual(y, max(0, min(1080 - 1080, expected_y)))

    def test_crop_clamped_to_source_bounds(self):
        """Crop box is clamped to not exceed source video boundaries."""
        segment = {"start_sec": 0.0, "end_sec": 10.0}
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 5.0,
                    "detections": [
                        {"centerpoint": [100, 100], "area": 1000.0}  # Near edge
                    ]
                }
            ]
        }
        x, y, w, h = calculate_segment_crop(segment, sidecar, 1920, 1080, 1440, 1080)
        self.assertGreaterEqual(x, 0)
        self.assertLessEqual(x + w, 1920)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(y + h, 1080)

    def test_source_smaller_than_target_scales_down(self):
        """When source is smaller than target, target is scaled down."""
        segment = {"start_sec": 0.0, "end_sec": 10.0}
        sidecar = {"entries": []}
        x, y, w, h = calculate_segment_crop(segment, sidecar, 640, 480, 1440, 1080)
        # Target should be scaled to fit within source
        self.assertLessEqual(w, 640)
        self.assertLessEqual(h, 480)


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
        self, mock_file, mock_resolve, mock_getsize, mock_rename, mock_isfile, mock_run_pynv
    ):
        """PyNv pipeline is called with tmp path; on success temp file is renamed to final output."""
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
        self.assertEqual(call_kw["tmp_output_path"], temp_path)
        self.assertEqual(call_kw["target_w"], 1440)
        self.assertEqual(call_kw["target_h"], 1080)
        mock_rename.assert_called_once_with(temp_path, output_path)

    @patch("frigate_buffer.services.video_compilation._run_pynv_compilation")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    @patch("frigate_buffer.services.video_compilation.os.rename")
    @patch("frigate_buffer.services.video_compilation.os.path.getsize")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    @patch("builtins.open", new_callable=mock_open)
    def test_pynv_pipeline_receives_slices_and_target_resolution(
        self, mock_file, mock_resolve, mock_getsize, mock_rename, mock_isfile, mock_run_pynv
    ):
        """_run_pynv_compilation is invoked with correct slices and target dimensions."""
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
        self.assertEqual(call_kw["target_w"], 1280)
        self.assertEqual(call_kw["target_h"], 720)
        self.assertEqual(len(call_kw["slices"]), 1)
        self.assertIn("crop_start", call_kw["slices"][0])
        self.assertIn("crop_end", call_kw["slices"][0])

    @patch("frigate_buffer.services.video_compilation._run_pynv_compilation")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    @patch("frigate_buffer.services.video_compilation.os.rename")
    @patch("frigate_buffer.services.video_compilation.os.path.getsize")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    @patch("builtins.open", new_callable=mock_open)
    def test_successful_completion_renames_temp_file(
        self, mock_file, mock_resolve, mock_getsize, mock_rename, mock_isfile, mock_run_pynv
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

        with self.assertRaises(RuntimeError):
            generate_compilation_video(segments, ce_dir, output_path)

    @patch("frigate_buffer.services.video_compilation._run_pynv_compilation")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    @patch("frigate_buffer.services.video_compilation.os.rename")
    @patch("frigate_buffer.services.video_compilation.os.path.getsize")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    @patch("builtins.open", new_callable=mock_open)
    def test_multiple_slices_passed_to_pynv(
        self, mock_file, mock_resolve, mock_getsize, mock_rename, mock_isfile, mock_run_pynv
    ):
        """Multiple slices are passed to _run_pynv_compilation (one segment per slice)."""
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
        self.assertEqual(len(call_kw["slices"]), 2)
        self.assertEqual(call_kw["slices"][0]["camera"], "doorbell")
        self.assertEqual(call_kw["slices"][0]["start_sec"], 0.0)
        self.assertEqual(call_kw["slices"][0]["end_sec"], 5.0)
        self.assertEqual(call_kw["slices"][1]["start_sec"], 5.0)
        self.assertEqual(call_kw["slices"][1]["end_sec"], 10.0)

    @patch("frigate_buffer.services.video_compilation._run_pynv_compilation")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    @patch("frigate_buffer.services.video_compilation.os.rename")
    @patch("frigate_buffer.services.video_compilation.os.path.getsize")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    @patch("builtins.open", new_callable=mock_open)
    def test_last_slice_of_camera_run_holds_crop(
        self, mock_file, mock_resolve, mock_getsize, mock_rename, mock_isfile, mock_run_pynv
    ):
        """Last slice before a camera switch must have crop_end == crop_start (smooth hold)."""
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
        self.assertEqual(slices[0]["crop_start"], slices[0]["crop_end"])

    @patch("frigate_buffer.services.video_compilation.GPU_LOCK", MagicMock())
    @patch("frigate_buffer.services.video_compilation.create_decoder")
    @patch("frigate_buffer.services.video_compilation._encode_frames_via_ffmpeg")
    @patch("frigate_buffer.services.video_compilation._get_video_metadata")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    def test_run_pynv_compilation_uses_metadata_fallback_when_len_zero(
        self, mock_isfile, mock_get_metadata, mock_encode_ffmpeg, mock_create_decoder
    ):
        """When decoder reports len 0, frame count uses _get_video_metadata fallback and _run_pynv_compilation completes; encode via FFmpeg."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_isfile.return_value = True
        mock_get_metadata.return_value = (640, 480, 20.0, 10.0)
        mock_ctx = _fake_create_decoder_context(frame_count=0, height=480, width=640)
        if mock_ctx is None:
            self.skipTest("torch not available")
        mock_ctx.get_index_from_time_in_seconds = lambda t: min(int(t * 20), 39)
        mock_ctx.get_frames = lambda indices: torch.zeros(
            (len(indices), 3, 480, 640), dtype=torch.uint8
        )
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_ctx)
        mock_cm.__exit__ = MagicMock(return_value=None)
        mock_create_decoder.return_value = mock_cm

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
            cuda_device_index=0,
        )

        mock_get_metadata.assert_called()
        mock_encode_ffmpeg.assert_called_once()
        args = mock_encode_ffmpeg.call_args[0]
        self.assertEqual(len(args), 4)
        self.assertEqual(args[1], 1440)
        self.assertEqual(args[2], 1080)
        self.assertEqual(args[3], "/out.mp4.tmp")

    @patch("frigate_buffer.services.video_compilation.GPU_LOCK", MagicMock())
    @patch("frigate_buffer.services.video_compilation.create_decoder")
    @patch("frigate_buffer.services.video_compilation._encode_frames_via_ffmpeg")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    def test_run_pynv_compilation_calls_ffmpeg_encode_with_h264_nvenc(
        self, mock_isfile, mock_encode_ffmpeg, mock_create_decoder
    ):
        """Compilation encode is via _encode_frames_via_ffmpeg (h264_nvenc only; no CPU fallback)."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_isfile.return_value = True
        mock_ctx = _fake_create_decoder_context(200, 480, 640)
        if mock_ctx is None:
            self.skipTest("torch not available")
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_ctx)
        mock_cm.__exit__ = MagicMock(return_value=None)
        mock_create_decoder.return_value = mock_cm

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
            cuda_device_index=0,
        )

        mock_encode_ffmpeg.assert_called_once()
        args = mock_encode_ffmpeg.call_args[0]
        self.assertEqual(args[1], 1440)
        self.assertEqual(args[2], 1080)
        self.assertEqual(args[3], "/out.mp4.tmp")

    @patch("frigate_buffer.services.video_compilation.GPU_LOCK", MagicMock())
    @patch("frigate_buffer.services.video_compilation.create_decoder")
    @patch("frigate_buffer.services.video_compilation._encode_frames_via_ffmpeg")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    def test_run_pynv_compilation_skips_slice_on_decoder_failure(
        self, mock_isfile, mock_encode_ffmpeg, mock_create_decoder
    ):
        """When create_decoder raises for a slice, that slice is skipped and encode is not called (no frames)."""
        mock_isfile.return_value = True
        mock_create_decoder.side_effect = RuntimeError("NVDEC init failed")

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
            cuda_device_index=0,
        )

        mock_encode_ffmpeg.assert_not_called()


class TestEncodeFramesViaFfmpeg(unittest.TestCase):
    """Tests for _encode_frames_via_ffmpeg: h264_nvenc only, descriptive error on failure."""

    def test_encode_uses_h264_nvenc_in_command(self):
        """FFmpeg is invoked with -c:v h264_nvenc (GPU only; no libx264) and -pix_fmt yuv420p."""
        import numpy as np
        frames = [np.zeros((1080, 1440, 3), dtype=np.uint8)]
        with patch("builtins.open", mock_open()):
            with patch("frigate_buffer.services.video_compilation.subprocess.Popen") as mock_popen:
                proc = MagicMock()
                proc.stdin = MagicMock()
                proc.returncode = 0
                mock_popen.return_value = proc
                _encode_frames_via_ffmpeg(frames, 1440, 1080, "/tmp/out.mp4")
        call_args = mock_popen.call_args[0][0]
        self.assertIn("h264_nvenc", call_args)
        self.assertNotIn("libx264", call_args)
        self.assertIn("yuv420p", call_args)

    def test_encode_uses_thread_queue_size_and_nvenc_tune(self):
        """FFmpeg command includes -thread_queue_size 512 and h264_nvenc preset/tune/rc/cq for stability."""
        import numpy as np
        frames = [np.zeros((1080, 1440, 3), dtype=np.uint8)]
        with patch("builtins.open", mock_open()):
            with patch("frigate_buffer.services.video_compilation.subprocess.Popen") as mock_popen:
                proc = MagicMock()
                proc.stdin = MagicMock()
                proc.returncode = 0
                mock_popen.return_value = proc
                _encode_frames_via_ffmpeg(frames, 1440, 1080, "/tmp/out.mp4")
        call_args = mock_popen.call_args[0][0]
        cmd_str = " ".join(call_args) if isinstance(call_args, list) else str(call_args)
        self.assertIn("thread_queue_size", cmd_str)
        self.assertIn("512", cmd_str)
        self.assertIn("p1", cmd_str)
        self.assertIn("hq", cmd_str)
        self.assertIn("vbr", cmd_str)
        self.assertIn("24", cmd_str)

    def test_encode_failure_raises_with_descriptive_message(self):
        """On non-zero exit, RuntimeError is raised advising to check ffmpeg_compile.log."""
        import numpy as np
        frames = [np.zeros((1080, 1440, 3), dtype=np.uint8)]
        with patch("builtins.open", mock_open()):
            with patch("frigate_buffer.services.video_compilation.subprocess.Popen") as mock_popen:
                proc = MagicMock()
                proc.stdin = MagicMock()
                proc.returncode = 1
                mock_popen.return_value = proc
                with self.assertRaises(RuntimeError) as ctx:
                    _encode_frames_via_ffmpeg(frames, 1440, 1080, "/tmp/out.mp4")
        self.assertIn("h264_nvenc", str(ctx.exception))
        self.assertIn("ffmpeg_compile.log", str(ctx.exception))

    def test_broken_pipe_logs_and_raises_with_log_path(self):
        """When FFmpeg crashes and closes stdin, BrokenPipeError is caught; RuntimeError advises checking ffmpeg_compile.log."""
        import numpy as np
        frames = [np.zeros((1080, 1440, 3), dtype=np.uint8)]
        with patch("builtins.open", mock_open()):
            with patch("frigate_buffer.services.video_compilation.subprocess.Popen") as mock_popen:
                proc = MagicMock()
                proc.stdin = MagicMock()
                proc.stdin.write.side_effect = BrokenPipeError
                mock_popen.return_value = proc
                with self.assertRaises(RuntimeError) as ctx:
                    _encode_frames_via_ffmpeg(frames, 1440, 1080, "/tmp/out.mp4")
        self.assertIn("broke pipe", str(ctx.exception))
        self.assertIn("ffmpeg_compile.log", str(ctx.exception))
        proc.wait.assert_called_once()


class TestCompileCeVideoConfig(unittest.TestCase):
    """Tests that compile_ce_video uses same timeline config as frame timeline."""

    @patch("frigate_buffer.services.video_compilation.generate_compilation_video")
    @patch("frigate_buffer.services.video_compilation.timeline_ema.build_phase1_assignments")
    @patch("frigate_buffer.services.video_compilation.timeline_ema.build_dense_times")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    def test_uses_max_multi_cam_frames_config_keys(
        self, mock_resolve, mock_build_dense, mock_build_assignments, mock_gen
    ):
        """compile_ce_video must use MAX_MULTI_CAM_FRAMES_SEC and MAX_MULTI_CAM_FRAMES_MIN."""
        mock_resolve.return_value = "cam1.mp4"
        mock_build_dense.return_value = [0.0, 1.0, 2.0]
        mock_build_assignments.return_value = [(0.0, "cam1"), (1.0, "cam1"), (2.0, "cam1")]
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
        with patch("frigate_buffer.services.video_compilation.os.scandir") as mock_scandir:
            with patch("frigate_buffer.services.video_compilation.os.path.isfile", return_value=True):
                with patch("builtins.open", mock_open(read_data=json.dumps({"entries": [], "native_width": 1920, "native_height": 1080}))):
                    mock_ent = MagicMock()
                    mock_ent.is_dir.return_value = True
                    mock_ent.name = "cam1"
                    mock_ent.path = "/ce/cam1"
                    mock_scandir.return_value.__enter__.return_value = [mock_ent]
                    compile_ce_video("/ce", 3.0, config, None)
        call_args = mock_build_dense.call_args[0]
        self.assertEqual(call_args[0], 2.5)
        self.assertEqual(call_args[1], 50)


class TestCompilationDecoderImport(unittest.TestCase):
    """Compilation uses gpu_decoder.create_decoder (PyNvVideoCodec); this test guards the import."""

    def test_video_compilation_imports_create_decoder(self):
        """video_compilation module uses create_decoder from gpu_decoder for decode."""
        from frigate_buffer.services import video_compilation
        self.assertTrue(hasattr(video_compilation, "create_decoder") or "create_decoder" in dir(video_compilation))
        from frigate_buffer.services.gpu_decoder import create_decoder
        self.assertIs(video_compilation.create_decoder, create_decoder)


if __name__ == "__main__":
    unittest.main()
