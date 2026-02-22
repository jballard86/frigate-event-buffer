"""Tests for video_compilation service."""

import json
import os
import unittest
from unittest.mock import MagicMock, patch, mock_open

from frigate_buffer.services.video_compilation import (
    assignments_to_slices,
    calculate_crop_at_time,
    calculate_segment_crop,
    convert_timeline_to_segments,
    generate_compilation_video,
    smooth_crop_centers_ema,
)


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
    """Tests for generate_compilation_video function."""

    @patch("frigate_buffer.services.video_compilation.subprocess.run")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    @patch("frigate_buffer.services.video_compilation.os.rename")
    @patch("frigate_buffer.services.video_compilation.os.path.getsize")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    @patch("builtins.open", new_callable=mock_open)
    def test_ffmpeg_command_includes_format_flag(
        self, mock_file, mock_resolve, mock_getsize, mock_rename, mock_isfile, mock_run
    ):
        """FFmpeg command must include -f mp4 to handle .tmp extension output."""
        mock_resolve.return_value = "doorbell-123.mp4"
        mock_isfile.return_value = True
        mock_getsize.return_value = 1000

        sidecar_data = {
            "native_width": 2560,
            "native_height": 1920,
            "entries": []
        }
        mock_file.return_value.read.return_value = json.dumps(sidecar_data)

        segments = [
            {"camera": "doorbell", "start_sec": 0.0, "end_sec": 10.0}
        ]
        ce_dir = "/app/storage/events/test36"
        output_path = "/app/storage/events/test36/test36_summary.mp4"

        mock_run.return_value = MagicMock(returncode=0)

        generate_compilation_video(segments, ce_dir, output_path)

        # Verify subprocess.run was called
        self.assertTrue(mock_run.called)

        # Get the command passed to subprocess.run
        call_args = mock_run.call_args
        cmd = call_args[0][0]

        # Verify -f mp4 is in the command
        self.assertIn("-f", cmd)
        mp4_index = cmd.index("-f")
        self.assertEqual(cmd[mp4_index + 1], "mp4")

        # Verify -f mp4 comes before the output path
        tmp_output_path = output_path + ".tmp"
        tmp_path_index = cmd.index(tmp_output_path)
        self.assertLess(mp4_index, tmp_path_index)

    @patch("frigate_buffer.services.video_compilation.subprocess.run")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    @patch("frigate_buffer.services.video_compilation.os.rename")
    @patch("frigate_buffer.services.video_compilation.os.path.getsize")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    @patch("builtins.open", new_callable=mock_open)
    def test_ffmpeg_filter_includes_format_standardization(
        self, mock_file, mock_resolve, mock_getsize, mock_rename, mock_isfile, mock_run
    ):
        """Filter chain must include format=yuv420p to standardize pixel formats before concat."""
        mock_resolve.return_value = "doorbell-123.mp4"
        mock_isfile.return_value = True
        mock_getsize.return_value = 1000

        sidecar_data = {
            "native_width": 2560,
            "native_height": 1920,
            "entries": []
        }
        mock_file.return_value.read.return_value = json.dumps(sidecar_data)

        segments = [
            {"camera": "doorbell", "start_sec": 0.0, "end_sec": 10.0}
        ]
        ce_dir = "/app/storage/events/test36"
        output_path = "/app/storage/events/test36/test36_summary.mp4"

        mock_run.return_value = MagicMock(returncode=0)

        generate_compilation_video(segments, ce_dir, output_path)

        cmd = mock_run.call_args[0][0]

        # Get the filter_complex value
        filter_complex_idx = cmd.index("-filter_complex")
        filter_complex = cmd[filter_complex_idx + 1]

        # Verify format=yuv420p is in the filter chain
        self.assertIn("format=yuv420p", filter_complex)

        # Verify it appears before the stream label [v0]
        format_idx = filter_complex.index("format=yuv420p")
        label_idx = filter_complex.index("[v0]")
        self.assertLess(format_idx, label_idx)

    @patch("frigate_buffer.services.video_compilation.subprocess.run")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    @patch("frigate_buffer.services.video_compilation.os.rename")
    @patch("frigate_buffer.services.video_compilation.os.path.getsize")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    @patch("builtins.open", new_callable=mock_open)
    def test_ffmpeg_command_structure(
        self, mock_file, mock_resolve, mock_getsize, mock_rename, mock_isfile, mock_run
    ):
        """FFmpeg command has correct structure with expected flags."""
        mock_resolve.return_value = "doorbell-123.mp4"
        mock_isfile.return_value = True
        mock_getsize.return_value = 1000

        sidecar_data = {
            "native_width": 2560,
            "native_height": 1920,
            "entries": []
        }
        mock_file.return_value.read.return_value = json.dumps(sidecar_data)

        segments = [
            {"camera": "doorbell", "start_sec": 0.0, "end_sec": 10.0}
        ]
        ce_dir = "/app/storage/events/test36"
        output_path = "/app/storage/events/test36/test36_summary.mp4"

        mock_run.return_value = MagicMock(returncode=0)

        generate_compilation_video(segments, ce_dir, output_path)

        cmd = mock_run.call_args[0][0]

        # Check expected flags are present (software decode, GPU encode)
        self.assertEqual(cmd[0], "ffmpeg")
        self.assertIn("-y", cmd)
        self.assertIn("-filter_complex", cmd)
        self.assertIn("-map", cmd)
        self.assertIn("[outv]", cmd)
        self.assertIn("-an", cmd)
        self.assertIn("-c:v", cmd)
        self.assertIn("h264_nvenc", cmd)
        self.assertIn("-pix_fmt", cmd)
        self.assertIn("yuv420p", cmd)
        self.assertIn("-f", cmd)
        self.assertIn("mp4", cmd)

    @patch("frigate_buffer.services.video_compilation.subprocess.run")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    @patch("frigate_buffer.services.video_compilation.os.rename")
    @patch("frigate_buffer.services.video_compilation.os.path.getsize")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    @patch("builtins.open", new_callable=mock_open)
    def test_successful_completion_renames_temp_file(
        self, mock_file, mock_resolve, mock_getsize, mock_rename, mock_isfile, mock_run
    ):
        """On success, temp file is renamed to final output path."""
        mock_resolve.return_value = "doorbell-123.mp4"
        mock_isfile.return_value = True
        mock_getsize.return_value = 1000

        sidecar_data = {
            "native_width": 2560,
            "native_height": 1920,
            "entries": []
        }
        mock_file.return_value.read.return_value = json.dumps(sidecar_data)

        segments = [
            {"camera": "doorbell", "start_sec": 0.0, "end_sec": 10.0}
        ]
        ce_dir = "/app/storage/events/test36"
        output_path = "/app/storage/events/test36/test36_summary.mp4"

        mock_run.return_value = MagicMock(returncode=0)

        generate_compilation_video(segments, ce_dir, output_path)

        # Verify rename was called with temp -> final
        tmp_output_path = output_path + ".tmp"
        mock_rename.assert_called_once_with(tmp_output_path, output_path)

    @patch("frigate_buffer.services.video_compilation.subprocess.run")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    @patch("builtins.open", new_callable=mock_open)
    def test_ffmpeg_failure_raises_exception(
        self, mock_file, mock_resolve, mock_isfile, mock_run
    ):
        """FFmpeg failure should raise CalledProcessError."""
        mock_resolve.return_value = "doorbell-123.mp4"
        mock_isfile.return_value = True

        sidecar_data = {
            "native_width": 2560,
            "native_height": 1920,
            "entries": []
        }
        mock_file.return_value.read.return_value = json.dumps(sidecar_data)

        segments = [
            {"camera": "doorbell", "start_sec": 0.0, "end_sec": 10.0}
        ]
        ce_dir = "/app/storage/events/test36"
        output_path = "/app/storage/events/test36/test36_summary.mp4"

        from subprocess import CalledProcessError
        mock_run.side_effect = CalledProcessError(1, cmd="ffmpeg", stderr="FFmpeg error")

        with self.assertRaises(CalledProcessError):
            generate_compilation_video(segments, ce_dir, output_path)

    @patch("frigate_buffer.services.video_compilation.subprocess.run")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    @patch("frigate_buffer.services.video_compilation.os.rename")
    @patch("frigate_buffer.services.video_compilation.os.path.getsize")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    @patch("builtins.open", new_callable=mock_open)
    def test_one_input_per_slice_and_zero_based_crop_expression(
        self, mock_file, mock_resolve, mock_getsize, mock_rename, mock_isfile, mock_run
    ):
        """Filter must use one input index per slice and 0-based t/duration in crop."""
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
        mock_run.return_value = MagicMock(returncode=0)
        generate_compilation_video(slices, ce_dir, output_path)
        cmd = mock_run.call_args[0][0]
        filter_complex_idx = cmd.index("-filter_complex")
        filter_complex = cmd[filter_complex_idx + 1]
        self.assertIn("[0:v]", filter_complex)
        self.assertIn("[1:v]", filter_complex)
        self.assertIn("(t/", filter_complex)
        self.assertNotIn("(t-", filter_complex)
        # Crop x/y expressions must be single-quoted so commas in min(max(...),...) are literal
        self.assertIn("'min(max(0,", filter_complex)

    @patch("frigate_buffer.services.video_compilation.subprocess.run")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    @patch("frigate_buffer.services.video_compilation.os.rename")
    @patch("frigate_buffer.services.video_compilation.os.path.getsize")
    @patch("frigate_buffer.services.query.resolve_clip_in_folder")
    @patch("builtins.open", new_callable=mock_open)
    def test_last_slice_of_camera_run_holds_crop(
        self, mock_file, mock_resolve, mock_getsize, mock_rename, mock_isfile, mock_run
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
        mock_run.return_value = MagicMock(returncode=0)
        generate_compilation_video(slices, ce_dir, output_path)
        self.assertEqual(slices[0]["crop_start"], slices[0]["crop_end"])


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


if __name__ == "__main__":
    unittest.main()
