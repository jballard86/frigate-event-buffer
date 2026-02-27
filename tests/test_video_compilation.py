"""Tests for video_compilation service."""

import json
import unittest
from unittest.mock import MagicMock, mock_open, patch

import pytest

from frigate_buffer.services.video_compilation import (
    _encode_frames_via_ffmpeg,
    _run_pynv_compilation,
    _trim_slices_to_action_window,
    assignments_to_slices,
    calculate_crop_at_time,
    calculate_segment_crop,
    convert_timeline_to_segments,
    generate_compilation_video,
    smooth_crop_centers_ema,
    smooth_zoom_ema,
)


def _fake_create_decoder_context(
    frame_count: int = 200, height: int = 480, width: int = 640
):
    """Build a mock DecoderContext for _run_pynv_compilation tests."""
    try:
        import torch
    except ImportError:
        return None
    mock_ctx = MagicMock()
    mock_ctx.__len__ = lambda self: frame_count
    # Called as ctx.get_index_from_time_in_seconds(t) -> one arg t
    mock_ctx.get_index_from_time_in_seconds = lambda t: min(
        max(0, int(t * 20)), max(0, frame_count - 1)
    )
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
        assert result == []

    def test_single_camera_continuous_timeline(self):
        """Single camera throughout creates one segment."""
        timeline = [
            (0.0, "doorbell"),
            (1.0, "doorbell"),
            (2.0, "doorbell"),
        ]
        result = convert_timeline_to_segments(timeline, 10.0)
        assert len(result) == 1
        assert result[0]["camera"] == "doorbell"
        assert result[0]["start_sec"] == 0.0
        assert result[0]["end_sec"] == 10.0

    def test_camera_switch_creates_multiple_segments(self):
        """Camera switches create multiple segments."""
        timeline = [
            (0.0, "doorbell"),
            (5.0, "doorbell"),
            (10.0, "carport"),
            (15.0, "carport"),
        ]
        result = convert_timeline_to_segments(timeline, 20.0)
        assert len(result) == 2
        assert result[0]["camera"] == "doorbell"
        assert result[0]["start_sec"] == 0.0
        assert result[0]["end_sec"] == 10.0
        assert result[1]["camera"] == "carport"
        assert result[1]["start_sec"] == 10.0
        assert result[1]["end_sec"] == 20.0

    def test_multiple_switches_create_correct_segments(self):
        """Multiple camera switches create correct segment boundaries."""
        timeline = [
            (0.0, "doorbell"),
            (5.0, "carport"),
            (10.0, "doorbell"),
            (15.0, "carport"),
        ]
        result = convert_timeline_to_segments(timeline, 20.0)
        assert len(result) == 4
        assert result[0]["end_sec"] == 5.0
        assert result[1]["start_sec"] == 5.0
        assert result[1]["end_sec"] == 10.0

    def test_global_end_fallback_when_zero(self):
        """When global_end is 0, fallback to last point + step."""
        timeline = [
            (0.0, "doorbell"),
            (1.0, "doorbell"),
            (2.0, "doorbell"),
        ]
        result = convert_timeline_to_segments(timeline, 0.0)
        assert len(result) == 1
        assert result[0]["end_sec"] == 3.0  # 2.0 + 1.0 step


class TestAssignmentsToSlices(unittest.TestCase):
    """Tests for assignments_to_slices function."""

    def test_empty_assignments_returns_empty_list(self):
        result = assignments_to_slices([], 60.0)
        assert result == []

    def test_one_assignment_creates_one_slice_with_global_end(self):
        assignments = [(0.0, "cam1")]
        result = assignments_to_slices(assignments, 10.0)
        assert len(result) == 1
        assert result[0]["camera"] == "cam1"
        assert result[0]["start_sec"] == 0.0
        assert result[0]["end_sec"] == 10.0

    def test_multiple_assignments_creates_slices_with_correct_boundaries(self):
        assignments = [(0.0, "cam1"), (2.0, "cam1"), (4.0, "cam2")]
        result = assignments_to_slices(assignments, 6.0)
        assert len(result) == 3
        assert result[0]["start_sec"] == 0.0
        assert result[0]["end_sec"] == 2.0
        assert result[1]["start_sec"] == 2.0
        assert result[1]["end_sec"] == 4.0
        assert result[2]["start_sec"] == 4.0
        assert result[2]["end_sec"] == 6.0


class TestTrimSlicesToActionWindow(unittest.TestCase):
    """Tests for _trim_slices_to_action_window (trim to first/last detection ± roll)."""

    def test_trims_to_action_window_and_clamps_overlapping_slices(self):
        """First 2s, last 10s, end=20 → [0,13]; outside dropped, overlapping clamped."""
        sidecars = {
            "cam1": [
                {
                    "timestamp_sec": 2.0,
                    "detections": [{"centerpoint": [100, 100], "area": 1.0}],
                },
                {
                    "timestamp_sec": 10.0,
                    "detections": [{"centerpoint": [200, 200], "area": 1.0}],
                },
            ],
        }
        slices = [
            {"camera": "cam1", "start_sec": 0.0, "end_sec": 5.0},
            {"camera": "cam1", "start_sec": 5.0, "end_sec": 15.0},
            {"camera": "cam1", "start_sec": 15.0, "end_sec": 20.0},
        ]
        result = _trim_slices_to_action_window(slices, sidecars, global_end=20.0)
        # action_start = max(0, 2-3)=0, action_end = min(20, 10+3)=13
        assert len(result) == 2
        assert result[0]["camera"] == "cam1"
        assert result[0]["start_sec"] == 0.0
        assert result[0]["end_sec"] == 5.0
        assert result[1]["start_sec"] == 5.0
        assert result[1]["end_sec"] == 13.0
        # Slice [15, 20] is entirely outside [0, 13] so dropped

    def test_no_detections_keeps_full_range(self):
        """When no sidecar has detections, no trimming (full 0..global_end)."""
        sidecars = {
            "cam1": [
                {"timestamp_sec": 1.0, "detections": []},
                {"timestamp_sec": 5.0, "detections": []},
            ]
        }
        slices = [{"camera": "cam1", "start_sec": 0.0, "end_sec": 10.0}]
        result = _trim_slices_to_action_window(slices, sidecars, global_end=10.0)
        assert len(result) == 1
        assert result[0]["start_sec"] == 0.0
        assert result[0]["end_sec"] == 10.0

    def test_discards_slice_entirely_after_action_end(self):
        """Slice entirely after action_end is discarded."""
        sidecars = {"cam1": [{"timestamp_sec": 1.0, "detections": [{"area": 1.0}]}]}
        slices = [{"camera": "cam1", "start_sec": 20.0, "end_sec": 30.0}]
        result = _trim_slices_to_action_window(slices, sidecars, global_end=30.0)
        assert len(result) == 0

    def test_discards_slice_entirely_before_action_start(self):
        """Slice entirely before action_start discarded when first detection is late."""
        sidecars = {"cam1": [{"timestamp_sec": 10.0, "detections": [{"area": 1.0}]}]}
        slices = [
            {"camera": "cam1", "start_sec": 0.0, "end_sec": 5.0},
            {"camera": "cam1", "start_sec": 7.0, "end_sec": 15.0},
        ]
        result = _trim_slices_to_action_window(slices, sidecars, global_end=20.0)
        # action_start = 10-3 = 7, action_end = 13.
        # First slice [0,5] entirely before 7 → dropped.
        assert len(result) == 1
        assert result[0]["start_sec"] == 7.0
        assert result[0]["end_sec"] == 13.0


class TestCalculateCropAtTime(unittest.TestCase):
    """Tests for calculate_crop_at_time function."""

    def test_no_entries_uses_center_crop(self):
        sidecar = {"entries": []}
        x, y, w, h = calculate_crop_at_time(sidecar, 5.0, 1920, 1080, 1440, 1080)
        assert w == 1440
        assert h == 1080
        assert x == 240
        assert y == 0

    def test_single_entry_detection_uses_weighted_center(self):
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 5.0,
                    "detections": [{"centerpoint": [1000, 500], "area": 1000.0}],
                }
            ]
        }
        x, y, w, h = calculate_crop_at_time(sidecar, 5.0, 1920, 1080, 1440, 1080)
        expected_x = max(0, min(1920 - 1440, int(1000 - 1440 / 2.0)))
        expected_y = max(0, min(1080 - 1080, int(500 - 1080 / 2.0)))
        assert x == expected_x
        assert y == expected_y

    def test_with_timestamps_sorted_uses_bisect(self):
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 1.0,
                    "detections": [{"centerpoint": [100, 100], "area": 100.0}],
                },
                {
                    "timestamp_sec": 3.0,
                    "detections": [{"centerpoint": [500, 500], "area": 100.0}],
                },
            ]
        }
        ts_sorted = [1.0, 3.0]
        x, y, w, h = calculate_crop_at_time(
            sidecar, 2.0, 1920, 1080, 1440, 1080, timestamps_sorted=ts_sorted
        )
        assert w == 1440
        assert h == 1080

    def test_hold_last_known_crop_when_nearest_has_no_detections(self):
        """When person leaves at t=4, query at t=6: nearest (t=6) has no detections;
        entry at t=4 has detections within 5s → crop holds at t=4 position."""
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 4.0,
                    "detections": [{"centerpoint": [800, 400], "area": 500.0}],
                },
                {"timestamp_sec": 5.0, "detections": []},
                {"timestamp_sec": 6.0, "detections": []},
            ]
        }
        x, y, w, h = calculate_crop_at_time(sidecar, 6.0, 1920, 1080, 1440, 1080)
        # Should use t=4 entry: center (800, 400), not frame center (960, 540)
        expected_x = max(0, min(1920 - 1440, int(800 - 1440 / 2.0)))
        expected_y = max(0, min(1080 - 1080, int(400 - 1080 / 2.0)))
        assert x == expected_x
        assert y == expected_y
        assert w == 1440
        assert h == 1080

    def test_hold_crop_fallback_to_center_when_no_detections_within_threshold(self):
        """When no entry with detections exists within 5s, crop remains center."""
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 0.0,
                    "detections": [{"centerpoint": [200, 200], "area": 100.0}],
                },
                {"timestamp_sec": 10.0, "detections": []},
            ]
        }
        x, y, w, h = calculate_crop_at_time(sidecar, 10.0, 1920, 1080, 1440, 1080)
        # t=0 is > 5s away from t=10, so fallback to center
        assert x == 240
        assert y == 0
        assert w == 1440
        assert h == 1080

    def test_nearest_entry_has_detections_unchanged(self):
        """Nearest entry has detections → behavior unchanged (no regression)."""
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 4.0,
                    "detections": [{"centerpoint": [800, 400], "area": 500.0}],
                },
                {
                    "timestamp_sec": 6.0,
                    "detections": [{"centerpoint": [1000, 500], "area": 500.0}],
                },
            ]
        }
        x, y, w, h = calculate_crop_at_time(sidecar, 5.5, 1920, 1080, 1440, 1080)
        # Nearest is t=6 (distance 0.5) with detections at (1000, 500)
        expected_x = max(0, min(1920 - 1440, int(1000 - 1440 / 2.0)))
        expected_y = max(0, min(1080 - 1080, int(500 - 1080 / 2.0)))
        assert x == expected_x
        assert y == expected_y

    def test_zoom_uses_content_area_when_tracking_target_frame_percent_set(self):
        """tracking_target_frame_percent>0 + detections → crop from content, target%."""
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 1.0,
                    "detections": [{"bbox": [100, 100, 200, 200], "area": 10000.0}],
                }
            ]
        }
        x, y, w, h = calculate_crop_at_time(
            sidecar,
            1.0,
            1920,
            1080,
            1440,
            1080,
            tracking_target_frame_percent=40,
        )
        # Content = bbox + 10% padding; crop area = content/0.4; aspect 1440/1080.
        # Crop (w,h) clamped to [0.4*1920,1920] and [0.4*1080,1080].
        assert w > 0
        assert w <= 1920
        assert h > 0
        assert h <= 1080
        assert w % 2 == 0
        assert h % 2 == 0
        assert w >= int(1920 * 0.4)
        assert h >= int(1080 * 0.4)
        assert x >= 0
        assert x + w <= 1920
        assert y >= 0
        assert y + h <= 1080

    def test_no_zoom_when_tracking_target_frame_percent_zero(self):
        """When tracking_target_frame_percent=0, crop is fixed target_w x target_h."""
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 1.0,
                    "detections": [{"bbox": [100, 100, 200, 200], "area": 10000.0}],
                }
            ]
        }
        x, y, w, h = calculate_crop_at_time(
            sidecar,
            1.0,
            1920,
            1080,
            1440,
            1080,
            tracking_target_frame_percent=0,
        )
        assert w == 1440
        assert h == 1080


class TestSmoothCropCentersEma(unittest.TestCase):
    """Tests for smooth_crop_centers_ema function."""

    def test_alpha_zero_or_one_does_not_change_slices(self):
        slices = [
            {"crop_start": (100, 50, 1440, 1080), "crop_end": (200, 60, 1440, 1080)},
        ]
        orig = [dict(s) for s in slices]
        smooth_crop_centers_ema(slices, 0.0)
        assert slices[0]["crop_start"] == orig[0]["crop_start"]
        smooth_crop_centers_ema(slices, 1.0)
        assert slices[0]["crop_start"] == orig[0]["crop_start"]

    def test_ema_smooths_center_trajectory(self):
        slices = [
            {"crop_start": (0, 0, 100, 100), "crop_end": (10, 10, 100, 100)},
            {"crop_start": (100, 100, 100, 100), "crop_end": (110, 110, 100, 100)},
        ]
        smooth_crop_centers_ema(slices, 0.5)
        assert "crop_start" in slices[0]
        assert "crop_end" in slices[0]
        xs0, ys0, _, _ = slices[0]["crop_start"]
        xs1, ys1, _, _ = slices[1]["crop_start"]
        assert xs1 < 100


class TestSmoothZoomEma(unittest.TestCase):
    """Tests for smooth_zoom_ema function."""

    def test_alpha_zero_or_one_does_not_change_zoom(self):
        """Alpha 0 or 1 leaves zoom dimensions unchanged (or no-op)."""
        slices = [
            {
                "crop_start": (0, 0, 800, 450),
                "crop_end": (0, 0, 800, 450),
                "native_width": 1920,
                "native_height": 1080,
            },
        ]
        orig_w, orig_h = slices[0]["crop_start"][2], slices[0]["crop_start"][3]
        smooth_zoom_ema(slices, 0.0, 1440, 1080)
        assert slices[0]["crop_start"][2] == orig_w
        assert slices[0]["crop_start"][3] == orig_h
        smooth_zoom_ema(slices, 1.0, 1440, 1080)
        assert slices[0]["crop_start"][2] == orig_w
        assert slices[0]["crop_start"][3] == orig_h

    def test_smooth_zoom_ema_updates_crop_size_in_place(self):
        """smooth_zoom_ema updates (w,h) per slice; dimensions even and in bounds."""
        slices = [
            {
                "crop_start": (100, 50, 960, 540),
                "crop_end": (120, 60, 960, 540),
                "native_width": 1920,
                "native_height": 1080,
            },
            {
                "crop_start": (200, 100, 800, 450),
                "crop_end": (220, 110, 800, 450),
                "native_width": 1920,
                "native_height": 1080,
            },
        ]
        smooth_zoom_ema(slices, 0.3, 1440, 1080)
        for sl in slices:
            xs, ys, w, h = sl["crop_start"]
            assert w > 0
            assert w <= 1920
            assert h > 0
            assert h <= 1080
            assert w % 2 == 0
            assert h % 2 == 0
            assert xs >= 0
            assert xs + w <= 1920
            assert ys >= 0
            assert ys + h <= 1080


class TestCalculateSegmentCrop(unittest.TestCase):
    """Tests for calculate_segment_crop function."""

    def test_no_detections_uses_center_crop(self):
        """When no detections, crop is centered on source frame."""
        segment = {"start_sec": 0.0, "end_sec": 10.0}
        sidecar = {"entries": []}
        x, y, w, h = calculate_segment_crop(segment, sidecar, 1920, 1080, 1440, 1080)
        assert w == 1440
        assert h == 1080
        # Should be centered
        assert x == 240  # (1920 - 1440) / 2
        assert y == 0  # (1080 - 1080) / 2

    def test_detections_use_weighted_center(self):
        """Detections are used for area-weighted center calculation."""
        segment = {"start_sec": 0.0, "end_sec": 10.0}
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 5.0,
                    "detections": [{"centerpoint": [1000, 500], "area": 1000.0}],
                }
            ]
        }
        x, y, w, h = calculate_segment_crop(segment, sidecar, 1920, 1080, 1440, 1080)
        assert w == 1440
        assert h == 1080
        # Center should be near the detection centerpoint
        expected_x = int(1000 - 1440 / 2.0)
        expected_y = int(500 - 1080 / 2.0)
        assert x == max(0, min(1920 - 1440, expected_x))
        assert y == max(0, min(1080 - 1080, expected_y))

    def test_bbox_fallback_when_no_centerpoint(self):
        """Uses bbox when centerpoint is not available."""
        segment = {"start_sec": 0.0, "end_sec": 10.0}
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 5.0,
                    "detections": [{"box": [800, 400, 1200, 600], "area": 40000.0}],
                }
            ]
        }
        x, y, w, h = calculate_segment_crop(segment, sidecar, 1920, 1080, 1440, 1080)
        # Center from bbox: (800 + 1200) / 2 = 1000, (400 + 600) / 2 = 500
        expected_x = int(1000 - 1440 / 2.0)
        expected_y = int(500 - 1080 / 2.0)
        assert x == max(0, min(1920 - 1440, expected_x))
        assert y == max(0, min(1080 - 1080, expected_y))

    def test_crop_clamped_to_source_bounds(self):
        """Crop box is clamped to not exceed source video boundaries."""
        segment = {"start_sec": 0.0, "end_sec": 10.0}
        sidecar = {
            "entries": [
                {
                    "timestamp_sec": 5.0,
                    "detections": [
                        {"centerpoint": [100, 100], "area": 1000.0}  # Near edge
                    ],
                }
            ]
        }
        x, y, w, h = calculate_segment_crop(segment, sidecar, 1920, 1080, 1440, 1080)
        assert x >= 0
        assert x + w <= 1920
        assert y >= 0
        assert y + h <= 1080

    def test_source_smaller_than_target_scales_down(self):
        """When source is smaller than target, target is scaled down."""
        segment = {"start_sec": 0.0, "end_sec": 10.0}
        sidecar = {"entries": []}
        x, y, w, h = calculate_segment_crop(segment, sidecar, 640, 480, 1440, 1080)
        # Target should be scaled to fit within source
        assert w <= 640
        assert h <= 480


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
    @patch("frigate_buffer.services.video_compilation.create_decoder")
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
        mock_create_decoder,
    ):
        """Decoder len 0 → _get_video_metadata; _run_pynv completes; FFmpeg encode."""
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
            cuda_device_index=0,
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
    @patch("frigate_buffer.services.video_compilation.create_decoder")
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
        mock_create_decoder,
    ):
        """Compilation encode via FFmpeg (h264_nvenc; no CPU fallback)."""
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch not available")
        mock_isfile.return_value = True
        mock_get_metadata.return_value = (640, 480, 20.0, 2.0)
        mock_ctx = _fake_create_decoder_context(200, 480, 640)
        if mock_ctx is None:
            self.skipTest("torch not available")
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_ctx)
        mock_cm.__exit__ = MagicMock(return_value=None)
        mock_create_decoder.return_value = mock_cm
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
            cuda_device_index=0,
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
    @patch("frigate_buffer.services.video_compilation.create_decoder")
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
        mock_create_decoder,
    ):
        """create_decoder raises → slice skipped; FFmpeg opened, no frames written."""
        mock_isfile.return_value = True
        mock_get_metadata.return_value = (640, 480, 20.0, 2.0)
        mock_create_decoder.side_effect = RuntimeError("NVDEC init failed")
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
            cuda_device_index=0,
        )

        mock_popen.assert_called_once()
        assert proc.stdin.write.call_count == 0
        proc.stdin.close.assert_called_once()
        proc.wait.assert_called()

    @patch("frigate_buffer.services.video_compilation.GPU_LOCK", MagicMock())
    @patch("frigate_buffer.services.video_compilation.create_decoder")
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
        mock_create_decoder,
    ):
        """get_frames returns 0 → slice skipped; log clarifies not hardware init."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_isfile.return_value = True
        mock_get_metadata.return_value = (640, 480, 20.0, 2.0)
        mock_ctx = _fake_create_decoder_context(200, height=480, width=640)
        if mock_ctx is None:
            self.skipTest("torch not available")
        mock_ctx.get_frames = lambda indices: torch.empty(
            0, 3, 480, 640, dtype=torch.uint8
        )
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_ctx)
        mock_cm.__exit__ = MagicMock(return_value=None)
        mock_create_decoder.return_value = mock_cm
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
            cuda_device_index=0,
        )

        assert proc.stdin.write.call_count == 0
        error_calls = [c for c in mock_logger.error.call_args_list if c[0]]
        assert len(error_calls) > 0
        msg = str(error_calls[0][0][0])
        assert "0 frames" in msg
        assert "Skipping chunk" in msg

    @patch("frigate_buffer.services.video_compilation.GPU_LOCK", MagicMock())
    @patch("frigate_buffer.services.video_compilation.create_decoder")
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
        mock_create_decoder,
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

        mock_ctx = _fake_create_decoder_context(200, height=480, width=640)
        if mock_ctx is None:
            self.skipTest("torch not available")
        mock_ctx.get_frames = record_get_frames
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_ctx)
        mock_cm.__exit__ = MagicMock(return_value=None)
        mock_create_decoder.return_value = mock_cm
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

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache", MagicMock()) as mock_empty_cache,
        ):
            _run_pynv_compilation(
                slices=slices,
                ce_dir="/ce",
                tmp_output_path="/out.mp4.tmp",
                target_w=1440,
                target_h=1080,
                resolve_clip_in_folder=resolve_clip,
                cuda_device_index=0,
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
    @patch("frigate_buffer.services.video_compilation.create_decoder")
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
        mock_create_decoder,
    ):
        """get_frames fewer than requested → repeat last frame; no IndexError."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_isfile.return_value = True
        mock_get_metadata.return_value = (640, 480, 20.0, 2.0)
        mock_ctx = _fake_create_decoder_context(200, height=480, width=640)
        if mock_ctx is None:
            self.skipTest("torch not available")
        mock_ctx.get_frames = lambda indices: torch.zeros(
            (1, 3, 480, 640), dtype=torch.uint8
        )
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_ctx)
        mock_cm.__exit__ = MagicMock(return_value=None)
        mock_create_decoder.return_value = mock_cm
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
            cuda_device_index=0,
        )

        n_frames_expected = 40
        assert proc.stdin.write.call_count == n_frames_expected

    @patch("frigate_buffer.services.video_compilation.GPU_LOCK", MagicMock())
    @patch("frigate_buffer.services.video_compilation.create_decoder")
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
        mock_create_decoder,
    ):
        """Decoder fewer than requested → DEBUG + one INFO per camera per event."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_isfile.return_value = True
        mock_get_metadata.return_value = (640, 480, 20.0, 2.0)
        mock_ctx = _fake_create_decoder_context(200, height=480, width=640)
        if mock_ctx is None:
            self.skipTest("torch not available")
        mock_ctx.get_frames = lambda indices: torch.zeros(
            (1, 3, 480, 640), dtype=torch.uint8
        )
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_ctx)
        mock_cm.__exit__ = MagicMock(return_value=None)
        mock_create_decoder.return_value = mock_cm
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
            cuda_device_index=0,
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
    @patch("frigate_buffer.services.video_compilation.create_decoder")
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
        mock_create_decoder,
    ):
        """_run_pynv uses crop_around_center_to_size; frames to FFmpeg at target res."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_isfile.return_value = True
        mock_get_metadata.return_value = (640, 480, 20.0, 2.0)
        # Decoder returns 640x480; slice has native 1920x1080 so scaling is applied.
        mock_ctx = _fake_create_decoder_context(200, height=480, width=640)
        if mock_ctx is None:
            self.skipTest("torch not available")
        mock_ctx.get_frames = lambda indices: torch.zeros(
            (len(indices), 3, 480, 640), dtype=torch.uint8
        )
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_ctx)
        mock_cm.__exit__ = MagicMock(return_value=None)
        mock_create_decoder.return_value = mock_cm
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
            cuda_device_index=0,
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
    @patch("frigate_buffer.services.video_compilation.create_decoder")
    @patch("frigate_buffer.services.video_compilation.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    @patch("frigate_buffer.services.video_compilation.logger")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    def test_run_pynv_compilation_logs_error_when_all_src_indices_identical(
        self, mock_isfile, mock_logger, mock_open_fn, mock_popen, mock_create_decoder
    ):
        """Same frame index for all → DEBUG and INFO once per camera (static)."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_isfile.return_value = True
        mock_ctx = _fake_create_decoder_context(200, height=480, width=640)
        if mock_ctx is None:
            self.skipTest("torch not available")
        mock_ctx.get_index_from_time_in_seconds = lambda t: 0
        mock_ctx.get_frames = lambda indices: torch.zeros(
            (len(indices), 3, 480, 640), dtype=torch.uint8
        )
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_ctx)
        mock_cm.__exit__ = MagicMock(return_value=None)
        mock_create_decoder.return_value = mock_cm
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
            cuda_device_index=0,
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
    @patch("frigate_buffer.services.video_compilation.create_decoder")
    @patch("frigate_buffer.services.video_compilation.subprocess.Popen")
    @patch("builtins.open", new_callable=mock_open)
    @patch("frigate_buffer.services.video_compilation.logger")
    @patch("frigate_buffer.services.video_compilation.os.path.isfile")
    def test_run_pynv_compilation_logs_once_per_camera_not_per_slice(
        self, mock_isfile, mock_logger, mock_open_fn, mock_popen, mock_create_decoder
    ):
        """DEBUG compilation log emitted once per camera, not per slice."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_isfile.return_value = True
        mock_ctx = _fake_create_decoder_context(200, height=480, width=640)
        if mock_ctx is None:
            self.skipTest("torch not available")
        mock_ctx.get_frames = lambda indices: torch.zeros(
            (len(indices), 3, 480, 640), dtype=torch.uint8
        )
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_ctx)
        mock_cm.__exit__ = MagicMock(return_value=None)
        mock_create_decoder.return_value = mock_cm
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
            cuda_device_index=0,
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
                _encode_frames_via_ffmpeg(frames, 1440, 1080, "/tmp/out.mp4")
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
                _encode_frames_via_ffmpeg(frames, 1440, 1080, "/tmp/out.mp4")
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
                with pytest.raises(RuntimeError) as ctx:
                    _encode_frames_via_ffmpeg(frames, 1440, 1080, "/tmp/out.mp4")
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
                with pytest.raises(RuntimeError) as ctx:
                    _encode_frames_via_ffmpeg(frames, 1440, 1080, "/tmp/out.mp4")
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


class TestCompilationDecoderImport(unittest.TestCase):
    """Compilation uses gpu_decoder.create_decoder; test guards import."""

    def test_video_compilation_imports_create_decoder(self):
        """video_compilation module uses create_decoder from gpu_decoder for decode."""
        from frigate_buffer.services import video_compilation

        assert hasattr(video_compilation, "create_decoder") or "create_decoder" in dir(
            video_compilation
        )
        from frigate_buffer.services.gpu_decoder import create_decoder

        assert video_compilation.create_decoder is create_decoder


if __name__ == "__main__":
    unittest.main()
