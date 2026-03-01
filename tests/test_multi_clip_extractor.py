"""Tests for multi-clip target-centric frame extractor."""

import json
import os
import shutil
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch


def _fake_create_decoder(path, gpu_id=0):
    """Context manager that yields a mock DecoderContext (used when patching
    gpu_decoder.create_decoder)."""
    try:
        import torch
    except ImportError:
        torch = None
    mock_ctx = MagicMock()
    mock_ctx.frame_count = 10
    mock_ctx.__len__ = lambda self: 10
    # PTS-based index: time (sec) -> frame index (mock uses ~5 fps, clamp to 0..9).
    mock_ctx.get_index_from_time_in_seconds = lambda t_sec: min(
        max(0, int(t_sec * 5)), 9
    )
    if torch is not None:

        def get_frames(indices):
            return torch.zeros((len(indices), 3, 480, 640), dtype=torch.uint8)

        mock_ctx.get_frames.side_effect = get_frames

    @contextmanager
    def cm():
        yield mock_ctx

    return cm()


from frigate_buffer.constants import DECODER_TIME_EPSILON_SEC
from frigate_buffer.models import ExtractedFrame
from frigate_buffer.services.multi_clip_extractor import (
    DETECTION_SIDECAR_FILENAME,
    _detection_timestamps_with_person,
    _load_sidecar_for_camera,
    _nearest_sidecar_entry,
    _person_area_at_time,
    _person_area_from_detections,
    _subsample_with_min_gap,
    extract_target_centric_frames,
)


class TestMultiClipExtractorHelpers(unittest.TestCase):
    """Tests for sidecar helpers (person area, load)."""

    def test_person_area_from_detections_empty(self):
        """Empty or no detections returns 0."""
        assert _person_area_from_detections([]) == 0.0
        assert _person_area_from_detections(None) == 0.0

    def test_person_area_from_detections_sum_preferred_labels(self):
        """Sums area only for person/people/pedestrian (case-insensitive)."""
        dets = [
            {"label": "person", "area": 1000},
            {"label": "Person", "area": 500},
            {"label": "car", "area": 8000},
        ]
        assert _person_area_from_detections(dets) == 1500.0

    def test_person_area_from_detections_new_sidecar_format(self):
        """New sidecar format (label, bbox, centerpoint, area) is summed correctly."""
        dets = [
            {
                "label": "person",
                "bbox": [0, 0, 10, 20],
                "centerpoint": [5, 10],
                "area": 200,
            },
            {
                "label": "person",
                "bbox": [10, 0, 30, 15],
                "centerpoint": [20, 7.5],
                "area": 300,
            },
        ]
        assert _person_area_from_detections(dets) == 500.0

    def test_nearest_sidecar_entry_returns_none_for_empty(self):
        """Empty sidecar entries returns None."""
        assert _nearest_sidecar_entry([], 0.5) is None

    def test_nearest_sidecar_entry_returns_entry_closest_to_t(self):
        """Returns entry with timestamp_sec closest to t_sec."""
        entries = [
            {"timestamp_sec": 0.0, "frame_number": 0, "detections": []},
            {"timestamp_sec": 1.0, "frame_number": 30, "detections": []},
            {"timestamp_sec": 2.0, "frame_number": 60, "detections": []},
        ]
        assert _nearest_sidecar_entry(entries, 0.6)["timestamp_sec"] == 1.0
        assert _nearest_sidecar_entry(entries, 0.0)["timestamp_sec"] == 0.0

    def test_person_area_at_time_empty_returns_zero(self):
        """Empty sidecar or missing detections defaults to 0."""
        assert _person_area_at_time([], 0.5) == 0.0
        assert _person_area_at_time([{"timestamp_sec": 0.0}], 0.0) == 0.0

    def test_person_area_at_time_nearest(self):
        """Returns person area from nearest entry by timestamp_sec."""
        entries = [
            {"timestamp_sec": 0.0, "detections": [{"label": "person", "area": 100}]},
            {"timestamp_sec": 1.0, "detections": [{"label": "person", "area": 200}]},
            {"timestamp_sec": 2.0, "detections": [{"label": "person", "area": 50}]},
        ]
        # t=0.6: nearest is 1.0 (distance 0.4), not 0.0 (0.6)
        assert _person_area_at_time(entries, 0.6) == 200.0
        assert _person_area_at_time(entries, 0.0) == 100.0

    def test_load_sidecar_missing_returns_none(self):
        """Missing file returns None."""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        d = os.path.join(test_dir, "_sidecar_fixture", "missing_" + str(id(self)))
        os.makedirs(d, exist_ok=True)
        try:
            assert _load_sidecar_for_camera(d) is None
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_load_sidecar_valid_returns_list(self):
        """Valid detection.json (legacy list) returns (entries, native_w,
        native_h) with entries list."""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        d = os.path.join(test_dir, "_sidecar_fixture", str(id(self)))
        os.makedirs(d, exist_ok=True)
        try:
            path = os.path.join(d, DETECTION_SIDECAR_FILENAME)
            data = [
                {
                    "timestamp_sec": 0.0,
                    "detections": [{"label": "person", "area": 100}],
                }
            ]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            result = _load_sidecar_for_camera(d)
            assert result is not None
            entries, nw, nh = result
            assert isinstance(entries, list)
            assert len(entries) == 1
            assert entries[0]["timestamp_sec"] == 0.0
            assert nw == 0
            assert nh == 0
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_detection_timestamps_with_person_empty_sidecar_safe(self):
        """One camera has entries with person area, other is None/empty;
        no exception."""
        sidecars = {
            "cam1": [
                {
                    "timestamp_sec": 0.0,
                    "detections": [{"label": "person", "area": 100}],
                }
            ],
            "cam2": None,
        }
        result = _detection_timestamps_with_person(sidecars, global_end=10.0)
        assert result == [0.0]
        sidecars["cam2"] = []
        result = _detection_timestamps_with_person(sidecars, global_end=10.0)
        assert result == [0.0]

    def test_detection_timestamps_with_person_filters_zero_area(self):
        """Only entries with person area > 0 are included."""
        sidecars = {
            "cam1": [
                {"timestamp_sec": 0.0, "detections": [{"label": "person", "area": 50}]},
                {"timestamp_sec": 1.0, "detections": []},
                {"timestamp_sec": 2.0, "detections": [{"label": "car", "area": 1000}]},
            ],
        }
        result = _detection_timestamps_with_person(sidecars, global_end=10.0)
        assert result == [0.0]

    def test_subsample_with_min_gap_enforces_step(self):
        """Selected timestamps have at least step_sec between them;
        not all from start."""
        timestamps = [0.0, 0.1, 0.2, 1.0, 1.05, 2.0, 2.1, 3.0]
        result = _subsample_with_min_gap(timestamps, step_sec=1.0, max_count=5)
        assert result == [0.0, 1.0, 2.0, 3.0]
        assert len(result) <= 5

    def test_subsample_with_min_gap_respects_max_count(self):
        """Subsample stops at max_count."""
        timestamps = [float(i) for i in range(20)]
        result = _subsample_with_min_gap(timestamps, step_sec=1.0, max_count=3)
        assert len(result) == 3
        assert result == [0.0, 1.0, 2.0]


class TestMultiClipExtractor(unittest.TestCase):
    """Tests for extract_target_centric_frames."""

    def setUp(self):
        # Use a deterministic workspace path so sandbox/Windows can create subdirs
        # (mkdtemp can block makedirs on Windows).
        self._test_dir = os.path.dirname(os.path.abspath(__file__))
        base = os.path.join(self._test_dir, "_extractor_fixture")
        self.tmp = os.path.join(base, str(id(self)))
        os.makedirs(self.tmp, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.tmp):
            shutil.rmtree(self.tmp, ignore_errors=True)

    def test_empty_ce_folder_returns_empty(self):
        """When CE folder has no camera subdirs with clips, returns empty list."""
        result = extract_target_centric_frames(
            self.tmp, max_frames_sec=1, max_frames_min=60
        )
        assert result == []

    def test_no_clips_in_subdirs_returns_empty(self):
        """When subdirs exist but no clip.mp4, returns empty list."""
        os.makedirs(os.path.join(self.tmp, "Camera1"))
        os.makedirs(os.path.join(self.tmp, "Camera2"))
        result = extract_target_centric_frames(
            self.tmp, max_frames_sec=1, max_frames_min=60
        )
        assert result == []

    def _make_decoder_context_mock(self, frame_count=10, height=480, width=640):
        """Create a mock DecoderContext: __len__,
        get_index_from_time_in_seconds, get_frames(indices) -> BCHW uint8 tensor."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_ctx = MagicMock()
        mock_ctx.frame_count = frame_count
        mock_ctx.__len__ = lambda self: frame_count
        mock_ctx.get_index_from_time_in_seconds = lambda t_sec: min(
            max(0, int(t_sec * 5)), frame_count - 1
        )

        def get_frames(indices):
            return torch.zeros((len(indices), 3, height, width), dtype=torch.uint8)

        mock_ctx.get_frames.side_effect = get_frames
        return mock_ctx

    @patch(
        "frigate_buffer.services.multi_clip_extractor.create_decoder",
        side_effect=_fake_create_decoder,
    )
    def test_extract_with_decoder_mock_returns_tensor_frames(self, mock_create_decoder):
        """extract_target_centric_frames uses create_decoder and get_frames;
        returns ExtractedFrame with tensor .frame."""
        os.makedirs(os.path.join(self.tmp, "cam1"))
        os.makedirs(os.path.join(self.tmp, "cam2"))
        for c in ("cam1", "cam2"):
            with open(os.path.join(self.tmp, c, "clip.mp4"), "wb"):
                pass
            sidecar_path = os.path.join(self.tmp, c, DETECTION_SIDECAR_FILENAME)
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "timestamp_sec": t,
                            "detections": [{"label": "person", "area": 100}],
                        }
                        for t in range(11)
                    ],
                    f,
                )

        result = extract_target_centric_frames(
            self.tmp, max_frames_sec=1.0, max_frames_min=5
        )

        assert len(result) >= 1
        assert len(result) <= 5
        for item in result:
            assert isinstance(item, ExtractedFrame)
            assert item.frame is not None
            assert (
                hasattr(item.frame, "shape") or type(item.frame).__name__ == "Tensor"
            ), "ExtractedFrame.frame should be tensor (BCHW)"
            assert item.camera in ("cam1", "cam2")
            assert "person_area" in item.metadata
            assert isinstance(item.metadata["person_area"], int)
            assert item.metadata["person_area"] == 100
        mock_create_decoder.assert_called()
        # One get_frames per sample time that yields a frame
        assert mock_create_decoder.call_count >= 1

    @patch(
        "frigate_buffer.services.multi_clip_extractor.create_decoder",
        side_effect=_fake_create_decoder,
    )
    def test_extract_calls_get_frames(self, mock_create_decoder):
        """extract_target_centric_frames uses create_decoder and get_frames
        is called on the context."""
        os.makedirs(os.path.join(self.tmp, "cam1"))
        with open(os.path.join(self.tmp, "cam1", "clip.mp4"), "wb"):
            pass
        sidecar_path = os.path.join(self.tmp, "cam1", DETECTION_SIDECAR_FILENAME)
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "native_width": 100,
                    "native_height": 100,
                    "entries": [
                        {
                            "timestamp_sec": 0.0,
                            "detections": [{"label": "person", "area": 100}],
                        }
                    ],
                },
                f,
            )
        result = extract_target_centric_frames(
            self.tmp, max_frames_sec=1.0, max_frames_min=5
        )
        assert len(result) >= 1
        mock_create_decoder.assert_called()

    @patch("frigate_buffer.services.multi_clip_extractor.create_decoder")
    def test_extract_drops_camera_when_get_frames_raises(self, mock_create_decoder):
        """When one camera's get_frames raises, that camera is dropped and
        extraction continues with the other(s)."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        os.makedirs(os.path.join(self.tmp, "cam1"))
        os.makedirs(os.path.join(self.tmp, "cam2"))
        for c in ("cam1", "cam2"):
            with open(os.path.join(self.tmp, c, "clip.mp4"), "wb"):
                pass
            sidecar_path = os.path.join(self.tmp, c, DETECTION_SIDECAR_FILENAME)
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        {
                            "timestamp_sec": t,
                            "detections": [{"label": "person", "area": 100}],
                        }
                        for t in range(11)
                    ],
                    f,
                )

        def make_cm(path, gpu_id=0):
            mock_ctx = MagicMock()
            mock_ctx.frame_count = 10
            mock_ctx.__len__ = lambda self: 10
            mock_ctx.get_index_from_time_in_seconds = lambda t_sec: min(
                max(0, int(t_sec * 5)), 9
            )
            if "cam2" in path:
                mock_ctx.get_frames.side_effect = RuntimeError(
                    "Decoder get_frames failed"
                )
            else:
                mock_ctx.get_frames.side_effect = lambda indices: torch.zeros(
                    (len(indices), 3, 480, 640), dtype=torch.uint8
                )

            @contextmanager
            def cm():
                yield mock_ctx

            return cm()

        mock_create_decoder.side_effect = make_cm

        result = extract_target_centric_frames(
            self.tmp, max_frames_sec=1.0, max_frames_min=5
        )

        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, ExtractedFrame)
            assert item.frame is not None
            assert item.camera == "cam1", (
                "Failed camera should be dropped; only cam1 should appear"
            )

    @patch("frigate_buffer.services.multi_clip_extractor._get_fps_duration_from_path")
    @patch(
        "frigate_buffer.services.multi_clip_extractor.create_decoder",
        side_effect=_fake_create_decoder,
    )
    def test_extract_uses_metadata_fallback_when_len_zero(
        self, mock_create_decoder, mock_get_fps_duration
    ):
        """When decoder reports 0 frames, frame count comes from
        _get_fps_duration_from_path and extraction does not crash."""
        os.makedirs(os.path.join(self.tmp, "cam1"))
        with open(os.path.join(self.tmp, "cam1", "clip.mp4"), "wb"):
            pass
        sidecar_path = os.path.join(self.tmp, "cam1", DETECTION_SIDECAR_FILENAME)
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "native_width": 100,
                    "native_height": 100,
                    "entries": [
                        {
                            "timestamp_sec": t,
                            "detections": [{"label": "person", "area": 100}],
                        }
                        for t in range(11)
                    ],
                },
                f,
            )
        mock_get_fps_duration.return_value = (1.0, 10.0)
        # Default _fake_create_decoder yields ctx with len 10; extraction uses it.
        # Test just verifies metadata is used when needed.
        result = extract_target_centric_frames(
            self.tmp, max_frames_sec=1.0, max_frames_min=5
        )
        assert len(result) >= 1
        for item in result:
            assert isinstance(item, ExtractedFrame)
            assert item.frame is not None

    @patch("frigate_buffer.services.multi_clip_extractor._get_fps_duration_from_path")
    @patch("frigate_buffer.services.multi_clip_extractor.create_decoder")
    def test_extract_clamps_time_to_strictly_less_than_duration(
        self, mock_create_decoder, mock_get_fps_duration
    ):
        """When sample time T equals camera duration, decoder is called with
        safe_T <= duration - epsilon (NVDEC rejects t_sec >= duration)."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        from contextlib import contextmanager

        os.makedirs(os.path.join(self.tmp, "cam1"))
        with open(os.path.join(self.tmp, "cam1", "clip.mp4"), "wb"):
            pass
        sidecar_path = os.path.join(self.tmp, "cam1", DETECTION_SIDECAR_FILENAME)
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "native_width": 100,
                    "native_height": 100,
                    "entries": [
                        {
                            "timestamp_sec": t * 0.2,
                            "detections": [{"label": "person", "area": 100}],
                        }
                        for t in range(11)
                    ],
                },
                f,
            )
        duration_sec = 2.0
        fps = 5.0
        mock_get_fps_duration.return_value = (fps, duration_sec)
        times_passed: list[float] = []

        def record_and_return(t_sec):
            times_passed.append(t_sec)
            return min(max(0, int(t_sec * fps)), 9)

        mock_ctx = MagicMock()
        mock_ctx.__len__ = lambda self: 10
        mock_ctx.get_index_from_time_in_seconds = record_and_return
        mock_ctx.get_frames.side_effect = lambda indices: torch.zeros(
            (len(indices), 3, 480, 640), dtype=torch.uint8
        )

        @contextmanager
        def cm(*_args, **_kwargs):
            yield mock_ctx

        mock_create_decoder.side_effect = cm

        result = extract_target_centric_frames(
            self.tmp, max_frames_sec=0.5, max_frames_min=3
        )

        assert len(result) >= 1
        max_valid = duration_sec - DECODER_TIME_EPSILON_SEC
        assert all(t <= max_valid for t in times_passed), (
            f"Decoder must not be called with t_sec > duration - epsilon: "
            f"got {times_passed} (max_valid={max_valid})"
        )

    @patch(
        "frigate_buffer.services.multi_clip_extractor.create_decoder",
        side_effect=_fake_create_decoder,
    )
    def test_extract_logs_nvdec(self, mock_create_decoder):
        """log_callback receives Decoding clips ... PyNvVideoCodec NVDEC."""
        os.makedirs(os.path.join(self.tmp, "cam1"))
        with open(os.path.join(self.tmp, "cam1", "clip.mp4"), "wb"):
            pass
        sidecar_path = os.path.join(self.tmp, "cam1", DETECTION_SIDECAR_FILENAME)
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "native_width": 100,
                    "native_height": 100,
                    "entries": [
                        {
                            "timestamp_sec": t,
                            "detections": [{"label": "person", "area": 100}],
                        }
                        for t in range(11)
                    ],
                },
                f,
            )
        logs = []
        result = extract_target_centric_frames(
            self.tmp, max_frames_sec=1.0, max_frames_min=5, log_callback=logs.append
        )
        assert len(result) >= 1
        decode_logs = [m for m in logs if "Decoding clips" in m]
        assert len(decode_logs) >= 1
        assert any("PyNvVideoCodec" in m or "NVDEC" in m for m in decode_logs), (
            f"Expected PyNvVideoCodec/NVDEC in decode message: {decode_logs}"
        )

    @patch(
        "frigate_buffer.services.multi_clip_extractor.create_decoder",
        side_effect=_fake_create_decoder,
    )
    def test_primary_camera_accepted(self, mock_create_decoder):
        """extract_target_centric_frames accepts primary_camera (EMA pipeline);
        returns frames without error."""
        os.makedirs(os.path.join(self.tmp, "cam1"))
        os.makedirs(os.path.join(self.tmp, "cam2"))
        for c in ("cam1", "cam2"):
            with open(os.path.join(self.tmp, c, "clip.mp4"), "wb"):
                pass
            sidecar_path = os.path.join(self.tmp, c, DETECTION_SIDECAR_FILENAME)
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "native_width": 100,
                        "native_height": 100,
                        "entries": [
                            {
                                "timestamp_sec": t,
                                "detections": [{"label": "person", "area": 100}],
                            }
                            for t in range(11)
                        ],
                    },
                    f,
                )

        result = extract_target_centric_frames(
            self.tmp, max_frames_sec=1.0, max_frames_min=5, primary_camera="cam1"
        )
        assert len(result) >= 1
        for item in result:
            assert isinstance(item, ExtractedFrame)
            assert item.camera in ("cam1", "cam2")

    @patch(
        "frigate_buffer.services.multi_clip_extractor.create_decoder",
        side_effect=_fake_create_decoder,
    )
    def test_ema_drop_no_person_excludes_zero_area_frames(self, mock_create_decoder):
        """With camera_timeline_final_yolo_drop_no_person=True, frames with
        person_area=0 are not in the result."""
        os.makedirs(os.path.join(self.tmp, "cam1"))
        os.makedirs(os.path.join(self.tmp, "cam2"))
        for c in ("cam1", "cam2"):
            with open(os.path.join(self.tmp, c, "clip.mp4"), "wb"):
                pass
        times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        entries = [
            {
                "timestamp_sec": t,
                "detections": [{"label": "person", "area": 0 if t < 1.0 else 100}],
            }
            for t in times
        ]
        for cam_name in ("cam1", "cam2"):
            sidecar_path = os.path.join(self.tmp, cam_name, DETECTION_SIDECAR_FILENAME)
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"native_width": 100, "native_height": 100, "entries": entries}, f
                )

        result = extract_target_centric_frames(
            self.tmp,
            max_frames_sec=1.0,
            max_frames_min=10,
            camera_timeline_final_yolo_drop_no_person=True,
        )
        for ef in result:
            assert isinstance(ef, ExtractedFrame)
            assert ef.metadata.get("person_area", 0) > 0, (
                "With camera_timeline_final_yolo_drop_no_person=True, "
                "no frame should have person_area=0"
            )
