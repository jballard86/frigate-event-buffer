"""Unit tests for GeminiAnalysisService. No proxy or video files required."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import requests

from frigate_buffer.models import FrameMetadata
from frigate_buffer.services.ai_analyzer import GeminiAnalysisService


class TestGeminiAnalysisServiceConfig(unittest.TestCase):
    """Test config validation and graceful behavior when config is missing/invalid."""

    def test_config_validation_missing_gemini_no_crash(self):
        config = {"GEMINI": None}
        service = GeminiAnalysisService(config)
        result = service.analyze_clip("evt1", "/nonexistent/clip.mp4")
        self.assertIsNone(result)

    def test_config_validation_disabled_no_crash(self):
        config = {"GEMINI": {"enabled": False, "proxy_url": "http://x", "api_key": "k"}}
        service = GeminiAnalysisService(config)
        result = service.analyze_clip("evt1", "/nonexistent/clip.mp4")
        self.assertIsNone(result)

    def test_config_validation_missing_proxy_url_no_crash(self):
        config = {"GEMINI": {"enabled": True, "proxy_url": "", "api_key": "k"}}
        service = GeminiAnalysisService(config)
        result = service.analyze_clip("evt1", "/nonexistent/clip.mp4")
        self.assertIsNone(result)


class TestGeminiAnalysisServicePayload(unittest.TestCase):
    """Test that the JSON payload sent to the proxy matches OpenAI schema."""

    @patch("frigate_buffer.services.ai_analyzer.requests.post")
    def test_payload_structure(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "title": "Test", "shortSummary": "S", "scene": "Sc", "confidence": 0.9, "potential_threat_level": 0
            })}}]
        }
        config = {"GEMINI": {"enabled": True, "proxy_url": "http://proxy", "api_key": "key", "model": "m"}}
        service = GeminiAnalysisService(config)
        # One dummy frame (numpy BGR)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        service.send_to_proxy("System prompt here", [frame])

        self.assertEqual(mock_post.call_count, 1)
        args, kwargs = mock_post.call_args
        body = kwargs.get("json")
        self.assertIsNotNone(body)
        self.assertIn("messages", body)
        messages = body["messages"]
        self.assertGreaterEqual(len(messages), 2)
        system_msg = next((m for m in messages if m.get("role") == "system"), None)
        user_msg = next((m for m in messages if m.get("role") == "user"), None)
        self.assertIsNotNone(system_msg)
        self.assertIsNotNone(user_msg)
        self.assertEqual(system_msg.get("content"), "System prompt here")
        user_content = user_msg.get("content")
        self.assertIsInstance(user_content, list)
        # First part text, then image parts with image_url
        image_parts = [p for p in user_content if p.get("type") == "image_url"]
        self.assertGreater(len(image_parts), 0)
        self.assertIn("image_url", image_parts[0])
        url = image_parts[0]["image_url"].get("url", "")
        self.assertTrue(url.startswith("data:image/jpeg;base64,"), f"Expected data URL, got: {url[:80]}")


class TestGeminiAnalysisServiceSendTextPrompt(unittest.TestCase):
    """Test send_text_prompt: text-only POST, no images; returns raw content string or None."""

    @patch("frigate_buffer.services.ai_analyzer.requests.post")
    def test_send_text_prompt_returns_content_on_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "# Report\nDone."}}]
        }
        config = {"GEMINI": {"enabled": True, "proxy_url": "http://proxy", "api_key": "key", "model": "m"}}
        service = GeminiAnalysisService(config)
        result = service.send_text_prompt("System", "User text")
        self.assertEqual(result, "# Report\nDone.")
        self.assertEqual(mock_post.call_count, 1)
        args, kwargs = mock_post.call_args
        self.assertIn("json", kwargs)
        body = kwargs["json"]
        messages = body["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "System")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "User text")
        self.assertNotIn("image_url", str(messages))

    @patch("frigate_buffer.services.ai_analyzer.requests.post")
    def test_send_text_prompt_returns_none_when_empty_content(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"choices": [{"message": {"content": ""}}]}
        config = {"GEMINI": {"enabled": True, "proxy_url": "http://proxy", "api_key": "key"}}
        service = GeminiAnalysisService(config)
        result = service.send_text_prompt("Sys", "User")
        self.assertIsNone(result)

    @patch("frigate_buffer.services.ai_analyzer.requests.post")
    def test_send_text_prompt_returns_none_on_5xx(self, mock_post):
        def raise_http_error(*args, **kwargs):
            raise requests.exceptions.HTTPError("500 Server Error")

        mock_post.side_effect = raise_http_error
        config = {"GEMINI": {"enabled": True, "proxy_url": "http://proxy", "api_key": "key"}}
        service = GeminiAnalysisService(config)
        result = service.send_text_prompt("Sys", "User")
        self.assertIsNone(result)

    def test_send_text_prompt_returns_none_when_proxy_not_configured(self):
        config = {"GEMINI": {"enabled": True, "proxy_url": "", "api_key": "key"}}
        service = GeminiAnalysisService(config)
        result = service.send_text_prompt("Sys", "User")
        self.assertIsNone(result)
        config2 = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": ""}}
        service2 = GeminiAnalysisService(config2)
        result2 = service2.send_text_prompt("Sys", "User")
        self.assertIsNone(result2)


class TestGeminiAnalysisServiceProxyFailure(unittest.TestCase):
    """Test that proxy failures are caught and do not crash the process."""

    @patch("frigate_buffer.services.ai_analyzer.requests.post")
    def test_proxy_failure_handling_500(self, mock_post):
        import requests as req
        mock_post.return_value.raise_for_status.side_effect = req.exceptions.HTTPError("500 Server Error")
        mock_post.return_value.status_code = 500
        config = {"GEMINI": {"enabled": True, "proxy_url": "http://proxy", "api_key": "key", "model": "m"}}
        service = GeminiAnalysisService(config)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        result = service.send_to_proxy("Prompt", [frame])
        self.assertIsNone(result)

    @patch("frigate_buffer.services.ai_analyzer.requests.post")
    def test_proxy_failure_handling_timeout(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("timeout")
        config = {"GEMINI": {"enabled": True, "proxy_url": "http://proxy", "api_key": "key", "model": "m"}}
        service = GeminiAnalysisService(config)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        result = service.send_to_proxy("Prompt", [frame])
        self.assertIsNone(result)

    @patch("frigate_buffer.services.ai_analyzer.requests.post")
    def test_proxy_failure_handling_invalid_json(self, mock_post):
        mock_post.return_value.raise_for_status.side_effect = None
        mock_post.return_value.json.return_value = {"choices": [{"message": {"content": "not valid json {"}}]}
        config = {"GEMINI": {"enabled": True, "proxy_url": "http://proxy", "api_key": "key", "model": "m"}}
        service = GeminiAnalysisService(config)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        result = service.send_to_proxy("Prompt", [frame])
        self.assertIsNone(result)


class TestGeminiAnalysisServiceReturnValue(unittest.TestCase):
    """Test that analyze_clip returns the parsed metadata dict (no MQTT publish)."""

    @patch("frigate_buffer.services.ai_analyzer.requests.post")
    @patch("frigate_buffer.services.ai_analyzer.ffmpegcv.VideoCaptureNV")
    def test_analyze_clip_returns_result_on_success(self, mock_vc, mock_post):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            clip_path = f.name
        self.addCleanup(lambda: os.path.exists(clip_path) and os.unlink(clip_path))
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.fps = 1.0
        mock_cap.__len__ = MagicMock(return_value=1)
        mock_cap.get.side_effect = [1.0, 1]
        mock_cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_cap.release = MagicMock()
        mock_vc.return_value = mock_cap
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "title": "T", "shortSummary": "S", "scene": "Sc", "confidence": 0.8, "potential_threat_level": 0
            })}}]
        }
        config = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k", "model": "m"}}
        service = GeminiAnalysisService(config)
        result = service.analyze_clip("evt1", clip_path)
        self.assertIsNotNone(result)
        self.assertEqual(result.get("title"), "T")
        self.assertEqual(result.get("shortSummary"), "S")
        self.assertEqual(result.get("potential_threat_level"), 0)


class TestGeminiAnalysisServiceFlatConfig(unittest.TestCase):
    """Test that flat config keys (GEMINI_PROXY_*, multi_cam) are used when set."""

    def test_flat_proxy_url_and_model_override_nested(self):
        config = {
            "GEMINI": {"enabled": True, "proxy_url": "http://nested", "api_key": "k", "model": "nested-model"},
            "GEMINI_PROXY_URL": "http://flat-url",
            "GEMINI_PROXY_MODEL": "flat-model",
        }
        service = GeminiAnalysisService(config)
        self.assertEqual(service._proxy_url, "http://flat-url")
        self.assertEqual(service._model, "flat-model")

    def test_flat_tuning_params_stored(self):
        config = {
            "GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"},
            "GEMINI_PROXY_TEMPERATURE": 0.7,
            "GEMINI_PROXY_TOP_P": 0.9,
            "GEMINI_PROXY_FREQUENCY_PENALTY": 0.2,
            "GEMINI_PROXY_PRESENCE_PENALTY": 0.1,
        }
        service = GeminiAnalysisService(config)
        self.assertEqual(service._temperature, 0.7)
        self.assertEqual(service._top_p, 0.9)
        self.assertEqual(service._frequency_penalty, 0.2)
        self.assertEqual(service._presence_penalty, 0.1)

    def test_multi_cam_crop_and_motion_from_flat_config(self):
        config = {
            "GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"},
            "CROP_WIDTH": 640,
            "CROP_HEIGHT": 360,
            "MOTION_THRESHOLD_PX": 100,
        }
        service = GeminiAnalysisService(config)
        self.assertEqual(service.crop_width, 640)
        self.assertEqual(service.crop_height, 360)
        self.assertEqual(service.motion_threshold_px, 100)


class TestGeminiAnalysisServiceProxyTuningPayload(unittest.TestCase):
    """Test that proxy request body includes temperature, top_p, frequency_penalty, presence_penalty."""

    @patch("frigate_buffer.services.ai_analyzer.requests.post")
    def test_payload_includes_tuning_params(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "title": "T", "shortSummary": "S", "scene": "Sc", "confidence": 0.9, "potential_threat_level": 0
            })}}]
        }
        config = {
            "GEMINI": {"enabled": True, "proxy_url": "http://proxy", "api_key": "key"},
            "GEMINI_PROXY_TEMPERATURE": 0.5,
            "GEMINI_PROXY_TOP_P": 0.95,
            "GEMINI_PROXY_FREQUENCY_PENALTY": 0.1,
            "GEMINI_PROXY_PRESENCE_PENALTY": 0.05,
        }
        service = GeminiAnalysisService(config)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        service.send_to_proxy("Prompt", [frame])
        body = mock_post.call_args[1].get("json")
        self.assertIn("temperature", body)
        self.assertEqual(body["temperature"], 0.5)
        self.assertIn("top_p", body)
        self.assertEqual(body["top_p"], 0.95)
        self.assertIn("frequency_penalty", body)
        self.assertEqual(body["frequency_penalty"], 0.1)
        self.assertIn("presence_penalty", body)
        self.assertEqual(body["presence_penalty"], 0.05)


class TestGeminiAnalysisServicePromptFile(unittest.TestCase):
    """Test MULTI_CAM_SYSTEM_PROMPT_FILE loads prompt from file when set."""

    def test_prompt_loaded_from_config_path_when_file_exists(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Custom system prompt with {image_count} and {global_event_camera_list}.")
            prompt_path = f.name
        self.addCleanup(lambda: os.path.exists(prompt_path) and os.unlink(prompt_path))
        config = {
            "GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"},
            "MULTI_CAM_SYSTEM_PROMPT_FILE": prompt_path,
        }
        service = GeminiAnalysisService(config)
        template = service._load_system_prompt_template()
        self.assertIn("Custom system prompt", template)
        self.assertIn("{image_count}", template)
        filled = service._build_system_prompt(
            image_count=3,
            camera_list="Doorbell",
            first_image_number=1,
            last_image_number=3,
            activity_start_str="2025-01-01 12:00:00",
            duration_str="10 seconds",
            zones_str="porch",
            labels_str="person",
        )
        self.assertIn("3", filled)
        self.assertIn("Doorbell", filled)

    def test_prompt_falls_back_when_config_path_empty(self):
        config = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"}, "MULTI_CAM_SYSTEM_PROMPT_FILE": ""}
        service = GeminiAnalysisService(config)
        template = service._load_system_prompt_template()
        self.assertIsNotNone(template)
        self.assertGreater(len(template), 0)


class TestGeminiAnalysisServiceCenterCrop(unittest.TestCase):
    """Test _center_crop behavior."""

    def test_center_crop_returns_exact_dimensions(self):
        config = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"}}
        service = GeminiAnalysisService(config)
        frame = np.zeros((200, 320, 3), dtype=np.uint8)
        crop = service._center_crop(frame, 160, 100)
        self.assertEqual(crop.shape, (100, 160, 3))

    def test_center_crop_passthrough_when_target_zero(self):
        config = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"}}
        service = GeminiAnalysisService(config)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = service._center_crop(frame, 0, 0)
        self.assertIs(out, frame)
        out = service._center_crop(frame, 50, 0)
        self.assertIs(out, frame)


class TestGeminiAnalysisServiceFirstLastFrames(unittest.TestCase):
    """Test that first and last frame of segment are always kept when over max_frames."""

    @patch("frigate_buffer.services.ai_analyzer.ffmpegcv.VideoCaptureNV")
    def test_first_and_last_frame_kept_when_candidates_exceed_max(self, mock_vc):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            clip_path = f.name
        self.addCleanup(lambda: os.path.exists(clip_path) and os.unlink(clip_path))
        # Sequential read: first 2 frames are pre-buffer (discarded), next 5 are segment. max_frames=3 -> first, 1 middle, last.
        dummy = np.zeros((60, 80, 3), dtype=np.uint8)
        frames_data = [
            dummy.copy(),
            dummy.copy(),
            np.full((60, 80, 3), 10, dtype=np.uint8),   # first candidate
            np.full((60, 80, 3), 20, dtype=np.uint8),
            np.full((60, 80, 3), 30, dtype=np.uint8),
            np.full((60, 80, 3), 40, dtype=np.uint8),
            np.full((60, 80, 3), 50, dtype=np.uint8),   # last candidate
        ]
        def read_side_effect():
            for fr in frames_data:
                yield (True, fr.copy())
            while True:
                yield (False, None)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.fps = 2.0
        mock_cap.__len__ = MagicMock(return_value=10)
        mock_cap.get.side_effect = lambda key: 2.0 if key == cv2.CAP_PROP_FPS else (10 if key == cv2.CAP_PROP_FRAME_COUNT else 0)
        mock_cap.read.side_effect = read_side_effect()
        mock_cap.release = MagicMock()
        mock_vc.return_value = mock_cap

        config = {
            "GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"},
            "EXPORT_BUFFER_BEFORE": 1,
            "MAX_MULTI_CAM_FRAMES_SEC": 2,
            "MAX_MULTI_CAM_FRAMES_MIN": 45,
            "FINAL_REVIEW_IMAGE_COUNT": 3,
        }
        service = GeminiAnalysisService(config)
        # event_start_ts > 0 → event-segment branch. buffer_offset=1s → start_frame=2; segment 2s → frames 2..6 (5 candidates).
        result = service._extract_frames(clip_path, event_start_ts=1.0, event_end_ts=3.0)
        self.assertEqual(len(result), 3, "Should cap at max_frames=3 (first + one middle + last)")
        # result is list of (frame, frame_time_sec)
        self.assertEqual(int(result[0][0][0, 0, 0]), 10, "First frame should be segment start")
        self.assertEqual(int(result[-1][0][0, 0, 0]), 50, "Last frame should be segment end")


class TestExtractFramesNoSeek(unittest.TestCase):
    """_extract_frames must not call cap.set(); ffmpegcv readers (e.g. FFmpegReaderNV) do not support it."""

    def test_extract_frames_event_segment_without_set(self):
        """Reader without .set (like FFmpegReaderNV): event-segment branch completes via sequential read."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            clip_path = f.name
        self.addCleanup(lambda: os.path.exists(clip_path) and os.unlink(clip_path))
        frames = [np.full((60, 80, 3), i, dtype=np.uint8) for i in (0, 0, 10, 20, 30, 40, 50)]

        class NoSetReader:
            def __init__(self, _path):
                self._idx = 0
                self._frames = frames

            def isOpened(self):
                return True

            def read(self):
                if self._idx >= len(self._frames):
                    return (False, None)
                ret = (True, self._frames[self._idx].copy())
                self._idx += 1
                return ret

            def release(self):
                pass

            @property
            def fps(self):
                return 2.0

            def __len__(self):
                return 10

            def get(self, key):
                return 2.0 if key == cv2.CAP_PROP_FPS else (10 if key == cv2.CAP_PROP_FRAME_COUNT else 0)

        with patch("frigate_buffer.services.ai_analyzer.ffmpegcv.VideoCaptureNV", NoSetReader):
            config = {
                "GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"},
                "EXPORT_BUFFER_BEFORE": 1,
                "MAX_MULTI_CAM_FRAMES_SEC": 2,
                "FINAL_REVIEW_IMAGE_COUNT": 3,
            }
            service = GeminiAnalysisService(config)
            result = service._extract_frames(clip_path, event_start_ts=1.0, event_end_ts=3.0)
        self.assertEqual(len(result), 3)
        self.assertEqual(int(result[0][0][0, 0, 0]), 10)
        self.assertEqual(int(result[-1][0][0, 0, 0]), 50)

    def test_extract_frames_uniform_sampling_without_set(self):
        """Reader without .set: uniform-sampling branch (event_start_ts=0) completes via sequential read."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            clip_path = f.name
        self.addCleanup(lambda: os.path.exists(clip_path) and os.unlink(clip_path))
        frames = [np.zeros((60, 80, 3), dtype=np.uint8) for _ in range(5)]

        class NoSetReader:
            def __init__(self, _path):
                self._idx = 0
                self._frames = frames

            def isOpened(self):
                return True

            def read(self):
                if self._idx >= len(self._frames):
                    return (False, None)
                ret = (True, self._frames[self._idx].copy())
                self._idx += 1
                return ret

            def release(self):
                pass

            @property
            def fps(self):
                return 1.0

            def __len__(self):
                return 5

            def get(self, key):
                return 1.0 if key == cv2.CAP_PROP_FPS else (5 if key == cv2.CAP_PROP_FRAME_COUNT else 0)

        with patch("frigate_buffer.services.ai_analyzer.ffmpegcv.VideoCaptureNV", NoSetReader):
            config = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"}, "FINAL_REVIEW_IMAGE_COUNT": 10}
            service = GeminiAnalysisService(config)
            result = service._extract_frames(clip_path, event_start_ts=0, event_end_ts=1.0)
        self.assertGreater(len(result), 0)
        self.assertLessEqual(len(result), 5)


class TestGeminiAnalysisServiceCropAppliedDuringExtraction(unittest.TestCase):
    """Test that when CROP_WIDTH/CROP_HEIGHT are set, extracted frames are cropped."""

    @patch("frigate_buffer.services.ai_analyzer.ffmpegcv.VideoCaptureNV")
    def test_extracted_frames_have_crop_dimensions(self, mock_vc):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            clip_path = f.name
        self.addCleanup(lambda: os.path.exists(clip_path) and os.unlink(clip_path))
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.fps = 1.0
        mock_cap.__len__ = MagicMock(return_value=5)
        mock_cap.get.side_effect = [1.0, 5]
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.release = MagicMock()
        mock_vc.return_value = mock_cap

        config = {
            "GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"},
            "EXPORT_BUFFER_BEFORE": 0,
            "MAX_MULTI_CAM_FRAMES_SEC": 1,
            "CROP_WIDTH": 320,
            "CROP_HEIGHT": 240,
        }
        service = GeminiAnalysisService(config)
        result = service._extract_frames(clip_path, event_start_ts=0, event_end_ts=1.0)
        self.assertGreater(len(result), 0)
        for frame, _ in result:
            self.assertEqual(frame.shape, (240, 320, 3), "Each frame should be center-cropped to 320x240")


class TestGeminiAnalysisServiceSmartCrop(unittest.TestCase):
    """Test _smart_crop with normalized box and padding."""

    def test_smart_crop_returns_exact_dimensions(self):
        config = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"}, "SMART_CROP_PADDING": 0.15}
        service = GeminiAnalysisService(config)
        frame = np.zeros((200, 320, 3), dtype=np.uint8)
        box = (0.25, 0.25, 0.75, 0.75)
        crop = service._smart_crop(frame, box, 160, 100)
        self.assertEqual(crop.shape, (100, 160, 3))

    def test_smart_crop_empty_box_falls_back_to_center_crop(self):
        config = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"}}
        service = GeminiAnalysisService(config)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = service._smart_crop(frame, [], 50, 50)
        self.assertEqual(out.shape, (50, 50, 3))
        out = service._smart_crop(frame, None, 50, 50)
        self.assertEqual(out.shape, (50, 50, 3))

    def test_smart_crop_with_padding_produces_larger_region(self):
        config = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"}, "SMART_CROP_PADDING": 0.2}
        service = GeminiAnalysisService(config)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        box = (0.4, 0.4, 0.6, 0.6)
        crop_padded = service._smart_crop(frame, box, 50, 50, padding=0.2)
        crop_no_pad = service._smart_crop(frame, box, 50, 50, padding=0.0)
        self.assertEqual(crop_padded.shape, (50, 50, 3))
        self.assertEqual(crop_no_pad.shape, (50, 50, 3))


class TestGeminiAnalysisServiceFrameMetadata(unittest.TestCase):
    """Test analyze_clip and _extract_frames with frame_metadata."""

    def test_smart_crop_padding_from_config(self):
        config = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"}, "SMART_CROP_PADDING": 0.25}
        service = GeminiAnalysisService(config)
        self.assertEqual(service.smart_crop_padding, 0.25)

    @patch("frigate_buffer.services.ai_analyzer.requests.post")
    @patch("frigate_buffer.services.ai_analyzer.ffmpegcv.VideoCaptureNV")
    def test_analyze_clip_with_frame_metadata_empty_does_not_crash(self, mock_vc, mock_post):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            clip_path = f.name
        self.addCleanup(lambda: os.path.exists(clip_path) and os.unlink(clip_path))
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.fps = 1.0
        mock_cap.__len__ = MagicMock(return_value=5)
        mock_cap.get.side_effect = lambda key: 1.0 if key == cv2.CAP_PROP_FPS else (5 if key == cv2.CAP_PROP_FRAME_COUNT else 0)
        mock_cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_cap.release = MagicMock()
        mock_vc.return_value = mock_cap
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": json.dumps({
                "title": "T", "shortSummary": "S", "scene": "Sc", "confidence": 0.8, "potential_threat_level": 0
            })}}]
        }
        config = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"}}
        service = GeminiAnalysisService(config)
        result = service.analyze_clip("evt1", clip_path, event_start_ts=0, event_end_ts=1.0, frame_metadata=[])
        self.assertIsNotNone(result)


class TestGeminiAnalysisServiceSaveAnalysisResult(unittest.TestCase):
    """Test _save_analysis_result: path handling and OSError does not crash."""

    def test_save_analysis_result_oserror_does_not_raise(self):
        """When writing analysis_result.json raises OSError (e.g. read-only fs), we log and do not raise."""
        event_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: os.path.exists(event_dir) and __import__("shutil").rmtree(event_dir, ignore_errors=True))
        clip_path = os.path.join(event_dir, "clip.mp4")
        with open(clip_path, "wb") as f:
            f.write(b"fake")
        config = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"}}
        service = GeminiAnalysisService(config)
        result = {"title": "T", "shortSummary": "S", "potential_threat_level": 0}
        out_path = os.path.join(event_dir, "analysis_result.json")
        with patch("frigate_buffer.services.ai_analyzer.open", side_effect=OSError(13, "Permission denied")):
            try:
                service._save_analysis_result("evt1", clip_path, result)
            except OSError:
                self.fail("_save_analysis_result should catch OSError and not re-raise")
        self.assertFalse(os.path.isfile(out_path), "File should not be created when open raises OSError")
