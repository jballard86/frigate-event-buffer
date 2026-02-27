"""Unit tests for GeminiAnalysisService. No proxy or video files required."""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import requests

from frigate_buffer.services.ai_analyzer import GeminiAnalysisService


def _minimal_config():
    return {
        "GEMINI": {
            "enabled": True,
            "proxy_url": "http://proxy",
            "api_key": "key",
            "model": "m",
        }
    }


def _success_response_json():
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "title": "Test",
                            "shortSummary": "S",
                            "scene": "Sc",
                            "confidence": 0.9,
                            "potential_threat_level": 0,
                        }
                    ),
                },
            }
        ],
    }


class _MockSuccessResponse:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return _success_response_json()


class TestGeminiAnalysisServiceConfig(unittest.TestCase):
    """Test config validation and graceful behavior when config is missing/invalid."""

    def test_config_validation_missing_gemini_no_crash(self):
        config = {"GEMINI": None}
        service = GeminiAnalysisService(config)
        result = service.analyze_multi_clip_ce("ce1", "/nonexistent/ce_folder", 0.0)
        assert result is None

    def test_config_validation_disabled_no_crash(self):
        config = {"GEMINI": {"enabled": False, "proxy_url": "http://x", "api_key": "k"}}
        service = GeminiAnalysisService(config)
        result = service.analyze_multi_clip_ce("ce1", "/nonexistent/ce_folder", 0.0)
        assert result is None

    def test_config_validation_missing_proxy_url_no_crash(self):
        config = {"GEMINI": {"enabled": True, "proxy_url": "", "api_key": "k"}}
        service = GeminiAnalysisService(config)
        result = service.analyze_multi_clip_ce("ce1", "/nonexistent/ce_folder", 0.0)
        assert result is None


class TestGeminiAnalysisServicePayload(unittest.TestCase):
    """Test that the JSON payload sent to the proxy matches OpenAI schema."""

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    def test_payload_structure(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "title": "Test",
                                "shortSummary": "S",
                                "scene": "Sc",
                                "confidence": 0.9,
                                "potential_threat_level": 0,
                            }
                        )
                    }
                }
            ]
        }
        config = {
            "GEMINI": {
                "enabled": True,
                "proxy_url": "http://proxy",
                "api_key": "key",
                "model": "m",
            }
        }
        service = GeminiAnalysisService(config)
        # One dummy frame (numpy BGR)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        service.send_to_proxy("System prompt here", [frame])

        assert mock_post.call_count == 1
        args, kwargs = mock_post.call_args
        body = kwargs.get("json")
        assert body is not None
        assert "messages" in body
        messages = body["messages"]
        assert len(messages) >= 2
        system_msg = next((m for m in messages if m.get("role") == "system"), None)
        user_msg = next((m for m in messages if m.get("role") == "user"), None)
        assert system_msg is not None
        assert user_msg is not None
        assert system_msg.get("content") == "System prompt here"
        user_content = user_msg.get("content")
        assert isinstance(user_content, list)
        # First part text, then image parts with image_url
        image_parts = [p for p in user_content if p.get("type") == "image_url"]
        assert len(image_parts) > 0
        assert "image_url" in image_parts[0]
        url = image_parts[0]["image_url"].get("url", "")
        assert url.startswith("data:image/jpeg;base64,"), (
            f"Expected data URL, got: {url[:80]}"
        )


class TestFrameToBase64Url(unittest.TestCase):
    """Test _frame_to_base64_url with numpy and tensor (Phase 4)."""

    def test_frame_to_base64_url_numpy_returns_data_url(self):
        config = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"}}
        service = GeminiAnalysisService(config)
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        url = service._frame_to_base64_url(frame)
        assert url.startswith("data:image/jpeg;base64,")
        assert len(url) > 50

    def test_frame_to_base64_url_tensor_returns_data_url(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        config = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"}}
        service = GeminiAnalysisService(config)
        # BCHW RGB uint8
        t = torch.zeros((1, 3, 60, 80), dtype=torch.uint8)
        url = service._frame_to_base64_url(t)
        assert url.startswith("data:image/jpeg;base64,")
        assert len(url) > 50

    def test_frame_to_base64_url_tensor_delegates_to_tensor_path(self):
        """Tensor input must use _frame_tensor_to_base64_url (GPU path),
        not cv2 path."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        config = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"}}
        service = GeminiAnalysisService(config)
        t = torch.zeros((1, 3, 60, 80), dtype=torch.uint8)
        with patch.object(
            service,
            "_frame_tensor_to_base64_url",
            return_value="data:image/jpeg;base64,MOCK",
        ) as mock_tensor_encode:
            url = service._frame_to_base64_url(t)
        mock_tensor_encode.assert_called_once_with(t)
        assert url == "data:image/jpeg;base64,MOCK"


class TestGeminiFrameCapAndLogging(unittest.TestCase):
    """Test rolling frame cap and API rate stats logging."""

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    def test_cap_disabled_sends_and_logs(self, mock_post):
        """When GEMINI_FRAMES_PER_HOUR_CAP is 0, request is sent and log
        mentions cap=disabled."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "title": "T",
                                "shortSummary": "S",
                                "scene": "Sc",
                                "confidence": 0.9,
                                "potential_threat_level": 0,
                            }
                        )
                    }
                }
            ]
        }
        config = {
            "GEMINI": {
                "enabled": True,
                "proxy_url": "http://p",
                "api_key": "k",
                "model": "m",
            },
            "GEMINI_FRAMES_PER_HOUR_CAP": 0,
        }
        service = GeminiAnalysisService(config)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        with patch("frigate_buffer.services.ai_analyzer.logger") as mock_log:
            result = service.send_to_proxy("Prompt", [frame])
        assert result is not None
        assert mock_post.call_count == 1
        log_calls = [str(c) for c in mock_log.info.call_args_list]
        assert any("cap=disabled" in c for c in log_calls), (
            f"Expected log to contain 'cap=disabled', got: {log_calls}"
        )

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    def test_cap_enforced_blocks_second_request(self, mock_post):
        """When cap is 2, first send (2 frames) succeeds; second send (1 frame)
        is blocked."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "title": "T",
                                "shortSummary": "S",
                                "scene": "Sc",
                                "confidence": 0.9,
                                "potential_threat_level": 0,
                            }
                        )
                    }
                }
            ]
        }
        config = {
            "GEMINI": {
                "enabled": True,
                "proxy_url": "http://p",
                "api_key": "k",
                "model": "m",
            },
            "GEMINI_FRAMES_PER_HOUR_CAP": 2,
        }
        service = GeminiAnalysisService(config)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        result1 = service.send_to_proxy("P", [frame, frame])
        assert result1 is not None
        assert mock_post.call_count == 1
        result2 = service.send_to_proxy("P", [frame])
        assert result2 is None, "Second request should be blocked by cap"
        assert mock_post.call_count == 1, "POST should not be called again when blocked"

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    def test_cap_logs_current_rate_status_blocked(self, mock_post):
        """When cap is enabled, info log contains current_frames, status, blocked."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "title": "T",
                                "shortSummary": "S",
                                "scene": "Sc",
                                "confidence": 0.9,
                                "potential_threat_level": 0,
                            }
                        )
                    }
                }
            ]
        }
        config = {
            "GEMINI": {
                "enabled": True,
                "proxy_url": "http://p",
                "api_key": "k",
                "model": "m",
            },
            "GEMINI_FRAMES_PER_HOUR_CAP": 10,
        }
        service = GeminiAnalysisService(config)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        with patch("frigate_buffer.services.ai_analyzer.logger") as mock_log:
            service.send_to_proxy("P", [frame])
        log_msg = mock_log.info.call_args[0][0]
        assert "current_frames" in log_msg
        assert "cap=" in log_msg
        assert "status=" in log_msg
        assert "blocked=" in log_msg


class TestGeminiAnalysisServiceSendTextPrompt(unittest.TestCase):
    """Test send_text_prompt: text-only POST, no images; returns raw content
    string or None."""

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    def test_send_text_prompt_returns_content_on_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "# Report\nDone."}}]
        }
        config = {
            "GEMINI": {
                "enabled": True,
                "proxy_url": "http://proxy",
                "api_key": "key",
                "model": "m",
            }
        }
        service = GeminiAnalysisService(config)
        result = service.send_text_prompt("System", "User text")
        assert result == "# Report\nDone."
        assert mock_post.call_count == 1
        args, kwargs = mock_post.call_args
        assert "json" in kwargs
        body = kwargs["json"]
        messages = body["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "User text"
        assert "image_url" not in str(messages)

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    def test_send_text_prompt_returns_none_when_empty_content(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": ""}}]
        }
        config = {
            "GEMINI": {"enabled": True, "proxy_url": "http://proxy", "api_key": "key"}
        }
        service = GeminiAnalysisService(config)
        result = service.send_text_prompt("Sys", "User")
        assert result is None

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    def test_send_text_prompt_returns_none_on_5xx(self, mock_post):
        def raise_http_error(*args, **kwargs):
            raise requests.exceptions.HTTPError("500 Server Error")

        mock_post.side_effect = raise_http_error
        config = {
            "GEMINI": {"enabled": True, "proxy_url": "http://proxy", "api_key": "key"}
        }
        service = GeminiAnalysisService(config)
        result = service.send_text_prompt("Sys", "User")
        assert result is None

    def test_send_text_prompt_returns_none_when_proxy_not_configured(self):
        config = {"GEMINI": {"enabled": True, "proxy_url": "", "api_key": "key"}}
        service = GeminiAnalysisService(config)
        result = service.send_text_prompt("Sys", "User")
        assert result is None
        config2 = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": ""}}
        service2 = GeminiAnalysisService(config2)
        result2 = service2.send_text_prompt("Sys", "User")
        assert result2 is None


class TestGeminiAnalysisServiceProxyFailure(unittest.TestCase):
    """Test that proxy failures are caught and do not crash the process."""

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    def test_proxy_failure_handling_500(self, mock_post):
        import requests as req

        mock_post.return_value.raise_for_status.side_effect = req.exceptions.HTTPError(
            "500 Server Error"
        )
        mock_post.return_value.status_code = 500
        config = {
            "GEMINI": {
                "enabled": True,
                "proxy_url": "http://proxy",
                "api_key": "key",
                "model": "m",
            }
        }
        service = GeminiAnalysisService(config)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        result = service.send_to_proxy("Prompt", [frame])
        assert result is None

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    def test_proxy_failure_handling_timeout(self, mock_post):
        import requests

        mock_post.side_effect = requests.exceptions.Timeout("timeout")
        config = {
            "GEMINI": {
                "enabled": True,
                "proxy_url": "http://proxy",
                "api_key": "key",
                "model": "m",
            }
        }
        service = GeminiAnalysisService(config)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        result = service.send_to_proxy("Prompt", [frame])
        assert result is None

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    def test_proxy_failure_handling_invalid_json(self, mock_post):
        mock_post.return_value.raise_for_status.side_effect = None
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "not valid json {"}}]
        }
        config = {
            "GEMINI": {
                "enabled": True,
                "proxy_url": "http://proxy",
                "api_key": "key",
                "model": "m",
            }
        }
        service = GeminiAnalysisService(config)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        result = service.send_to_proxy("Prompt", [frame])
        assert result is None


class TestGeminiAnalysisServiceFlatConfig(unittest.TestCase):
    """Test that flat config keys (GEMINI_PROXY_*, multi_cam) are used when set."""

    def test_flat_proxy_url_and_model_override_nested(self):
        config = {
            "GEMINI": {
                "enabled": True,
                "proxy_url": "http://nested",
                "api_key": "k",
                "model": "nested-model",
            },
            "GEMINI_PROXY_URL": "http://flat-url",
            "GEMINI_PROXY_MODEL": "flat-model",
        }
        service = GeminiAnalysisService(config)
        assert service._proxy_url == "http://flat-url"
        assert service._model == "flat-model"

    def test_flat_tuning_params_stored(self):
        config = {
            "GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"},
            "GEMINI_PROXY_TEMPERATURE": 0.7,
            "GEMINI_PROXY_TOP_P": 0.9,
            "GEMINI_PROXY_FREQUENCY_PENALTY": 0.2,
            "GEMINI_PROXY_PRESENCE_PENALTY": 0.1,
        }
        service = GeminiAnalysisService(config)
        assert service._temperature == 0.7
        assert service._top_p == 0.9
        assert service._frequency_penalty == 0.2
        assert service._presence_penalty == 0.1

    def test_multi_cam_crop_and_motion_from_flat_config(self):
        config = {
            "GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"},
            "CROP_WIDTH": 640,
            "CROP_HEIGHT": 360,
            "MOTION_THRESHOLD_PX": 100,
        }
        service = GeminiAnalysisService(config)
        assert service.crop_width == 640
        assert service.crop_height == 360
        assert service.motion_threshold_px == 100


class TestGeminiAnalysisServiceProxyTuningPayload(unittest.TestCase):
    """Test that proxy request body includes temperature, top_p,
    frequency_penalty, presence_penalty."""

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    def test_payload_includes_tuning_params(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "title": "T",
                                "shortSummary": "S",
                                "scene": "Sc",
                                "confidence": 0.9,
                                "potential_threat_level": 0,
                            }
                        )
                    }
                }
            ]
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
        assert "temperature" in body
        assert body["temperature"] == 0.5
        assert "top_p" in body
        assert body["top_p"] == 0.95
        assert "frequency_penalty" in body
        assert body["frequency_penalty"] == 0.1
        assert "presence_penalty" in body
        assert body["presence_penalty"] == 0.05


class TestGeminiAnalysisServicePromptFile(unittest.TestCase):
    """Test MULTI_CAM_SYSTEM_PROMPT_FILE loads prompt from file when set."""

    def test_prompt_loaded_from_config_path_when_file_exists(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(
                "Custom system prompt with {image_count} and "
                "{global_event_camera_list}."
            )
            prompt_path = f.name
        self.addCleanup(lambda: os.path.exists(prompt_path) and os.unlink(prompt_path))
        config = {
            "GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"},
            "MULTI_CAM_SYSTEM_PROMPT_FILE": prompt_path,
        }
        service = GeminiAnalysisService(config)
        template = service._load_system_prompt_template()
        assert "Custom system prompt" in template
        assert "{image_count}" in template
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
        assert "3" in filled
        assert "Doorbell" in filled

    def test_prompt_falls_back_when_config_path_empty(self):
        config = {
            "GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"},
            "MULTI_CAM_SYSTEM_PROMPT_FILE": "",
        }
        service = GeminiAnalysisService(config)
        template = service._load_system_prompt_template()
        assert template is not None
        assert len(template) > 0


class TestGeminiAnalysisServiceConfigMisc(unittest.TestCase):
    """Misc config tests (e.g. smart_crop_padding)."""

    def test_smart_crop_padding_from_config(self):
        config = {
            "GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"},
            "SMART_CROP_PADDING": 0.25,
        }
        service = GeminiAnalysisService(config)
        assert service.smart_crop_padding == 0.25


class TestGeminiAnalysisServiceAnalyzeMultiClipCe(unittest.TestCase):
    """Test analyze_multi_clip_ce: exception handling returns None and
    does not propagate."""

    def test_analyze_multi_clip_ce_returns_none_when_extraction_raises(self):
        """When extract_target_centric_frames raises, analyze_multi_clip_ce
        catches and returns None."""
        ce_dir = tempfile.mkdtemp()
        self.addCleanup(
            lambda: (
                os.path.exists(ce_dir)
                and __import__("shutil").rmtree(ce_dir, ignore_errors=True)
            )
        )
        config = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"}}
        service = GeminiAnalysisService(config)
        with patch(
            "frigate_buffer.services.ai_analyzer.extract_target_centric_frames",
            side_effect=ValueError("read of closed file"),
        ):
            result = service.analyze_multi_clip_ce("ce_123", ce_dir, ce_start_time=0.0)
        assert result is None


class TestBuildMultiCamPayloadForPreview(unittest.TestCase):
    """Test build_multi_cam_payload_for_preview return (result, error_message)
    and log_messages."""

    def test_nonexistent_folder_returns_error_and_no_logs(self):
        """When CE folder does not exist, returns (None, error) and does not
        append to log_messages."""
        config = {"GEMINI": {"enabled": False}}
        service = GeminiAnalysisService(config)
        log_messages = []
        result, err = service.build_multi_cam_payload_for_preview(
            "/nonexistent/ce_folder", 0.0, log_messages=log_messages
        )
        assert result is None
        assert err == "CE folder not found"
        assert log_messages == []

    def test_no_frames_extracted_returns_error_and_appends_logs(self):
        """When extraction returns no frames, returns (None, error) and
        log_messages contains frame selection line."""
        ce_dir = tempfile.mkdtemp()
        self.addCleanup(
            lambda: (
                os.path.exists(ce_dir)
                and __import__("shutil").rmtree(ce_dir, ignore_errors=True)
            )
        )
        config = {"GEMINI": {"enabled": False}}
        service = GeminiAnalysisService(config)
        log_messages = []
        with patch(
            "frigate_buffer.services.ai_analyzer.extract_target_centric_frames",
            return_value=[],
        ):
            result, err = service.build_multi_cam_payload_for_preview(
                ce_dir, 0.0, log_messages=log_messages
            )
        assert result is None
        assert "No frames extracted" in err
        assert len(log_messages) > 0
        assert any("Frame selection:" in m for m in log_messages), (
            f"Expected a 'Frame selection:' message in {log_messages}"
        )


class TestProxyRetryOnChunkedEncodingError(unittest.TestCase):
    """Retry: first call raises ChunkedEncodingError, second succeeds."""

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    def test_send_to_proxy_retries_and_succeeds(self, mock_post):
        mock_post.side_effect = [
            requests.exceptions.ChunkedEncodingError(
                "Connection broken: InvalidChunkLength(...)"
            ),
            _MockSuccessResponse(),
        ]
        service = GeminiAnalysisService(_minimal_config())
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        result = service.send_to_proxy("System prompt", [frame])

        assert mock_post.call_count == 2, "requests.post should be called twice (retry)"
        assert result is not None
        assert result.get("title") == "Test"

        first_kw = mock_post.call_args_list[0][1]
        second_kw = mock_post.call_args_list[1][1]
        assert "Accept-Encoding" not in first_kw.get("headers", {})
        assert second_kw.get("headers", {}).get("Accept-Encoding") == "identity"
        assert second_kw.get("headers", {}).get("Connection") == "close"

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    def test_send_text_prompt_retries_and_succeeds(self, mock_post):
        class _TextResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "OK"}}]}

        mock_post.side_effect = [
            requests.exceptions.ChunkedEncodingError("Connection broken"),
            _TextResponse(),
        ]
        service = GeminiAnalysisService(_minimal_config())

        result = service.send_text_prompt("System", "User")

        assert mock_post.call_count == 2
        assert result == "OK"


class TestProxyFailureAfterTwoAttempts(unittest.TestCase):
    """Both attempts raise ChunkedEncodingError; returns None and logs."""

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    @patch("frigate_buffer.services.gemini_proxy_client.logger")
    def test_send_to_proxy_returns_none_when_both_fail(self, mock_logger, mock_post):
        mock_post.side_effect = [
            requests.exceptions.ChunkedEncodingError("first"),
            requests.exceptions.ChunkedEncodingError("second"),
        ]
        service = GeminiAnalysisService(_minimal_config())
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        result = service.send_to_proxy("System", [frame])

        assert mock_post.call_count == 2
        assert result is None
        assert mock_logger.warning.call_count >= 1

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    @patch("frigate_buffer.services.gemini_proxy_client.logger")
    def test_send_text_prompt_returns_none_when_both_fail(self, mock_logger, mock_post):
        mock_post.side_effect = [
            requests.exceptions.ChunkedEncodingError("first"),
            requests.exceptions.ChunkedEncodingError("second"),
        ]
        service = GeminiAnalysisService(_minimal_config())

        result = service.send_text_prompt("System", "User")

        assert mock_post.call_count == 2
        assert result is None
        assert mock_logger.warning.call_count >= 1


class TestSendToProxyNativeGeminiFormat(unittest.TestCase):
    """send_to_proxy parses native Gemini API response
    (candidates[].content.parts[].text)."""

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    def test_send_to_proxy_parses_native_gemini_response(self, mock_post):
        gemini_payload = {
            "title": "G",
            "shortSummary": "S",
            "scene": "Sc",
            "confidence": 0.9,
            "potential_threat_level": 0,
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = lambda: None
        mock_post.return_value.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": json.dumps(gemini_payload)}],
                    },
                },
            ],
        }
        service = GeminiAnalysisService(_minimal_config())
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        result = service.send_to_proxy("System prompt", [frame])

        assert result is not None
        assert result.get("title") == "G"
        assert result.get("scene") == "Sc"
        assert result.get("shortSummary") == "S"
        assert result.get("confidence") == 0.9
        assert result.get("potential_threat_level") == 0


def _mock_imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"dummy")
    return True


class TestAiFrameAnalysisWriting(unittest.TestCase):
    """write_ai_frame_analysis_multi_cam: multi-cam frames, manifest, zip;
    numpy and tensor frames."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.test_dir, ignore_errors=True))

    @patch("frigate_buffer.managers.file.cv2")
    def test_multi_cam_writing(self, mock_cv2):
        mock_cv2.imwrite.side_effect = _mock_imwrite
        from frigate_buffer.managers.file import write_ai_frame_analysis_multi_cam

        with patch("frigate_buffer.managers.file._CV2_AVAILABLE", True):
            event_dir = os.path.join(self.test_dir, "event2")
            os.makedirs(event_dir)

            frame = np.zeros((100, 100, 3), dtype=np.uint8)

            class MockExtractedFrame:
                def __init__(self, frame, timestamp_sec, camera, metadata=None):
                    self.frame = frame
                    self.timestamp_sec = timestamp_sec
                    self.camera = camera
                    self.metadata = metadata or {}

            frames = [
                MockExtractedFrame(
                    frame, 2000.0, "cam1", {"is_full_frame_resize": True}
                ),
                MockExtractedFrame(frame, 2001.0, "cam 2", {}),
            ]

            write_ai_frame_analysis_multi_cam(
                event_dir,
                frames,
                write_manifest=True,
                create_zip_flag=True,
                save_frames=True,
            )

            frames_dir = os.path.join(event_dir, "ai_frame_analysis", "frames")
            assert os.path.exists(os.path.join(frames_dir, "frame_001_cam1.jpg"))
            assert os.path.exists(os.path.join(frames_dir, "frame_002_cam_2.jpg"))

            manifest_path = os.path.join(
                event_dir, "ai_frame_analysis", "manifest.json"
            )
            assert os.path.exists(manifest_path)
            with open(manifest_path) as f:
                manifest = json.load(f)

            assert len(manifest) == 2
            assert manifest[0]["filename"] == "frame_001_cam1.jpg"
            assert manifest[0]["camera"] == "cam1"
            assert manifest[0]["is_full_frame_resize"]
            assert manifest[1]["filename"] == "frame_002_cam_2.jpg"
            assert manifest[1]["camera"] == "cam 2"
            assert "is_full_frame_resize" not in manifest[1]

    @patch("frigate_buffer.managers.file.cv2")
    def test_multi_cam_writing_tensor_frames(self, mock_cv2):
        """write_ai_frame_analysis_multi_cam accepts ExtractedFrame with tensor
        .frame (BCHW RGB)."""
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")
        mock_cv2.imwrite.side_effect = _mock_imwrite
        from frigate_buffer.managers.file import write_ai_frame_analysis_multi_cam

        with patch("frigate_buffer.managers.file._CV2_AVAILABLE", True):
            event_dir = os.path.join(self.test_dir, "event_tensor")
            os.makedirs(event_dir)

            class MockExtractedFrame:
                def __init__(self, frame, timestamp_sec, camera, metadata=None):
                    self.frame = frame
                    self.timestamp_sec = timestamp_sec
                    self.camera = camera
                    self.metadata = metadata or {}

            t1 = torch.zeros((1, 3, 100, 100), dtype=torch.uint8)
            t2 = torch.zeros((1, 3, 100, 100), dtype=torch.uint8)
            frames = [
                MockExtractedFrame(t1, 2000.0, "cam1", {"is_full_frame_resize": True}),
                MockExtractedFrame(t2, 2001.0, "cam2", {}),
            ]

            write_ai_frame_analysis_multi_cam(
                event_dir,
                frames,
                write_manifest=True,
                create_zip_flag=False,
                save_frames=True,
            )

            frames_dir = os.path.join(event_dir, "ai_frame_analysis", "frames")
            assert os.path.exists(os.path.join(frames_dir, "frame_001_cam1.jpg"))
            assert os.path.exists(os.path.join(frames_dir, "frame_002_cam2.jpg"))
            manifest_path = os.path.join(
                event_dir, "ai_frame_analysis", "manifest.json"
            )
            assert os.path.exists(manifest_path)
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            assert len(manifest) == 2
            assert manifest[0]["camera"] == "cam1"
            assert manifest[0]["is_full_frame_resize"]
