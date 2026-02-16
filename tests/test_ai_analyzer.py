"""Unit tests for GeminiAnalysisService. No proxy or video files required."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

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
    @patch("frigate_buffer.services.ai_analyzer.cv2.VideoCapture")
    def test_analyze_clip_returns_result_on_success(self, mock_vc, mock_post):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            clip_path = f.name
        self.addCleanup(lambda: os.path.exists(clip_path) and os.unlink(clip_path))
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
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
