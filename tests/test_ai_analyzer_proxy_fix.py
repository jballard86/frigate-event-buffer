"""Tests for proxy retry behavior (ChunkedEncodingError / ProtocolError)."""

import json
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
        "choices": [{
            "message": {
                "content": json.dumps({
                    "title": "Test",
                    "shortSummary": "S",
                    "scene": "Sc",
                    "confidence": 0.9,
                    "potential_threat_level": 0,
                }),
            },
        }],
    }


class _MockSuccessResponse:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return _success_response_json()


class TestProxyRetryOnChunkedEncodingError(unittest.TestCase):
    """Retry: first call raises ChunkedEncodingError, second succeeds."""

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    def test_send_to_proxy_retries_and_succeeds(self, mock_post):
        mock_post.side_effect = [
            requests.exceptions.ChunkedEncodingError("Connection broken: InvalidChunkLength(...)"),
            _MockSuccessResponse(),
        ]
        service = GeminiAnalysisService(_minimal_config())
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        result = service.send_to_proxy("System prompt", [frame])

        self.assertEqual(mock_post.call_count, 2, "requests.post should be called twice (retry)")
        self.assertIsNotNone(result)
        self.assertEqual(result.get("title"), "Test")

        # Second call should use retry headers
        first_kw = mock_post.call_args_list[0][1]
        second_kw = mock_post.call_args_list[1][1]
        self.assertNotIn("Accept-Encoding", first_kw.get("headers", {}))
        self.assertEqual(second_kw.get("headers", {}).get("Accept-Encoding"), "identity")
        self.assertEqual(second_kw.get("headers", {}).get("Connection"), "close")

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    def test_send_text_prompt_retries_and_succeeds(self, mock_post):
        class _TextResponse:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"choices": [{"message": {"content": "OK"}}]}
        mock_post.side_effect = [
            requests.exceptions.ChunkedEncodingError("Connection broken"),
            _TextResponse(),
        ]
        service = GeminiAnalysisService(_minimal_config())

        result = service.send_text_prompt("System", "User")

        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(result, "OK")


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

        self.assertEqual(mock_post.call_count, 2)
        self.assertIsNone(result)
        # ChunkedEncodingError path logs warning each time (no generic Exception)
        self.assertGreaterEqual(mock_logger.warning.call_count, 1)

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    @patch("frigate_buffer.services.gemini_proxy_client.logger")
    def test_send_text_prompt_returns_none_when_both_fail(self, mock_logger, mock_post):
        mock_post.side_effect = [
            requests.exceptions.ChunkedEncodingError("first"),
            requests.exceptions.ChunkedEncodingError("second"),
        ]
        service = GeminiAnalysisService(_minimal_config())

        result = service.send_text_prompt("System", "User")

        self.assertEqual(mock_post.call_count, 2)
        self.assertIsNone(result)
        self.assertGreaterEqual(mock_logger.warning.call_count, 1)


class TestSendToProxyNativeGeminiFormat(unittest.TestCase):
    """send_to_proxy parses native Gemini API response (candidates[].content.parts[].text)."""

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

        self.assertIsNotNone(result)
        self.assertEqual(result.get("title"), "G")
        self.assertEqual(result.get("scene"), "Sc")
        self.assertEqual(result.get("shortSummary"), "S")
        self.assertEqual(result.get("confidence"), 0.9)
        self.assertEqual(result.get("potential_threat_level"), 0)
