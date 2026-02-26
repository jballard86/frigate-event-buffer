"""
Integration tests for AI analyzer: persistence of analysis_result.json,
orchestrator hand-off to Frigate and HA, and error handling when proxy
returns invalid JSON or 5xx.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from frigate_buffer.services.ai_analyzer import GeminiAnalysisService

# --- Test Case 1: Persistence ---


def _make_mock_extracted_frame():
    """One frame for analyze_multi_clip_ce: object with .frame, .timestamp_sec,
    .camera, .metadata."""

    class _EF:
        __slots__ = ("frame", "timestamp_sec", "camera", "metadata")

        def __init__(self):
            self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
            self.timestamp_sec = 0.0
            self.camera = "cam1"
            self.metadata = {}

    return _EF()


def _make_mock_extracted_frame_tensor():
    """Phase 5: One frame with tensor .frame (BCHW RGB) for analyze_multi_clip_ce."""
    try:
        import torch
    except ImportError:
        return None

    class _EF:
        __slots__ = ("frame", "timestamp_sec", "camera", "metadata")

        def __init__(self):
            self.frame = torch.zeros((1, 3, 100, 100), dtype=torch.uint8)
            self.timestamp_sec = 0.0
            self.camera = "cam1"
            self.metadata = {}

    return _EF()


class TestIntegrationStep5Persistence(unittest.TestCase):
    """Verify analysis_result.json is created in the CE folder with expected content."""

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    @patch("frigate_buffer.services.ai_analyzer.extract_target_centric_frames")
    def test_analysis_result_json_created_with_expected_fields(
        self, mock_extract, mock_post
    ):
        ce_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: os.path.exists(ce_dir) and shutil.rmtree(ce_dir))
        mock_extract.return_value = [_make_mock_extracted_frame()]

        payload = {
            "title": "Person at door",
            "shortSummary": "A person approached the front door.",
            "scene": "Front porch",
            "confidence": 0.9,
            "potential_threat_level": 1,
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": json.dumps(payload)}}]
        }

        config = {
            "GEMINI": {
                "enabled": True,
                "proxy_url": "http://proxy",
                "api_key": "key",
            }
        }
        service = GeminiAnalysisService(config)
        result = service.analyze_multi_clip_ce("ce_123", ce_dir, ce_start_time=0.0)

        self.assertIsNotNone(result)
        self.assertEqual(result.get("title"), "Person at door")
        self.assertEqual(
            result.get("shortSummary"), "A person approached the front door."
        )
        self.assertEqual(result.get("potential_threat_level"), 1)

        out_path = os.path.join(ce_dir, "analysis_result.json")
        self.assertTrue(os.path.isfile(out_path), f"Expected {out_path} to exist")
        with open(out_path, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertIn("shortSummary", saved)
        self.assertIn("title", saved)
        self.assertIn("potential_threat_level", saved)
        self.assertEqual(saved["title"], payload["title"])
        self.assertEqual(saved["shortSummary"], payload["shortSummary"])
        self.assertEqual(
            saved["potential_threat_level"], payload["potential_threat_level"]
        )

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    @patch("frigate_buffer.services.ai_analyzer.extract_target_centric_frames")
    def test_analysis_result_saved_even_when_required_fields_missing(
        self, mock_extract, mock_post
    ):
        """When proxy returns partial result (e.g. missing potential_threat_level),
        we still save the dict."""
        ce_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: os.path.exists(ce_dir) and shutil.rmtree(ce_dir))
        mock_extract.return_value = [_make_mock_extracted_frame()]

        partial_payload = {
            "title": "Partial",
            "shortSummary": "No threat level in response.",
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": json.dumps(partial_payload)}}]
        }

        config = {
            "GEMINI": {
                "enabled": True,
                "proxy_url": "http://proxy",
                "api_key": "key",
            }
        }
        service = GeminiAnalysisService(config)
        result = service.analyze_multi_clip_ce("ce_partial", ce_dir, ce_start_time=0.0)

        self.assertIsNotNone(result)
        out_path = os.path.join(ce_dir, "analysis_result.json")
        self.assertTrue(
            os.path.isfile(out_path),
            "Should save analysis_result.json even when fields missing",
        )
        with open(out_path, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["title"], "Partial")
        self.assertEqual(saved["shortSummary"], "No threat level in response.")

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    @patch("frigate_buffer.services.ai_analyzer.extract_target_centric_frames")
    def test_analysis_result_json_created_with_tensor_frames(
        self, mock_extract, mock_post
    ):
        """Phase 5: analyze_multi_clip_ce accepts ExtractedFrame with tensor .frame;
        saves analysis_result.json."""
        tensor_frame = _make_mock_extracted_frame_tensor()
        if tensor_frame is None:
            self.skipTest("torch not available")
        ce_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: os.path.exists(ce_dir) and shutil.rmtree(ce_dir))
        mock_extract.return_value = [tensor_frame]

        payload = {
            "title": "Tensor frame test",
            "shortSummary": "Pipeline accepts tensor frames.",
            "scene": "Test",
            "confidence": 0.9,
            "potential_threat_level": 0,
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": json.dumps(payload)}}]
        }

        config = {
            "GEMINI": {
                "enabled": True,
                "proxy_url": "http://proxy",
                "api_key": "key",
            }
        }
        service = GeminiAnalysisService(config)
        result = service.analyze_multi_clip_ce("ce_tensor", ce_dir, ce_start_time=0.0)

        self.assertIsNotNone(result)
        self.assertEqual(result.get("title"), "Tensor frame test")
        out_path = os.path.join(ce_dir, "analysis_result.json")
        self.assertTrue(os.path.isfile(out_path), f"Expected {out_path} to exist")
        with open(out_path, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["title"], payload["title"])


# --- Test Case 2: Orchestrator hand-off ---


class TestIntegrationStep6OrchestratorHandoff(unittest.TestCase):
    """Verify _handle_analysis_result triggers Frigate API and HA notification."""

    def test_handle_analysis_result_calls_post_event_description_and_publish_notification(  # noqa: E501
        self,
    ):
        from frigate_buffer.models import EventState
        from frigate_buffer.orchestrator import StateAwareOrchestrator

        config = {
            "MQTT_BROKER": "localhost",
            "MQTT_PORT": 1883,
            "MQTT_USER": None,
            "MQTT_PASSWORD": None,
            "FRIGATE_URL": "http://frigate",
            "STORAGE_PATH": tempfile.gettempdir(),
            "RETENTION_DAYS": 7,
            "BUFFER_IP": "127.0.0.1",
            "FLASK_PORT": 5000,
            "GEMINI": {"enabled": False},
        }
        with (
            patch("frigate_buffer.orchestrator.MqttClientWrapper"),
            patch("frigate_buffer.orchestrator.NotificationDispatcher"),
            patch("frigate_buffer.orchestrator.EventLifecycleService"),
        ):
            orch = StateAwareOrchestrator(config)
        orch.flask_app = MagicMock()

        orch.download_service = MagicMock()
        orch.download_service.post_event_description = MagicMock(return_value=True)
        orch.notifier = MagicMock()
        orch.notifier.publish_notification = MagicMock()

        event_id = "evt-456"
        event = EventState(event_id, "Doorbell", "person", 1000.0)
        event.folder_path = "/tmp/events/evt-456"
        event.end_time = 1010.0
        orch.state_manager.get_event = MagicMock(return_value=event)
        orch.state_manager.set_genai_metadata = MagicMock(return_value=True)
        orch.consolidated_manager.get_by_frigate_event = MagicMock(return_value=None)
        orch.file_manager.write_summary = MagicMock(return_value=True)
        orch.file_manager.write_metadata_json = MagicMock()

        result = {
            "title": "Test Title",
            "shortSummary": "Test summary for Frigate and HA.",
            "scene": "Porch",
            "potential_threat_level": 0,
        }
        orch._handle_analysis_result(event_id, result)

        orch.download_service.post_event_description.assert_called_once()
        call_args = orch.download_service.post_event_description.call_args[0]
        self.assertEqual(call_args[0], event_id)
        self.assertIn("Test summary", call_args[1])

        orch.notifier.publish_notification.assert_called_once()
        notify_args = orch.notifier.publish_notification.call_args[0]
        self.assertEqual(notify_args[1], "finalized")


# --- Test Case 3: Error handling ---


class TestIntegrationStep5ErrorHandling(unittest.TestCase):
    """Verify invalid JSON or 5xx does not create analysis_result.json or crash."""

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    @patch("frigate_buffer.services.ai_analyzer.extract_target_centric_frames")
    def test_invalid_json_does_not_create_analysis_result_file(
        self, mock_extract, mock_post
    ):
        ce_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: os.path.exists(ce_dir) and shutil.rmtree(ce_dir))
        mock_extract.return_value = [_make_mock_extracted_frame()]

        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "not valid json {"}}]
        }
        mock_post.return_value.raise_for_status = MagicMock()

        config = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"}}
        service = GeminiAnalysisService(config)
        result = service.analyze_multi_clip_ce("ce_err", ce_dir, ce_start_time=0.0)

        self.assertIsNone(result)
        out_path = os.path.join(ce_dir, "analysis_result.json")
        self.assertFalse(
            os.path.isfile(out_path),
            "Should not create analysis_result.json on invalid JSON",
        )

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    def test_proxy_500_send_to_proxy_returns_none(self, mock_post):
        import requests as req

        mock_post.return_value.status_code = 500
        mock_post.return_value.raise_for_status.side_effect = req.exceptions.HTTPError(
            "500 Server Error"
        )

        config = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"}}
        service = GeminiAnalysisService(config)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        result = service.send_to_proxy("Prompt", [frame])
        self.assertIsNone(result)

    @patch("frigate_buffer.services.gemini_proxy_client.requests.post")
    @patch("frigate_buffer.services.ai_analyzer.extract_target_centric_frames")
    def test_proxy_500_analyze_multi_clip_ce_does_not_create_analysis_result(
        self, mock_extract, mock_post
    ):
        import requests as req

        ce_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: os.path.exists(ce_dir) and shutil.rmtree(ce_dir))
        mock_extract.return_value = [_make_mock_extracted_frame()]

        mock_post.return_value.status_code = 500
        mock_post.return_value.raise_for_status.side_effect = req.exceptions.HTTPError(
            "500 Server Error"
        )

        config = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"}}
        service = GeminiAnalysisService(config)
        result = service.analyze_multi_clip_ce("ce_500", ce_dir, ce_start_time=0.0)
        self.assertIsNone(result)
        out_path = os.path.join(ce_dir, "analysis_result.json")
        self.assertFalse(
            os.path.isfile(out_path),
            "Should not create analysis_result.json on proxy 500",
        )
