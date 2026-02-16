"""
Integration tests for Step 5 (Proxy Response Bridge) and Step 6 (Orchestrator Integration).

Verifies: persistence of analysis_result.json, orchestrator hand-off to Frigate and HA,
and error handling when proxy returns invalid JSON or 5xx.
"""

import json
import shutil
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from frigate_buffer.services.ai_analyzer import GeminiAnalysisService


# --- Test Case 1: Persistence ---


class TestIntegrationStep5Persistence(unittest.TestCase):
    """Verify analysis_result.json is created in the event folder with expected content."""

    @patch("frigate_buffer.services.ai_analyzer.requests.post")
    @patch("frigate_buffer.services.ai_analyzer.cv2.VideoCapture")
    def test_analysis_result_json_created_with_expected_fields(self, mock_vc, mock_post):
        event_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: os.path.exists(event_dir) and shutil.rmtree(event_dir))
        clip_path = os.path.join(event_dir, "clip.mp4")
        with open(clip_path, "wb") as f:
            f.write(b"fake mp4")
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda k: 1.0 if k == 5 else (3 if k == 7 else 0)  # FPS, frame count
        mock_cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_cap.release = MagicMock()
        mock_vc.return_value = mock_cap

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

        config = {"GEMINI": {"enabled": True, "proxy_url": "http://proxy", "api_key": "key"}}
        service = GeminiAnalysisService(config)
        result = service.analyze_clip("evt-123", clip_path)

        self.assertIsNotNone(result)
        self.assertEqual(result.get("title"), "Person at door")
        self.assertEqual(result.get("shortSummary"), "A person approached the front door.")
        self.assertEqual(result.get("potential_threat_level"), 1)

        out_path = os.path.join(event_dir, "analysis_result.json")
        self.assertTrue(os.path.isfile(out_path), f"Expected {out_path} to exist")
        with open(out_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertIn("shortSummary", saved)
        self.assertIn("title", saved)
        self.assertIn("potential_threat_level", saved)
        self.assertEqual(saved["title"], payload["title"])
        self.assertEqual(saved["shortSummary"], payload["shortSummary"])
        self.assertEqual(saved["potential_threat_level"], payload["potential_threat_level"])

    @patch("frigate_buffer.services.ai_analyzer.requests.post")
    @patch("frigate_buffer.services.ai_analyzer.cv2.VideoCapture")
    def test_analysis_result_saved_even_when_required_fields_missing(self, mock_vc, mock_post):
        """When proxy returns partial result (e.g. missing potential_threat_level), we still save the dict."""
        event_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: os.path.exists(event_dir) and shutil.rmtree(event_dir))
        clip_path = os.path.join(event_dir, "clip.mp4")
        with open(clip_path, "wb") as f:
            f.write(b"fake mp4")
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda k: 1.0 if k == 5 else (2 if k == 7 else 0)
        mock_cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
        mock_cap.release = MagicMock()
        mock_vc.return_value = mock_cap

        partial_payload = {"title": "Partial", "shortSummary": "No threat level in response."}
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": json.dumps(partial_payload)}}]
        }

        config = {"GEMINI": {"enabled": True, "proxy_url": "http://proxy", "api_key": "key"}}
        service = GeminiAnalysisService(config)
        result = service.analyze_clip("evt-partial", clip_path)

        self.assertIsNotNone(result)
        out_path = os.path.join(event_dir, "analysis_result.json")
        self.assertTrue(os.path.isfile(out_path), "Should save analysis_result.json even when fields missing")
        with open(out_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["title"], "Partial")
        self.assertEqual(saved["shortSummary"], "No threat level in response.")


# --- Test Case 2: Orchestrator hand-off ---


class TestIntegrationStep6OrchestratorHandoff(unittest.TestCase):
    """Verify _handle_analysis_result triggers Frigate API and HA notification."""

    def test_handle_analysis_result_calls_post_event_description_and_publish_notification(self):
        from frigate_buffer.orchestrator import StateAwareOrchestrator
        from frigate_buffer.models import EventState

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
        with patch("frigate_buffer.orchestrator.MqttClientWrapper"), \
             patch("frigate_buffer.orchestrator.NotificationPublisher"), \
             patch("frigate_buffer.orchestrator.EventLifecycleService"):
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

    @patch("frigate_buffer.services.ai_analyzer.requests.post")
    @patch("frigate_buffer.services.ai_analyzer.cv2.VideoCapture")
    def test_invalid_json_does_not_create_analysis_result_file(self, mock_vc, mock_post):
        event_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: os.path.exists(event_dir) and shutil.rmtree(event_dir))
        clip_path = os.path.join(event_dir, "clip.mp4")
        with open(clip_path, "wb") as f:
            f.write(b"fake")
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda k: 1.0 if k == 5 else (2 if k == 7 else 0)
        mock_cap.read.return_value = (True, np.zeros((50, 50, 3), dtype=np.uint8))
        mock_cap.release = MagicMock()
        mock_vc.return_value = mock_cap

        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "not valid json {"}}]}
        mock_post.return_value.raise_for_status = MagicMock()

        config = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"}}
        service = GeminiAnalysisService(config)
        result = service.analyze_clip("evt-err", clip_path)

        self.assertIsNone(result)
        out_path = os.path.join(event_dir, "analysis_result.json")
        self.assertFalse(os.path.isfile(out_path), "Should not create analysis_result.json on invalid JSON")

    @patch("frigate_buffer.services.ai_analyzer.requests.post")
    def test_proxy_500_send_to_proxy_returns_none(self, mock_post):
        import requests as req
        mock_post.return_value.status_code = 500
        mock_post.return_value.raise_for_status.side_effect = req.exceptions.HTTPError("500 Server Error")

        config = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"}}
        service = GeminiAnalysisService(config)
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        result = service.send_to_proxy("Prompt", [frame])
        self.assertIsNone(result)

    @patch("frigate_buffer.services.ai_analyzer.requests.post")
    @patch("frigate_buffer.services.ai_analyzer.cv2.VideoCapture")
    def test_proxy_500_analyze_clip_does_not_create_analysis_result(self, mock_vc, mock_post):
        import requests as req
        event_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: os.path.exists(event_dir) and shutil.rmtree(event_dir))
        clip_path = os.path.join(event_dir, "clip.mp4")
        with open(clip_path, "wb") as f:
            f.write(b"fake")
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda k: 1.0 if k == 5 else (2 if k == 7 else 0)
        mock_cap.read.return_value = (True, np.zeros((50, 50, 3), dtype=np.uint8))
        mock_cap.release = MagicMock()
        mock_vc.return_value = mock_cap
        mock_post.return_value.status_code = 500
        mock_post.return_value.raise_for_status.side_effect = req.exceptions.HTTPError("500 Server Error")

        config = {"GEMINI": {"enabled": True, "proxy_url": "http://p", "api_key": "k"}}
        service = GeminiAnalysisService(config)
        result = service.analyze_clip("evt-500", clip_path)
        self.assertIsNone(result)
        out_path = os.path.join(event_dir, "analysis_result.json")
        self.assertFalse(os.path.isfile(out_path), "Should not create analysis_result.json on proxy 500")
