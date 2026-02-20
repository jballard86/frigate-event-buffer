"""
Tests for max event length (cancel long events): config, FileManager rename/canceled summary,
lifecycle and orchestrator ensuring API is never sent for over-max events, and cleanup.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from frigate_buffer.managers.file import FileManager
from frigate_buffer.models import EventState


class TestFileManagerMaxEventLength(unittest.TestCase):
    """Test rename_event_folder and write_canceled_summary."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def test_rename_event_folder_appends_suffix(self):
        """rename_event_folder renames to basename + suffix and returns new path."""
        fm = FileManager(storage_path=self.tmp, retention_days=1)
        folder = os.path.join(self.tmp, "cam1", "1700000000_ev123")
        os.makedirs(folder, exist_ok=True)
        new_path = fm.rename_event_folder(folder)
        self.assertEqual(os.path.basename(new_path), "1700000000_ev123-canceled")
        self.assertFalse(os.path.isdir(folder))
        self.assertTrue(os.path.isdir(new_path))

    def test_rename_event_folder_idempotent_if_already_suffixed(self):
        """If folder already ends with suffix, rename returns same path."""
        fm = FileManager(storage_path=self.tmp, retention_days=1)
        folder = os.path.join(self.tmp, "events", "1700000000_ev123-canceled")
        os.makedirs(folder, exist_ok=True)
        new_path = fm.rename_event_folder(folder)
        self.assertEqual(new_path, folder)
        self.assertTrue(os.path.isdir(folder))

    def test_write_canceled_summary_writes_title_and_description(self):
        """write_canceled_summary writes summary.txt with cancel title for event view."""
        fm = FileManager(storage_path=self.tmp, retention_days=1)
        folder = os.path.join(self.tmp, "events", "1700000000_ce1")
        os.makedirs(folder, exist_ok=True)
        result = fm.write_canceled_summary(folder)
        self.assertTrue(result)
        summary_path = os.path.join(folder, "summary.txt")
        self.assertTrue(os.path.isfile(summary_path))
        with open(summary_path) as f:
            content = f.read()
        self.assertIn("Title: Canceled event: max event length exceeded", content)
        self.assertIn("Event exceeded max_event_length_seconds", content)


class TestOrchestratorDefenseInDepth(unittest.TestCase):
    """Orchestrator must not call analyze_clip when event duration >= MAX_EVENT_LENGTH_SECONDS."""

    @patch('frigate_buffer.orchestrator.threading.Thread')
    def test_on_clip_ready_does_not_call_analyze_clip_when_duration_over_max(self, mock_thread):
        """When state returns event with duration >= max, analyze_clip is never called (defense in depth)."""
        fake_thread = MagicMock()
        def run_target_and_return_mock(target=None, args=(), kwargs=None, **kw):
            if target:
                target()
            return fake_thread
        mock_thread.side_effect = run_target_and_return_mock

        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        config = {
            'cameras': [{'name': 'cam1'}],
            'MQTT_BROKER': 'localhost', 'MQTT_PORT': 1883,
            'FRIGATE_URL': 'http://frigate', 'BUFFER_IP': 'localhost',
            'STORAGE_PATH': storage, 'RETENTION_DAYS': 3, 'FLASK_PORT': 5055,
            'EVENT_GAP_SECONDS': 120, 'MAX_EVENT_LENGTH_SECONDS': 120,
            'GEMINI': {'enabled': True, 'proxy_url': 'http://proxy', 'api_key': 'key', 'model': 'x'},
            'GEMINI_PROXY_URL': 'http://proxy',
        }
        with patch('frigate_buffer.orchestrator.GeminiAnalysisService') as mock_gemini_class:
            mock_ai = MagicMock()
            mock_gemini_class.return_value = mock_ai
            from frigate_buffer.orchestrator import StateAwareOrchestrator
            orch = StateAwareOrchestrator(config)
            orch.state_manager.create_event("evt1", "cam1", "person", 100.0)
            ev = orch.state_manager.get_event("evt1")
            ev.end_time = 250.0

            orch.lifecycle_service.on_clip_ready_for_analysis("evt1", "/tmp/clip.mp4")

            mock_ai.analyze_clip.assert_not_called()
