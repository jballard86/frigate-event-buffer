"""Unit tests for Frigate Export Watchdog."""

import json
import logging
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frigate_buffer.services.frigate_export_watchdog import (
    DELETE_EXPORT_TIMEOUT,
    MAX_HEAD_REQUESTS,
    MAX_LINK_CHECK_FOLDERS,
    run_once,
)


class TestFrigateExportWatchdog(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("frigate-buffer")
        self.logger.setLevel(logging.DEBUG)
        self.log_capture = []
        self.log_handler = logging.Handler()
        self.log_handler.emit = lambda record: self.log_capture.append(record)
        self.logger.addHandler(self.log_handler)

    def tearDown(self):
        self.logger.removeHandler(self.log_handler)

    def test_watchdog_constants_capped(self):
        """HEAD requests for link verification are capped (20 folders, 120 requests)."""
        self.assertEqual(MAX_LINK_CHECK_FOLDERS, 20)
        self.assertEqual(MAX_HEAD_REQUESTS, 120)

    def _make_timeline_with_export_response(
        self,
        export_id="test-export-id-123",
        label="Clip export response (from Frigate API)",
    ):
        return {
            "event_id": "evt1",
            "entries": [
                {
                    "source": "frigate_api",
                    "direction": "out",
                    "label": "Clip export request (to Frigate API)",
                    "ts": "2025-01-01T12:00:00",
                    "data": {},
                },
                {
                    "source": "frigate_api",
                    "direction": "in",
                    "label": label,
                    "ts": "2025-01-01T12:00:05",
                    "data": {
                        "frigate_response": {"export_id": export_id, "success": True}
                    },
                },
            ],
        }

    @patch("frigate_buffer.services.frigate_export_watchdog.requests.delete")
    @patch("frigate_buffer.services.frigate_export_watchdog.requests.head")
    def test_delete_called_when_clip_present_and_logs_success(
        self, mock_head, mock_delete
    ):
        mock_delete.return_value = MagicMock(status_code=200)
        mock_head.return_value = MagicMock(status_code=200)

        with tempfile.TemporaryDirectory() as tmp:
            storage = os.path.join(tmp, "storage")
            cam = os.path.join(storage, "doorbell")
            event_dir = os.path.join(cam, "1234567890_evt1")
            os.makedirs(event_dir, exist_ok=True)
            with open(os.path.join(event_dir, "notification_timeline.json"), "w") as f:
                json.dump(
                    self._make_timeline_with_export_response(export_id="exp-1"),
                    f,
                    indent=2,
                )
            with open(os.path.join(event_dir, "clip.mp4"), "wb") as f:
                f.write(b"fake")
            config = {
                "STORAGE_PATH": storage,
                "FRIGATE_URL": "http://frigate:5000",
                "BUFFER_IP": "127.0.0.1",
                "FLASK_PORT": "5055",
            }
            run_once(config)
            self.assertEqual(mock_delete.call_count, 1)
            self.assertIn("/api/export/exp-1", mock_delete.call_args[0][0])
            self.assertEqual(
                mock_delete.call_args[1].get("timeout"),
                DELETE_EXPORT_TIMEOUT,
                "DELETE should be called with timeout",
            )
            found_success = any(
                record.levelno == logging.INFO
                and "Frigate export removed" in record.getMessage()
                and "exp-1" in record.getMessage()
                and "success" in record.getMessage()
                for record in self.log_capture
            )
            self.assertTrue(found_success, "Should log success for DELETE")

    @patch("frigate_buffer.services.frigate_export_watchdog.requests.delete")
    def test_delete_not_called_when_clip_missing(self, mock_delete):
        with tempfile.TemporaryDirectory() as tmp:
            storage = os.path.join(tmp, "storage")
            cam = os.path.join(storage, "doorbell")
            event_dir = os.path.join(cam, "1234567890_evt1")
            os.makedirs(event_dir, exist_ok=True)
            with open(os.path.join(event_dir, "notification_timeline.json"), "w") as f:
                json.dump(self._make_timeline_with_export_response(), f, indent=2)
            # no clip.mp4
            config = {"STORAGE_PATH": storage, "FRIGATE_URL": "http://frigate:5000"}
            run_once(config)
            self.assertEqual(mock_delete.call_count, 0)

    @patch("frigate_buffer.services.frigate_export_watchdog.requests.delete")
    @patch("frigate_buffer.services.frigate_export_watchdog.requests.head")
    def test_folder_with_only_append_timeline_is_considered(
        self, mock_head, mock_delete
    ):
        """Watchdog considers folder when only notification_timeline_append.jsonl exists."""
        mock_delete.return_value = MagicMock(status_code=200)
        mock_head.return_value = MagicMock(status_code=200)
        with tempfile.TemporaryDirectory() as tmp:
            storage = os.path.join(tmp, "storage")
            cam = os.path.join(storage, "doorbell")
            event_dir = os.path.join(cam, "1234567890_evt1")
            os.makedirs(event_dir, exist_ok=True)
            # No notification_timeline.json; only append file (app behavior).
            entry = {
                "source": "frigate_api",
                "direction": "in",
                "label": "Clip export response (1-cam)",
                "ts": "2025-01-01T12:00:05",
                "data": {
                    "frigate_response": {
                        "export_id": "exp-append-only",
                        "success": True,
                    }
                },
            }
            with open(
                os.path.join(event_dir, "notification_timeline_append.jsonl"), "w"
            ) as f:
                f.write(json.dumps(entry) + "\n")
            with open(os.path.join(event_dir, "clip.mp4"), "wb") as f:
                f.write(b"x")
            run_once({"STORAGE_PATH": storage, "FRIGATE_URL": "http://f:5000"})
            self.assertEqual(mock_delete.call_count, 1)
            self.assertIn("exp-append-only", mock_delete.call_args[0][0])

    @patch("frigate_buffer.services.frigate_export_watchdog.requests.delete")
    @patch("frigate_buffer.services.frigate_export_watchdog.requests.head")
    def test_delete_error_logs_status_and_reason(self, mock_head, mock_delete):
        mock_delete.return_value = MagicMock(
            status_code=500,
            headers={"content-type": "application/json"},
            json=MagicMock(return_value={"message": "Internal server error"}),
            text='{"message": "Internal server error"}',
        )
        mock_head.return_value = MagicMock(status_code=200)

        with tempfile.TemporaryDirectory() as tmp:
            storage = os.path.join(tmp, "storage")
            cam = os.path.join(storage, "doorbell")
            event_dir = os.path.join(cam, "1234567890_evt1")
            os.makedirs(event_dir, exist_ok=True)
            with open(os.path.join(event_dir, "notification_timeline.json"), "w") as f:
                json.dump(
                    self._make_timeline_with_export_response(export_id="exp-500"),
                    f,
                    indent=2,
                )
            with open(os.path.join(event_dir, "clip.mp4"), "wb") as f:
                f.write(b"fake")
            config = {
                "STORAGE_PATH": storage,
                "FRIGATE_URL": "http://frigate:5000",
                "BUFFER_IP": "127.0.0.1",
                "FLASK_PORT": "5055",
            }
            run_once(config)
            self.assertEqual(mock_delete.call_count, 1)
            found_warning = any(
                record.levelno == logging.WARNING
                and "Frigate export delete error" in record.getMessage()
                and "exp-500" in record.getMessage()
                and "500" in record.getMessage()
                for record in self.log_capture
            )
            self.assertTrue(
                found_warning, "Should log WARNING with status for DELETE error"
            )
            found_reason = any(
                "reason=" in record.getMessage()
                or "Internal server error" in record.getMessage()
                for record in self.log_capture
                if "Frigate export delete error" in record.getMessage()
            )
            self.assertTrue(found_reason, "Should include error reason in log")

    @patch("frigate_buffer.services.frigate_export_watchdog.requests.delete")
    def test_consolidated_event_camera_subdir_clip(self, mock_delete):
        mock_delete.return_value = MagicMock(status_code=200)
        with tempfile.TemporaryDirectory() as tmp:
            storage = os.path.join(tmp, "storage")
            events_dir = os.path.join(storage, "events")
            ce_dir = os.path.join(events_dir, "1234567890_ce1")
            os.makedirs(ce_dir, exist_ok=True)
            doorbell_dir = os.path.join(ce_dir, "doorbell")
            os.makedirs(doorbell_dir, exist_ok=True)
            with open(os.path.join(ce_dir, "notification_timeline.json"), "w") as f:
                json.dump(
                    {
                        "event_id": "ce1",
                        "entries": [
                            {
                                "source": "frigate_api",
                                "direction": "in",
                                "label": "Clip export response for Doorbell",
                                "ts": "2025-01-01T12:00:05",
                                "data": {
                                    "frigate_response": {
                                        "export_id": "exp-doorbell",
                                        "id": "exp-doorbell",
                                    }
                                },
                            },
                        ],
                    },
                    f,
                    indent=2,
                )
            with open(os.path.join(doorbell_dir, "clip.mp4"), "wb") as f:
                f.write(b"fake")
            config = {"STORAGE_PATH": storage, "FRIGATE_URL": "http://frigate:5000"}
            run_once(config)
            self.assertEqual(mock_delete.call_count, 1)
            self.assertIn("exp-doorbell", mock_delete.call_args[0][0])

    @patch("frigate_buffer.services.frigate_export_watchdog.requests.delete")
    @patch("frigate_buffer.services.frigate_export_watchdog.requests.head")
    def test_delete_200_with_json_body_logs_success_and_debug_body(
        self, mock_head, mock_delete
    ):
        mock_delete.return_value = MagicMock(
            status_code=200,
            headers={"content-type": "application/json"},
            json=MagicMock(return_value={"deleted": True}),
            text='{"deleted": true}',
        )
        mock_head.return_value = MagicMock(status_code=200)
        with tempfile.TemporaryDirectory() as tmp:
            storage = os.path.join(tmp, "storage")
            event_dir = os.path.join(storage, "cam", "123_evt1")
            os.makedirs(event_dir, exist_ok=True)
            with open(os.path.join(event_dir, "notification_timeline.json"), "w") as f:
                json.dump(
                    self._make_timeline_with_export_response(export_id="exp-body"),
                    f,
                    indent=2,
                )
            with open(os.path.join(event_dir, "clip.mp4"), "wb") as f:
                f.write(b"x")
            run_once(
                {
                    "STORAGE_PATH": storage,
                    "FRIGATE_URL": "http://f:5000",
                    "BUFFER_IP": "x",
                    "FLASK_PORT": "5055",
                }
            )
        self.assertEqual(mock_delete.call_count, 1)
        found_info = any(
            record.levelno == logging.INFO
            and "Frigate export removed" in record.getMessage()
            and "exp-body" in record.getMessage()
            for record in self.log_capture
        )
        self.assertTrue(found_info, "Should log success at INFO")
        found_debug_body = any(
            record.levelno == logging.DEBUG
            and "delete response" in record.getMessage()
            and "200" in record.getMessage()
            for record in self.log_capture
        )
        self.assertTrue(found_debug_body, "Should log response body at DEBUG")

    @patch("frigate_buffer.services.frigate_export_watchdog.requests.delete")
    @patch("frigate_buffer.services.frigate_export_watchdog.requests.head")
    def test_delete_404_logs_already_removed_at_debug_summary_at_info(
        self, mock_head, mock_delete
    ):
        mock_delete.return_value = MagicMock(
            status_code=404,
            headers={"content-type": "application/json"},
            json=MagicMock(return_value={"message": "Export not found"}),
            text='{"message": "Export not found"}',
        )
        mock_head.return_value = MagicMock(status_code=200)
        with tempfile.TemporaryDirectory() as tmp:
            storage = os.path.join(tmp, "storage")
            event_dir = os.path.join(storage, "cam", "123_evt1")
            os.makedirs(event_dir, exist_ok=True)
            with open(os.path.join(event_dir, "notification_timeline.json"), "w") as f:
                json.dump(
                    self._make_timeline_with_export_response(export_id="exp-404"),
                    f,
                    indent=2,
                )
            with open(os.path.join(event_dir, "clip.mp4"), "wb") as f:
                f.write(b"x")
            run_once(
                {
                    "STORAGE_PATH": storage,
                    "FRIGATE_URL": "http://f:5000",
                    "BUFFER_IP": "x",
                    "FLASK_PORT": "5055",
                }
            )
        self.assertEqual(mock_delete.call_count, 1)
        found_debug = any(
            record.levelno == logging.DEBUG
            and "already removed" in record.getMessage()
            and "exp-404" in record.getMessage()
            for record in self.log_capture
        )
        self.assertTrue(found_debug, "Should log already removed at DEBUG for 404")
        summary_logs = [
            r
            for r in self.log_capture
            if r.levelno == logging.INFO
            and "Export watchdog complete" in r.getMessage()
            and "already removed" in r.getMessage()
        ]
        self.assertEqual(
            len(summary_logs), 1, "Run summary at INFO should mention already removed"
        )
        found_no_warning = not any(
            record.levelno == logging.WARNING and "exp-404" in record.getMessage()
            for record in self.log_capture
        )
        self.assertTrue(found_no_warning, "404 should not be logged as WARNING")

    @patch("frigate_buffer.services.frigate_export_watchdog.requests.delete")
    @patch("frigate_buffer.services.frigate_export_watchdog.requests.head")
    def test_run_once_logs_summary_with_succeeded_failed_already_removed(
        self, mock_head, mock_delete
    ):
        mock_delete.return_value = MagicMock(status_code=200)
        mock_head.return_value = MagicMock(status_code=200)
        with tempfile.TemporaryDirectory() as tmp:
            storage = os.path.join(tmp, "storage")
            event_dir = os.path.join(storage, "cam", "123_evt1")
            os.makedirs(event_dir, exist_ok=True)
            with open(os.path.join(event_dir, "notification_timeline.json"), "w") as f:
                json.dump(
                    self._make_timeline_with_export_response(export_id="exp-summary"),
                    f,
                    indent=2,
                )
            with open(os.path.join(event_dir, "clip.mp4"), "wb") as f:
                f.write(b"x")
            run_once(
                {
                    "STORAGE_PATH": storage,
                    "FRIGATE_URL": "http://f:5000",
                    "BUFFER_IP": "x",
                    "FLASK_PORT": "5055",
                }
            )
        summary_logs = [
            r
            for r in self.log_capture
            if r.levelno == logging.INFO
            and "Export watchdog complete" in r.getMessage()
        ]
        self.assertEqual(
            len(summary_logs), 1, "Should log exactly one run summary at INFO"
        )
        msg = summary_logs[0].getMessage()
        self.assertIn("succeeded", msg)
        self.assertIn("failed", msg)
        self.assertIn("already removed", msg)


if __name__ == "__main__":
    unittest.main()
