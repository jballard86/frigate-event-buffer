"""Tests for QuickTitleService: fetch latest.jpg, YOLO, crop, AI title,
state update, notify."""

import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import requests

from frigate_buffer.services.quick_title_service import QuickTitleService

# Capture real RequestException at import time so the test can raise it even if
# sys.modules["requests"] is later replaced by other test modules.
_RequestException = requests.RequestException


class TestQuickTitleService(unittest.TestCase):
    """Setup → Execute → Verify tests for run_quick_title."""

    def _make_service(
        self,
        config=None,
        state_manager=None,
        file_manager=None,
        consolidated_manager=None,
        video_service=None,
        ai_analyzer=None,
        notifier=None,
        snooze_manager=None,
    ):
        config = config or {"FRIGATE_URL": "http://frigate:5000"}
        state_manager = state_manager or MagicMock()
        file_manager = file_manager or MagicMock()
        consolidated_manager = consolidated_manager or MagicMock()
        video_service = video_service or MagicMock()
        ai_analyzer = ai_analyzer or MagicMock()
        notifier = notifier or MagicMock()
        return QuickTitleService(
            config,
            state_manager,
            file_manager,
            consolidated_manager,
            video_service,
            ai_analyzer,
            notifier,
            snooze_manager=snooze_manager,
        )

    def test_run_quick_title_skips_when_no_ai_analyzer(self):
        """When ai_analyzer is None, run_quick_title returns without calling
        fetch or notifier."""
        notifier = MagicMock()
        service = self._make_service(ai_analyzer=None, notifier=notifier)
        service.run_quick_title(
            "ev1", "cam1", "person", "ce1", "/storage/events/ce1/cam1", None
        )
        notifier.publish_notification.assert_not_called()

    def test_run_quick_title_skips_when_no_frigate_url(self):
        """When FRIGATE_URL is empty, run_quick_title returns without fetching."""
        notifier = MagicMock()
        service = self._make_service(config={}, notifier=notifier)
        service.run_quick_title(
            "ev1", "cam1", "person", "ce1", "/storage/events/ce1/cam1", None
        )
        notifier.publish_notification.assert_not_called()

    @patch("frigate_buffer.services.quick_title_service.requests.get")
    def test_run_quick_title_fetch_failure_returns_early(self, mock_get):
        """When requests.get raises, run_quick_title returns without notifying."""

        def raise_request_exception(*args, **kwargs):
            raise _RequestException("network error")

        mock_get.side_effect = raise_request_exception
        notifier = MagicMock()
        service = self._make_service(notifier=notifier)
        service.run_quick_title(
            "ev1", "cam1", "person", "ce1", "/storage/events/ce1/cam1", None
        )
        notifier.publish_notification.assert_not_called()

    @patch("frigate_buffer.services.quick_title_service.cv2.imdecode")
    @patch("frigate_buffer.services.quick_title_service.requests.get")
    def test_run_quick_title_decode_failure_returns_early(
        self, mock_get, mock_imdecode
    ):
        """When imdecode returns None, run_quick_title returns without notifying."""
        mock_get.return_value.content = b"\xff\xd8\xff"
        mock_get.return_value.raise_for_status = MagicMock()
        mock_imdecode.return_value = None
        notifier = MagicMock()
        service = self._make_service(notifier=notifier)
        service.run_quick_title(
            "ev1", "cam1", "person", "ce1", "/storage/events/ce1/cam1", None
        )
        notifier.publish_notification.assert_not_called()

    @patch("frigate_buffer.services.quick_title_service.requests.get")
    def test_run_quick_title_no_title_returned_does_not_notify(self, mock_get):
        """When generate_quick_title returns None or empty title,
        no notification is sent."""
        mock_get.return_value.content = b"\xff\xd8\xff"
        mock_get.return_value.raise_for_status = MagicMock()
        with patch(
            "frigate_buffer.services.quick_title_service.cv2.imdecode"
        ) as mock_decode:
            mock_decode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            video_service = MagicMock()
            video_service.run_detection_on_image.return_value = []
            ai_analyzer = MagicMock()
            ai_analyzer.generate_quick_title.return_value = None
            notifier = MagicMock()
            service = self._make_service(
                video_service=video_service,
                ai_analyzer=ai_analyzer,
                notifier=notifier,
            )
            state_manager = MagicMock()
            state_manager.get_event.return_value = MagicMock(
                event_id="ev1",
                genai_description="",
                severity="detection",
                threat_level=0,
                genai_scene=None,
                folder_path="/storage/cam1/ev1",
            )
            service._state_manager = state_manager
            service.run_quick_title(
                "ev1", "cam1", "person", "ce1", "/storage/events/ce1/cam1", None
            )
            notifier.publish_notification.assert_not_called()

    def test_run_quick_title_skips_when_ai_snoozed(self):
        """When snooze_manager is set and camera is AI-snoozed, run_quick_title
        returns without fetching or notifying."""
        snooze_manager = MagicMock()
        snooze_manager.is_ai_snoozed.return_value = True
        notifier = MagicMock()
        service = self._make_service(notifier=notifier, snooze_manager=snooze_manager)
        service.run_quick_title(
            "ev1", "cam1", "person", "ce1", "/storage/events/ce1/cam1", None
        )
        snooze_manager.is_ai_snoozed.assert_called_once_with("cam1")
        notifier.publish_notification.assert_not_called()

    @patch("frigate_buffer.services.quick_title_service.requests.get")
    def test_run_quick_title_writes_snapshot_cropped_when_detections(self, mock_get):
        """When detections exist, write_stitched_frame is called with
        snapshot_cropped.jpg in the camera folder."""
        mock_get.return_value.content = b"\xff\xd8\xff"
        mock_get.return_value.raise_for_status = MagicMock()
        with patch(
            "frigate_buffer.services.quick_title_service.cv2.imdecode"
        ) as mock_decode:
            mock_decode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            video_service = MagicMock()
            video_service.run_detection_on_image.return_value = [
                {"bbox": [10, 10, 50, 50]}
            ]
            ai_analyzer = MagicMock()
            ai_analyzer.generate_quick_title.return_value = {
                "title": "Person",
                "description": "",
            }
            file_manager = MagicMock()
            file_manager.write_stitched_frame.return_value = True
            state_manager = MagicMock()
            event = MagicMock(
                event_id="ev1",
                genai_description="",
                severity="detection",
                threat_level=0,
                genai_scene=None,
                folder_path="/storage/cam1/ev1",
                created_at=1000.0,
                end_time=1010.0,
            )
            state_manager.get_event.return_value = event
            consolidated_manager = MagicMock()
            ce = MagicMock(
                consolidated_id="ce1",
                camera="cam1",
                label="person",
                folder_path="/storage/events/ce1",
                primary_camera="cam1",
                primary_event_id="ev1",
                start_time=1000.0,
                end_time=1010.0,
                final_threat_level=0,
                severity="detection",
                cameras=["cam1"],
            )
            consolidated_manager.get_by_frigate_event.return_value = ce
            notifier = MagicMock()
            service = self._make_service(
                state_manager=state_manager,
                file_manager=file_manager,
                consolidated_manager=consolidated_manager,
                video_service=video_service,
                ai_analyzer=ai_analyzer,
                notifier=notifier,
            )
            camera_folder = "/storage/events/ce1/cam1"
            service.run_quick_title("ev1", "cam1", "person", "ce1", camera_folder, None)
            file_manager.write_stitched_frame.assert_called_once()
            call_args = file_manager.write_stitched_frame.call_args[0]
            expected_path = os.path.join(camera_folder, "snapshot_cropped.jpg")
            assert call_args[1] == expected_path

    @patch("frigate_buffer.services.quick_title_service.requests.get")
    def test_run_quick_title_updates_state_and_notifies_with_tag(self, mock_get):
        """Happy path: state updated, write_summary and write_metadata_json called,
        notification with title and description."""
        mock_get.return_value.content = b"\xff\xd8\xff"
        mock_get.return_value.raise_for_status = MagicMock()
        with patch(
            "frigate_buffer.services.quick_title_service.cv2.imdecode"
        ) as mock_decode:
            mock_decode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            video_service = MagicMock()
            video_service.run_detection_on_image.return_value = []
            ai_analyzer = MagicMock()
            ai_analyzer.generate_quick_title.return_value = {
                "title": "Person at door",
                "description": "A person is standing at the front door.",
            }
            state_manager = MagicMock()
            event = MagicMock(
                event_id="ev1",
                genai_description="",
                severity="detection",
                threat_level=0,
                genai_scene=None,
                folder_path="/storage/cam1/ev1",
                created_at=1000.0,
                end_time=1010.0,
            )
            state_manager.get_event.return_value = event
            file_manager = MagicMock()
            consolidated_manager = MagicMock()
            ce = MagicMock(
                consolidated_id="ce1",
                camera="cam1",
                label="person",
                folder_path="/storage/events/ce1",
                primary_camera="cam1",
                primary_event_id="ev1",
                start_time=1000.0,
                end_time=1010.0,
                final_description="",
                final_threat_level=0,
                severity="detection",
            )
            consolidated_manager.get_by_frigate_event.return_value = ce
            notifier = MagicMock()
            service = self._make_service(
                state_manager=state_manager,
                file_manager=file_manager,
                consolidated_manager=consolidated_manager,
                video_service=video_service,
                ai_analyzer=ai_analyzer,
                notifier=notifier,
            )
            service.run_quick_title(
                "ev1",
                "cam1",
                "person",
                "ce1",
                "/storage/events/ce1/cam1",
                tag_override=None,
            )
            state_manager.set_genai_metadata.assert_called_once()
            file_manager.write_summary.assert_called_once()
            file_manager.write_metadata_json.assert_called_once()
            # Quick title does not set CE.final_* (final = CE analysis only for
            # external_api)
            notifier.publish_notification.assert_called_once()
            call_args = notifier.publish_notification.call_args
            assert call_args[0][1] == "snapshot_ready"
            assert call_args[1]["tag_override"] == "frigate_ce1"
            notify_target = call_args[0][0]
            assert getattr(notify_target, "genai_title", None) == "Person at door"
            assert (
                getattr(notify_target, "genai_description", None)
                == "A person is standing at the front door."
            )

    @patch("frigate_buffer.services.quick_title_service.requests.get")
    def test_run_quick_title_event_gone_does_not_notify(self, mock_get):
        """When event is no longer in state (first get_event after title returns None),
        do not notify."""
        mock_get.return_value.content = b"\xff\xd8\xff"
        mock_get.return_value.raise_for_status = MagicMock()
        with patch(
            "frigate_buffer.services.quick_title_service.cv2.imdecode"
        ) as mock_decode:
            mock_decode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            video_service = MagicMock()
            video_service.run_detection_on_image.return_value = []
            ai_analyzer = MagicMock()
            ai_analyzer.generate_quick_title.return_value = {
                "title": "Title",
                "description": "",
            }
            state_manager = MagicMock()
            state_manager.get_event.return_value = None
            notifier = MagicMock()
            service = self._make_service(
                state_manager=state_manager,
                video_service=video_service,
                ai_analyzer=ai_analyzer,
                notifier=notifier,
            )
            service.run_quick_title(
                "ev1", "cam1", "person", "ce1", "/storage/events/ce1/cam1", None
            )
            notifier.publish_notification.assert_not_called()
