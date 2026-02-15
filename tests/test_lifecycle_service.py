import unittest
from unittest.mock import MagicMock, patch, ANY
import time
from frigate_buffer.services.lifecycle import EventLifecycleService
from frigate_buffer.models import EventState, ConsolidatedEvent

class TestEventLifecycleService(unittest.TestCase):
    def setUp(self):
        self.config = {
            'NOTIFICATION_DELAY': 0.1,
            'EXPORT_BUFFER_BEFORE': 5,
            'EXPORT_BUFFER_AFTER': 30,
            'SUMMARY_PADDING_BEFORE': 15,
            'SUMMARY_PADDING_AFTER': 15,
            'FRIGATE_URL': 'http://frigate',
            'BUFFER_IP': 'localhost',
            'FLASK_PORT': 5000
        }
        self.state_manager = MagicMock()
        self.file_manager = MagicMock()
        self.consolidated_manager = MagicMock()
        self.consolidated_manager._lock = MagicMock()
        self.consolidated_manager._lock.__enter__ = MagicMock()
        self.consolidated_manager._lock.__exit__ = MagicMock()
        self.video_service = MagicMock()
        self.notifier = MagicMock()
        self.timeline_logger = MagicMock()

        self.service = EventLifecycleService(
            self.config,
            self.state_manager,
            self.file_manager,
            self.consolidated_manager,
            self.video_service,
            self.notifier,
            self.timeline_logger
        )

    def test_handle_event_new_new_event(self):
        # Setup
        event_id = "123456.789-new"
        camera = "front_door"
        label = "person"
        start_time = 123456.789

        event = EventState(event_id, camera, label, start_time)
        self.state_manager.create_event.return_value = event

        ce = ConsolidatedEvent("ce_id", "folder", "path", start_time, start_time)
        # get_or_create returns (ce, is_new, camera_folder)
        self.consolidated_manager.get_or_create.return_value = (ce, True, "path/cam")

        # Act
        self.service.handle_event_new(event_id, camera, label, start_time, {})

        # Assert
        self.state_manager.create_event.assert_called_with(event_id, camera, label, start_time)
        self.consolidated_manager.get_or_create.assert_called_with(event_id, camera, label, start_time)

    @patch('frigate_buffer.services.lifecycle.threading.Thread')
    def test_handle_event_new_threads(self, mock_thread):
        # Setup
        event_id = "123456.789-new"
        camera = "front_door"
        label = "person"
        start_time = 123456.789
        event = EventState(event_id, camera, label, start_time)
        self.state_manager.create_event.return_value = event
        ce = ConsolidatedEvent("ce_id", "folder", "path", start_time, start_time)
        self.consolidated_manager.get_or_create.return_value = (ce, True, "path/cam")

        # Act
        self.service.handle_event_new(event_id, camera, label, start_time, {})

        # Assert
        # Check if notification thread started
        mock_thread.assert_called()
        args, kwargs = mock_thread.call_args
        self.assertEqual(kwargs['target'], self.service._send_initial_notification)

    def test_process_event_end(self):
        # Setup
        event = EventState("evt1", "cam1", "person", 100.0)
        event.folder_path = "/tmp/evt1"
        event.has_clip = True
        event.has_snapshot = True
        event.end_time = 110.0

        # Mock CE found
        ce = ConsolidatedEvent("ce1", "f1", "p1", 100.0, 110.0)
        self.consolidated_manager.get_by_frigate_event.return_value = ce

        # Return value for cleanup
        self.file_manager.cleanup_old_events.return_value = 5

        # Act
        self.service.process_event_end(event)

        # Assert
        self.file_manager.download_snapshot.assert_called_with("evt1", "/tmp/evt1")
        self.consolidated_manager.update_activity.assert_called()
        self.consolidated_manager.schedule_close_timer.assert_called_with("ce1")
        # Cleanup called
        self.file_manager.cleanup_old_events.assert_called()
        self.assertIsNotNone(self.service.last_cleanup_time)

    def test_finalize_consolidated_event(self):
        # Setup
        ce_id = "ce1"
        ce = ConsolidatedEvent(ce_id, "folder", "path", 100.0, 110.0)

        # Ensure mark_closing returns True to proceed
        self.consolidated_manager.mark_closing.return_value = True

        ce.frigate_event_ids = ["evt1"]
        ce.cameras = ["cam1"]
        ce.primary_camera = "cam1"

        self.consolidated_manager._events = {ce_id: ce}
        self.consolidated_manager._lock = MagicMock()

        evt1 = EventState("evt1", "cam1", "person", 100.0)
        evt1.end_time = 110.0
        self.state_manager.get_event.return_value = evt1

        self.file_manager.ensure_consolidated_camera_folder.return_value = "/tmp/ce1/cam1"
        self.file_manager.export_and_transcode_clip.return_value = {"success": True}

        # Act
        self.service.finalize_consolidated_event(ce_id)

        # Assert
        self.consolidated_manager.mark_closing.assert_called_with(ce_id)
        self.file_manager.export_and_transcode_clip.assert_called()
        self.consolidated_manager.remove.assert_called_with(ce_id)

if __name__ == '__main__':
    unittest.main()
