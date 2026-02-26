import sys
from unittest.mock import MagicMock

# Keys to mock so this module can import timeline without pulling in heavy deps.
# We mock in setup_module and restore in teardown_module so other test modules
# never see the mocks.
_MODULE_KEYS = (
    "requests",
    "flask",
    "paho",
    "paho.mqtt",
    "paho.mqtt.client",
    "schedule",
    "yaml",
    "voluptuous",
)
_saved_modules = {}

import unittest


def setup_module():
    for k in _MODULE_KEYS:
        _saved_modules[k] = sys.modules.get(k)
        sys.modules[k] = MagicMock()
    from frigate_buffer.services.timeline import TimelineLogger as TL

    mod = sys.modules[__name__]
    mod.TimelineLogger = TL


def teardown_module():
    for k in _MODULE_KEYS:
        if _saved_modules.get(k) is not None:
            sys.modules[k] = _saved_modules[k]
        elif k in sys.modules:
            del sys.modules[k]


TimelineLogger = None


def _ensure_imports():
    """Run when module is executed without pytest (e.g. python -m unittest)."""
    global TimelineLogger
    if TimelineLogger is None:
        from frigate_buffer.services.timeline import TimelineLogger as TL

        globals()["TimelineLogger"] = TL


class TestTimelineLogger(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        _ensure_imports()
        self.mock_file_manager = MagicMock()
        self.mock_consolidated_manager = MagicMock()
        self.logger = TimelineLogger(
            self.mock_file_manager, self.mock_consolidated_manager
        )

    def test_folder_for_event_none(self):
        """Test folder_for_event with None event."""
        self.assertIsNone(self.logger.folder_for_event(None))

    def test_folder_for_event_consolidated(self):
        """Test folder_for_event with an event that is part of a consolidated event."""
        mock_event = MagicMock()
        mock_event.event_id = "test_event"
        mock_event.folder_path = "/path/to/event"

        mock_ce = MagicMock()
        mock_ce.folder_path = "/path/to/ce"

        self.mock_consolidated_manager.get_by_frigate_event.return_value = mock_ce

        folder = self.logger.folder_for_event(mock_event)

        self.assertEqual(folder, "/path/to/ce")
        self.mock_consolidated_manager.get_by_frigate_event.assert_called_once_with(
            "test_event"
        )

    def test_folder_for_event_not_consolidated(self):
        """Test folder_for_event with an event that is NOT part of a
        consolidated event."""
        mock_event = MagicMock()
        mock_event.event_id = "test_event"
        mock_event.folder_path = "/path/to/event"

        self.mock_consolidated_manager.get_by_frigate_event.return_value = None

        folder = self.logger.folder_for_event(mock_event)

        self.assertEqual(folder, "/path/to/event")
        self.mock_consolidated_manager.get_by_frigate_event.assert_called_once_with(
            "test_event"
        )

    def test_log_ha_success(self):
        """Test log_ha successfully appends timeline entry."""
        mock_event = MagicMock()
        mock_event.event_id = "test_event"
        mock_event.folder_path = "/path/to/event"
        self.mock_consolidated_manager.get_by_frigate_event.return_value = None

        status = "delivered"
        payload = {"message": "Person detected"}

        self.logger.log_ha(mock_event, status, payload)

        self.mock_file_manager.append_timeline_entry.assert_called_once_with(
            "/path/to/event",
            {
                "source": "ha_notification",
                "direction": "out",
                "label": f"Sent to Home Assistant: {status}",
                "data": payload,
            },
        )

    def test_log_ha_no_folder(self):
        """Test log_ha does nothing if no folder is found."""
        self.logger.log_ha(None, "status", {})
        self.mock_file_manager.append_timeline_entry.assert_not_called()

    def test_log_mqtt_success(self):
        """Test log_mqtt successfully appends timeline entry."""
        folder_path = "/path/to/event"
        topic = "frigate/events"
        payload = {"id": "123"}
        label = "MQTT Event"

        self.logger.log_mqtt(folder_path, topic, payload, label)

        self.mock_file_manager.append_timeline_entry.assert_called_once_with(
            folder_path,
            {
                "source": "frigate_mqtt",
                "direction": "in",
                "label": label,
                "data": {"topic": topic, "payload": payload},
            },
        )

    def test_log_mqtt_no_folder(self):
        """Test log_mqtt does nothing if no folder_path is provided."""
        self.logger.log_mqtt(None, "topic", {}, "label")
        self.mock_file_manager.append_timeline_entry.assert_not_called()

    def test_log_frigate_api_success(self):
        """Test log_frigate_api successfully appends timeline entry."""
        folder_path = "/path/to/event"
        direction = "out"
        label = "Get Event"
        data = {"event_id": "123"}

        self.logger.log_frigate_api(folder_path, direction, label, data)

        self.mock_file_manager.append_timeline_entry.assert_called_once_with(
            folder_path,
            {
                "source": "frigate_api",
                "direction": direction,
                "label": label,
                "data": data,
            },
        )

    def test_log_frigate_api_no_folder(self):
        """Test log_frigate_api does nothing if no folder_path is provided."""
        self.logger.log_frigate_api(None, "in", "label", {})
        self.mock_file_manager.append_timeline_entry.assert_not_called()

    def test_log_dispatch_results_success(self):
        """Test log_dispatch_results appends generic notification_dispatch entry."""
        mock_event = MagicMock()
        mock_event.event_id = "ev_123"
        mock_event.folder_path = "/path/to/event"
        self.mock_consolidated_manager.get_by_frigate_event.return_value = None

        results = [
            {"provider": "HA_MQTT", "status": "success"},
            {"provider": "OTHER", "status": "failure", "message": "timeout"},
        ]
        self.logger.log_dispatch_results(mock_event, "finalized", results)

        self.mock_file_manager.append_timeline_entry.assert_called_once_with(
            "/path/to/event",
            {
                "source": "notification_dispatch",
                "direction": "out",
                "label": "Notification: finalized — HA_MQTT: success; OTHER: failure",
                "data": {"event_phase": "finalized", "results": results},
            },
        )

    def test_log_dispatch_results_no_folder(self):
        """Test log_dispatch_results does nothing when event has no folder."""
        self.logger.log_dispatch_results(None, "finalized", [])
        self.mock_file_manager.append_timeline_entry.assert_not_called()


if __name__ == "__main__":
    unittest.main()
