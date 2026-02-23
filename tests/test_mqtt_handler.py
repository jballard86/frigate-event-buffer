"""Tests for MqttMessageHandler: topic routing, filtering, and delegate calls."""

import json
import unittest
from unittest.mock import MagicMock

from frigate_buffer.services.mqtt_handler import MqttMessageHandler


def _make_msg(topic: str, payload: dict) -> MagicMock:
    msg = MagicMock()
    msg.topic = topic
    msg.payload = json.dumps(payload).encode("utf-8")
    return msg


class TestMqttHandlerRouting(unittest.TestCase):
    """Verify on_message routes by topic to the correct logic."""

    def _make_handler(self, **kwargs):
        defaults = {
            "config": {},
            "state_manager": MagicMock(),
            "zone_filter": MagicMock(),
            "lifecycle_service": MagicMock(),
            "timeline_logger": MagicMock(),
            "notifier": MagicMock(),
            "file_manager": MagicMock(),
            "consolidated_manager": MagicMock(),
            "download_service": MagicMock(),
        }
        defaults.update(kwargs)
        return MqttMessageHandler(**defaults)

    def test_on_message_frigate_events_calls_handle_frigate_event_path(self):
        """frigate/events payload is processed; zone filter and lifecycle used."""
        config = {"CAMERA_LABEL_MAP": {}}
        zone_filter = MagicMock()
        zone_filter.should_start_event.return_value = True
        state_manager = MagicMock()
        state_manager.get_event.return_value = None
        lifecycle_service = MagicMock()
        handler = self._make_handler(
            config=config,
            state_manager=state_manager,
            zone_filter=zone_filter,
            lifecycle_service=lifecycle_service,
        )
        msg = _make_msg(
            "frigate/events",
            {
                "type": "new",
                "after": {
                    "id": "ev1",
                    "camera": "cam1",
                    "label": "person",
                    "start_time": 1000.0,
                    "entered_zones": ["zone1"],
                    "current_zones": ["zone1"],
                },
            },
        )
        handler.on_message(None, None, msg)
        zone_filter.should_start_event.assert_called_once()
        lifecycle_service.handle_event_new.assert_called_once()
        args = lifecycle_service.handle_event_new.call_args[0]
        self.assertEqual(args[0], "ev1")
        self.assertEqual(args[1], "cam1")
        self.assertEqual(args[2], "person")

    def test_on_message_frigate_events_filtered_camera_does_not_create(self):
        """When camera not in CAMERA_LABEL_MAP, handle_event_new is not called."""
        config = {"CAMERA_LABEL_MAP": {"other_cam": ["person"]}}
        state_manager = MagicMock()
        state_manager.get_event.return_value = None
        lifecycle_service = MagicMock()
        handler = self._make_handler(
            config=config,
            state_manager=state_manager,
            lifecycle_service=lifecycle_service,
        )
        msg = _make_msg(
            "frigate/events",
            {
                "type": "new",
                "after": {
                    "id": "ev1",
                    "camera": "filtered_cam",
                    "label": "person",
                    "start_time": 1000.0,
                },
            },
        )
        handler.on_message(None, None, msg)
        lifecycle_service.handle_event_new.assert_not_called()

    def test_on_message_frigate_events_zone_filter_rejects(self):
        """When zone filter returns False, handle_event_new is not called."""
        config = {"CAMERA_LABEL_MAP": {}}
        zone_filter = MagicMock()
        zone_filter.should_start_event.return_value = False
        state_manager = MagicMock()
        state_manager.get_event.return_value = None
        lifecycle_service = MagicMock()
        handler = self._make_handler(
            config=config,
            state_manager=state_manager,
            zone_filter=zone_filter,
            lifecycle_service=lifecycle_service,
        )
        msg = _make_msg(
            "frigate/events",
            {
                "type": "new",
                "after": {
                    "id": "ev1",
                    "camera": "cam1",
                    "label": "person",
                    "start_time": 1000.0,
                    "entered_zones": [],
                    "current_zones": [],
                },
            },
        )
        handler.on_message(None, None, msg)
        lifecycle_service.handle_event_new.assert_not_called()

    def test_on_message_tracked_object_update_calls_description_path(self):
        """tracked_object_update with type description updates state and notifier."""
        state_manager = MagicMock()
        state_manager.get_event.return_value = MagicMock(folder_path="/ev/ev1")
        timeline_logger = MagicMock()
        timeline_logger.folder_for_event.return_value = "/ev/ev1"
        notifier = MagicMock()
        file_manager = MagicMock()
        handler = self._make_handler(
            state_manager=state_manager,
            timeline_logger=timeline_logger,
            notifier=notifier,
            file_manager=file_manager,
        )
        msg = _make_msg(
            "frigate/cam1/tracked_object_update",
            {
                "id": "ev1",
                "type": "description",
                "description": "A person walking",
            },
        )
        state_manager.set_ai_description.return_value = True
        state_manager.get_event.return_value = MagicMock(folder_path="/ev/ev1")
        handler.on_message(None, None, msg)
        state_manager.set_ai_description.assert_called_once_with("ev1", "A person walking")
        notifier.publish_notification.assert_called_once()

    def test_on_message_reviews_calls_handle_review(self):
        """frigate/reviews with update type is processed; state and notifier used."""
        state_manager = MagicMock()
        state_manager.get_event.return_value = MagicMock(folder_path="/ev/ev1")
        state_manager.set_genai_metadata.return_value = True
        timeline_logger = MagicMock()
        timeline_logger.folder_for_event.return_value = "/ev/ev1"
        consolidated_manager = MagicMock()
        consolidated_manager.get_by_frigate_event.return_value = None
        notifier = MagicMock()
        file_manager = MagicMock()
        handler = self._make_handler(
            state_manager=state_manager,
            timeline_logger=timeline_logger,
            consolidated_manager=consolidated_manager,
            notifier=notifier,
            file_manager=file_manager,
        )
        msg = _make_msg(
            "frigate/reviews",
            {
                "type": "genai",
                "after": {
                    "data": {
                        "detections": ["ev1"],
                        "metadata": {
                            "title": "Person",
                            "shortSummary": "Summary",
                            "potential_threat_level": 0,
                        },
                    },
                    "severity": "detection",
                },
            },
        )
        handler.on_message(None, None, msg)
        state_manager.set_genai_metadata.assert_called_once()
        notifier.publish_notification.assert_called_once()
        self.assertEqual(notifier.publish_notification.call_args[0][1], "finalized")

    def test_on_message_invalid_json_does_not_crash(self):
        """Invalid JSON in payload does not raise; handler completes."""
        handler = self._make_handler()
        msg = MagicMock()
        msg.topic = "frigate/events"
        msg.payload = b"not valid json {"
        handler.on_message(None, None, msg)
        handler._lifecycle_service.handle_event_new.assert_not_called()

    def test_on_message_unknown_topic_ignored(self):
        """Unknown topic does not call any handler path."""
        lifecycle_service = MagicMock()
        handler = self._make_handler(lifecycle_service=lifecycle_service)
        msg = _make_msg("other/topic", {"foo": "bar"})
        handler.on_message(None, None, msg)
        lifecycle_service.handle_event_new.assert_not_called()
