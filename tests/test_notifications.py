"""Tests for NotificationDispatcher, HomeAssistantMqttProvider, and
NotificationEvent protocol compliance."""

import json
import unittest
from unittest.mock import MagicMock, patch

from frigate_buffer.models import ConsolidatedEvent, EventPhase, EventState
from frigate_buffer.services.notifications import (
    BaseNotificationProvider,
    HomeAssistantMqttProvider,
    NotificationDispatcher,
    NotificationResult,
)

try:
    from frigate_buffer.models import NotificationEvent
except ImportError:
    NotificationEvent = None


def _make_event(event_id: str = "evt1", created_at: float = 100.0) -> EventState:
    """Minimal EventState for notification payload building."""
    return EventState(
        event_id=event_id,
        camera="cam1",
        label="person",
        created_at=created_at,
        phase=EventPhase.NEW,
        folder_path=None,
    )


def _publish_success():
    """Return a result that paho treats as success."""
    r = MagicMock()
    r.rc = 0
    return r


class TestHomeAssistantMqttProvider(unittest.TestCase):
    """HomeAssistantMqttProvider: payload shape, send() and send_overflow()
    return NotificationResult."""

    def setUp(self):
        self.mqtt = MagicMock()
        self.mqtt.publish.return_value = _publish_success()
        self.provider = HomeAssistantMqttProvider(self.mqtt, "127.0.0.1", 5055, "", "")

    def _last_payload(self) -> dict:
        """Last JSON payload passed to mqtt.publish."""
        self.mqtt.publish.assert_called()
        args = self.mqtt.publish.call_args[0]
        return json.loads(args[1])

    def test_send_returns_notification_result_success(self):
        result = self.provider.send(_make_event("evt1"), "new")
        assert isinstance(result, dict)
        assert result.get("provider") == "HA_MQTT"
        assert result.get("status") == "success"
        assert "payload" in result
        assert result["payload"]["event_id"] == "evt1"
        assert result["payload"]["tag"] == "frigate_evt1"

    def test_send_payload_has_tag_phase_event_id(self):
        self.provider.send(_make_event("evt1"), "new")
        payload = self._last_payload()
        assert payload["event_id"] == "evt1"
        assert payload["tag"] == "frigate_evt1"
        assert "phase" in payload
        assert "clear_tag" not in payload

    def test_second_send_same_tag_includes_clear_tag(self):
        self.provider.send(_make_event("evt1"), "new")
        self.provider.send(_make_event("evt1"), "snapshot_ready")
        payload = self._last_payload()
        assert "clear_tag" in payload
        assert payload["clear_tag"] == "frigate_evt1"
        assert payload["tag"] == "frigate_evt1"

    def test_mark_last_event_ended_next_send_no_clear_tag(self):
        self.provider.send(_make_event("evt1"), "new")
        self.provider.mark_last_event_ended()
        self.provider.send(_make_event("evt2"), "new")
        payload = self._last_payload()
        assert "clear_tag" not in payload
        assert payload["tag"] == "frigate_evt2"

    def test_tag_override_used(self):
        self.provider.send(_make_event("evt1"), "new", tag_override="frigate_ce_abc")
        payload = self._last_payload()
        assert payload["tag"] == "frigate_ce_abc"

    def test_send_accepts_phase_as_string(self):
        """CE path can pass event-like object with phase as string."""
        event_like = type(
            "NotifyTarget",
            (),
            {
                "event_id": "ce_123",
                "camera": "events",
                "label": "person",
                "folder_path": "/storage/events/ce_123/doorbell",
                "created_at": 1234567890.0,
                "end_time": 1234567900.0,
                "phase": "finalized",
                "threat_level": 0,
                "snapshot_downloaded": True,
                "clip_downloaded": True,
                "genai_title": "Title",
                "genai_description": None,
                "ai_description": None,
                "review_summary": None,
                "image_url_override": None,
            },
        )()
        with patch(
            "frigate_buffer.services.query.resolve_clip_in_folder", return_value=None
        ):
            result = self.provider.send(
                event_like, "finalized", tag_override="frigate_ce_123"
            )
        assert result.get("provider") == "HA_MQTT"
        assert result.get("status") == "success"
        payload = self._last_payload()
        assert payload["phase"] == "finalized"
        assert payload["event_id"] == "ce_123"
        assert payload["tag"] == "frigate_ce_123"

    def test_send_overflow_returns_notification_result(self):
        result = self.provider.send_overflow()
        assert isinstance(result, dict)
        assert result.get("provider") == "HA_MQTT"
        assert result.get("status") == "success"
        self.mqtt.publish.assert_called_once()
        args = self.mqtt.publish.call_args[0]
        assert args[0] == HomeAssistantMqttProvider.TOPIC
        payload = json.loads(args[1])
        assert payload.get("event_id") == "overflow_summary"
        assert payload.get("tag") == "frigate_overflow"


class TestNotificationDispatcher(unittest.TestCase):
    """NotificationDispatcher: log_dispatch_results called with results;
    clear_tag via HA provider; empty providers."""

    def setUp(self):
        self.mqtt = MagicMock()
        self.mqtt.publish.return_value = _publish_success()
        self.timeline_logger = MagicMock()
        self.ha_provider = HomeAssistantMqttProvider(
            self.mqtt, "127.0.0.1", 5055, "", ""
        )
        self.dispatcher = NotificationDispatcher(
            providers=[self.ha_provider],
            timeline_logger=self.timeline_logger,
        )
        # Avoid rate limiting in tests
        self.dispatcher._notification_times.clear()

    def test_publish_calls_log_dispatch_results_with_event_status_results(self):
        event = _make_event("evt1")
        self.dispatcher.publish_notification(event, "new")
        self.timeline_logger.log_dispatch_results.assert_called_once()
        call_args = self.timeline_logger.log_dispatch_results.call_args[0]
        assert call_args[0] is event
        assert call_args[1] == "new"
        results = call_args[2]
        assert isinstance(results, list)
        assert len(results) > 0
        r = results[0]
        assert r.get("provider") == "HA_MQTT"
        assert r.get("status") == "success"

    def test_dispatcher_clear_tag_semantics_via_ha_provider(self):
        event = _make_event("evt1")
        self.dispatcher.publish_notification(event, "new")
        self.dispatcher.publish_notification(event, "snapshot_ready")
        payload = json.loads(self.mqtt.publish.call_args[0][1])
        assert "clear_tag" in payload
        assert payload["clear_tag"] == "frigate_evt1"

    def test_mark_last_event_ended_forwarded_to_ha_provider(self):
        event1 = _make_event("evt1")
        event2 = _make_event("evt2")
        self.dispatcher.publish_notification(event1, "new")
        self.dispatcher.mark_last_event_ended()
        self.dispatcher.publish_notification(event2, "new")
        payload = json.loads(self.mqtt.publish.call_args[0][1])
        assert "clear_tag" not in payload
        assert payload["tag"] == "frigate_evt2"

    def test_dispatcher_empty_providers_calls_log_dispatch_results_no_raise(self):
        """When HA is disabled (empty providers), publish_notification
        still logs and does not raise."""
        empty_dispatcher = NotificationDispatcher(
            providers=[],
            timeline_logger=self.timeline_logger,
        )
        event = _make_event("evt1")
        result = empty_dispatcher.publish_notification(event, "new")
        self.timeline_logger.log_dispatch_results.assert_called_once()
        call_args = self.timeline_logger.log_dispatch_results.call_args[0]
        assert call_args[0] is event
        assert call_args[1] == "new"
        assert call_args[2] == []
        assert not result  # No provider sent

    def test_mock_provider_result_passed_to_log_dispatch_results(self):
        """Dispatcher collects NotificationResult from each provider and
        passes to log_dispatch_results."""
        mock_result: NotificationResult = {"provider": "MOCK", "status": "success"}
        mock_provider = MagicMock(spec=BaseNotificationProvider)
        mock_provider.send.return_value = mock_result
        mock_provider.send_overflow.return_value = {
            "provider": "MOCK",
            "status": "success",
        }
        timeline = MagicMock()
        dispatcher = NotificationDispatcher(
            providers=[mock_provider], timeline_logger=timeline
        )
        dispatcher._notification_times.clear()

        event = _make_event("evt1")
        dispatcher.publish_notification(event, "new")

        mock_provider.send.assert_called_once()
        timeline.log_dispatch_results.assert_called_once()
        call_args = timeline.log_dispatch_results.call_args[0]
        assert call_args[2] == [mock_result]

    def test_queue_size(self):
        assert self.dispatcher.queue_size == 0

    def test_snooze_skip_when_all_cameras_snoozed_returns_false(self):
        """When snooze_manager is set and all cameras are notifications-snoozed,
        publish_notification returns False and does not call providers."""
        snooze_manager = MagicMock()
        snooze_manager.is_notifications_snoozed.return_value = True
        dispatcher = NotificationDispatcher(
            providers=[self.ha_provider],
            timeline_logger=self.timeline_logger,
            snooze_manager=snooze_manager,
        )
        dispatcher._notification_times.clear()
        event = type(
            "Event",
            (),
            {"event_id": "ce1", "camera": "cam1", "cameras": ["cam1"]},
        )()
        result = dispatcher.publish_notification(event, "finalized")
        assert result is False
        self.mqtt.publish.assert_not_called()
        self.timeline_logger.log_dispatch_results.assert_not_called()

    def test_snooze_no_skip_when_at_least_one_camera_not_snoozed(self):
        """When at least one camera in CE is not snoozed, notification is sent."""
        snooze_manager = MagicMock()
        snooze_manager.is_notifications_snoozed.side_effect = lambda c: c == "cam1"
        dispatcher = NotificationDispatcher(
            providers=[self.ha_provider],
            timeline_logger=self.timeline_logger,
            snooze_manager=snooze_manager,
        )
        dispatcher._notification_times.clear()
        event = type(
            "NotifyTarget",
            (),
            {
                "event_id": "ce1",
                "camera": "events",
                "cameras": ["cam1", "cam2"],
                "label": "person",
                "folder_path": "/storage/events/ce1/doorbell",
                "created_at": 1234567890.0,
                "end_time": 1234567900.0,
                "phase": "finalized",
                "threat_level": 0,
                "snapshot_downloaded": True,
                "clip_downloaded": True,
                "genai_title": "Title",
                "genai_description": None,
                "ai_description": None,
                "review_summary": None,
                "image_url_override": None,
            },
        )()
        result = dispatcher.publish_notification(event, "finalized")
        assert result is True
        self.mqtt.publish.assert_called_once()
        self.timeline_logger.log_dispatch_results.assert_called_once()

    def test_snooze_none_dispatcher_sends_as_before(self):
        """When snooze_manager is None, dispatcher does not skip (backward compat)."""
        event = _make_event("evt1")
        result = self.dispatcher.publish_notification(event, "new")
        assert result is True
        self.timeline_logger.log_dispatch_results.assert_called_once()


class TestNotificationEventCompliance(unittest.TestCase):
    """EventState and ConsolidatedEvent implement NotificationEvent
    protocol used by dispatcher/provider."""

    def setUp(self):
        if NotificationEvent is None:
            self.skipTest("NotificationEvent protocol not yet defined in models.py")

    def test_event_state_implements_protocol(self):
        event = EventState(
            event_id="test_event",
            camera="test_cam",
            label="person",
            created_at=1234567890.0,
        )
        assert isinstance(event, NotificationEvent)
        assert hasattr(event, "image_url_override")
        assert hasattr(event, "ai_description")
        assert hasattr(event, "genai_title")
        assert hasattr(event, "genai_description")
        assert hasattr(event, "review_summary")
        assert hasattr(event, "folder_path")
        assert hasattr(event, "clip_downloaded")
        assert hasattr(event, "snapshot_downloaded")
        assert hasattr(event, "threat_level")

    def test_consolidated_event_implements_protocol(self):
        ce = ConsolidatedEvent(
            consolidated_id="ce_123",
            folder_name="123_uuid",
            folder_path="/tmp/events/123_uuid",
            start_time=1234567890.0,
            last_activity_time=1234567900.0,
        )
        assert isinstance(ce, NotificationEvent)
        assert hasattr(ce, "image_url_override")
        assert hasattr(ce, "ai_description")
        assert hasattr(ce, "genai_title")
        assert hasattr(ce, "genai_description")
        assert hasattr(ce, "review_summary")
        assert ce.image_url_override is None
        assert ce.ai_description is None
        assert ce.review_summary is None

    def test_event_state_has_slots(self):
        if hasattr(EventState, "__slots__"):
            assert isinstance(EventState.__slots__, (tuple, list))

    def test_consolidated_event_has_slots(self):
        if hasattr(ConsolidatedEvent, "__slots__"):
            assert isinstance(ConsolidatedEvent.__slots__, (tuple, list))


if __name__ == "__main__":
    unittest.main()
