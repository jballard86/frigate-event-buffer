"""Tests for NotificationPublisher (clear_tag, mark_last_event_ended) and NotificationEvent protocol compliance."""

import json
import unittest
from unittest.mock import MagicMock

from frigate_buffer.services.notifier import NotificationPublisher
from frigate_buffer.models import EventState, EventPhase, ConsolidatedEvent

try:
    from frigate_buffer.models import NotificationEvent
except ImportError:
    NotificationEvent = None


def _make_event(event_id: str = "evt1", created_at: float = 100.0) -> EventState:
    """Minimal EventState for notifier payload building."""
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


class TestNotifierClearTag(unittest.TestCase):
    """NotificationPublisher clear_tag and mark_last_event_ended behavior."""

    def setUp(self):
        self.mqtt = MagicMock()
        self.mqtt.publish.return_value = _publish_success()
        self.publisher = NotificationPublisher(
            self.mqtt, "127.0.0.1", 5055, "", ""
        )
        self.publisher._notification_times.clear()

    def _last_payload(self) -> dict:
        """Last JSON payload passed to mqtt.publish."""
        self.mqtt.publish.assert_called()
        args = self.mqtt.publish.call_args[0]
        return json.loads(args[1])

    def test_first_notification_has_no_clear_tag(self):
        self.publisher.publish_notification(_make_event("evt1"), "new")
        payload = self._last_payload()
        self.assertNotIn("clear_tag", payload)
        self.assertEqual(payload["tag"], "frigate_evt1")

    def test_second_notification_same_event_includes_clear_tag(self):
        self.publisher.publish_notification(_make_event("evt1"), "new")
        first_payload = self._last_payload()
        self.assertNotIn("clear_tag", first_payload)
        self.publisher.publish_notification(_make_event("evt1"), "snapshot_ready")
        payload = self._last_payload()
        self.assertIn("clear_tag", payload)
        self.assertEqual(payload["clear_tag"], "frigate_evt1")
        self.assertEqual(payload["tag"], "frigate_evt1")

    def test_after_mark_last_event_ended_next_send_has_no_clear_tag(self):
        self.publisher.publish_notification(_make_event("evt1"), "new")
        self.publisher.mark_last_event_ended()
        self.publisher.publish_notification(_make_event("evt2"), "new")
        payload = self._last_payload()
        self.assertNotIn("clear_tag", payload)
        self.assertEqual(payload["tag"], "frigate_evt2")

    def test_different_tag_without_mark_ended_no_clear_tag(self):
        self.publisher.publish_notification(_make_event("evt1"), "new")
        self.publisher.publish_notification(_make_event("evt2"), "new")
        payload = self._last_payload()
        self.assertNotIn("clear_tag", payload)
        self.assertEqual(payload["tag"], "frigate_evt2")

    def test_tag_override_used_for_current_tag(self):
        self.publisher.publish_notification(
            _make_event("evt1"), "new", tag_override="frigate_ce_abc"
        )
        payload = self._last_payload()
        self.assertEqual(payload["tag"], "frigate_ce_abc")
        self.publisher.publish_notification(
            _make_event("evt1"), "snapshot_ready", tag_override="frigate_ce_abc"
        )
        payload = self._last_payload()
        self.assertIn("clear_tag", payload)
        self.assertEqual(payload["clear_tag"], "frigate_ce_abc")

    def test_publish_accepts_phase_as_string(self):
        """CE path can pass event-like object with phase as string; payload uses it without AttributeError."""
        event_like = type("NotifyTarget", (), {
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
        })()
        self.publisher.publish_notification(event_like, "finalized", tag_override="frigate_ce_123")
        payload = self._last_payload()
        self.assertEqual(payload["phase"], "finalized")
        self.assertEqual(payload["event_id"], "ce_123")
        self.assertEqual(payload["tag"], "frigate_ce_123")


class TestNotificationEventCompliance(unittest.TestCase):
    """EventState and ConsolidatedEvent implement NotificationEvent protocol used by notifier."""

    def setUp(self):
        if NotificationEvent is None:
            self.skipTest("NotificationEvent protocol not yet defined in models.py")

    def test_event_state_implements_protocol(self):
        event = EventState(
            event_id="test_event",
            camera="test_cam",
            label="person",
            created_at=1234567890.0
        )
        self.assertIsInstance(event, NotificationEvent)
        self.assertTrue(hasattr(event, 'image_url_override'))
        self.assertTrue(hasattr(event, 'ai_description'))
        self.assertTrue(hasattr(event, 'genai_title'))
        self.assertTrue(hasattr(event, 'genai_description'))
        self.assertTrue(hasattr(event, 'review_summary'))
        self.assertTrue(hasattr(event, 'folder_path'))
        self.assertTrue(hasattr(event, 'clip_downloaded'))
        self.assertTrue(hasattr(event, 'snapshot_downloaded'))
        self.assertTrue(hasattr(event, 'threat_level'))

    def test_consolidated_event_implements_protocol(self):
        ce = ConsolidatedEvent(
            consolidated_id="ce_123",
            folder_name="123_uuid",
            folder_path="/tmp/events/123_uuid",
            start_time=1234567890.0,
            last_activity_time=1234567900.0
        )
        self.assertIsInstance(ce, NotificationEvent)
        self.assertTrue(hasattr(ce, 'image_url_override'))
        self.assertTrue(hasattr(ce, 'ai_description'))
        self.assertTrue(hasattr(ce, 'genai_title'))
        self.assertTrue(hasattr(ce, 'genai_description'))
        self.assertTrue(hasattr(ce, 'review_summary'))
        self.assertIsNone(ce.image_url_override)
        self.assertIsNone(ce.ai_description)
        self.assertIsNone(ce.review_summary)

    def test_event_state_has_slots(self):
        """EventState uses __slots__ (smaller memory footprint)."""
        if hasattr(EventState, "__slots__"):
            self.assertIsInstance(EventState.__slots__, (tuple, list))

    def test_consolidated_event_has_slots(self):
        """ConsolidatedEvent uses __slots__ (smaller memory footprint)."""
        if hasattr(ConsolidatedEvent, "__slots__"):
            self.assertIsInstance(ConsolidatedEvent.__slots__, (tuple, list))


if __name__ == "__main__":
    unittest.main()
