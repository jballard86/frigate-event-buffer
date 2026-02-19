"""Tests for NotificationPublisher clear_tag and mark_last_event_ended behavior."""

import json
import unittest
from unittest.mock import MagicMock

from frigate_buffer.services.notifier import NotificationPublisher
from frigate_buffer.models import EventState, EventPhase


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
    def setUp(self):
        self.mqtt = MagicMock()
        self.mqtt.publish.return_value = _publish_success()
        self.publisher = NotificationPublisher(
            self.mqtt, "127.0.0.1", 5055, "", ""
        )
        # Bypass rate limiting so each call sends immediately
        self.publisher._notification_times.clear()

    def _last_payload(self) -> dict:
        """Last JSON payload passed to mqtt.publish."""
        self.mqtt.publish.assert_called()
        args = self.mqtt.publish.call_args[0]
        return json.loads(args[1])

    def test_first_notification_has_no_clear_tag(self):
        # Setup: no prior notification
        # Execute
        self.publisher.publish_notification(_make_event("evt1"), "new")
        # Verify
        payload = self._last_payload()
        self.assertNotIn("clear_tag", payload)
        self.assertEqual(payload["tag"], "frigate_evt1")

    def test_second_notification_same_event_includes_clear_tag(self):
        # Setup: send first notification
        self.publisher.publish_notification(_make_event("evt1"), "new")
        first_payload = self._last_payload()
        self.assertNotIn("clear_tag", first_payload)
        # Execute: same event, update (e.g. snapshot_ready)
        self.publisher.publish_notification(_make_event("evt1"), "snapshot_ready")
        # Verify
        payload = self._last_payload()
        self.assertIn("clear_tag", payload)
        self.assertEqual(payload["clear_tag"], "frigate_evt1")
        self.assertEqual(payload["tag"], "frigate_evt1")

    def test_after_mark_last_event_ended_next_send_has_no_clear_tag(self):
        # Setup: send one notification, then mark event ended
        self.publisher.publish_notification(_make_event("evt1"), "new")
        self.publisher.mark_last_event_ended()
        # Execute: next notification (new event)
        self.publisher.publish_notification(_make_event("evt2"), "new")
        # Verify: next send is "new event", so no clear_tag
        payload = self._last_payload()
        self.assertNotIn("clear_tag", payload)
        self.assertEqual(payload["tag"], "frigate_evt2")

    def test_different_tag_without_mark_ended_no_clear_tag(self):
        # Setup: send for evt1
        self.publisher.publish_notification(_make_event("evt1"), "new")
        # Execute: send for different event (evt2), without calling mark_last_event_ended
        self.publisher.publish_notification(_make_event("evt2"), "new")
        # Verify: different event so last_tag != current_tag -> no clear_tag
        payload = self._last_payload()
        self.assertNotIn("clear_tag", payload)
        self.assertEqual(payload["tag"], "frigate_evt2")

    def test_tag_override_used_for_current_tag(self):
        # Setup: send with tag_override (e.g. CE tag)
        self.publisher.publish_notification(
            _make_event("evt1"), "new", tag_override="frigate_ce_abc"
        )
        payload = self._last_payload()
        self.assertEqual(payload["tag"], "frigate_ce_abc")
        # Execute: same CE, update
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


if __name__ == "__main__":
    unittest.main()
