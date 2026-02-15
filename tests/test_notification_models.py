import unittest
from typing import Protocol, runtime_checkable, Optional
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from frigate_buffer.models import EventState, ConsolidatedEvent, NotificationEvent, EventPhase
except ImportError:
    # Handle case where NotificationEvent is not yet defined
    NotificationEvent = None
    from frigate_buffer.models import EventState, ConsolidatedEvent, EventPhase

class TestNotificationEventCompliance(unittest.TestCase):
    def setUp(self):
        if NotificationEvent is None:
            self.skipTest("NotificationEvent protocol not yet defined in models.py")

    def test_event_state_implements_protocol(self):
        """Verify EventState implements NotificationEvent protocol."""
        # Create a dummy EventState
        event = EventState(
            event_id="test_event",
            camera="test_cam",
            label="person",
            created_at=1234567890.0
        )
        # Manually set fields if needed or rely on defaults
        # verification
        self.assertIsInstance(event, NotificationEvent)

        # Check specific fields required by notifier
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
        """Verify ConsolidatedEvent implements NotificationEvent protocol."""
        # Create a dummy ConsolidatedEvent
        ce = ConsolidatedEvent(
            consolidated_id="ce_123",
            folder_name="123_uuid",
            folder_path="/tmp/events/123_uuid",
            start_time=1234567890.0,
            last_activity_time=1234567900.0
        )

        self.assertIsInstance(ce, NotificationEvent)

        # Verify specific fields that were missing or problematic
        self.assertTrue(hasattr(ce, 'image_url_override'))
        self.assertTrue(hasattr(ce, 'ai_description'))
        self.assertTrue(hasattr(ce, 'genai_title'))
        self.assertTrue(hasattr(ce, 'genai_description'))
        self.assertTrue(hasattr(ce, 'review_summary'))

        # Check values
        self.assertIsNone(ce.image_url_override)
        self.assertIsNone(ce.ai_description)
        self.assertIsNone(ce.review_summary)

if __name__ == '__main__':
    unittest.main()
