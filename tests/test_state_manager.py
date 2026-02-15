import sys
from unittest.mock import MagicMock

# Mock dependencies before importing project modules
sys.modules['requests'] = MagicMock()
sys.modules['flask'] = MagicMock()
sys.modules['paho'] = MagicMock()
sys.modules['paho.mqtt'] = MagicMock()
sys.modules['paho.mqtt.client'] = MagicMock()
sys.modules['schedule'] = MagicMock()
sys.modules['yaml'] = MagicMock()
sys.modules['voluptuous'] = MagicMock()

import unittest
from frigate_buffer.managers.state import EventStateManager
from frigate_buffer.models import EventState, EventPhase

class TestEventStateManager(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        self.manager = EventStateManager()
        self.event_id = "123456.789-new"
        self.camera = "front_door"
        self.label = "person"
        self.start_time = 123456.789

    def test_create_event_new(self):
        """Test creating a new event in NEW phase."""
        event = self.manager.create_event(self.event_id, self.camera, self.label, self.start_time)

        self.assertIsInstance(event, EventState)
        self.assertEqual(event.event_id, self.event_id)
        self.assertEqual(event.camera, self.camera)
        self.assertEqual(event.label, self.label)
        self.assertEqual(event.phase, EventPhase.NEW)
        self.assertEqual(event.created_at, self.start_time)

        # Verify it's stored
        stored_event = self.manager.get_event(self.event_id)
        self.assertEqual(stored_event, event)

    def test_create_event_existing(self):
        """Test creating an event that already exists returns the existing one."""
        event1 = self.manager.create_event(self.event_id, self.camera, self.label, self.start_time)
        event1.phase = EventPhase.DESCRIBED  # Modify to distinguish

        # Try creating again with same ID
        event2 = self.manager.create_event(self.event_id, self.camera, self.label, self.start_time)

        self.assertEqual(event1, event2)
        self.assertEqual(event2.phase, EventPhase.DESCRIBED)  # Should still be DESCRIBED

    def test_get_event(self):
        """Test retrieving events."""
        # Non-existent
        self.assertIsNone(self.manager.get_event("nonexistent"))

        # Existent
        self.manager.create_event(self.event_id, self.camera, self.label, self.start_time)
        self.assertIsNotNone(self.manager.get_event(self.event_id))

    def test_set_ai_description(self):
        """Test setting AI description and advancing to DESCRIBED phase."""
        self.manager.create_event(self.event_id, self.camera, self.label, self.start_time)

        description = "A person walking towards the door."
        success = self.manager.set_ai_description(self.event_id, description)

        self.assertTrue(success)
        event = self.manager.get_event(self.event_id)
        self.assertEqual(event.ai_description, description)
        self.assertEqual(event.phase, EventPhase.DESCRIBED)

    def test_set_ai_description_nonexistent(self):
        """Test setting AI description for non-existent event."""
        success = self.manager.set_ai_description("nonexistent", "desc")
        self.assertFalse(success)

    def test_set_genai_metadata(self):
        """Test setting GenAI metadata and advancing to FINALIZED phase."""
        self.manager.create_event(self.event_id, self.camera, self.label, self.start_time)

        title = "Suspicious Person"
        desc = "A person loitering near the door."
        severity = "suspicious"
        threat_level = 1
        scene = "Front porch scene"

        success = self.manager.set_genai_metadata(
            self.event_id, title, desc, severity, threat_level, scene
        )

        self.assertTrue(success)
        event = self.manager.get_event(self.event_id)
        self.assertEqual(event.genai_title, title)
        self.assertEqual(event.genai_description, desc)
        self.assertEqual(event.severity, severity)
        self.assertEqual(event.threat_level, threat_level)
        self.assertEqual(event.genai_scene, scene)
        self.assertEqual(event.phase, EventPhase.FINALIZED)

    def test_set_genai_metadata_nonexistent(self):
        """Test setting GenAI metadata for non-existent event."""
        success = self.manager.set_genai_metadata("nonexistent", "title", "desc", "severity")
        self.assertFalse(success)

    def test_set_review_summary(self):
        """Test setting review summary and advancing to SUMMARIZED phase."""
        self.manager.create_event(self.event_id, self.camera, self.label, self.start_time)

        summary = "Daily summary: nothing happened."
        success = self.manager.set_review_summary(self.event_id, summary)

        self.assertTrue(success)
        event = self.manager.get_event(self.event_id)
        self.assertEqual(event.review_summary, summary)
        self.assertEqual(event.phase, EventPhase.SUMMARIZED)

    def test_set_review_summary_nonexistent(self):
        """Test setting review summary for non-existent event."""
        success = self.manager.set_review_summary("nonexistent", "summary")
        self.assertFalse(success)

    def test_mark_event_ended(self):
        """Test marking event as ended."""
        self.manager.create_event(self.event_id, self.camera, self.label, self.start_time)

        end_time = self.start_time + 10.0
        event = self.manager.mark_event_ended(self.event_id, end_time, has_clip=True, has_snapshot=False)

        self.assertIsNotNone(event)
        self.assertEqual(event.end_time, end_time)
        self.assertTrue(event.has_clip)
        self.assertFalse(event.has_snapshot)

    def test_remove_event(self):
        """Test removing an event."""
        self.manager.create_event(self.event_id, self.camera, self.label, self.start_time)

        removed = self.manager.remove_event(self.event_id)
        self.assertIsNotNone(removed)
        self.assertEqual(removed.event_id, self.event_id)

        # Verify it's gone
        self.assertIsNone(self.manager.get_event(self.event_id))

        # Remove again
        self.assertIsNone(self.manager.remove_event(self.event_id))

    def test_get_active_event_ids(self):
        """Test getting list of active event IDs."""
        ids = self.manager.get_active_event_ids()
        self.assertEqual(len(ids), 0)

        self.manager.create_event("evt1", "cam1", "person", 100.0)
        self.manager.create_event("evt2", "cam2", "car", 100.0)

        ids = self.manager.get_active_event_ids()
        self.assertEqual(len(ids), 2)
        self.assertIn("evt1", ids)
        self.assertIn("evt2", ids)

    def test_get_stats(self):
        """Test getting statistics."""
        # Create events in different phases
        # NEW
        self.manager.create_event("evt1", "cam1", "person", 100.0)

        # DESCRIBED
        self.manager.create_event("evt2", "cam1", "car", 100.0)
        self.manager.set_ai_description("evt2", "desc")

        # FINALIZED
        self.manager.create_event("evt3", "cam2", "person", 100.0)
        self.manager.set_genai_metadata("evt3", "title", "desc", "severity")

        stats = self.manager.get_stats()

        self.assertEqual(stats["total_active"], 3)
        self.assertEqual(stats["by_phase"]["NEW"], 1)
        self.assertEqual(stats["by_phase"]["DESCRIBED"], 1)
        self.assertEqual(stats["by_phase"]["FINALIZED"], 1)
        self.assertEqual(stats["by_camera"]["cam1"], 2)
        self.assertEqual(stats["by_camera"]["cam2"], 1)

if __name__ == '__main__':
    unittest.main()
