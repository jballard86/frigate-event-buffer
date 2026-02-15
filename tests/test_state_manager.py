import unittest
import time
from frigate_buffer.managers.state import EventStateManager
from frigate_buffer.models import EventPhase

class TestEventStateManager(unittest.TestCase):
    def setUp(self):
        self.manager = EventStateManager()

    def test_create_event_success(self):
        event_id = "test_event_1"
        camera = "front_door"
        label = "person"
        start_time = 123456789.0

        event = self.manager.create_event(event_id, camera, label, start_time)

        self.assertEqual(event.event_id, event_id)
        self.assertEqual(event.camera, camera)
        self.assertEqual(event.label, label)
        self.assertEqual(event.created_at, start_time)
        self.assertEqual(event.phase, EventPhase.NEW)

        # Verify it's in the manager
        self.assertEqual(self.manager.get_event(event_id), event)

    def test_create_event_duplicate(self):
        event_id = "test_event_1"
        camera = "front_door"
        label = "person"
        start_time = 123456789.0

        event1 = self.manager.create_event(event_id, camera, label, start_time)

        # Try creating again with same ID but different parameters
        event2 = self.manager.create_event(event_id, "other_camera", "other_label", start_time + 10)

        # Should return the exact same object
        self.assertIs(event1, event2)

        # Parameters of the first creation should be preserved (idempotency)
        self.assertEqual(event2.camera, camera)
        self.assertEqual(event2.label, label)
        self.assertEqual(event2.created_at, start_time)

    def test_get_event(self):
        self.assertIsNone(self.manager.get_event("non_existent"))

        event_id = "test_event_1"
        self.manager.create_event(event_id, "cam", "label", 100.0)

        event = self.manager.get_event(event_id)
        self.assertIsNotNone(event)
        self.assertEqual(event.event_id, event_id)

    def test_remove_event(self):
        event_id = "test_event_1"
        self.manager.create_event(event_id, "cam", "label", 100.0)

        removed = self.manager.remove_event(event_id)
        self.assertIsNotNone(removed)
        self.assertEqual(removed.event_id, event_id)

        self.assertIsNone(self.manager.get_event(event_id))

        # Removing non-existent
        self.assertIsNone(self.manager.remove_event("non_existent"))

    def test_set_ai_description(self):
        event_id = "test_event_1"
        self.manager.create_event(event_id, "cam", "label", 100.0)

        description = "A person carrying a box"
        success = self.manager.set_ai_description(event_id, description)

        self.assertTrue(success)
        event = self.manager.get_event(event_id)
        self.assertEqual(event.ai_description, description)
        self.assertEqual(event.phase, EventPhase.DESCRIBED)

        # Update description again
        new_description = "A person wearing a red shirt"
        success = self.manager.set_ai_description(event_id, new_description)
        self.assertTrue(success)
        self.assertEqual(event.ai_description, new_description)
        self.assertEqual(event.phase, EventPhase.DESCRIBED)

    def test_set_genai_metadata(self):
        event_id = "test_event_1"
        self.manager.create_event(event_id, "cam", "label", 100.0)

        success = self.manager.set_genai_metadata(
            event_id,
            title="Suspicious Activity",
            description="Person looking into windows",
            severity="suspicious",
            threat_level=1,
            scene="Front yard at night"
        )

        self.assertTrue(success)
        event = self.manager.get_event(event_id)
        self.assertEqual(event.genai_title, "Suspicious Activity")
        self.assertEqual(event.genai_description, "Person looking into windows")
        self.assertEqual(event.severity, "suspicious")
        self.assertEqual(event.threat_level, 1)
        self.assertEqual(event.genai_scene, "Front yard at night")
        self.assertEqual(event.phase, EventPhase.FINALIZED)

    def test_set_review_summary(self):
        event_id = "test_event_1"
        self.manager.create_event(event_id, "cam", "label", 100.0)

        summary = "Summary of the event"
        success = self.manager.set_review_summary(event_id, summary)

        self.assertTrue(success)
        event = self.manager.get_event(event_id)
        self.assertEqual(event.review_summary, summary)
        self.assertEqual(event.phase, EventPhase.SUMMARIZED)

    def test_mark_event_ended(self):
        event_id = "test_event_1"
        self.manager.create_event(event_id, "cam", "label", 100.0)

        event = self.manager.mark_event_ended(event_id, 150.0, True, False)

        self.assertIsNotNone(event)
        self.assertEqual(event.end_time, 150.0)
        self.assertTrue(event.has_clip)
        self.assertFalse(event.has_snapshot)

    def test_get_active_event_ids(self):
        self.manager.create_event("evt1", "cam", "label", 100.0)
        self.manager.create_event("evt2", "cam", "label", 110.0)

        ids = self.manager.get_active_event_ids()
        self.assertIn("evt1", ids)
        self.assertIn("evt2", ids)
        self.assertEqual(len(ids), 2)

    def test_get_stats(self):
        self.manager.create_event("evt1", "cam1", "person", 100.0)
        self.manager.create_event("evt2", "cam2", "dog", 110.0)
        self.manager.set_ai_description("evt1", "desc")

        stats = self.manager.get_stats()
        self.assertEqual(stats["total_active"], 2)
        self.assertEqual(stats["by_phase"]["NEW"], 1)
        self.assertEqual(stats["by_phase"]["DESCRIBED"], 1)
        self.assertEqual(stats["by_camera"]["cam1"], 1)
        self.assertEqual(stats["by_camera"]["cam2"], 1)

if __name__ == '__main__':
    unittest.main()
