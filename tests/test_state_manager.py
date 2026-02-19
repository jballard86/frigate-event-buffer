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
from frigate_buffer.managers.state import EventStateManager, _normalize_box
from frigate_buffer.models import EventPhase, FrameMetadata

class TestEventStateManager(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        self.manager = EventStateManager()

    def test_create_event_success(self):
        """Test creating a new event successfully."""
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
        """Test creating an event that already exists preserves original (idempotency)."""
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

    def test_create_event_existing(self):
        """Test creating an event that already exists returns the existing one."""
        event_id = "123456.789-new"
        camera = "front_door"
        label = "person"
        start_time = 123456.789

        event1 = self.manager.create_event(event_id, camera, label, start_time)
        event1.phase = EventPhase.DESCRIBED  # Modify to distinguish

        # Try creating again with same ID
        event2 = self.manager.create_event(event_id, camera, label, start_time)

        self.assertEqual(event1, event2)
        self.assertEqual(event2.phase, EventPhase.DESCRIBED)  # Should still be DESCRIBED

    def test_get_event(self):
        """Test retrieving events."""
        # Non-existent
        self.assertIsNone(self.manager.get_event("non_existent"))

        # Existent
        event_id = "test_event_1"
        self.manager.create_event(event_id, "cam", "label", 100.0)

        event = self.manager.get_event(event_id)
        self.assertIsNotNone(event)
        self.assertEqual(event.event_id, event_id)

    def test_remove_event(self):
        """Test removing an event."""
        event_id = "test_event_1"
        self.manager.create_event(event_id, "cam", "label", 100.0)

        removed = self.manager.remove_event(event_id)
        self.assertIsNotNone(removed)
        self.assertEqual(removed.event_id, event_id)

        self.assertIsNone(self.manager.get_event(event_id))

        # Removing non-existent
        self.assertIsNone(self.manager.remove_event("non_existent"))

    def test_set_ai_description(self):
        """Test setting AI description and advancing to DESCRIBED phase."""
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

    def test_set_ai_description_nonexistent(self):
        """Test setting AI description for non-existent event."""
        success = self.manager.set_ai_description("nonexistent", "desc")
        self.assertFalse(success)

    def test_set_genai_metadata(self):
        """Test setting GenAI metadata and advancing to FINALIZED phase."""
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

    def test_set_genai_metadata_nonexistent(self):
        """Test setting GenAI metadata for non-existent event."""
        success = self.manager.set_genai_metadata("nonexistent", "title", "desc", "severity")
        self.assertFalse(success)

    def test_set_review_summary(self):
        """Test setting review summary and advancing to SUMMARIZED phase."""
        event_id = "test_event_1"
        self.manager.create_event(event_id, "cam", "label", 100.0)

        summary = "Summary of the event"
        success = self.manager.set_review_summary(event_id, summary)

        self.assertTrue(success)
        event = self.manager.get_event(event_id)
        self.assertEqual(event.review_summary, summary)
        self.assertEqual(event.phase, EventPhase.SUMMARIZED)

    def test_set_review_summary_nonexistent(self):
        """Test setting review summary for non-existent event."""
        success = self.manager.set_review_summary("nonexistent", "summary")
        self.assertFalse(success)

    def test_mark_event_ended(self):
        """Test marking event as ended."""
        event_id = "test_event_1"
        self.manager.create_event(event_id, "cam", "label", 100.0)

        event = self.manager.mark_event_ended(event_id, 150.0, True, False)

        self.assertIsNotNone(event)
        self.assertEqual(event.end_time, 150.0)
        self.assertTrue(event.has_clip)
        self.assertFalse(event.has_snapshot)

    def test_get_active_event_ids(self):
        """Test getting list of active event IDs."""
        # Empty initially
        ids = self.manager.get_active_event_ids()
        self.assertEqual(len(ids), 0)

        self.manager.create_event("evt1", "cam", "label", 100.0)
        self.manager.create_event("evt2", "cam", "label", 110.0)

        ids = self.manager.get_active_event_ids()
        self.assertIn("evt1", ids)
        self.assertIn("evt2", ids)
        self.assertEqual(len(ids), 2)

    def test_get_stats(self):
        """Test getting statistics with multiple events across phases and cameras."""
        self.manager.create_event("evt1", "cam1", "person", 100.0)
        self.manager.create_event("evt2", "cam2", "dog", 110.0)
        self.manager.set_ai_description("evt1", "desc")

        stats = self.manager.get_stats()
        self.assertEqual(stats["total_active"], 2)
        self.assertEqual(stats["by_phase"]["NEW"], 1)
        self.assertEqual(stats["by_phase"]["DESCRIBED"], 1)
        self.assertEqual(stats["by_camera"]["cam1"], 1)
        self.assertEqual(stats["by_camera"]["cam2"], 1)

    def test_get_stats_comprehensive(self):
        """Test getting statistics with events in all phases."""
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

    def test_normalize_box_pixels(self):
        """_normalize_box converts pixel coords to normalized [ymin, xmin, ymax, xmax]."""
        # Frigate [x1, y1, x2, y2] pixels
        out = _normalize_box([100, 200, 300, 400], frame_width=1000, frame_height=800)
        self.assertIsNotNone(out)
        ymin, xmin, ymax, xmax = out
        self.assertAlmostEqual(xmin, 0.1)
        self.assertAlmostEqual(ymin, 0.25)
        self.assertAlmostEqual(xmax, 0.3)
        self.assertAlmostEqual(ymax, 0.5)
        self.assertTrue(all(0 <= v <= 1 for v in out))

    def test_normalize_box_invalid(self):
        """_normalize_box returns None for invalid input."""
        self.assertIsNone(_normalize_box(None))
        self.assertIsNone(_normalize_box([]))
        self.assertIsNone(_normalize_box([1, 2, 3]))
        self.assertIsNone(_normalize_box("not a list"))

    def test_add_and_get_frame_metadata(self):
        """add_frame_metadata stores entries; get_frame_metadata returns copy."""
        self.manager.create_event("e1", "cam", "person", 100.0)
        self.manager.add_frame_metadata("e1", 101.0, [0.1, 0.2, 0.5, 0.6], 1000.0, 0.9)
        self.manager.add_frame_metadata("e1", 102.0, [0.2, 0.3, 0.6, 0.7], 1200.0, 0.95)
        lst = self.manager.get_frame_metadata("e1")
        self.assertEqual(len(lst), 2)
        self.assertEqual(lst[0].frame_time, 101.0)
        self.assertEqual(lst[0].score, 0.9)
        self.assertEqual(lst[1].frame_time, 102.0)
        # Return is a copy
        lst.append(FrameMetadata(0, (0, 0, 1, 1), 0, 0))
        self.assertEqual(len(self.manager.get_frame_metadata("e1")), 2)

    def test_get_frame_metadata_empty(self):
        """get_frame_metadata returns empty list for unknown or empty event."""
        self.assertEqual(self.manager.get_frame_metadata("unknown"), [])
        self.manager.create_event("e1", "cam", "person", 100.0)
        self.assertEqual(self.manager.get_frame_metadata("e1"), [])

    def test_clear_frame_metadata(self):
        """clear_frame_metadata removes all entries for the event."""
        self.manager.create_event("e1", "cam", "person", 100.0)
        self.manager.add_frame_metadata("e1", 101.0, [0.1, 0.1, 0.5, 0.5], 100.0, 0.8)
        self.manager.clear_frame_metadata("e1")
        self.assertEqual(self.manager.get_frame_metadata("e1"), [])

    def test_remove_event_clears_frame_metadata(self):
        """remove_event also clears frame metadata for that event."""
        self.manager.create_event("e1", "cam", "person", 100.0)
        self.manager.add_frame_metadata("e1", 101.0, [0.1, 0.1, 0.5, 0.5], 100.0, 0.8)
        self.manager.remove_event("e1")
        self.assertEqual(self.manager.get_frame_metadata("e1"), [])

if __name__ == '__main__':
    unittest.main()
