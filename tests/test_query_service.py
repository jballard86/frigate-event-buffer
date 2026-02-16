import unittest
import os
import shutil
import tempfile
import json
import time
from frigate_buffer.services.query import EventQueryService

class TestEventQueryService(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.service = EventQueryService(self.test_dir)

        # Create structure
        # Camera 1
        self.cam1 = os.path.join(self.test_dir, "front_door")
        os.makedirs(self.cam1)

        # Event 1 for Camera 1
        ts1 = str(int(time.time() - 100))
        self.ev1_id = "123456"
        self.ev1_dir = os.path.join(self.cam1, f"{ts1}_{self.ev1_id}")
        os.makedirs(self.ev1_dir)

        with open(os.path.join(self.ev1_dir, "summary.txt"), "w") as f:
            f.write("Title: Person detected\nDescription: A person at the door.")

        with open(os.path.join(self.ev1_dir, "metadata.json"), "w") as f:
            json.dump({"label": "person", "threat_level": 5}, f)

        with open(os.path.join(self.ev1_dir, "clip.mp4"), "w") as f:
            f.write("dummy content")

        # Consolidated Event
        self.events_dir = os.path.join(self.test_dir, "events")
        os.makedirs(self.events_dir)

        ts2 = str(int(time.time() - 50))
        self.ce_id = f"{ts2}_ce_1"
        self.ce_dir = os.path.join(self.events_dir, self.ce_id)
        os.makedirs(self.ce_dir)

        # Camera subdir in CE
        os.makedirs(os.path.join(self.ce_dir, "front_door"))
        with open(os.path.join(self.ce_dir, "front_door", "clip.mp4"), "w") as f:
            f.write("dummy content")

        with open(os.path.join(self.ce_dir, "summary.txt"), "w") as f:
            f.write("Title: Consolidated Event\nDescription: Something happened.")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_get_cameras(self):
        cameras = self.service.get_cameras()
        self.assertIn("front_door", cameras)
        self.assertIn("events", cameras)

    def test_get_camera_events(self):
        events = self.service.get_events("front_door")
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["event_id"], self.ev1_id)
        self.assertEqual(ev["title"], "Person detected")
        self.assertTrue(ev["has_clip"])
        self.assertEqual(ev["threat_level"], 5)
        # No timeline or metadata end_time => end_timestamp absent (event ongoing)
        self.assertNotIn("end_timestamp", ev)

    def test_get_consolidated_events(self):
        events = self.service.get_events("events")
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["event_id"], self.ce_id)
        self.assertEqual(ev["title"], "Consolidated Event")
        self.assertTrue(ev["consolidated"])
        self.assertTrue(ev["has_clip"])

    def test_get_all_events(self):
        events, cameras = self.service.get_all_events()
        self.assertIn("front_door", cameras)
        self.assertIn("events", cameras)

        # Check if we have both events
        ids = [e["event_id"] for e in events]
        self.assertIn(self.ev1_id, ids)
        self.assertIn(self.ce_id, ids)

        # Check consolidated event properties
        ce = next(e for e in events if e["event_id"] == self.ce_id)
        self.assertTrue(ce.get("consolidated"), "Consolidated event should have consolidated=True")
        self.assertTrue(ce.get("has_clip"), "Consolidated event should have has_clip=True")

    def test_camera_event_includes_end_timestamp_when_in_timeline(self):
        """When timeline has an entry with payload.after.end_time, event dict includes end_timestamp."""
        timeline = {
            "event_id": self.ev1_id,
            "entries": [
                {
                    "source": "frigate_mqtt",
                    "data": {
                        "payload": {
                            "after": {"end_time": 1234567890.5},
                        }
                    },
                }
            ],
        }
        with open(os.path.join(self.ev1_dir, "notification_timeline.json"), "w") as f:
            json.dump(timeline, f)
        events = self.service.get_events("front_door")
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertIn("end_timestamp", ev)
        self.assertEqual(ev["end_timestamp"], 1234567890.5)

    def test_camera_event_end_timestamp_fallback_from_metadata(self):
        """When metadata.json has end_time but timeline has none, event gets end_timestamp from metadata."""
        with open(os.path.join(self.ev1_dir, "metadata.json"), "w") as f:
            json.dump({"label": "person", "threat_level": 0, "end_time": 1234567895.25}, f)
        events = self.service.get_events("front_door")
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertIn("end_timestamp", ev)
        self.assertEqual(ev["end_timestamp"], 1234567895.25)

if __name__ == '__main__':
    unittest.main()
