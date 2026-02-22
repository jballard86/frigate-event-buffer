import unittest
import os
import shutil
import tempfile
import json
import time
from frigate_buffer.services.query import EventQueryService

class TestEventQueryService(unittest.TestCase):
    def setUp(self):
        # Use a unique subdir per test to avoid Windows rmtree permission errors on open files.
        base = os.path.join(os.path.dirname(__file__), "_query_test_fixture")
        self.test_dir = os.path.join(base, str(id(self)))
        os.makedirs(self.test_dir, exist_ok=True)
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

        # Camera subdirs in CE (multi-cam: front_door and back_door)
        os.makedirs(os.path.join(self.ce_dir, "front_door"))
        with open(os.path.join(self.ce_dir, "front_door", "clip.mp4"), "w") as f:
            f.write("dummy content")
        os.makedirs(os.path.join(self.ce_dir, "back_door"))
        with open(os.path.join(self.ce_dir, "back_door", "clip.mp4"), "w") as f:
            f.write("dummy content")

        with open(os.path.join(self.ce_dir, "summary.txt"), "w") as f:
            f.write("Title: Consolidated Event\nDescription: Something happened.")

    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir)
        except OSError:
            pass

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
        self.assertIn("hosted_clips", ev)
        self.assertEqual(len(ev["hosted_clips"]), 2)  # front_door and back_door
        cameras = [c["camera"] for c in ev["hosted_clips"]]
        self.assertIn("front_door", cameras)
        self.assertIn("back_door", cameras)
        self.assertIn("clip.mp4", ev["hosted_clips"][0]["url"])

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

    def test_ultralytics_folder_excluded_from_events(self):
        """The ultralytics config folder should not appear as an event."""
        # Create an ultralytics folder in the events directory (simulating config folder misplaced or created there)
        ultralytics_dir = os.path.join(self.events_dir, "ultralytics")
        os.makedirs(ultralytics_dir, exist_ok=True)

        # Create some files that Ultralytics would create
        with open(os.path.join(ultralytics_dir, "settings.json"), "w") as f:
            json.dump({"settings_version": "0.0.6", "datasets_dir": "/app/datasets"}, f)

        with open(os.path.join(ultralytics_dir, "persistent_cache.json"), "w") as f:
            json.dump({"cpu_info": "Test CPU"}, f)

        # Get consolidated events - should still only return 1 event (the real CE)
        events = self.service.get_events("events")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_id"], self.ce_id)

        # Verify ultralytics is not in the event list
        for ev in events:
            self.assertNotEqual(ev.get("event_id"), "ultralytics")

    def test_ultralytics_capital_u_excluded_from_events(self):
        """The Ultralytics folder with capital U should not appear as an event (case sensitivity test)."""
        # Create an Ultralytics folder with capital U (as shown in bug report)
        ultralytics_dir = os.path.join(self.events_dir, "Ultralytics")
        os.makedirs(ultralytics_dir, exist_ok=True)

        # Create the exact files from the bug report
        with open(os.path.join(ultralytics_dir, "settings.json"), "w") as f:
            json.dump({"settings_version": "0.0.6", "datasets_dir": "/app/datasets"}, f)

        with open(os.path.join(ultralytics_dir, "persistent_cache.json"), "w") as f:
            json.dump({"cpu_info": "AMD Ryzen Threadripper 1900X 8-Core Processor"}, f)

        # Get consolidated events - should still only return 1 event (the real CE)
        events = self.service.get_events("events")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_id"], self.ce_id)

        # Verify Ultralytics (capital U) is not in the event list
        for ev in events:
            self.assertNotEqual(ev.get("event_id"), "Ultralytics")

    def test_ultralytics_excluded_from_cameras(self):
        """The ultralytics folder should not appear as a camera."""
        # Create ultralytics folder at storage root (where YOLO_CONFIG_DIR points)
        ultralytics_dir = os.path.join(self.test_dir, "ultralytics")
        os.makedirs(ultralytics_dir, exist_ok=True)

        with open(os.path.join(ultralytics_dir, "settings.json"), "w") as f:
            json.dump({"settings_version": "0.0.6"}, f)

        # Get cameras - should not include ultralytics
        cameras = self.service.get_cameras()
        self.assertNotIn("ultralytics", cameras)
        self.assertNotIn("Ultralytics", cameras)
        # Should still have front_door and events
        self.assertIn("front_door", cameras)
        self.assertIn("events", cameras)

    def test_yolo_models_excluded_from_cameras(self):
        """The yolo_models folder should not appear as a camera."""
        # Create yolo_models folder at storage root
        yolo_dir = os.path.join(self.test_dir, "yolo_models")
        os.makedirs(yolo_dir, exist_ok=True)

        with open(os.path.join(yolo_dir, "yolov8n.pt"), "w") as f:
            f.write("dummy model")

        # Get cameras - should not include yolo_models
        cameras = self.service.get_cameras()
        self.assertNotIn("yolo_models", cameras)
        # Should still have front_door and events
        self.assertIn("front_door", cameras)
        self.assertIn("events", cameras)

    def test_get_all_events_excludes_ultralytics_folder(self):
        """get_all_events() must not include any events from the ultralytics directory."""
        # Create ultralytics at storage root with a subdir (simulates Ultralytics lib creating e.g. Ultralytics/runs)
        ultralytics_dir = os.path.join(self.test_dir, "ultralytics")
        os.makedirs(ultralytics_dir, exist_ok=True)
        subdir = os.path.join(ultralytics_dir, "Ultralytics")
        os.makedirs(subdir, exist_ok=True)
        with open(os.path.join(subdir, "settings.json"), "w") as f:
            json.dump({"settings_version": "0.0.6"}, f)

        all_events, cameras_found = self.service.get_all_events()

        # ultralytics must not be in cameras_found
        self.assertNotIn("ultralytics", cameras_found)
        # No event may have camera == "ultralytics"
        for ev in all_events:
            self.assertNotEqual(ev.get("camera"), "ultralytics", f"Ghost event from ultralytics: {ev.get('event_id')}")
        # We should still have our real events (1 camera event + 1 consolidated)
        self.assertGreaterEqual(len(all_events), 2)

if __name__ == '__main__':
    unittest.main()
