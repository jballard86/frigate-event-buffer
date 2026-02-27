import json
import os
import shutil
import tempfile
import time
import unittest
from collections import OrderedDict

from frigate_buffer.services.query import EventQueryService, read_timeline_merged


class TestEventQueryService(unittest.TestCase):
    def setUp(self):
        # Use a unique subdir per test to avoid Windows rmtree permission
        # errors on open files.
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
        assert "front_door" in cameras
        assert "events" in cameras

    def test_get_camera_events(self):
        events = self.service.get_events("front_door")
        assert len(events) == 1
        ev = events[0]
        assert ev["event_id"] == self.ev1_id
        assert ev["title"] == "Person detected"
        assert ev["has_clip"]
        assert ev["threat_level"] == 5
        # No timeline or metadata end_time => end_timestamp absent (event ongoing)
        assert "end_timestamp" not in ev

    def test_get_consolidated_events(self):
        events = self.service.get_events("events")
        assert len(events) == 1
        ev = events[0]
        assert ev["event_id"] == self.ce_id
        assert ev["title"] == "Consolidated Event"
        assert ev["consolidated"]
        assert ev["has_clip"]
        assert "hosted_clips" in ev
        assert len(ev["hosted_clips"]) == 2  # front_door and back_door
        cameras = [c["camera"] for c in ev["hosted_clips"]]
        assert "front_door" in cameras
        assert "back_door" in cameras
        assert "clip.mp4" in ev["hosted_clips"][0]["url"]

    def test_consolidated_event_includes_summary_video_when_file_exists(self):
        """When {ce_id}_summary.mp4 exists in the CE root, event has
        Summary video in hosted_clips and as hosted_clip."""
        summary_basename = f"{self.ce_id}_summary.mp4"
        with open(os.path.join(self.ce_dir, summary_basename), "w") as f:
            f.write("dummy")
        self.service._cache.clear()
        self.service._event_cache.clear()
        events = self.service.get_events("events")
        assert len(events) == 1
        ev = events[0]
        assert "hosted_clips" in ev
        summary_entries = [
            c for c in ev["hosted_clips"] if c.get("camera") == "Summary video"
        ]
        assert len(summary_entries) == 1, (
            "hosted_clips should include Summary video entry"
        )
        summary_url = summary_entries[0]["url"]
        assert f"/files/events/{self.ce_id}/{summary_basename}" in summary_url
        assert ev["hosted_clip"] == summary_url, (
            "hosted_clip should be the summary URL when summary exists"
        )

    def test_get_all_events(self):
        events, cameras = self.service.get_all_events()
        assert "front_door" in cameras
        assert "events" in cameras

        # Check if we have both events
        ids = [e["event_id"] for e in events]
        assert self.ev1_id in ids
        assert self.ce_id in ids

        # Check consolidated event properties
        ce = next(e for e in events if e["event_id"] == self.ce_id)
        assert ce.get("consolidated"), (
            "Consolidated event should have consolidated=True"
        )
        assert ce.get("has_clip"), "Consolidated event should have has_clip=True"

    def test_camera_event_includes_end_timestamp_when_in_timeline(self):
        """When timeline has an entry with payload.after.end_time, event
        dict includes end_timestamp."""
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
        assert len(events) == 1
        ev = events[0]
        assert "end_timestamp" in ev
        assert ev["end_timestamp"] == 1234567890.5

    def test_camera_event_end_timestamp_fallback_from_metadata(self):
        """When metadata.json has end_time but timeline has none, event
        gets end_timestamp from metadata."""
        with open(os.path.join(self.ev1_dir, "metadata.json"), "w") as f:
            json.dump(
                {"label": "person", "threat_level": 0, "end_time": 1234567895.25}, f
            )
        events = self.service.get_events("front_door")
        assert len(events) == 1
        ev = events[0]
        assert "end_timestamp" in ev
        assert ev["end_timestamp"] == 1234567895.25

    def test_ultralytics_folder_excluded_from_events(self):
        """The ultralytics config folder should not appear as an event."""
        # Create an ultralytics folder in the events directory (simulating
        # config folder misplaced or created there)
        ultralytics_dir = os.path.join(self.events_dir, "ultralytics")
        os.makedirs(ultralytics_dir, exist_ok=True)

        # Create some files that Ultralytics would create
        with open(os.path.join(ultralytics_dir, "settings.json"), "w") as f:
            json.dump({"settings_version": "0.0.6", "datasets_dir": "/app/datasets"}, f)

        with open(os.path.join(ultralytics_dir, "persistent_cache.json"), "w") as f:
            json.dump({"cpu_info": "Test CPU"}, f)

        # Get consolidated events - should still only return 1 event (the real CE)
        events = self.service.get_events("events")
        assert len(events) == 1
        assert events[0]["event_id"] == self.ce_id

        # Verify ultralytics is not in the event list
        for ev in events:
            assert ev.get("event_id") != "ultralytics"

    def test_ultralytics_capital_u_excluded_from_events(self):
        """The Ultralytics folder with capital U should not appear as an
        event (case sensitivity test)."""
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
        assert len(events) == 1
        assert events[0]["event_id"] == self.ce_id

        # Verify Ultralytics (capital U) is not in the event list
        for ev in events:
            assert ev.get("event_id") != "Ultralytics"

    def test_ultralytics_excluded_from_cameras(self):
        """The ultralytics folder should not appear as a camera."""
        # Create ultralytics folder at storage root (where YOLO_CONFIG_DIR points)
        ultralytics_dir = os.path.join(self.test_dir, "ultralytics")
        os.makedirs(ultralytics_dir, exist_ok=True)

        with open(os.path.join(ultralytics_dir, "settings.json"), "w") as f:
            json.dump({"settings_version": "0.0.6"}, f)

        # Get cameras - should not include ultralytics
        cameras = self.service.get_cameras()
        assert "ultralytics" not in cameras
        assert "Ultralytics" not in cameras
        # Should still have front_door and events
        assert "front_door" in cameras
        assert "events" in cameras

    def test_yolo_models_excluded_from_cameras(self):
        """The yolo_models folder should not appear as a camera."""
        # Create yolo_models folder at storage root
        yolo_dir = os.path.join(self.test_dir, "yolo_models")
        os.makedirs(yolo_dir, exist_ok=True)

        with open(os.path.join(yolo_dir, "yolov8n.pt"), "w") as f:
            f.write("dummy model")

        # Get cameras - should not include yolo_models
        cameras = self.service.get_cameras()
        assert "yolo_models" not in cameras
        # Should still have front_door and events
        assert "front_door" in cameras
        assert "events" in cameras

    def test_get_all_events_excludes_ultralytics_folder(self):
        """get_all_events() must not include any events from the
        ultralytics directory."""
        # Create ultralytics at storage root with a subdir (simulates
        # Ultralytics lib creating e.g. Ultralytics/runs)
        ultralytics_dir = os.path.join(self.test_dir, "ultralytics")
        os.makedirs(ultralytics_dir, exist_ok=True)
        subdir = os.path.join(ultralytics_dir, "Ultralytics")
        os.makedirs(subdir, exist_ok=True)
        with open(os.path.join(subdir, "settings.json"), "w") as f:
            json.dump({"settings_version": "0.0.6"}, f)

        all_events, cameras_found = self.service.get_all_events()

        # ultralytics must not be in cameras_found
        assert "ultralytics" not in cameras_found
        # No event may have camera == "ultralytics"
        for ev in all_events:
            assert ev.get("camera") != "ultralytics", (
                f"Ghost event from ultralytics: {ev.get('event_id')}"
            )
        # We should still have our real events (1 camera event + 1 consolidated)
        assert len(all_events) >= 2


class TestQueryCaching(unittest.TestCase):
    """EventQueryService cache TTL and per-folder caching behavior."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.ttl = 2
        self.service = EventQueryService(self.test_dir, cache_ttl=self.ttl)

        self.cam = "test_cam"
        os.makedirs(os.path.join(self.test_dir, self.cam))
        self.event_id = "event1"
        self.event_dir = os.path.join(
            self.test_dir, self.cam, f"{int(time.time())}_{self.event_id}"
        )
        os.makedirs(self.event_dir)

        self.summary_path = os.path.join(self.event_dir, "summary.txt")
        with open(self.summary_path, "w") as f:
            f.write("Title: Initial Title")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_caching_behavior(self):
        events = self.service.get_events(self.cam)
        assert events[0]["title"] == "Initial Title"

        with open(self.summary_path, "w") as f:
            f.write("Title: Modified Title")

        st = os.stat(self.event_dir)
        os.utime(self.event_dir, (st.st_atime, st.st_mtime + 1.0))

        events_cached = self.service.get_events(self.cam)
        assert events_cached[0]["title"] == "Initial Title", (
            "Should return cached data immediately"
        )

        time.sleep(self.ttl + 0.1)

        events_fresh = self.service.get_events(self.cam)
        assert events_fresh[0]["title"] == "Modified Title", (
            "Should return fresh data after TTL"
        )

    def test_event_cache_evicts_when_over_max(self):
        """Event cache is bounded with LRU eviction (event_cache_max)."""
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        os.makedirs(os.path.join(storage, "cam1"), exist_ok=True)
        for i in range(5):
            d = os.path.join(storage, "cam1", f"100{i}_ev{i}")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "summary.txt"), "w") as f:
                f.write("Event")
        svc = EventQueryService(storage, cache_ttl=5, event_cache_max=3)
        assert isinstance(svc._event_cache, OrderedDict)
        assert svc._event_cache_max == 3
        for _ in range(5):
            svc.get_events("cam1")
        assert len(svc._event_cache) <= 3, "event cache must not exceed max size"


class TestExtractEndTimestampFromTimeline(unittest.TestCase):
    """_extract_end_timestamp_from_timeline: Frigate payload.after.end_time
    and test_ai_prompt data.end_time (test events only)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        self.service = EventQueryService(self.tmp)

    def test_returns_none_for_empty_timeline(self):
        assert (
            self.service._extract_end_timestamp_from_timeline({"entries": []}) is None
        )

    def test_returns_none_when_no_entries_have_end_time(self):
        """Entries present but all end_time null or missing → None."""
        timeline = {
            "entries": [
                {
                    "source": "frigate_mqtt",
                    "data": {"payload": {"after": {"end_time": None}}},
                },
                {"source": "frigate_mqtt", "data": {"payload": {"after": {}}}},
            ]
        }
        assert self.service._extract_end_timestamp_from_timeline(timeline) is None

    def test_returns_end_time_from_frigate_payload_after(self):
        """Regular events: end_time from payload.after (unchanged behavior)."""
        timeline = {
            "entries": [
                {
                    "source": "frigate_mqtt",
                    "data": {"payload": {"after": {"end_time": 1700000000.5}}},
                },
            ]
        }
        assert (
            self.service._extract_end_timestamp_from_timeline(timeline) == 1700000000.5
        )

    def test_returns_end_time_from_test_ai_prompt_entry(self):
        """Test events only: end_time from source=test_ai_prompt and data.end_time."""
        timeline = {
            "entries": [
                {
                    "source": "test_ai_prompt",
                    "data": {"title": "Test", "end_time": 1700000100.0},
                },
            ]
        }
        assert (
            self.service._extract_end_timestamp_from_timeline(timeline) == 1700000100.0
        )

    def test_returns_max_when_multiple_frigate_entries_have_different_end_times(self):
        """Multiple Frigate event-end entries: result is the latest (max) end_time."""
        timeline = {
            "entries": [
                {
                    "source": "frigate_mqtt",
                    "data": {"payload": {"after": {"end_time": 1700000000.0}}},
                },
                {
                    "source": "frigate_mqtt",
                    "data": {"payload": {"after": {"end_time": 1700000050.0}}},
                },
                {
                    "source": "frigate_mqtt",
                    "data": {"payload": {"after": {"end_time": 1700000025.0}}},
                },
            ]
        }
        assert (
            self.service._extract_end_timestamp_from_timeline(timeline) == 1700000050.0
        )

    def test_returns_max_when_frigate_and_test_ai_prompt_both_have_end_time(self):
        """When both Frigate and test_ai_prompt have end_time, result is
        the latest (max)."""
        timeline = {
            "entries": [
                {
                    "source": "frigate_mqtt",
                    "data": {"payload": {"after": {"end_time": 1700000000.0}}},
                },
                {"source": "test_ai_prompt", "data": {"end_time": 1700000100.0}},
            ]
        }
        assert (
            self.service._extract_end_timestamp_from_timeline(timeline) == 1700000100.0
        )


class TestEvictCache(unittest.TestCase):
    """evict_cache removes a key so next request refetches (e.g. test_events
    after Send prompt to AI)."""

    def test_evict_cache_removes_key(self):
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        svc = EventQueryService(storage, cache_ttl=60)
        svc._set_cache("test_events", [{"id": "old"}])
        assert svc._get_cached("test_events") is not None
        svc.evict_cache("test_events")
        assert svc._get_cached("test_events") is None


class TestGetTestEventsSortAndTimestamp(unittest.TestCase):
    """get_test_events: sorted by folder mtime desc; timestamp from
    content_mtime (test events only)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        self.events_dir = os.path.join(self.tmp, "events")
        os.makedirs(self.events_dir, exist_ok=True)
        self.service = EventQueryService(self.tmp)

    def test_get_test_events_sorted_by_mtime_desc_and_timestamp_from_mtime(self):
        # Create test1 and test2 with different mtimes (test2 older)
        test1 = os.path.join(self.events_dir, "test1")
        test2 = os.path.join(self.events_dir, "test2")
        os.makedirs(test1)
        os.makedirs(os.path.join(test1, "cam1"))
        with open(os.path.join(test1, "cam1", "clip.mp4"), "w") as f:
            f.write("x")
        os.makedirs(test2)
        os.makedirs(os.path.join(test2, "cam1"))
        with open(os.path.join(test2, "cam1", "clip.mp4"), "w") as f:
            f.write("x")
        # Make test2 older (e.g. 100s ago); set mtime on folder and contents
        # so content_mtime is t_old
        t_old = time.time() - 100
        os.utime(test2, (t_old, t_old))
        os.utime(os.path.join(test2, "cam1"), (t_old, t_old))
        os.utime(os.path.join(test2, "cam1", "clip.mp4"), (t_old, t_old))
        events = self.service.get_test_events()
        assert len(events) == 2
        # First event should be the one with newer mtime (test1)
        assert events[0]["event_id"] == "test1"
        assert events[1]["event_id"] == "test2"
        # Timestamp should be content_mtime (not "0")
        assert events[0]["timestamp"] != "0"
        assert events[1]["timestamp"] != "0"
        assert int(events[1]["timestamp"]) == int(t_old)


class TestReadTimelineMerged(unittest.TestCase):
    """read_timeline_merged merges base notification_timeline.json and append JSONL."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def test_read_timeline_merged_merges_base_and_append(self):
        folder = os.path.join(self.tmp, "cam1", "123_ev1")
        os.makedirs(folder, exist_ok=True)
        base_path = os.path.join(folder, "notification_timeline.json")
        append_path = os.path.join(folder, "notification_timeline_append.jsonl")
        with open(base_path, "w") as f:
            json.dump(
                {"event_id": "ev1", "entries": [{"ts": "T1", "label": "base"}]}, f
            )
        with open(append_path, "a") as f:
            f.write(json.dumps({"ts": "T2", "label": "append"}) + "\n")
        data = read_timeline_merged(folder)
        assert len(data["entries"]) == 2
        assert data["entries"][0]["label"] == "base"
        assert data["entries"][1]["label"] == "append"


if __name__ == "__main__":
    unittest.main()
