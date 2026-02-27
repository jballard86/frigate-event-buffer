import sys
from unittest.mock import MagicMock

# Keys to mock so this module can import state without pulling in heavy deps.
# We mock in setup_module and restore in teardown_module so other test modules
# never see the mocks.
_MODULE_KEYS = (
    "requests",
    "flask",
    "paho",
    "paho.mqtt",
    "paho.mqtt.client",
    "schedule",
    "yaml",
    "voluptuous",
)
_saved_modules = {}

import unittest


def setup_module():
    for k in _MODULE_KEYS:
        _saved_modules[k] = sys.modules.get(k)
        sys.modules[k] = MagicMock()
    # Import after mocks so state/models don't pull in flask etc.
    from frigate_buffer.managers.state import (
        EventStateManager as ESM,
    )
    from frigate_buffer.managers.state import (
        _normalize_box as nb,
    )
    from frigate_buffer.models import EventPhase as EP
    from frigate_buffer.models import FrameMetadata as FM

    mod = sys.modules[__name__]
    mod.EventStateManager = ESM
    mod._normalize_box = nb
    mod.EventPhase = EP
    mod.FrameMetadata = FM


def teardown_module():
    for k in _MODULE_KEYS:
        if _saved_modules.get(k) is not None:
            sys.modules[k] = _saved_modules[k]
        elif k in sys.modules:
            del sys.modules[k]


# Import for use in test methods (set by setup_module when running under pytest)
EventStateManager = _normalize_box = EventPhase = FrameMetadata = None


def _ensure_imports():
    """Run when module is executed without pytest (e.g. python -m unittest)."""
    global EventStateManager, _normalize_box, EventPhase, FrameMetadata
    if EventStateManager is None:
        from frigate_buffer.managers.state import (
            EventStateManager as _ESM,
        )
        from frigate_buffer.managers.state import (
            _normalize_box as _nb,
        )
        from frigate_buffer.models import EventPhase as _EP
        from frigate_buffer.models import FrameMetadata as _FM

        globals()["EventStateManager"] = _ESM
        globals()["_normalize_box"] = _nb
        globals()["EventPhase"] = _EP
        globals()["FrameMetadata"] = _FM


class TestEventStateManager(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        _ensure_imports()
        self.manager = EventStateManager()

    def test_create_event_success(self):
        """Test creating a new event successfully."""
        event_id = "test_event_1"
        camera = "front_door"
        label = "person"
        start_time = 123456789.0

        event = self.manager.create_event(event_id, camera, label, start_time)

        assert event.event_id == event_id
        assert event.camera == camera
        assert event.label == label
        assert event.created_at == start_time
        assert event.phase == EventPhase.NEW

        # Verify it's in the manager
        assert self.manager.get_event(event_id) == event

    def test_create_event_duplicate(self):
        """Test creating an event that already exists preserves original
        (idempotency)."""
        event_id = "test_event_1"
        camera = "front_door"
        label = "person"
        start_time = 123456789.0

        event1 = self.manager.create_event(event_id, camera, label, start_time)

        # Try creating again with same ID but different parameters
        event2 = self.manager.create_event(
            event_id, "other_camera", "other_label", start_time + 10
        )

        # Should return the exact same object
        assert event1 is event2

        # Parameters of the first creation should be preserved (idempotency)
        assert event2.camera == camera
        assert event2.label == label
        assert event2.created_at == start_time

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

        assert event1 == event2
        assert event2.phase == EventPhase.DESCRIBED  # Should still be DESCRIBED

    def test_get_event(self):
        """Test retrieving events."""
        # Non-existent
        assert self.manager.get_event("non_existent") is None

        # Existent
        event_id = "test_event_1"
        self.manager.create_event(event_id, "cam", "label", 100.0)

        event = self.manager.get_event(event_id)
        assert event is not None
        assert event.event_id == event_id

    def test_remove_event(self):
        """Test removing an event."""
        event_id = "test_event_1"
        self.manager.create_event(event_id, "cam", "label", 100.0)

        removed = self.manager.remove_event(event_id)
        assert removed is not None
        assert removed.event_id == event_id

        assert self.manager.get_event(event_id) is None

        # Removing non-existent
        assert self.manager.remove_event("non_existent") is None

    def test_set_ai_description(self):
        """Test setting AI description and advancing to DESCRIBED phase."""
        event_id = "test_event_1"
        self.manager.create_event(event_id, "cam", "label", 100.0)

        description = "A person carrying a box"
        success = self.manager.set_ai_description(event_id, description)

        assert success
        event = self.manager.get_event(event_id)
        assert event.ai_description == description
        assert event.phase == EventPhase.DESCRIBED

        # Update description again
        new_description = "A person wearing a red shirt"
        success = self.manager.set_ai_description(event_id, new_description)
        assert success
        assert event.ai_description == new_description
        assert event.phase == EventPhase.DESCRIBED

    def test_set_ai_description_nonexistent(self):
        """Test setting AI description for non-existent event."""
        success = self.manager.set_ai_description("nonexistent", "desc")
        assert not success

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
            scene="Front yard at night",
        )

        assert success
        event = self.manager.get_event(event_id)
        assert event.genai_title == "Suspicious Activity"
        assert event.genai_description == "Person looking into windows"
        assert event.severity == "suspicious"
        assert event.threat_level == 1
        assert event.genai_scene == "Front yard at night"
        assert event.phase == EventPhase.FINALIZED

    def test_set_genai_metadata_nonexistent(self):
        """Test setting GenAI metadata for non-existent event."""
        success = self.manager.set_genai_metadata(
            "nonexistent", "title", "desc", "severity"
        )
        assert not success

    def test_set_review_summary(self):
        """Test setting review summary and advancing to SUMMARIZED phase."""
        event_id = "test_event_1"
        self.manager.create_event(event_id, "cam", "label", 100.0)

        summary = "Summary of the event"
        success = self.manager.set_review_summary(event_id, summary)

        assert success
        event = self.manager.get_event(event_id)
        assert event.review_summary == summary
        assert event.phase == EventPhase.SUMMARIZED

    def test_set_review_summary_nonexistent(self):
        """Test setting review summary for non-existent event."""
        success = self.manager.set_review_summary("nonexistent", "summary")
        assert not success

    def test_mark_event_ended(self):
        """Test marking event as ended."""
        event_id = "test_event_1"
        self.manager.create_event(event_id, "cam", "label", 100.0)

        event = self.manager.mark_event_ended(event_id, 150.0, True, False)

        assert event is not None
        assert event.end_time == 150.0
        assert event.has_clip
        assert not event.has_snapshot

    def test_get_active_event_ids(self):
        """Test getting list of active event IDs."""
        # Empty initially
        ids = self.manager.get_active_event_ids()
        assert len(ids) == 0

        self.manager.create_event("evt1", "cam", "label", 100.0)
        self.manager.create_event("evt2", "cam", "label", 110.0)

        ids = self.manager.get_active_event_ids()
        assert "evt1" in ids
        assert "evt2" in ids
        assert len(ids) == 2

    def test_get_stats(self):
        """Test getting statistics with multiple events across phases and cameras."""
        self.manager.create_event("evt1", "cam1", "person", 100.0)
        self.manager.create_event("evt2", "cam2", "dog", 110.0)
        self.manager.set_ai_description("evt1", "desc")

        stats = self.manager.get_stats()
        assert stats["total_active"] == 2
        assert stats["by_phase"]["NEW"] == 1
        assert stats["by_phase"]["DESCRIBED"] == 1
        assert stats["by_camera"]["cam1"] == 1
        assert stats["by_camera"]["cam2"] == 1

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

        assert stats["total_active"] == 3
        assert stats["by_phase"]["NEW"] == 1
        assert stats["by_phase"]["DESCRIBED"] == 1
        assert stats["by_phase"]["FINALIZED"] == 1
        assert stats["by_camera"]["cam1"] == 2
        assert stats["by_camera"]["cam2"] == 1

    def test_normalize_box_pixels(self):
        """_normalize_box converts pixel coords to normalized
        [ymin, xmin, ymax, xmax]."""
        # Frigate [x1, y1, x2, y2] pixels
        out = _normalize_box([100, 200, 300, 400], frame_width=1000, frame_height=800)
        assert out is not None
        ymin, xmin, ymax, xmax = out
        assert abs(xmin - 0.1) < 1e-7
        assert abs(ymin - 0.25) < 1e-7
        assert abs(xmax - 0.3) < 1e-7
        assert abs(ymax - 0.5) < 1e-7
        assert all(0 <= v <= 1 for v in out)

    def test_normalize_box_invalid(self):
        """_normalize_box returns None for invalid input."""
        assert _normalize_box(None) is None
        assert _normalize_box([]) is None
        assert _normalize_box([1, 2, 3]) is None
        assert _normalize_box("not a list") is None

    def test_add_and_get_frame_metadata(self):
        """add_frame_metadata stores entries; get_frame_metadata returns copy."""
        self.manager.create_event("e1", "cam", "person", 100.0)
        self.manager.add_frame_metadata("e1", 101.0, [0.1, 0.2, 0.5, 0.6], 1000.0, 0.9)
        self.manager.add_frame_metadata("e1", 102.0, [0.2, 0.3, 0.6, 0.7], 1200.0, 0.95)
        lst = self.manager.get_frame_metadata("e1")
        assert len(lst) == 2
        assert lst[0].frame_time == 101.0
        assert lst[0].score == 0.9
        assert lst[1].frame_time == 102.0
        # Return is a copy
        lst.append(FrameMetadata(0, (0, 0, 1, 1), 0, 0))
        assert len(self.manager.get_frame_metadata("e1")) == 2

    def test_get_frame_metadata_empty(self):
        """get_frame_metadata returns empty list for unknown or empty event."""
        assert self.manager.get_frame_metadata("unknown") == []
        self.manager.create_event("e1", "cam", "person", 100.0)
        assert self.manager.get_frame_metadata("e1") == []

    def test_clear_frame_metadata(self):
        """clear_frame_metadata removes all entries for the event."""
        self.manager.create_event("e1", "cam", "person", 100.0)
        self.manager.add_frame_metadata("e1", 101.0, [0.1, 0.1, 0.5, 0.5], 100.0, 0.8)
        self.manager.clear_frame_metadata("e1")
        assert self.manager.get_frame_metadata("e1") == []

    def test_remove_event_clears_frame_metadata(self):
        """remove_event also clears frame metadata for that event."""
        self.manager.create_event("e1", "cam", "person", 100.0)
        self.manager.add_frame_metadata("e1", 101.0, [0.1, 0.1, 0.5, 0.5], 100.0, 0.8)
        self.manager.remove_event("e1")
        assert self.manager.get_frame_metadata("e1") == []


if __name__ == "__main__":
    unittest.main()
