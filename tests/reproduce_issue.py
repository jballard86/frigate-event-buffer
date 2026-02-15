
import os
import sys
import time
from unittest.mock import MagicMock, patch

# Ensure we can import from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frigate_buffer.orchestrator import StateAwareOrchestrator
from frigate_buffer.models import EventState, ConsolidatedEvent, EventPhase

class MockFileManager:
    def __init__(self):
        self.export_and_transcode_clip_calls = []

    def ensure_consolidated_camera_folder(self, folder_path, camera):
        return f"{folder_path}/{camera}"

    def export_and_transcode_clip(self, event_id, folder_path, camera, start_time, end_time, export_buffer_before, export_buffer_after):
        self.export_and_transcode_clip_calls.append({
            'event_id': event_id,
            'folder_path': folder_path,
            'camera': camera,
            'start_time': start_time,
            'end_time': end_time,
            'export_buffer_before': export_buffer_before,
            'export_buffer_after': export_buffer_after
        })
        return {'success': True}

    def generate_gif_from_clip(self, *args, **kwargs):
        return True

    def fetch_review_summary(self, *args, **kwargs):
        return "Mock summary"

    def write_review_summary(self, *args, **kwargs):
        return True

    def sanitize_camera_name(self, camera):
        return camera

    def append_timeline_entry(self, folder_path, entry):
        pass

class MockConsolidatedEventManager:
    def __init__(self):
        self._events = {}
        self._lock = MagicMock()
        self._lock.__enter__ = MagicMock()
        self._lock.__exit__ = MagicMock()

    def remove(self, ce_id):
        pass

class MockEventStateManager:
    def __init__(self):
        self._events = {}

    def get_event(self, event_id):
        return self._events.get(event_id)

    def remove_event(self, event_id):
        pass

def test_on_consolidated_event_close():
    config = {
        'STORAGE_PATH': '/tmp/test_storage',
        'FRIGATE_URL': 'http://localhost:5000',
        'RETENTION_DAYS': 7,
        'BUFFER_IP': 'localhost',
        'FLASK_PORT': 5055,
        'EXPORT_BUFFER_BEFORE': 5,
        'EXPORT_BUFFER_AFTER': 30,
        'SUMMARY_PADDING_BEFORE': 15,
        'SUMMARY_PADDING_AFTER': 15,
        'MQTT_BROKER': 'localhost',  # Add missing config keys
        'MQTT_PORT': 1883,
    }

    # Patch orchestrator to prevent side effects in init
    with patch('frigate_buffer.orchestrator.mqtt.Client'), \
         patch('frigate_buffer.orchestrator.NotificationPublisher'), \
         patch('frigate_buffer.orchestrator.DailyReviewManager'), \
         patch('frigate_buffer.orchestrator.FileManager'), \
         patch('frigate_buffer.orchestrator.ConsolidatedEventManager'), \
         patch('frigate_buffer.orchestrator.EventStateManager'):

        orchestrator = StateAwareOrchestrator(config)

    # Replace managers with our mocks
    orchestrator.file_manager = MockFileManager()
    orchestrator.consolidated_manager = MockConsolidatedEventManager()
    orchestrator.state_manager = MockEventStateManager()
    orchestrator.notifier = MagicMock() # Mock notifier

    # Setup test data
    ce_id = "ce_123"
    ce_start_time = 1000.0

    # Two events on camera1: one short, one long
    # Event 1: 1000 - 1010 (duration 10s)
    # Event 2: 1020 - 1050 (duration 30s) -> Should be representative
    event1 = EventState(event_id="evt_cam1_1", camera="camera1", label="person", created_at=1000.0)
    event1.end_time = 1010.0

    event2 = EventState(event_id="evt_cam1_2", camera="camera1", label="person", created_at=1020.0)
    event2.end_time = 1050.0

    # One event on camera2
    # Event 3: 1005 - 1035 (duration 30s)
    event3 = EventState(event_id="evt_cam2_1", camera="camera2", label="car", created_at=1005.0)
    event3.end_time = 1035.0

    orchestrator.state_manager._events = {
        "evt_cam1_1": event1,
        "evt_cam1_2": event2,
        "evt_cam2_1": event3
    }

    ce = ConsolidatedEvent(
        consolidated_id=ce_id,
        folder_name="folder_123",
        folder_path="/tmp/test_storage/events/folder_123",
        start_time=ce_start_time,
        last_activity_time=1050.0,
        end_time_max=1050.0,
        cameras=["camera1", "camera2"],
        frigate_event_ids=["evt_cam1_1", "evt_cam1_2", "evt_cam2_1"],
        labels=["person", "car"],
        closed=True
    )

    orchestrator.consolidated_manager._events = {ce_id: ce}

    # Run the method under test
    print(f"Running _on_consolidated_event_close for {ce_id}...")
    orchestrator._on_consolidated_event_close(ce_id)

    # Verify calls
    calls = orchestrator.file_manager.export_and_transcode_clip_calls
    print(f"Captured {len(calls)} calls to export_and_transcode_clip")

    expected_calls_by_camera = {}
    for call in calls:
        expected_calls_by_camera[call['camera']] = call

    if "camera1" not in expected_calls_by_camera:
        print("FAIL: No export call for camera1")
        return False

    call1 = expected_calls_by_camera["camera1"]
    # Camera 1 expected: min_start=1000.0, max_end=1050.0 + buffer (which is added inside export_and_transcode_clip if handled correctly, but wait...)
    # In my proposed fix, I calculate max_end based on event end times.
    # The existing code (which I'm testing against now) uses CE global times.
    # So CE global start=1000.0, CE global end=1050.0.
    # This test is designed to verify the FIX, so let's assert what we EXPECT after the fix.

    # After fix:
    # Camera 1: start=1000.0 (min of 1000 and 1020), end=1050.0 (max of 1010 and 1050)
    # Representative event: evt_cam1_2 (longest duration)

    # Camera 2: start=1005.0, end=1035.0
    # Representative event: evt_cam2_1

    print(f"Camera 1 Call: start={call1['start_time']}, end={call1['end_time']}, event_id={call1['event_id']}")

    success = True
    if call1['start_time'] != 1000.0:
        print(f"FAIL: Camera 1 start time mismatch. Expected 1000.0, got {call1['start_time']}")
        success = False
    if call1['end_time'] != 1050.0:
        print(f"FAIL: Camera 1 end time mismatch. Expected 1050.0, got {call1['end_time']}")
        success = False
    if call1['event_id'] != "evt_cam1_2": # Should pick the longer event
        print(f"FAIL: Camera 1 event ID mismatch. Expected evt_cam1_2, got {call1['event_id']}")
        success = False

    if "camera2" not in expected_calls_by_camera:
        print("FAIL: No export call for camera2")
        return False

    call2 = expected_calls_by_camera["camera2"]
    print(f"Camera 2 Call: start={call2['start_time']}, end={call2['end_time']}, event_id={call2['event_id']}")

    if call2['start_time'] != 1005.0:
        print(f"FAIL: Camera 2 start time mismatch. Expected 1005.0, got {call2['start_time']}")
        success = False
    if call2['end_time'] != 1035.0:
        print(f"FAIL: Camera 2 end time mismatch. Expected 1035.0, got {call2['end_time']}")
        success = False
    if call2['event_id'] != "evt_cam2_1":
        print(f"FAIL: Camera 2 event ID mismatch. Expected evt_cam2_1, got {call2['event_id']}")
        success = False

    if success:
        print("SUCCESS: All checks passed!")
    else:
        print("FAILURE: Some checks failed.")

    return success

if __name__ == "__main__":
    if test_on_consolidated_event_close():
        sys.exit(0)
    else:
        sys.exit(1)
