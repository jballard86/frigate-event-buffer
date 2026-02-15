import unittest
import os
import shutil
import tempfile
import time
import json
from frigate_buffer.services.query import EventQueryService

class TestQueryCaching(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # shorter TTL for testing
        self.ttl = 2
        self.service = EventQueryService(self.test_dir, cache_ttl=self.ttl)

        # Setup dummy camera and event
        self.cam = "test_cam"
        os.makedirs(os.path.join(self.test_dir, self.cam))
        self.event_id = "event1"
        self.event_dir = os.path.join(self.test_dir, self.cam, f"{int(time.time())}_{self.event_id}")
        os.makedirs(self.event_dir)

        self.summary_path = os.path.join(self.event_dir, "summary.txt")
        with open(self.summary_path, "w") as f:
            f.write("Title: Initial Title")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_caching_behavior(self):
        # 1. Initial fetch
        events = self.service.get_events(self.cam)
        self.assertEqual(events[0]['title'], "Initial Title")

        # 2. Modify file on disk
        with open(self.summary_path, "w") as f:
            f.write("Title: Modified Title")

        # Simulate directory modification (required for mtime-based caching)
        # Ensure mtime actually changes by forcing it forward
        st = os.stat(self.event_dir)
        os.utime(self.event_dir, (st.st_atime, st.st_mtime + 1.0))

        # 3. Fetch immediately - should be cached (stale)
        events_cached = self.service.get_events(self.cam)
        self.assertEqual(events_cached[0]['title'], "Initial Title", "Should return cached data immediately")

        # 4. Wait for TTL expiry
        time.sleep(self.ttl + 0.1)

        # 5. Fetch again - should be fresh
        events_fresh = self.service.get_events(self.cam)
        self.assertEqual(events_fresh[0]['title'], "Modified Title", "Should return fresh data after TTL")

if __name__ == '__main__':
    unittest.main()
