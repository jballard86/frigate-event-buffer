"""Tests for SmartZoneFilter."""

import os
import sys
import unittest
from unittest.mock import MagicMock

# Mock requests module before importing frigate_buffer managers
sys.modules["requests"] = MagicMock()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frigate_buffer.managers.zone_filter import SmartZoneFilter


class TestSmartZoneFilter(unittest.TestCase):
    def setUp(self):
        self.config = {
            "CAMERA_EVENT_FILTERS": {
                "cam1": {
                    "exceptions": ["exception_label", "exception_sub_label"],
                    "tracked_zones": ["zone1", "zone2"],
                },
                "cam2": {"tracked_zones": ["zone1"]},
                "cam3": {"exceptions": ["exception_label"]},
                "cam4": {},
            }
        }
        self.filter = SmartZoneFilter(self.config)

    def test_normalize_sub_label(self):
        """Test normalize_sub_label handles various input formats."""

        # 1. None
        self.assertIsNone(SmartZoneFilter.normalize_sub_label(None))

        # 2. String
        self.assertEqual(SmartZoneFilter.normalize_sub_label("valid"), "valid")
        self.assertIsNone(SmartZoneFilter.normalize_sub_label(""))
        self.assertIsNone(SmartZoneFilter.normalize_sub_label("   "))
        self.assertEqual(SmartZoneFilter.normalize_sub_label("  valid  "), "valid")

        # 3. List/Tuple
        self.assertEqual(SmartZoneFilter.normalize_sub_label(["valid", 0.9]), "valid")
        self.assertEqual(SmartZoneFilter.normalize_sub_label(("valid", 0.9)), "valid")
        self.assertIsNone(SmartZoneFilter.normalize_sub_label([]))
        self.assertIsNone(SmartZoneFilter.normalize_sub_label(()))
        self.assertIsNone(SmartZoneFilter.normalize_sub_label([None, 0.9]))
        self.assertIsNone(SmartZoneFilter.normalize_sub_label(["", 0.9]))
        self.assertIsNone(SmartZoneFilter.normalize_sub_label(["   ", 0.9]))

        # Non-string first element in list/tuple
        # The implementation converts non-None non-string first elements to string
        self.assertEqual(SmartZoneFilter.normalize_sub_label([123, 0.9]), "123")
        self.assertEqual(SmartZoneFilter.normalize_sub_label([123]), "123")

        # 4. Unexpected types
        self.assertIsNone(SmartZoneFilter.normalize_sub_label(123))
        self.assertIsNone(SmartZoneFilter.normalize_sub_label({}))
        self.assertIsNone(SmartZoneFilter.normalize_sub_label(object()))

    def test_should_start_event_no_config(self):
        """Test default behavior when no filters are defined."""
        # Camera not in config
        self.assertTrue(
            self.filter.should_start_event("cam5", "label", "sub", ["zone1"])
        )
        # Camera in config but empty filters
        self.assertTrue(
            self.filter.should_start_event("cam4", "label", "sub", ["zone1"])
        )

    def test_should_start_event_exceptions(self):
        """Test exception logic (label and sub_label matching, case insensitivity)."""
        # Label match
        self.assertTrue(
            self.filter.should_start_event("cam1", "exception_label", "sub", [])
        )
        self.assertTrue(
            self.filter.should_start_event("cam1", "EXCEPTION_LABEL", "sub", [])
        )

        # Sub label match
        self.assertTrue(
            self.filter.should_start_event("cam1", "label", "exception_sub_label", [])
        )
        self.assertTrue(
            self.filter.should_start_event(
                "cam1", "label", ["exception_sub_label", 0.9], []
            )
        )
        self.assertTrue(
            self.filter.should_start_event("cam1", "label", "EXCEPTION_SUB_LABEL", [])
        )

        # No match
        self.assertFalse(self.filter.should_start_event("cam1", "label", "sub", []))

    def test_should_start_event_tracked_zones(self):
        """Test zone logic (matching, non-matching, empty entered zones)."""
        # Match zone
        self.assertTrue(
            self.filter.should_start_event("cam1", "label", "sub", ["zone1"])
        )
        self.assertTrue(
            self.filter.should_start_event("cam1", "label", "sub", ["ZONE1"])
        )
        self.assertTrue(
            self.filter.should_start_event("cam1", "label", "sub", ["zone2"])
        )
        self.assertTrue(
            self.filter.should_start_event("cam1", "label", "sub", ["other", "zone1"])
        )

        # No match zone
        self.assertFalse(
            self.filter.should_start_event("cam1", "label", "sub", ["other_zone"])
        )
        self.assertFalse(self.filter.should_start_event("cam1", "label", "sub", []))
        self.assertFalse(self.filter.should_start_event("cam1", "label", "sub", None))

        # Only tracked zones defined
        self.assertTrue(
            self.filter.should_start_event("cam2", "label", "sub", ["zone1"])
        )
        self.assertFalse(
            self.filter.should_start_event("cam2", "label", "sub", ["other"])
        )

    def test_should_start_event_interaction(self):
        """Test interaction between exceptions and tracked zones."""
        # Exception matched, zone not matched -> True
        self.assertTrue(
            self.filter.should_start_event(
                "cam1", "exception_label", "sub", ["other_zone"]
            )
        )
        # Exception matched, no zones -> True
        self.assertTrue(
            self.filter.should_start_event("cam1", "exception_label", "sub", [])
        )

        # Exception not matched, zone matched -> True
        self.assertTrue(
            self.filter.should_start_event("cam1", "label", "sub", ["zone1"])
        )

        # Neither matched -> False
        self.assertFalse(
            self.filter.should_start_event("cam1", "label", "sub", ["other_zone"])
        )

    def test_should_start_event_edge_cases(self):
        """Test edge cases."""
        # None label/sub_label
        self.assertFalse(self.filter.should_start_event("cam1", None, None, ["other"]))
        self.assertTrue(self.filter.should_start_event("cam1", None, None, ["zone1"]))

        # Empty/whitespace values in config
        config = {
            "CAMERA_EVENT_FILTERS": {
                "cam_bad": {
                    "exceptions": [None, "", " "],
                    "tracked_zones": [None, "", " "],
                }
            }
        }
        f = SmartZoneFilter(config)

        # If tracked_zones list is not empty but contains only invalid values,
        # it effectively tracks nothing, so should return False.
        self.assertFalse(f.should_start_event("cam_bad", "label", "sub", ["zone1"]))

        # Test with empty tracked_zones list explicitly
        config_empty = {"CAMERA_EVENT_FILTERS": {"cam_empty": {"tracked_zones": []}}}
        f_empty = SmartZoneFilter(config_empty)
        # Empty tracked_zones means track everything (default behavior)
        self.assertTrue(
            f_empty.should_start_event("cam_empty", "label", "sub", ["any"])
        )

    def test_should_start_event_current_zones(self):
        """Test that current_zones alone can satisfy tracked_zones
        (entered_zones empty)."""
        # current_zones only, tracked zone present -> True
        self.assertTrue(
            self.filter.should_start_event("cam1", "label", "sub", [], ["zone1"])
        )
        self.assertTrue(
            self.filter.should_start_event("cam2", "label", "sub", [], ["zone1"])
        )
        # current_zones only, no tracked zone -> False
        self.assertFalse(
            self.filter.should_start_event("cam1", "label", "sub", [], ["other"])
        )
        # Both entered and current empty -> False
        self.assertFalse(self.filter.should_start_event("cam1", "label", "sub", [], []))
        # Backward compat: 4 args only (current_zones defaults to empty)
        # -> same as before
        self.assertFalse(self.filter.should_start_event("cam1", "label", "sub", []))
        self.assertTrue(
            self.filter.should_start_event("cam1", "label", "sub", ["zone1"])
        )


if __name__ == "__main__":
    unittest.main()
