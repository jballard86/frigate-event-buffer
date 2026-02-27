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
        assert SmartZoneFilter.normalize_sub_label(None) is None

        # 2. String
        assert SmartZoneFilter.normalize_sub_label("valid") == "valid"
        assert SmartZoneFilter.normalize_sub_label("") is None
        assert SmartZoneFilter.normalize_sub_label("   ") is None
        assert SmartZoneFilter.normalize_sub_label("  valid  ") == "valid"

        # 3. List/Tuple
        assert SmartZoneFilter.normalize_sub_label(["valid", 0.9]) == "valid"
        assert SmartZoneFilter.normalize_sub_label(("valid", 0.9)) == "valid"
        assert SmartZoneFilter.normalize_sub_label([]) is None
        assert SmartZoneFilter.normalize_sub_label(()) is None
        assert SmartZoneFilter.normalize_sub_label([None, 0.9]) is None
        assert SmartZoneFilter.normalize_sub_label(["", 0.9]) is None
        assert SmartZoneFilter.normalize_sub_label(["   ", 0.9]) is None

        # Non-string first element in list/tuple
        # The implementation converts non-None non-string first elements to string
        assert SmartZoneFilter.normalize_sub_label([123, 0.9]) == "123"
        assert SmartZoneFilter.normalize_sub_label([123]) == "123"

        # 4. Unexpected types
        assert SmartZoneFilter.normalize_sub_label(123) is None
        assert SmartZoneFilter.normalize_sub_label({}) is None
        assert SmartZoneFilter.normalize_sub_label(object()) is None

    def test_should_start_event_no_config(self):
        """Test default behavior when no filters are defined."""
        # Camera not in config
        assert self.filter.should_start_event("cam5", "label", "sub", ["zone1"])
        # Camera in config but empty filters
        assert self.filter.should_start_event("cam4", "label", "sub", ["zone1"])

    def test_should_start_event_exceptions(self):
        """Test exception logic (label and sub_label matching, case insensitivity)."""
        # Label match
        assert self.filter.should_start_event("cam1", "exception_label", "sub", [])
        assert self.filter.should_start_event("cam1", "EXCEPTION_LABEL", "sub", [])

        # Sub label match
        assert self.filter.should_start_event(
            "cam1", "label", "exception_sub_label", []
        )
        assert self.filter.should_start_event(
            "cam1", "label", ["exception_sub_label", 0.9], []
        )
        assert self.filter.should_start_event(
            "cam1", "label", "EXCEPTION_SUB_LABEL", []
        )

        # No match
        assert not self.filter.should_start_event("cam1", "label", "sub", [])

    def test_should_start_event_tracked_zones(self):
        """Test zone logic (matching, non-matching, empty entered zones)."""
        # Match zone
        assert self.filter.should_start_event("cam1", "label", "sub", ["zone1"])
        assert self.filter.should_start_event("cam1", "label", "sub", ["ZONE1"])
        assert self.filter.should_start_event("cam1", "label", "sub", ["zone2"])
        assert self.filter.should_start_event(
            "cam1", "label", "sub", ["other", "zone1"]
        )

        # No match zone
        assert not self.filter.should_start_event(
            "cam1", "label", "sub", ["other_zone"]
        )
        assert not self.filter.should_start_event("cam1", "label", "sub", [])
        assert not self.filter.should_start_event("cam1", "label", "sub", None)

        # Only tracked zones defined
        assert self.filter.should_start_event("cam2", "label", "sub", ["zone1"])
        assert not self.filter.should_start_event("cam2", "label", "sub", ["other"])

    def test_should_start_event_interaction(self):
        """Test interaction between exceptions and tracked zones."""
        # Exception matched, zone not matched -> True
        assert self.filter.should_start_event(
            "cam1", "exception_label", "sub", ["other_zone"]
        )
        # Exception matched, no zones -> True
        assert self.filter.should_start_event("cam1", "exception_label", "sub", [])

        # Exception not matched, zone matched -> True
        assert self.filter.should_start_event("cam1", "label", "sub", ["zone1"])

        # Neither matched -> False
        assert not self.filter.should_start_event(
            "cam1", "label", "sub", ["other_zone"]
        )

    def test_should_start_event_edge_cases(self):
        """Test edge cases."""
        # None label/sub_label
        assert not self.filter.should_start_event("cam1", None, None, ["other"])
        assert self.filter.should_start_event("cam1", None, None, ["zone1"])

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
        assert not f.should_start_event("cam_bad", "label", "sub", ["zone1"])

        # Test with empty tracked_zones list explicitly
        config_empty = {"CAMERA_EVENT_FILTERS": {"cam_empty": {"tracked_zones": []}}}
        f_empty = SmartZoneFilter(config_empty)
        # Empty tracked_zones means track everything (default behavior)
        assert f_empty.should_start_event("cam_empty", "label", "sub", ["any"])

    def test_should_start_event_current_zones(self):
        """Test that current_zones alone can satisfy tracked_zones
        (entered_zones empty)."""
        # current_zones only, tracked zone present -> True
        assert self.filter.should_start_event("cam1", "label", "sub", [], ["zone1"])
        assert self.filter.should_start_event("cam2", "label", "sub", [], ["zone1"])
        # current_zones only, no tracked zone -> False
        assert not self.filter.should_start_event("cam1", "label", "sub", [], ["other"])
        # Both entered and current empty -> False
        assert not self.filter.should_start_event("cam1", "label", "sub", [], [])
        # Backward compat: 4 args only (current_zones defaults to empty)
        # -> same as before
        assert not self.filter.should_start_event("cam1", "label", "sub", [])
        assert self.filter.should_start_event("cam1", "label", "sub", ["zone1"])


if __name__ == "__main__":
    unittest.main()
