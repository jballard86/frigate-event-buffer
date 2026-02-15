"""Tests for SmartZoneFilter."""
import os
import sys
import unittest
from unittest.mock import MagicMock

# Mock requests module before importing frigate_buffer managers
sys.modules["requests"] = MagicMock()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frigate_buffer.managers.zone_filter import SmartZoneFilter

class TestSmartZoneFilter(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()
