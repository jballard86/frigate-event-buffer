"""Tests for logging_utils: ErrorBuffer and suppress_review_debug_logs flag."""

import unittest

from frigate_buffer.logging_utils import (
    set_suppress_review_debug_logs,
    should_suppress_review_debug_logs,
)


class TestSuppressReviewDebugLogs(unittest.TestCase):
    """Verify test-run flag is set and read correctly; always reset in teardown."""

    def tearDown(self) -> None:
        set_suppress_review_debug_logs(False)

    def test_default_is_false(self) -> None:
        set_suppress_review_debug_logs(False)
        self.assertFalse(should_suppress_review_debug_logs())

    def test_set_true_then_read(self) -> None:
        set_suppress_review_debug_logs(True)
        self.assertTrue(should_suppress_review_debug_logs())

    def test_set_false_after_true(self) -> None:
        set_suppress_review_debug_logs(True)
        set_suppress_review_debug_logs(False)
        self.assertFalse(should_suppress_review_debug_logs())
