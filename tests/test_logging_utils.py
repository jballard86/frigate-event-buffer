"""Tests for logging_utils: ErrorBuffer, suppress_review_debug_logs flag,
and StreamCaptureHandler."""

import logging
import unittest

from frigate_buffer.logging_utils import (
    StreamCaptureHandler,
    set_suppress_review_debug_logs,
    should_suppress_review_debug_logs,
)


class TestSuppressReviewDebugLogs(unittest.TestCase):
    """Verify test-run flag is set and read correctly; always reset in teardown."""

    def tearDown(self) -> None:
        set_suppress_review_debug_logs(False)

    def test_default_is_false(self) -> None:
        set_suppress_review_debug_logs(False)
        assert not should_suppress_review_debug_logs()

    def test_set_true_then_read(self) -> None:
        set_suppress_review_debug_logs(True)
        assert should_suppress_review_debug_logs()

    def test_set_false_after_true(self) -> None:
        set_suppress_review_debug_logs(True)
        set_suppress_review_debug_logs(False)
        assert not should_suppress_review_debug_logs()


class TestStreamCaptureHandlerFilter(unittest.TestCase):
    """StreamCaptureHandler captures non-MQTT logs and skips MQTT module records."""

    def test_captures_non_mqtt_log(self) -> None:
        captured: list[str] = []
        handler = StreamCaptureHandler(captured=captured)
        logger = logging.getLogger("frigate-buffer")
        logger.addHandler(handler)
        prev_level = logger.level
        try:
            logger.setLevel(logging.DEBUG)
            logger.info("Hello from video")
            assert len(captured) == 1
            assert "Hello from video" in captured[0]
            assert "INFO" in captured[0]
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prev_level)

    def test_skips_mqtt_handler_records(self) -> None:
        captured: list[str] = []
        handler = StreamCaptureHandler(captured=captured)
        record = logging.LogRecord(
            name="frigate-buffer",
            level=logging.DEBUG,
            pathname="/path/to/mqtt_handler.py",
            lineno=1,
            msg="MQTT message received",
            args=(),
            exc_info=None,
        )
        record.module = "mqtt_handler"
        handler.emit(record)
        assert len(captured) == 0

    def test_skips_mqtt_client_records(self) -> None:
        captured: list[str] = []
        handler = StreamCaptureHandler(captured=captured)
        record = logging.LogRecord(
            name="frigate-buffer",
            level=logging.DEBUG,
            pathname="/path/to/mqtt_client.py",
            lineno=1,
            msg="Connected",
            args=(),
            exc_info=None,
        )
        record.module = "mqtt_client"
        handler.emit(record)
        assert len(captured) == 0

    def test_captures_when_pathname_has_no_mqtt(self) -> None:
        captured: list[str] = []
        handler = StreamCaptureHandler(captured=captured)
        record = logging.LogRecord(
            name="frigate-buffer",
            level=logging.INFO,
            pathname="/path/to/video_compilation.py",
            lineno=1,
            msg="Compilation step",
            args=(),
            exc_info=None,
        )
        record.module = "video_compilation"
        handler.emit(record)
        assert len(captured) == 1
        assert "Compilation step" in captured[0]
