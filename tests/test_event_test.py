"""
Tests for event_test package: ce_start_time helper and orchestrator validation.
"""

import os
import shutil

from frigate_buffer.event_test.event_test_orchestrator import (
    _yield_done,
    _yield_error,
    _yield_log,
    get_ce_start_time_from_folder_path,
    run_test_pipeline,
)


class TestGetCeStartTimeFromFolderPath:
    """Tests for get_ce_start_time_from_folder_path."""

    def test_timestamp_id_returns_timestamp(self):
        # Path that looks like CE folder name 1730000123_abc (path need not exist)
        path = os.path.join("events", "1730000123_abc")
        assert get_ce_start_time_from_folder_path(path) == 1730000123.0

    def test_test_folder_returns_zero(self):
        assert get_ce_start_time_from_folder_path("/some/events/test1") == 0.0
        assert get_ce_start_time_from_folder_path("/events/test42") == 0.0

    def test_plain_name_returns_zero(self):
        assert get_ce_start_time_from_folder_path("/path/to/MyEvent") == 0.0


class TestRunTestPipelineValidation:
    """Tests that run_test_pipeline yields error when source is invalid
    or incomplete."""

    def test_nonexistent_source_yields_error(self):
        tmp = os.path.join(os.path.dirname(__file__), "tmp_event_test")
        os.makedirs(tmp, exist_ok=True)
        try:
            source = os.path.join(tmp, "events", "nonexistent")
            events = list(
                run_test_pipeline(
                    source,
                    tmp,
                    None,
                    None,
                    None,
                    None,
                    {},
                )
            )
            errs = [e for e in events if e.get("type") == "error"]
            assert len(errs) == 1
            assert (
                "readable" in errs[0].get("message", "").lower()
                or "not found" in errs[0].get("message", "").lower()
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_clip_yields_error(self):
        tmp = os.path.join(os.path.dirname(__file__), "tmp_event_test_miss")
        events_dir = os.path.join(tmp, "events")
        os.makedirs(events_dir, exist_ok=True)
        source = os.path.join(events_dir, "1730000000_ev")
        cam_dir = os.path.join(source, "camera1")
        os.makedirs(cam_dir, exist_ok=True)
        try:
            # No clip.mp4
            events = list(
                run_test_pipeline(
                    source,
                    tmp,
                    None,
                    None,
                    None,
                    None,
                    {},
                )
            )
            errs = [e for e in events if e.get("type") == "error"]
            assert len(errs) == 1
            assert (
                "camera" in errs[0].get("message", "").lower()
                or "clip" in errs[0].get("message", "").lower()
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestEventHelpers:
    """Test yield helpers."""

    def test_yield_log(self):
        ev = _yield_log("hello")
        assert ev["type"] == "log"
        assert ev["message"] == "hello"

    def test_yield_done(self):
        ev = _yield_done("test1", "/files/events/test1/ai_request.html")
        assert ev["type"] == "done"
        assert ev["test_run_id"] == "test1"
        assert ev["ai_request_url"] == "/files/events/test1/ai_request.html"

    def test_yield_error(self):
        ev = _yield_error("fail")
        assert ev["type"] == "error"
        assert ev["message"] == "fail"
