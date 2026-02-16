"""
Tests for DailyReporterService: scan for analysis_result.json, aggregate event lines,
fill prompt template, call send_text_prompt, save Markdown to daily_reports/.
"""

import json
import os
import shutil
import tempfile
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

from frigate_buffer.services.daily_reporter import DailyReporterService


class TestDailyReporterServiceScan(unittest.TestCase):
    """Test that only analysis_result.json for target_date are collected."""

    def test_collects_only_events_matching_target_date(self):
        # 1234567890 (Unix) -> 2009-02-13 in UTC/local. 1000000000 -> 2001-09-08/09.
        target = date(2009, 2, 13)
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        cam = os.path.join(storage, "doorbell")
        in_date_dir = os.path.join(cam, "1234567890_evt1")
        out_date_dir = os.path.join(cam, "1000000000_evt2")
        os.makedirs(in_date_dir, exist_ok=True)
        os.makedirs(out_date_dir, exist_ok=True)
        with open(os.path.join(in_date_dir, "analysis_result.json"), "w", encoding="utf-8") as f:
            json.dump({
                "title": "Person at door",
                "shortSummary": "Someone rang.",
                "potential_threat_level": 0,
            }, f)
        with open(os.path.join(out_date_dir, "analysis_result.json"), "w", encoding="utf-8") as f:
            json.dump({
                "title": "Other",
                "shortSummary": "Other day.",
                "potential_threat_level": 1,
            }, f)
        config = {}
        mock_analyzer = MagicMock()
        service = DailyReporterService(config, storage, mock_analyzer)
        events = service._collect_events_for_date(target)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][1].get("title"), "Person at door")

    def test_consolidated_event_folder_date_parsed(self):
        # events/{ts}_{uuid}/camera/analysis_result.json -> folder name ts_uuid; 1234567890 -> 2009-02-13
        target = date(2009, 2, 13)
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        events_dir = os.path.join(storage, "events")
        ce_dir = os.path.join(events_dir, "1234567890_abc12")
        cam_dir = os.path.join(ce_dir, "doorbell")
        os.makedirs(cam_dir, exist_ok=True)
        with open(os.path.join(cam_dir, "analysis_result.json"), "w", encoding="utf-8") as f:
            json.dump({
                "title": "CE event",
                "shortSummary": "Consolidated.",
                "potential_threat_level": 1,
            }, f)
        config = {}
        mock_analyzer = MagicMock()
        service = DailyReporterService(config, storage, mock_analyzer)
        events = service._collect_events_for_date(target)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][1].get("title"), "CE event")


class TestDailyReporterServiceAggregate(unittest.TestCase):
    """Test event line format: [{time}] {title}: {shortSummary} (Threat: {level})."""

    def test_aggregate_event_lines_format(self):
        config = {}
        mock_analyzer = MagicMock()
        service = DailyReporterService(config, "/tmp", mock_analyzer)
        events = [
            ("/path/to/1234567890_evt/analysis_result.json", {
                "title": "Delivery",
                "shortSummary": "Package left.",
                "potential_threat_level": 0,
            }, 1234567890),
        ]
        lines = service._aggregate_event_lines(events)
        self.assertEqual(len(lines), 1)
        self.assertIn("[", lines[0])
        self.assertIn("] Delivery: Package left. (Threat: 0)", lines[0])


class TestDailyReporterServicePrompt(unittest.TestCase):
    """Test that template placeholders {date} and {event_list} are replaced."""

    def test_load_prompt_replaces_date_and_event_list(self):
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Date: {date}\nEvents:\n{event_list}")
            prompt_path = f.name
        self.addCleanup(lambda: os.path.exists(prompt_path) and os.unlink(prompt_path))
        config = {"REPORT_PROMPT_FILE": prompt_path}
        mock_analyzer = MagicMock()
        mock_analyzer.send_text_prompt.return_value = "# Done"
        service = DailyReporterService(config, storage, mock_analyzer)
        result = service.generate_report(date(2025, 1, 15))
        self.assertTrue(result)
        call_args = mock_analyzer.send_text_prompt.call_args
        self.assertIsNotNone(call_args)
        system_prompt = call_args[0][0]
        self.assertIn("2025-01-15", system_prompt)
        self.assertIn("(No events for this date.)", system_prompt)


class TestDailyReporterServiceSave(unittest.TestCase):
    """Test that report Markdown is written to daily_reports/{date}_report.md."""

    def test_generate_report_writes_md_file(self):
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        cam_dir = os.path.join(storage, "cam")
        event_dir = os.path.join(cam_dir, "1739617200_evt1")  # 2025-02-15
        os.makedirs(event_dir, exist_ok=True)
        with open(os.path.join(event_dir, "analysis_result.json"), "w", encoding="utf-8") as f:
            json.dump({
                "title": "Test",
                "shortSummary": "Summary.",
                "potential_threat_level": 0,
            }, f)
        config = {}
        mock_analyzer = MagicMock()
        mock_analyzer.send_text_prompt.return_value = "# Report\nDone."
        service = DailyReporterService(config, storage, mock_analyzer)
        target = date(2025, 2, 15)
        result = service.generate_report(target)
        self.assertTrue(result)
        out_path = os.path.join(storage, "daily_reports", "2025-02-15_report.md")
        self.assertTrue(os.path.isfile(out_path))
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertEqual(content, "# Report\nDone.")


class TestDailyReporterServiceEdgeCases(unittest.TestCase):
    """Test no events for date and proxy returning None."""

    def test_no_events_for_date_does_not_crash(self):
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        config = {}
        mock_analyzer = MagicMock()
        mock_analyzer.send_text_prompt.return_value = "# No activity\nNo events."
        service = DailyReporterService(config, storage, mock_analyzer)
        result = service.generate_report(date(2020, 6, 1))
        self.assertTrue(result)
        mock_analyzer.send_text_prompt.assert_called_once()

    def test_proxy_returns_none_returns_false(self):
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        cam_dir = os.path.join(storage, "cam")
        event_dir = os.path.join(cam_dir, "1739617200_evt1")
        os.makedirs(event_dir, exist_ok=True)
        with open(os.path.join(event_dir, "analysis_result.json"), "w", encoding="utf-8") as f:
            json.dump({"title": "T", "shortSummary": "S", "potential_threat_level": 0}, f)
        config = {}
        mock_analyzer = MagicMock()
        mock_analyzer.send_text_prompt.return_value = None
        service = DailyReporterService(config, storage, mock_analyzer)
        result = service.generate_report(date(2025, 2, 15))
        self.assertFalse(result)
        out_path = os.path.join(storage, "daily_reports", "2025-02-15_report.md")
        self.assertFalse(os.path.isfile(out_path))
