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
        events_list, total_seen, total_matched = service._collect_events_for_date(target)
        self.assertEqual(len(events_list), 1)
        self.assertEqual(total_matched, 1)
        self.assertEqual(events_list[0][1].get("title"), "Person at door")

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
        events_list, total_seen, total_matched = service._collect_events_for_date(target)
        self.assertEqual(len(events_list), 1)
        self.assertEqual(total_matched, 1)
        self.assertEqual(events_list[0][1].get("title"), "CE event")

    def test_collect_events_for_date_returns_tuple_of_list_and_counts(self):
        """_collect_events_for_date returns (events_list, total_seen, total_matched)."""
        import datetime
        config = {}
        mock_analyzer = MagicMock()
        service = DailyReporterService(config, tempfile.gettempdir(), mock_analyzer)
        result = service._collect_events_for_date(datetime.date.today())
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        events_list, total_seen, total_matched = result
        self.assertIsInstance(events_list, list)
        self.assertIsInstance(total_seen, int)
        self.assertIsInstance(total_matched, int)
        self.assertEqual(total_matched, len(events_list))


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
    """Test that template placeholders are replaced (date, event_list, known_person_name, report_*)."""

    def test_known_person_name_replaced_in_system_prompt(self):
        """REPORT_KNOWN_PERSON_NAME from config is used for {known_person_name} placeholder."""
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Report for {date}. Known person: {known_person_name}.")
            prompt_path = f.name
        self.addCleanup(lambda: os.path.exists(prompt_path) and os.unlink(prompt_path))
        config = {"REPORT_PROMPT_FILE": prompt_path, "REPORT_KNOWN_PERSON_NAME": "Joe"}
        mock_analyzer = MagicMock()
        mock_analyzer.send_text_prompt.return_value = "# Done"
        service = DailyReporterService(config, storage, mock_analyzer)
        service.generate_report(date(2025, 1, 15))
        call_args = mock_analyzer.send_text_prompt.call_args
        system_prompt = call_args[0][0]
        self.assertIn("Known person: Joe.", system_prompt)

    def test_known_person_name_defaults_to_unspecified_when_empty(self):
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Known: {known_person_name}")
            prompt_path = f.name
        self.addCleanup(lambda: os.path.exists(prompt_path) and os.unlink(prompt_path))
        config = {"REPORT_PROMPT_FILE": prompt_path}
        mock_analyzer = MagicMock()
        mock_analyzer.send_text_prompt.return_value = "# Done"
        service = DailyReporterService(config, storage, mock_analyzer)
        service.generate_report(date(2025, 1, 15))
        system_prompt = mock_analyzer.send_text_prompt.call_args[0][0]
        self.assertIn("Known: Unspecified", system_prompt)

    def test_load_prompt_replaces_date_and_event_list(self):
        """With no events, {date} and {event_list} are replaced; {event_list} gets JSON []."""
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
        self.assertIn("[]", system_prompt)
        self.assertNotIn("{event_list}", system_prompt)

    def test_new_placeholders_replaced_in_system_prompt(self):
        """report_date_string, report_start_time, report_end_time, list_of_event_json_objects are replaced."""
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(
                "Range: {report_start_time} to {report_end_time}. "
                "Title: {report_date_string}. Events: {list_of_event_json_objects}."
            )
            prompt_path = f.name
        self.addCleanup(lambda: os.path.exists(prompt_path) and os.unlink(prompt_path))
        config = {"REPORT_PROMPT_FILE": prompt_path}
        mock_analyzer = MagicMock()
        mock_analyzer.send_text_prompt.return_value = "# Done"
        service = DailyReporterService(config, storage, mock_analyzer)
        service.generate_report(date(2025, 1, 15))
        system_prompt = mock_analyzer.send_text_prompt.call_args[0][0]
        self.assertIn("2025-01-15 00:00:00", system_prompt)
        self.assertIn("2025-01-15 23:59:59", system_prompt)
        self.assertIn("Title: 2025-01-15.", system_prompt)
        self.assertIn("Events: [].", system_prompt)
        self.assertNotIn("{report_start_time}", system_prompt)
        self.assertNotIn("{report_end_time}", system_prompt)
        self.assertNotIn("{report_date_string}", system_prompt)
        self.assertNotIn("{list_of_event_json_objects}", system_prompt)

    def test_missing_prompt_file_returns_false_and_does_not_call_proxy(self):
        """When report prompt file is missing, generate_report returns False and send_text_prompt is not called."""
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        nonexistent = os.path.join(storage, "nonexistent_report_prompt.txt")
        self.assertFalse(os.path.isfile(nonexistent))
        config = {"REPORT_PROMPT_FILE": nonexistent}
        mock_analyzer = MagicMock()
        service = DailyReporterService(config, storage, mock_analyzer)
        result = service.generate_report(date(2025, 1, 15))
        self.assertFalse(result)
        mock_analyzer.send_text_prompt.assert_not_called()

    def test_multiple_events_appear_in_list_of_event_json_objects(self):
        """Two event folders for same date produce two event objects in the prompt."""
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        for i, (cam, evt_id) in enumerate([("cam1", "1739617200_evt1"), ("cam2", "1739617300_evt2")]):
            event_dir = os.path.join(storage, cam, evt_id)
            os.makedirs(event_dir, exist_ok=True)
            with open(os.path.join(event_dir, "analysis_result.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "title": f"Event {i+1}",
                    "shortSummary": f"Summary {i+1}.",
                    "potential_threat_level": 0,
                }, f)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Events: {list_of_event_json_objects}")
            prompt_path = f.name
        self.addCleanup(lambda: os.path.exists(prompt_path) and os.unlink(prompt_path))
        config = {"REPORT_PROMPT_FILE": prompt_path}
        mock_analyzer = MagicMock()
        mock_analyzer.send_text_prompt.return_value = "# Done"
        service = DailyReporterService(config, storage, mock_analyzer)
        service.generate_report(date(2025, 2, 15))
        system_prompt = mock_analyzer.send_text_prompt.call_args[0][0]
        self.assertIn("Event 1", system_prompt)
        self.assertIn("Event 2", system_prompt)
        start = system_prompt.index("Events: ") + len("Events: ")
        parsed = json.loads(system_prompt[start:].strip())
        self.assertEqual(len(parsed), 2)
        titles = {o.get("title") for o in parsed}
        self.assertEqual(titles, {"Event 1", "Event 2"})


class TestDailyReporterServiceAggregateFile(unittest.TestCase):
    """Test aggregate file: read when present, fallback to scan, delete on success."""

    def test_report_uses_aggregate_file_when_present(self):
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        os.makedirs(os.path.join(storage, "daily_reports"), exist_ok=True)
        aggregate_path = os.path.join(storage, "daily_reports", "aggregate_2025-02-15.jsonl")
        with open(aggregate_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"title": "From aggregate", "camera": "cam1", "time": "2025-02-15 10:00:00", "threat_level": 0, "scene": "", "confidence": 0, "context": []}) + "\n")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Events: {list_of_event_json_objects}")
            prompt_path = f.name
        self.addCleanup(lambda: os.path.exists(prompt_path) and os.unlink(prompt_path))
        config = {"REPORT_PROMPT_FILE": prompt_path}
        mock_analyzer = MagicMock()
        mock_analyzer.send_text_prompt.return_value = "# Done"
        service = DailyReporterService(config, storage, mock_analyzer)
        service.generate_report(date(2025, 2, 15))
        system_prompt = mock_analyzer.send_text_prompt.call_args[0][0]
        self.assertIn("From aggregate", system_prompt)

    def test_report_falls_back_to_scan_when_aggregate_missing(self):
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        event_dir = os.path.join(storage, "cam", "1739617200_evt1")
        os.makedirs(event_dir, exist_ok=True)
        with open(os.path.join(event_dir, "analysis_result.json"), "w", encoding="utf-8") as f:
            json.dump({"title": "From scan", "potential_threat_level": 0}, f)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Events: {list_of_event_json_objects}")
            prompt_path = f.name
        self.addCleanup(lambda: os.path.exists(prompt_path) and os.unlink(prompt_path))
        config = {"REPORT_PROMPT_FILE": prompt_path}
        mock_analyzer = MagicMock()
        mock_analyzer.send_text_prompt.return_value = "# Done"
        service = DailyReporterService(config, storage, mock_analyzer)
        service.generate_report(date(2025, 2, 15))
        system_prompt = mock_analyzer.send_text_prompt.call_args[0][0]
        self.assertIn("From scan", system_prompt)

    def test_successful_report_deletes_aggregate_file(self):
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        os.makedirs(os.path.join(storage, "daily_reports"), exist_ok=True)
        aggregate_path = os.path.join(storage, "daily_reports", "aggregate_2025-02-15.jsonl")
        with open(aggregate_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"title": "One", "camera": "c", "time": "2025-02-15 10:00:00", "threat_level": 0, "scene": "", "confidence": 0, "context": []}) + "\n")
        self.assertTrue(os.path.isfile(aggregate_path))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Report")
            prompt_path = f.name
        self.addCleanup(lambda: os.path.exists(prompt_path) and os.unlink(prompt_path))
        config = {"REPORT_PROMPT_FILE": prompt_path}
        mock_analyzer = MagicMock()
        mock_analyzer.send_text_prompt.return_value = "# Report"
        service = DailyReporterService(config, storage, mock_analyzer)
        result = service.generate_report(date(2025, 2, 15))
        self.assertTrue(result)
        self.assertFalse(os.path.isfile(aggregate_path))


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


class TestDailyReporterServiceFolderNameAndTimestamp(unittest.TestCase):
    """Edge cases for _folder_name_and_timestamp (invalid or odd folder names)."""

    def test_folder_name_no_underscore_returns_none_ts(self):
        config = {}
        mock_analyzer = MagicMock()
        service = DailyReporterService(config, "/tmp", mock_analyzer)
        # Folder name "single" has no underscore -> split("_")[0] is "single" -> int("single") raises ValueError -> returns (None, None)
        json_path = os.path.join("/tmp", "camera", "single", "analysis_result.json")
        folder_name, unix_ts = service._folder_name_and_timestamp(json_path)
        self.assertIsNone(unix_ts)
        self.assertIsNone(folder_name)

    def test_folder_name_non_numeric_prefix_returns_none_ts(self):
        config = {}
        mock_analyzer = MagicMock()
        service = DailyReporterService(config, "/tmp", mock_analyzer)
        json_path = os.path.join("/tmp", "camera", "abc_evt1", "analysis_result.json")
        folder_name, unix_ts = service._folder_name_and_timestamp(json_path)
        self.assertIsNone(unix_ts)

    def test_folder_name_events_layout_uses_parent_basename(self):
        config = {}
        mock_analyzer = MagicMock()
        service = DailyReporterService(config, "/tmp", mock_analyzer)
        json_path = os.path.join("/tmp", "events", "1739617200_abc12", "doorbell", "analysis_result.json")
        folder_name, unix_ts = service._folder_name_and_timestamp(json_path)
        self.assertEqual(folder_name, "1739617200_abc12")
        self.assertEqual(unix_ts, 1739617200)


class TestDailyReporterServiceAggregateEdgeCases(unittest.TestCase):
    """Edge cases for _aggregate_event_lines: missing or invalid potential_threat_level."""

    def test_aggregate_handles_missing_threat_level(self):
        config = {}
        mock_analyzer = MagicMock()
        service = DailyReporterService(config, "/tmp", mock_analyzer)
        events = [
            ("/path/a.json", {"title": "T", "shortSummary": "S"}, 1234567890),
        ]
        # potential_threat_level missing -> default 0
        lines = service._aggregate_event_lines(events)
        self.assertEqual(len(lines), 1)
        self.assertIn("(Threat: 0)", lines[0])

    def test_aggregate_handles_invalid_threat_level_gracefully(self):
        config = {}
        mock_analyzer = MagicMock()
        service = DailyReporterService(config, "/tmp", mock_analyzer)
        events = [
            ("/path/a.json", {"title": "T", "shortSummary": "S", "potential_threat_level": "high"}, 1234567890),
        ]
        # int("high") would raise ValueError; we need the code to tolerate that
        try:
            lines = service._aggregate_event_lines(events)
            self.assertEqual(len(lines), 1)
        except ValueError:
            self.fail("_aggregate_event_lines should not raise for non-int potential_threat_level")


class TestDailyReporterServiceCleanupOldReports(unittest.TestCase):
    """Test cleanup_old_reports deletes reports older than retention_days."""

    def test_cleanup_old_reports_deletes_old_keeps_recent(self):
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        reports_dir = os.path.join(storage, "daily_reports")
        os.makedirs(reports_dir, exist_ok=True)
        # Old: 100 days ago
        old_date = date(2020, 1, 1)
        old_path = os.path.join(reports_dir, f"{old_date.isoformat()}_report.md")
        with open(old_path, "w", encoding="utf-8") as f:
            f.write("# Old report\n")
        # Recent: 5 days ago (we'll use retention_days=10 so this is kept)
        from datetime import timedelta
        recent = date.today() - timedelta(days=5)
        recent_path = os.path.join(reports_dir, f"{recent.isoformat()}_report.md")
        with open(recent_path, "w", encoding="utf-8") as f:
            f.write("# Recent report\n")
        config = {}
        mock_analyzer = MagicMock()
        service = DailyReporterService(config, storage, mock_analyzer)
        deleted = service.cleanup_old_reports(retention_days=10)
        self.assertEqual(deleted, 1)
        self.assertFalse(os.path.isfile(old_path))
        self.assertTrue(os.path.isfile(recent_path))
