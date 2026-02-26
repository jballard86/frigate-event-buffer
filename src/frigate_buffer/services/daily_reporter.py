"""
Daily Report Service - Aggregates analysis_result.json by date and generates an AI daily report.

Reads from daily_reports/aggregate_YYYY-MM-DD.jsonl when present (events appended as they are
analyzed); otherwise scans STORAGE_PATH for analysis_result.json. Fills the report prompt template,
calls the Gemini proxy via GeminiAnalysisService.send_text_prompt, saves Markdown to daily_reports/,
and deletes the aggregate file on success.
"""

import json
import logging
import os
from datetime import date, datetime, timedelta
from datetime import time as dt_time
from typing import Any

from frigate_buffer.services.ai_analyzer import GeminiAnalysisService

logger = logging.getLogger("frigate-buffer")

AGGREGATE_FILENAME_PREFIX = "aggregate_"
AGGREGATE_FILENAME_SUFFIX = ".jsonl"


class DailyReporterService:
    """Generates daily security reports from aggregated analysis_result.json files."""

    def __init__(
        self, config: dict, storage_path: str, ai_analyzer: GeminiAnalysisService
    ):
        self.config = config
        self.storage_path = os.path.abspath(storage_path)
        self.ai_analyzer = ai_analyzer
        self._prompt_template: str | None = None
        self._report_prompt_file = (config.get("REPORT_PROMPT_FILE") or "").strip()
        if not self._report_prompt_file:
            self._report_prompt_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "report_prompt.txt"
            )
        self._known_person_name = (
            config.get("REPORT_KNOWN_PERSON_NAME") or ""
        ).strip() or "Unspecified"

    def _aggregate_file_path(self, target_date: date) -> str:
        """Path to the per-day aggregate JSONL file."""
        out_dir = os.path.join(self.storage_path, "daily_reports")
        return os.path.join(
            out_dir,
            f"{AGGREGATE_FILENAME_PREFIX}{target_date.isoformat()}{AGGREGATE_FILENAME_SUFFIX}",
        )

    def _load_events_from_aggregate(self, target_date: date) -> list[dict] | None:
        """
        Load event objects from aggregate_YYYY-MM-DD.jsonl if present.
        Returns None if file missing or unreadable; returns list (maybe empty) if file exists.
        """
        path = self._aggregate_file_path(target_date)
        if not os.path.isfile(path):
            return None
        events: list[dict] = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            events.append(obj)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return None
        return events

    def _delete_aggregate_for_date(self, target_date: date) -> None:
        """Remove the aggregate file for target_date after a successful report."""
        path = self._aggregate_file_path(target_date)
        if os.path.isfile(path):
            try:
                os.remove(path)
                logger.debug("Removed aggregate file %s after successful report", path)
            except OSError as e:
                logger.warning("Could not remove aggregate file %s: %s", path, e)

    def append_event_to_daily_aggregate(
        self, event_date: date, event_payload: dict[str, Any]
    ) -> None:
        """
        Append one event to the per-day aggregate JSONL file (for report-time read).
        Call this when an analysis result is persisted. event_payload must have keys
        compatible with the report: title, scene, confidence, threat_level, camera, time, context.
        """
        out_dir = os.path.join(self.storage_path, "daily_reports")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            logger.warning("Could not create daily_reports dir for aggregate: %s", e)
            return
        path = self._aggregate_file_path(event_date)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_payload, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("Could not append to aggregate %s: %s", path, e)

    def _load_prompt_template(self) -> str | None:
        """Load report prompt from file. Returns None if file is missing or unreadable (no fallback)."""
        if self._prompt_template is not None:
            return self._prompt_template
        if not self._report_prompt_file:
            logger.error("Daily report skipped: report prompt file path is empty")
            return None
        if not os.path.isfile(self._report_prompt_file):
            logger.error(
                "Daily report skipped: report prompt file not found: %s",
                self._report_prompt_file,
            )
            return None
        try:
            with open(self._report_prompt_file, encoding="utf-8") as f:
                self._prompt_template = f.read()
            return self._prompt_template
        except OSError as e:
            logger.error(
                "Daily report skipped: could not read report prompt file %s: %s",
                self._report_prompt_file,
                e,
            )
            return None

    def _folder_name_and_timestamp(
        self, json_path: str
    ) -> tuple[str | None, int | None]:
        """Return (folder_name, unix_timestamp) for the event folder containing this analysis_result.json."""
        event_dir = os.path.dirname(os.path.abspath(json_path))
        parent_dir = os.path.dirname(event_dir)
        parent_basename = os.path.basename(parent_dir)
        grandparent_basename = os.path.basename(os.path.dirname(parent_dir))
        if grandparent_basename == "events":
            folder_name = parent_basename
        else:
            folder_name = os.path.basename(event_dir)
        try:
            ts_str = folder_name.split("_")[0]
            ts_int = int(ts_str)
            return folder_name, ts_int
        except (ValueError, IndexError):
            return None, None

    def _collect_events_for_date(
        self, target_date: date
    ) -> tuple[list[tuple[str, dict, int]], int, int]:
        """
        Scan storage for analysis_result.json; return (events, total_seen, total_matched).
        total_seen = number of analysis_result.json paths found; total_matched = events for target_date.
        """
        events: list[tuple[str, dict, int]] = []
        total_seen = 0
        if not os.path.isdir(self.storage_path):
            return (events, 0, 0)
        for root, _dirs, files in os.walk(self.storage_path):
            if "analysis_result.json" not in files:
                continue
            json_path = os.path.join(root, "analysis_result.json")
            total_seen += 1
            _folder_name, unix_ts = self._folder_name_and_timestamp(json_path)
            if unix_ts is None:
                continue
            try:
                event_date = date.fromtimestamp(unix_ts)
            except (ValueError, OSError):
                continue
            if event_date != target_date:
                continue
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.debug("Skip invalid or unreadable %s: %s", json_path, e)
                continue
            events.append((json_path, data, unix_ts))
        return (events, total_seen, len(events))

    def _aggregate_event_lines(self, events: list[tuple[str, dict, int]]) -> list[str]:
        """Build sorted list of lines: '[{time}] {title}: {shortSummary} (Threat: {level})'."""
        lines = []
        for _path, data, unix_ts in events:
            title = (data.get("title") or "").strip()
            short_summary = (
                data.get("shortSummary") or data.get("description") or ""
            ).strip()
            try:
                level = int(data.get("potential_threat_level", 0))
            except (TypeError, ValueError):
                level = 0
            time_str = datetime.fromtimestamp(unix_ts).strftime("%H:%M")
            line = f"[{time_str}] {title}: {short_summary} (Threat: {level})"
            lines.append((unix_ts, line))
        lines.sort(key=lambda x: (x[0], x[1]))
        return [line for _ts, line in lines]

    def _build_event_json_objects(self, events: list[tuple[str, dict, int]]) -> str:
        """Build JSON array string for list_of_event_json_objects placeholder."""
        objects = []
        for json_path, data, unix_ts in events:
            event_dir = os.path.dirname(json_path)
            parent_basename = os.path.basename(os.path.dirname(event_dir))
            if parent_basename == "events":
                camera = os.path.basename(event_dir)
            else:
                camera = (
                    os.path.basename(os.path.dirname(event_dir))
                    if os.path.dirname(event_dir)
                    else "unknown"
                )
            try:
                threat_level = int(data.get("potential_threat_level", 0))
            except (TypeError, ValueError):
                threat_level = 0
            obj = {
                "title": data.get("title", ""),
                "scene": data.get("scene", ""),
                "confidence": data.get("confidence", 0),
                "threat_level": threat_level,
                "camera": camera,
                "time": datetime.fromtimestamp(unix_ts).strftime("%Y-%m-%d %H:%M:%S"),
                "context": data.get("context", []),
            }
            objects.append(obj)
        return json.dumps(objects, indent=2, ensure_ascii=False)

    def generate_report(self, target_date: date) -> bool:
        """
        Build event list from aggregate file (if present) or scan; call AI, save Markdown.
        Returns True if a report was written, False otherwise.
        On success, deletes the aggregate file for target_date.
        """
        date_str = target_date.isoformat()
        event_objects: list[dict] = []

        # Prefer aggregate file written as events are analyzed.
        from_aggregate = self._load_events_from_aggregate(target_date)
        if from_aggregate is not None:
            event_objects = sorted(from_aggregate, key=lambda o: o.get("time", ""))
            logger.info(
                "Daily report for %s: including %d event(s) from aggregate file",
                date_str,
                len(event_objects),
            )
        else:
            events_list, total_seen, total_matched = self._collect_events_for_date(
                target_date
            )
            logger.info(
                "Daily report scan: %d analysis_result.json found, %d for date %s",
                total_seen,
                total_matched,
                date_str,
            )
            for json_path, data, unix_ts in events_list:
                event_dir = os.path.dirname(json_path)
                parent_basename = os.path.basename(os.path.dirname(event_dir))
                camera = (
                    os.path.basename(event_dir)
                    if parent_basename == "events"
                    else (
                        os.path.basename(os.path.dirname(event_dir))
                        if os.path.dirname(event_dir)
                        else "unknown"
                    )
                )
                try:
                    threat_level = int(data.get("potential_threat_level", 0))
                except (TypeError, ValueError):
                    threat_level = 0
                event_objects.append(
                    {
                        "title": data.get("title", ""),
                        "scene": data.get("scene", ""),
                        "confidence": data.get("confidence", 0),
                        "threat_level": threat_level,
                        "camera": camera,
                        "time": datetime.fromtimestamp(unix_ts).strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "context": data.get("context", []),
                    }
                )
            logger.info(
                "Daily report for %s: including %d event(s) from analysis_result.json",
                date_str,
                len(event_objects),
            )
            for json_path, _data, _ts in events_list:
                logger.debug("Included event from %s", json_path)

        # Same value for both {event_list} and {list_of_event_json_objects} (JSON array).
        list_of_event_json_objects = (
            json.dumps(event_objects, indent=2, ensure_ascii=False)
            if event_objects
            else "[]"
        )
        event_list_str = list_of_event_json_objects

        report_date_string = date_str
        start_of_day = datetime.combine(target_date, dt_time.min)
        end_of_day = datetime.combine(target_date, dt_time.max)
        report_start_time = start_of_day.strftime("%Y-%m-%d %H:%M:%S")
        report_end_time = end_of_day.strftime("%Y-%m-%d %H:%M:%S")

        template = self._load_prompt_template()
        if template is None:
            logger.info("Daily report for %s skipped (prompt unavailable)", date_str)
            return False
        system_prompt = (
            template.replace("{date}", date_str)
            .replace("{event_list}", event_list_str)
            .replace("{report_date_string}", report_date_string)
            .replace("{report_start_time}", report_start_time)
            .replace("{report_end_time}", report_end_time)
            .replace("{list_of_event_json_objects}", list_of_event_json_objects)
            .replace("{known_person_name}", self._known_person_name)
        )
        user_prompt = (
            list_of_event_json_objects
            if event_objects
            else "No events recorded for this date."
        )

        report_md = self.ai_analyzer.send_text_prompt(system_prompt, user_prompt)
        if not report_md:
            logger.warning("Daily report for %s: no response from proxy", date_str)
            return False

        out_dir = os.path.join(self.storage_path, "daily_reports")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            logger.error("Could not create daily_reports dir %s: %s", out_dir, e)
            return False
        out_path = os.path.join(out_dir, f"{target_date.isoformat()}_report.md")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report_md)
            logger.info("Daily report written to %s", out_path)
            self._delete_aggregate_for_date(target_date)
            return True
        except OSError as e:
            logger.error("Could not write daily report to %s: %s", out_path, e)
            return False

    def cleanup_old_reports(self, retention_days: int) -> int:
        """
        Remove report files older than retention_days. Returns count deleted.
        Only deletes files matching YYYY-MM-DD_report.md.
        """
        out_dir = os.path.join(self.storage_path, "daily_reports")
        if not os.path.isdir(out_dir):
            return 0
        cutoff = date.today() - timedelta(days=retention_days)
        deleted = 0
        for name in os.listdir(out_dir):
            if not name.endswith("_report.md"):
                continue
            date_str = name.replace("_report.md", "")
            if len(date_str) != 10 or date_str[4] != "-" or date_str[7] != "-":
                continue
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d").date()
                if d < cutoff:
                    path = os.path.join(out_dir, name)
                    os.remove(path)
                    deleted += 1
                    logger.info("Cleaned up old daily report: %s", name)
            except ValueError:
                pass
        return deleted
