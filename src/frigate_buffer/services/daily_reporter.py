"""
Daily Report Service - Aggregates analysis_result.json by date and generates an AI daily report.

Scans STORAGE_PATH for analysis_result.json in event folders, filters by target_date,
builds an event list, fills the report prompt template, calls the Gemini proxy via
GeminiAnalysisService.send_text_prompt, and saves Markdown to daily_reports/.
"""

import json
import logging
import os
from datetime import date, datetime, time as dt_time
from typing import Iterator

from frigate_buffer.services.ai_analyzer import GeminiAnalysisService

logger = logging.getLogger("frigate-buffer")


class DailyReporterService:
    """Generates daily security reports from aggregated analysis_result.json files."""

    def __init__(self, config: dict, storage_path: str, ai_analyzer: GeminiAnalysisService):
        self.config = config
        self.storage_path = os.path.abspath(storage_path)
        self.ai_analyzer = ai_analyzer
        self._prompt_template: str | None = None
        self._report_prompt_file = (config.get("REPORT_PROMPT_FILE") or "").strip()
        if not self._report_prompt_file:
            self._report_prompt_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "report_prompt.txt"
            )
        self._known_person_name = (config.get("REPORT_KNOWN_PERSON_NAME") or "").strip() or "Unspecified"

    def _load_prompt_template(self) -> str:
        if self._prompt_template is not None:
            return self._prompt_template
        if self._report_prompt_file and os.path.isfile(self._report_prompt_file):
            try:
                with open(self._report_prompt_file, "r", encoding="utf-8") as f:
                    self._prompt_template = f.read()
                return self._prompt_template
            except OSError as e:
                logger.warning("Could not read report prompt file %s: %s", self._report_prompt_file, e)
        self._prompt_template = (
            "You are a security report writer. Time range: {report_start_time} to {report_end_time}. "
            "Date: {report_date_string}. Known person: {known_person_name}. Events:\n{list_of_event_json_objects}\n\n"
            "Produce a concise daily security report in Markdown."
        )
        return self._prompt_template

    def _folder_name_and_timestamp(self, json_path: str) -> tuple[str | None, int | None]:
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

    def _collect_events_for_date(self, target_date: date) -> Iterator[tuple[str, dict, int]]:
        """Scan storage for analysis_result.json; yield (json_path, data, unix_ts) for target_date (generator to limit peak memory)."""
        if not os.path.isdir(self.storage_path):
            return
        for root, _dirs, files in os.walk(self.storage_path):
            if "analysis_result.json" not in files:
                continue
            json_path = os.path.join(root, "analysis_result.json")
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
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.debug("Skip invalid or unreadable %s: %s", json_path, e)
                continue
            yield (json_path, data, unix_ts)

    def _aggregate_event_lines(self, events: list[tuple[str, dict, int]]) -> list[str]:
        """Build sorted list of lines: '[{time}] {title}: {shortSummary} (Threat: {level})'."""
        lines = []
        for _path, data, unix_ts in events:
            title = (data.get("title") or "").strip()
            short_summary = (data.get("shortSummary") or data.get("description") or "").strip()
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
                camera = os.path.basename(os.path.dirname(event_dir)) if os.path.dirname(event_dir) else "unknown"
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
        Scan for analysis_result.json for target_date, build event list, call AI, save Markdown.
        Returns True if a report was written, False otherwise.
        Consumes _collect_events_for_date generator in one pass to limit peak memory.
        """
        event_objects: list[dict] = []
        for json_path, data, unix_ts in self._collect_events_for_date(target_date):
            event_dir = os.path.dirname(json_path)
            parent_basename = os.path.basename(os.path.dirname(event_dir))
            camera = os.path.basename(event_dir) if parent_basename == "events" else (os.path.basename(os.path.dirname(event_dir)) if os.path.dirname(event_dir) else "unknown")
            try:
                threat_level = int(data.get("potential_threat_level", 0))
            except (TypeError, ValueError):
                threat_level = 0
            event_objects.append({
                "title": data.get("title", ""),
                "scene": data.get("scene", ""),
                "confidence": data.get("confidence", 0),
                "threat_level": threat_level,
                "camera": camera,
                "time": datetime.fromtimestamp(unix_ts).strftime("%Y-%m-%d %H:%M:%S"),
                "context": data.get("context", []),
            })
        # Same value for both {event_list} and {list_of_event_json_objects} (JSON array).
        list_of_event_json_objects = json.dumps(event_objects, indent=2, ensure_ascii=False) if event_objects else "[]"
        event_list_str = list_of_event_json_objects

        date_str = target_date.isoformat()
        report_date_string = date_str
        start_of_day = datetime.combine(target_date, dt_time.min)
        end_of_day = datetime.combine(target_date, dt_time.max)
        report_start_time = start_of_day.strftime("%Y-%m-%d %H:%M:%S")
        report_end_time = end_of_day.strftime("%Y-%m-%d %H:%M:%S")

        template = self._load_prompt_template()
        system_prompt = (
            template.replace("{date}", date_str)
            .replace("{event_list}", event_list_str)
            .replace("{report_date_string}", report_date_string)
            .replace("{report_start_time}", report_start_time)
            .replace("{report_end_time}", report_end_time)
            .replace("{list_of_event_json_objects}", list_of_event_json_objects)
            .replace("{known_person_name}", self._known_person_name)
        )
        user_prompt = list_of_event_json_objects if event_objects else "No events recorded for this date."

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
            return True
        except OSError as e:
            logger.error("Could not write daily report to %s: %s", out_path, e)
            return False
