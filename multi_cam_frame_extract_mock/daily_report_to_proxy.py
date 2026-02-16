# Lives in multi_cam_frame_extract_mock/ for standalone development; plan to merge into frigate_buffer later (see PLAN_TO_MERGE_LATER.md).
#
# Purpose: At 1am (configurable), compile all multi-cam recap responses from the previous
# calendar day and send them to the Gemini proxy to produce a single AI daily report.
# Prompt is loaded from report_prompt.txt.

import os
import json
import time
import yaml
import logging

# Optional for 1am scheduling: use 'schedule' or threading.Timer / system cron

# --- Mock: features, timelines, inputs, outputs ---------------------------------
#
# FEATURES:
# - Scheduled run at 1am (config: daily_report_schedule_hour, default 1).
# - Define "previous day" as calendar day (midnight to midnight) in local time or config TZ.
# - Gather all recap/summary content from the previous day (see INPUTS).
# - Load system prompt from report_prompt.txt (same dir as script or config path).
# - Build one OpenAI-format request: system = report_prompt.txt, user = compiled list of recaps.
# - POST to Gemini proxy (same URL/api_key as frame extractor); no images, text only.
# - Parse response: choices[0].message.content = daily report text.
# - Save report to a known path (see OUTPUTS) and optionally log to timeline or notify.
#
# TIMELINES:
# - Run once per day at daily_report_schedule_hour (e.g. 1am).
# - "Previous day" = yesterday 00:00:00 to 23:59:59 (local or config timezone).
# - If no recaps exist for that day, either skip proxy call and write "No activity" or still call with empty list per prompt.
#
# INPUTS:
# - report_prompt.txt: system prompt for the report (edit file to change).
# - Previous day's recaps: need a defined storage layout. Options:
#   A) Multi-cam frame extractor writes a small JSON or text file per recap (e.g. multi_cam_frames/recap_<event_id>_<ts>.json with {"summary": "...", "event_id": "...", "camera": "...", "timestamp": ...}).
#   B) Or scan event folders for multi_cam_frames/ and read a recap.txt or similar written by the extractor after each proxy response.
#   C) Or main app stores Gemini recap responses in a daily index (e.g. storage/daily_recaps/YYYY-MM-DD/recaps.json).
#   Mock: assume we have a function get_previous_day_recaps(storage_path, date) that returns a list of {"summary": str, "event_id": str, "camera": str, "timestamp": float} (or paths to read from).
# - Config: GEMINI_PROXY_URL, GEMINI_PROXY_API_KEY, STORAGE_PATH, DAILY_REPORT_SCHEDULE_HOUR, REPORT_PROMPT_FILE, REPORT_OUTPUT_DIR.
#
# OUTPUTS:
# - Proxy response: choices[0].message.content (daily report markdown or plain text).
# - Save to: REPORT_OUTPUT_DIR / daily_reports / YYYY-MM-DD_report.md (or .txt). Configurable.
# - Optional: append to a "last report" path for dashboard, or send to HA / notification.
# ----------------------------------------------------------------------------------

logger = logging.getLogger("DailyReportService")

CONFIG_FILE = os.getenv("CONFIG_FILE", "config.yaml")
DEFAULT_CONFIG = {
    "gemini_proxy_url": "REDACTED_LOCAL_IP:5050",
    "gemini_proxy_api_key": "",
    "gemini_proxy_model": "gemini-2.5-flash-lite",
    "daily_report_schedule_hour": 1,
    "report_prompt_file": "",
    "storage_path": os.getenv("STORAGE_PATH", "/app/storage"),
    "report_output_dir": "",  # default: storage_path/daily_reports or same dir as script
    "known_person_name": "",  # e.g. "Joe" — from config or Frigate sub_label; used in report_prompt.txt
}


def load_config():
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = yaml.safe_load(f)
                if data:
                    # Flatten nested keys if present
                    for key in ("gemini_proxy_url", "gemini_proxy_api_key", "gemini_proxy_model",
                                "daily_report_schedule_hour", "report_prompt_file", "storage_path", "report_output_dir", "known_person_name"):
                        if key in data:
                            config[key] = data[key]
                    if "network" in data:
                        config["gemini_proxy_url"] = data["network"].get("gemini_proxy_url", config["gemini_proxy_url"])
                        config["storage_path"] = data["network"].get("storage_path", config["storage_path"])
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    return config


def load_report_prompt_template():
    """Load raw template from report_prompt.txt (contains {report_start_time}, {report_end_time}, etc.)."""
    prompt_file = CONF.get("report_prompt_file") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "report_prompt.txt"
    )
    if os.path.isfile(prompt_file):
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not read report prompt file {prompt_file}: {e}")
    return "You are a security report writer. Summarize the provided list of daily recap summaries into one concise daily report."


def build_report_prompt_final(
    report_start_time: str,
    report_end_time: str,
    report_date_string: str,
    known_person_name: str,
    list_of_event_json_objects: str,
) -> str:
    """
    Fill report_prompt.txt placeholders. Data from config and get_previous_day_recaps.
    Placeholders: {report_start_time}, {report_end_time}, {report_date_string}, {known_person_name}, {list_of_event_json_objects}.
    list_of_event_json_objects should be a JSON array string (each object: title, scene, confidence, threat_level, camera, time, context).
    """
    template = load_report_prompt_template()
    return (
        template.replace("{report_start_time}", report_start_time)
        .replace("{report_end_time}", report_end_time)
        .replace("{report_date_string}", report_date_string)
        .replace("{known_person_name}", known_person_name or "Unspecified")
        .replace("{list_of_event_json_objects}", list_of_event_json_objects)
    )


# --- Mock: data gathering --------------------------------------------------------
def get_previous_day_recaps(storage_path: str, date_str: str):
    """
    Return list of event objects for the given calendar day (date_str YYYY-MM-DD).
    Scans storage for analysis_result.json files written by the Extractor (multi_cam_frame_extracter.py)
    inside multi_cam_frames/ in event folders. Folder date is derived from event folder name (timestamp_* or ce_id).
    Each item: title, scene, confidence, threat_level, camera, time, context (array; empty in single-file scan).
    """
    from datetime import datetime as dt
    events = []
    if not os.path.isdir(storage_path):
        return events
    for root, dirs, files in os.walk(storage_path):
        if "analysis_result.json" not in files:
            continue
        json_path = os.path.join(root, "analysis_result.json")
        # multi_cam_frames dir contains analysis_result.json; parent is event folder (camera/ts_id or events/ce_id/camera)
        multi_cam_dir = os.path.dirname(json_path)
        event_folder = os.path.basename(os.path.dirname(multi_cam_dir))
        if not event_folder:
            continue
        # Event folder name: {timestamp}_{event_id} or {ts}_{short_uuid}; first part is Unix timestamp
        try:
            ts_str = event_folder.split("_")[0]
            ts_int = int(ts_str)
            event_date = dt.fromtimestamp(ts_int).strftime("%Y-%m-%d")
        except (ValueError, OSError):
            continue
        if event_date != date_str:
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        events.append({
            "title": data.get("title", ""),
            "scene": data.get("scene", ""),
            "confidence": data.get("confidence", 0),
            "threat_level": data.get("potential_threat_level", 0),
            "camera": data.get("camera", "unknown"),
            "time": data.get("start_time"),
            "context": data.get("context", []),
        })
    return events


def build_user_message(recaps: list) -> str:
    """Turn list of recaps into one user message for the proxy."""
    if not recaps:
        return "No multi-camera recap summaries were recorded for this day."
    lines = []
    for i, r in enumerate(recaps, 1):
        summary = r.get("summary", "")
        camera = r.get("camera", "unknown")
        ts = r.get("timestamp")
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else ""
        lines.append(f"[{i}] ({camera}) {time_str}\n{summary}")
    return "\n\n---\n\n".join(lines)


# --- Mock: proxy call and save ---------------------------------------------------
def send_report_to_proxy(system_prompt: str, user_message: str) -> str:
    """
    POST to Gemini proxy (OpenAI-format, text only). Returns choices[0].message.content.
    Mock: no actual HTTP call; return placeholder.
    """
    # TODO: requests.post(CONF["gemini_proxy_url"] + "/v1/chat/completions", json={...}, headers={"Authorization": "Bearer " + CONF["gemini_proxy_api_key"]})
    _ = system_prompt, user_message
    return "[Mock] Daily report would be returned from proxy here."


def save_report(report_text: str, date_str: str):
    """Write report to REPORT_OUTPUT_DIR / daily_reports / {date_str}_report.md."""
    out_dir = CONF.get("report_output_dir") or os.path.join(CONF["storage_path"], "daily_reports")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{date_str}_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"Saved daily report to {path}")


# --- Entry: run for previous day -------------------------------------------------
def run_daily_report():
    """Compile previous day's recaps, fill report prompt placeholders, send to proxy, save report. Call at 1am."""
    from datetime import datetime, timedelta
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    report_start_time = f"{yesterday} 00:00:00"
    report_end_time = f"{yesterday} 23:59:59"
    report_date_string = yesterday
    known_person_name = CONF.get("known_person_name") or ""

    recaps = get_previous_day_recaps(CONF["storage_path"], yesterday)
    # Event objects for prompt: title, scene, confidence, threat_level, camera, time, context (array)
    list_of_event_json_objects = json.dumps(recaps, indent=2) if recaps else "[]"

    system_prompt = build_report_prompt_final(
        report_start_time=report_start_time,
        report_end_time=report_end_time,
        report_date_string=report_date_string,
        known_person_name=known_person_name,
        list_of_event_json_objects=list_of_event_json_objects,
    )
    user_message = "Generate the report." if recaps else "No events for this day. State that in the report."
    report_text = send_report_to_proxy(system_prompt, user_message)
    save_report(report_text, yesterday)


def main():
    """Either run once (for testing) or schedule at daily_report_schedule_hour."""
    global CONF
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    CONF = load_config()
    # For testing: run immediately. For production: schedule at CONF["daily_report_schedule_hour"] (e.g. 1am)
    run_daily_report()


if __name__ == "__main__":
    CONF = load_config()
    main()
