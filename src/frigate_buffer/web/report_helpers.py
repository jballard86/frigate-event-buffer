"""
Daily report helpers for the web layer.

Listing and reading report markdown from daily_reports/ (YYYY-MM-DD_report.md).
Path safety for reading uses path_helpers.resolve_under_storage.
"""

import os
from datetime import date, datetime

from frigate_buffer.web.path_helpers import resolve_under_storage


def daily_reports_dir(storage_path: str) -> str:
    """Return the path to the daily_reports directory under storage."""
    return os.path.join(storage_path, "daily_reports")


def list_report_dates(storage_path: str) -> list[str]:
    """
    Return sorted list of YYYY-MM-DD for which we have a report file, newest first.

    Scans daily_reports/ for *_report.md and parses the date prefix.
    """
    reports_dir = daily_reports_dir(storage_path)
    if not os.path.isdir(reports_dir):
        return []
    dates: list[str] = []
    for name in os.listdir(reports_dir):
        if not name.endswith("_report.md"):
            continue
        date_str = name.replace("_report.md", "")
        if len(date_str) == 10 and date_str[4] == "-" and date_str[7] == "-":
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
                dates.append(date_str)
            except ValueError:
                pass
    return sorted(dates, reverse=True)


def get_report_for_date(storage_path: str, d: date) -> dict | None:
    """
    Read report markdown for date. Returns {'summary': str} or None if missing.

    Uses resolve_under_storage so the path cannot escape storage.
    """
    path = resolve_under_storage(
        storage_path, "daily_reports", f"{d.isoformat()}_report.md"
    )
    if path is None or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return {"summary": f.read()}
    except OSError:
        return None
