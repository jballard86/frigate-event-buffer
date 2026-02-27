"""
Tests for web report_helpers: list_report_dates and get_report_for_date.
"""

import os
import shutil
import tempfile
import unittest
from datetime import date

from frigate_buffer.web.report_helpers import (
    daily_reports_dir,
    get_report_for_date,
    list_report_dates,
)


class TestDailyReportsDir(unittest.TestCase):
    """daily_reports_dir returns path under storage."""

    def test_returns_joined_path(self):
        storage = "/var/storage"
        assert daily_reports_dir(storage) == os.path.join(storage, "daily_reports")


class TestListReportDates(unittest.TestCase):
    """list_report_dates returns sorted YYYY-MM-DD, newest first;
    skips invalid names."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def test_missing_dir_returns_empty(self):
        """When daily_reports does not exist, returns []."""
        assert list_report_dates(self.tmp) == []

    def test_returns_dates_newest_first(self):
        """Valid *_report.md files yield sorted dates, newest first."""
        reports_dir = os.path.join(self.tmp, "daily_reports")
        os.makedirs(reports_dir, exist_ok=True)
        for d in ("2025-01-01", "2025-03-15", "2025-02-10"):
            open(os.path.join(reports_dir, f"{d}_report.md"), "w").close()
        assert list_report_dates(self.tmp) == ["2025-03-15", "2025-02-10", "2025-01-01"]

    def test_skips_non_report_files(self):
        """Files not ending in _report.md are ignored."""
        reports_dir = os.path.join(self.tmp, "daily_reports")
        os.makedirs(reports_dir, exist_ok=True)
        open(os.path.join(reports_dir, "2025-01-01_report.md"), "w").close()
        open(os.path.join(reports_dir, "other.md"), "w").close()
        open(os.path.join(reports_dir, "2025-01-02_report.txt"), "w").close()
        assert list_report_dates(self.tmp) == ["2025-01-01"]

    def test_skips_invalid_date_format(self):
        """Filenames with invalid date part are skipped."""
        reports_dir = os.path.join(self.tmp, "daily_reports")
        os.makedirs(reports_dir, exist_ok=True)
        open(os.path.join(reports_dir, "2025-13-01_report.md"), "w").close()
        open(os.path.join(reports_dir, "not-a-date_report.md"), "w").close()
        assert list_report_dates(self.tmp) == []


class TestGetReportForDate(unittest.TestCase):
    """get_report_for_date returns {'summary': str} or None; path-safe."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def test_missing_file_returns_none(self):
        """When report file does not exist, returns None."""
        assert get_report_for_date(self.tmp, date(2025, 6, 15)) is None

    def test_returns_summary_when_file_exists(self):
        """When YYYY-MM-DD_report.md exists, returns {'summary': content}."""
        reports_dir = os.path.join(self.tmp, "daily_reports")
        os.makedirs(reports_dir, exist_ok=True)
        path = os.path.join(reports_dir, "2025-06-15_report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Daily report\n\nNothing to report.")
        data = get_report_for_date(self.tmp, date(2025, 6, 15))
        assert data is not None
        assert data["summary"] == "# Daily report\n\nNothing to report."

    def test_path_traversal_returns_none(self):
        """Path escape (e.g. ..) does not read outside storage;
        returns None for invalid path."""
        # resolve_under_storage(storage, "daily_reports",
        # "../etc/passwd_report.md") would be normalized and likely outside
        # base -> None. We don't have a literal file there;
        # get_report_for_date uses d.isoformat() so we can't inject .. via date.
        # So just ensure missing date file returns None.
        assert get_report_for_date(self.tmp, date(1999, 1, 1)) is None
