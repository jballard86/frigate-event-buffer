"""
Tests that cleanup_old_events deletes test1, test2, ... folders under events/ by mtime.
Uses workspace tests/tmp_cleanup for temp dirs to avoid sandbox permission issues.
"""
import os
import time
import shutil
import pytest

from frigate_buffer.managers.file import FileManager


def _tmp_dir():
    base = os.path.join(os.path.dirname(__file__), "tmp_cleanup")
    os.makedirs(base, exist_ok=True)
    return base


def test_cleanup_deletes_old_test_folder():
    """Test that events/test1 (and testN) are deleted when older than retention."""
    tmp = _tmp_dir()
    sub = os.path.join(tmp, "old_test")
    try:
        fm = FileManager(storage_path=sub, retention_days=1)
        events_dir = os.path.join(sub, "events")
        os.makedirs(events_dir, exist_ok=True)
        test1 = os.path.join(events_dir, "test1")
        os.makedirs(test1, exist_ok=True)
        old = time.time() - (2 * 86400)
        os.utime(test1, (old, old))
        deleted = fm.cleanup_old_events(active_event_ids=[], active_ce_folder_names=[])
        assert deleted == 1
        assert not os.path.isdir(test1)
    finally:
        shutil.rmtree(sub, ignore_errors=True)


def test_cleanup_keeps_recent_test_folder():
    """Test that events/test1 is kept when younger than retention."""
    tmp = _tmp_dir()
    sub = os.path.join(tmp, "recent_test")
    try:
        fm = FileManager(storage_path=sub, retention_days=1)
        events_dir = os.path.join(sub, "events")
        os.makedirs(events_dir, exist_ok=True)
        test1 = os.path.join(events_dir, "test1")
        os.makedirs(test1, exist_ok=True)
        deleted = fm.cleanup_old_events(active_event_ids=[], active_ce_folder_names=[])
        assert deleted == 0
        assert os.path.isdir(test1)
    finally:
        shutil.rmtree(sub, ignore_errors=True)


def test_cleanup_ignores_non_test_events_folders():
    """Test that events/ subdirs that are not testN are still parsed as timestamp_id."""
    tmp = _tmp_dir()
    sub = os.path.join(tmp, "ce_style")
    try:
        fm = FileManager(storage_path=sub, retention_days=1)
        events_dir = os.path.join(sub, "events")
        os.makedirs(events_dir, exist_ok=True)
        old_ce = os.path.join(events_dir, "1730000000_myce")
        os.makedirs(old_ce, exist_ok=True)
        os.utime(old_ce, (0, 0))
        deleted = fm.cleanup_old_events(active_event_ids=[], active_ce_folder_names=[])
        assert deleted == 1
        assert not os.path.isdir(old_ce)
    finally:
        shutil.rmtree(sub, ignore_errors=True)
