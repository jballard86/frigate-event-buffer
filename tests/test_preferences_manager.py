"""Tests for PreferencesManager and POST /api/mobile/register."""

import json
import os
import shutil
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from frigate_buffer.managers.preferences import (
    PREFERENCES_FILENAME,
    PreferencesManager,
)
from frigate_buffer.services.query import EventQueryService


class TestPreferencesManager(unittest.TestCase):
    """Unit tests for PreferencesManager."""

    def test_get_fcm_token_when_file_missing_returns_none(self):
        """When mobile_preferences.json does not exist, get_fcm_token returns None."""
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        mgr = PreferencesManager(storage)
        assert mgr.get_fcm_token() is None

    def test_set_fcm_token_then_get_returns_token(self):
        """After set_fcm_token, get_fcm_token returns the same token."""
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        mgr = PreferencesManager(storage)
        mgr.set_fcm_token("my_fcm_token_123")
        assert mgr.get_fcm_token() == "my_fcm_token_123"

    def test_first_write_creates_file(self):
        """First set_fcm_token creates mobile_preferences.json under storage_path."""
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        mgr = PreferencesManager(storage)
        mgr.set_fcm_token("token")
        path = os.path.join(storage, PREFERENCES_FILENAME)
        assert os.path.isfile(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("token") == "token"

    def test_get_fcm_token_invalid_json_returns_none(self):
        """If file exists but is invalid JSON, get_fcm_token returns None."""
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        path = os.path.join(storage, PREFERENCES_FILENAME)
        os.makedirs(storage, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("not valid json {")
        mgr = PreferencesManager(storage)
        assert mgr.get_fcm_token() is None

    def test_get_fcm_token_missing_key_returns_none(self):
        """If file has no 'token' key, get_fcm_token returns None."""
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        path = os.path.join(storage, PREFERENCES_FILENAME)
        os.makedirs(storage, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"other": "value"}, f)
        mgr = PreferencesManager(storage)
        assert mgr.get_fcm_token() is None

    def test_get_fcm_token_empty_string_returns_none(self):
        """If token is empty string, get_fcm_token returns None."""
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        path = os.path.join(storage, PREFERENCES_FILENAME)
        os.makedirs(storage, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"token": ""}, f)
        mgr = PreferencesManager(storage)
        assert mgr.get_fcm_token() is None

    def test_set_fcm_token_overwrites_previous(self):
        """set_fcm_token overwrites the previous token."""
        storage = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(storage, ignore_errors=True))
        mgr = PreferencesManager(storage)
        mgr.set_fcm_token("first")
        mgr.set_fcm_token("second")
        assert mgr.get_fcm_token() == "second"


# Pytest-style tests for the same manager (and for API)
def test_get_fcm_token_when_file_missing_returns_none():
    """When mobile_preferences.json does not exist, get_fcm_token returns None."""
    with tempfile.TemporaryDirectory() as storage:
        mgr = PreferencesManager(storage)
        assert mgr.get_fcm_token() is None


def test_set_then_get_fcm_token():
    """After set_fcm_token, get_fcm_token returns the token."""
    with tempfile.TemporaryDirectory() as storage:
        mgr = PreferencesManager(storage)
        mgr.set_fcm_token("abc")
        assert mgr.get_fcm_token() == "abc"


def test_register_mobile_success():
    """POST /api/mobile/register with valid token returns 200 and persists token."""
    from frigate_buffer.web.server import create_app

    storage = tempfile.mkdtemp()
    try:
        prefs = PreferencesManager(storage)
        orchestrator = SimpleNamespace(
            config={
                "STORAGE_PATH": storage,
                "ALLOWED_CAMERAS": [],
                "STATS_REFRESH_SECONDS": 60,
            },
            _request_count_lock=threading.Lock(),
            _request_count=0,
            state_manager=SimpleNamespace(),
            file_manager=SimpleNamespace(),
            query_service=EventQueryService(storage),
            download_service=MagicMock(),
            consolidated_manager=MagicMock(),
            preferences_manager=prefs,
        )
        app = create_app(orchestrator)
        client = app.test_client()
        r = client.post(
            "/api/mobile/register",
            data=json.dumps({"token": "fcm_abc123"}),
            content_type="application/json",
        )
        assert r.status_code == 200
        data = r.get_json()
        assert data == {"status": "success"}
        assert prefs.get_fcm_token() == "fcm_abc123"
    finally:
        shutil.rmtree(storage, ignore_errors=True)


def test_register_mobile_missing_token_returns_400():
    """POST /api/mobile/register without token returns 400."""
    from frigate_buffer.web.server import create_app

    storage = tempfile.mkdtemp()
    try:
        prefs = PreferencesManager(storage)
        orchestrator = SimpleNamespace(
            config={
                "STORAGE_PATH": storage,
                "ALLOWED_CAMERAS": [],
                "STATS_REFRESH_SECONDS": 60,
            },
            _request_count_lock=threading.Lock(),
            _request_count=0,
            state_manager=SimpleNamespace(),
            file_manager=SimpleNamespace(),
            query_service=EventQueryService(storage),
            download_service=MagicMock(),
            consolidated_manager=MagicMock(),
            preferences_manager=prefs,
        )
        app = create_app(orchestrator)
        client = app.test_client()
        r = client.post(
            "/api/mobile/register",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert r.status_code == 400
        data = r.get_json()
        assert "error" in data
        assert "token" in data["error"].lower()
    finally:
        shutil.rmtree(storage, ignore_errors=True)


def test_register_mobile_empty_token_returns_400():
    """POST /api/mobile/register with empty token returns 400."""
    from frigate_buffer.web.server import create_app

    storage = tempfile.mkdtemp()
    try:
        prefs = PreferencesManager(storage)
        orchestrator = SimpleNamespace(
            config={
                "STORAGE_PATH": storage,
                "ALLOWED_CAMERAS": [],
                "STATS_REFRESH_SECONDS": 60,
            },
            _request_count_lock=threading.Lock(),
            _request_count=0,
            state_manager=SimpleNamespace(),
            file_manager=SimpleNamespace(),
            query_service=EventQueryService(storage),
            download_service=MagicMock(),
            consolidated_manager=MagicMock(),
            preferences_manager=prefs,
        )
        app = create_app(orchestrator)
        client = app.test_client()
        r = client.post(
            "/api/mobile/register",
            data=json.dumps({"token": "   "}),
            content_type="application/json",
        )
        assert r.status_code == 400
    finally:
        shutil.rmtree(storage, ignore_errors=True)
