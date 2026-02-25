"""Tests for PushoverProvider: phase filter, priority, overflow, emergency params."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from frigate_buffer.models import EventState, EventPhase
from frigate_buffer.services.notifications.providers.pushover import (
    PushoverProvider,
    PUSHOVER_API_URL,
    OVERFLOW_MESSAGE,
    OVERFLOW_TITLE,
)


def _make_event(
    event_id: str = "evt1",
    camera: str = "cam1",
    created_at: float = 100.0,
    threat_level: int = 0,
    folder_path: str | None = None,
) -> EventState:
    """Minimal EventState for Pushover tests."""
    return EventState(
        event_id=event_id,
        camera=camera,
        label="person",
        created_at=created_at,
        phase=EventPhase.NEW,
        threat_level=threat_level,
        folder_path=folder_path,
    )


def _ce_like_event(event_id: str = "ce_100_abc", folder_name: str = "100_abc"):
    """Event-like object with folder_name (ConsolidatedEvent style). Uses a simple mock."""
    m = MagicMock()
    m.event_id = event_id
    m.camera = "events"
    m.label = "person"
    m.created_at = 100.0
    m.threat_level = 0
    m.folder_path = f"/storage/events/{folder_name}"
    m.folder_name = folder_name
    m.genai_title = None
    m.genai_description = None
    m.ai_description = None
    m.clip_downloaded = False
    return m


class TestPushoverPhaseFilter(unittest.TestCase):
    """Phase filter: only snapshot_ready, clip_ready, finalized are sent; others skipped."""

    def setUp(self):
        self.config = {"pushover_api_token": "tok", "pushover_user_key": "uk"}
        self.provider = PushoverProvider(self.config, "http://buf:5055")

    @patch("frigate_buffer.services.notifications.providers.pushover.requests.post")
    def test_new_status_skipped(self, mock_post):
        result = self.provider.send(_make_event(), "new")
        self.assertIsNotNone(result)
        self.assertEqual(result["provider"], "PUSHOVER")
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result.get("message"), "Filtered intermediate phase")
        mock_post.assert_not_called()

    @patch("frigate_buffer.services.notifications.providers.pushover.requests.post")
    def test_described_status_skipped(self, mock_post):
        result = self.provider.send(_make_event(), "described")
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result.get("message"), "Filtered intermediate phase")
        mock_post.assert_not_called()

    @patch("frigate_buffer.services.notifications.providers.pushover.requests.post")
    def test_summarized_status_skipped(self, mock_post):
        result = self.provider.send(_make_event(), "summarized")
        self.assertEqual(result["status"], "skipped")
        mock_post.assert_not_called()


class TestPushoverSnapshotReady(unittest.TestCase):
    """snapshot_ready: priority 0 or 1 by threat_level; optional latest.jpg attachment."""

    def setUp(self):
        self.config = {"pushover_api_token": "tok", "pushover_user_key": "uk"}
        self.provider = PushoverProvider(self.config, "http://buf:5055")

    @patch("frigate_buffer.services.notifications.providers.pushover.requests.post")
    def test_snapshot_ready_priority_normal(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"status": 1})
        event = _make_event(threat_level=0)
        result = self.provider.send(event, "snapshot_ready")
        self.assertEqual(result["provider"], "PUSHOVER")
        self.assertEqual(result["status"], "success")
        call_kw = mock_post.call_args[1]
        self.assertEqual(call_kw["data"]["priority"], 0)

    @patch("frigate_buffer.services.notifications.providers.pushover.requests.post")
    def test_snapshot_ready_priority_high_when_threat_critical(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"status": 1})
        event = _make_event(threat_level=2)
        result = self.provider.send(event, "snapshot_ready")
        self.assertEqual(result["status"], "success")
        call_kw = mock_post.call_args[1]
        self.assertEqual(call_kw["data"]["priority"], 1)

    @patch("frigate_buffer.services.notifications.providers.pushover.requests.post")
    def test_snapshot_ready_attaches_latest_jpg_when_present(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"status": 1})
        with tempfile.TemporaryDirectory() as tmp:
            latest = os.path.join(tmp, "latest.jpg")
            with open(latest, "wb") as f:
                f.write(b"\xff\xd8\xff")
            event = _make_event(folder_path=tmp)
            result = self.provider.send(event, "snapshot_ready")
        self.assertEqual(result["status"], "success")
        call_kw = mock_post.call_args[1]
        self.assertIn("files", call_kw)
        self.assertIn("attachment", call_kw["files"])


class TestPushoverClipReadyFinalized(unittest.TestCase):
    """clip_ready and finalized: priority -1; optional notification.gif."""

    def setUp(self):
        self.config = {"pushover_api_token": "tok", "pushover_user_key": "uk"}
        self.provider = PushoverProvider(self.config, "http://buf:5055")

    @patch("frigate_buffer.services.notifications.providers.pushover.requests.post")
    def test_clip_ready_priority_low(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"status": 1})
        result = self.provider.send(_make_event(), "clip_ready")
        self.assertEqual(result["status"], "success")
        call_kw = mock_post.call_args[1]
        self.assertEqual(call_kw["data"]["priority"], -1)

    @patch("frigate_buffer.services.notifications.providers.pushover.requests.post")
    def test_finalized_priority_low(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"status": 1})
        result = self.provider.send(_make_event(), "finalized")
        self.assertEqual(result["status"], "success")
        call_kw = mock_post.call_args[1]
        self.assertEqual(call_kw["data"]["priority"], -1)

    @patch("frigate_buffer.services.notifications.providers.pushover.requests.post")
    def test_finalized_attaches_notification_gif_when_present(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"status": 1})
        with tempfile.TemporaryDirectory() as tmp:
            gif_path = os.path.join(tmp, "notification.gif")
            with open(gif_path, "wb") as f:
                f.write(b"GIF89a\x00\x00")
            event = _make_event(folder_path=tmp)
            result = self.provider.send(event, "finalized")
        self.assertEqual(result["status"], "success")
        call_kw = mock_post.call_args[1]
        self.assertIn("files", call_kw)
        self.assertIn("attachment", call_kw["files"])


class TestPushoverEmergencyParams(unittest.TestCase):
    """Emergency constants and priority 2 payload (retry/expire)."""

    def test_emergency_constants_defined(self):
        from frigate_buffer.services.notifications.providers import pushover as po_module
        self.assertEqual(po_module.EMERGENCY_RETRY, 30)
        self.assertEqual(po_module.EMERGENCY_EXPIRE, 3600)


class TestPushoverSendOverflow(unittest.TestCase):
    """send_overflow: priority 0, fixed message and title."""

    def setUp(self):
        self.config = {"pushover_api_token": "tok", "pushover_user_key": "uk"}
        self.provider = PushoverProvider(self.config, "http://buf:5055")

    @patch("frigate_buffer.services.notifications.providers.pushover.requests.post")
    def test_send_overflow_sends_priority_zero(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"status": 1})
        result = self.provider.send_overflow()
        self.assertIsNotNone(result)
        self.assertEqual(result["provider"], "PUSHOVER")
        self.assertEqual(result["status"], "success")
        call_kw = mock_post.call_args[1]
        self.assertEqual(call_kw["data"]["priority"], 0)
        self.assertEqual(call_kw["data"]["message"], OVERFLOW_MESSAGE)
        self.assertEqual(call_kw["data"]["title"], OVERFLOW_TITLE)
        self.assertEqual(mock_post.call_args[0][0], PUSHOVER_API_URL)


class TestPushoverPlayerUrl(unittest.TestCase):
    """Player URL built from event properties only (no folder_path parsing)."""

    def setUp(self):
        self.config = {"pushover_api_token": "tok", "pushover_user_key": "uk"}
        self.provider = PushoverProvider(self.config, "http://buf:5055")

    @patch("frigate_buffer.services.notifications.providers.pushover.requests.post")
    def test_player_url_uses_folder_name_for_ce_style(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"status": 1})
        event = _ce_like_event(event_id="ce_100_abc", folder_name="100_abc")
        self.provider.send(event, "finalized")
        call_kw = mock_post.call_args[1]
        url = call_kw["data"]["url"]
        self.assertIn("camera=events", url)
        self.assertIn("subdir=100_abc", url)

    @patch("frigate_buffer.services.notifications.providers.pushover.requests.post")
    def test_player_url_uses_camera_and_event_id_when_no_folder_name(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"status": 1})
        event = _make_event(event_id="evt1", camera="front_door")
        self.provider.send(event, "snapshot_ready")
        call_kw = mock_post.call_args[1]
        url = call_kw["data"]["url"]
        self.assertIn("camera=front_door", url)
        self.assertIn("subdir=evt1", url)


class TestPushoverApiFailure(unittest.TestCase):
    """API errors return failure result."""

    def setUp(self):
        self.config = {"pushover_api_token": "tok", "pushover_user_key": "uk"}
        self.provider = PushoverProvider(self.config, "http://buf:5055")

    @patch("frigate_buffer.services.notifications.providers.pushover.requests.post")
    def test_api_4xx_returns_failure(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=400,
            text="Bad Request",
            json=lambda: {"status": 0, "errors": ["user invalid"]},
            headers={"content-type": "application/json"},
        )
        result = self.provider.send(_make_event(), "snapshot_ready")
        self.assertEqual(result["provider"], "PUSHOVER")
        self.assertEqual(result["status"], "failure")
        self.assertIn("message", result)


if __name__ == "__main__":
    unittest.main()
