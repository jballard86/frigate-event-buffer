import unittest
from unittest.mock import MagicMock, patch, ANY
import logging
import sys
import os
import requests

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frigate_buffer.services.download import DownloadService

class TestDownloadService(unittest.TestCase):
    def setUp(self):
        self.frigate_url = "http://mock-frigate:5000"
        self.mock_video_service = MagicMock()
        self.download_service = DownloadService(self.frigate_url, self.mock_video_service)

        # Configure logging to capture output
        self.logger = logging.getLogger('frigate-buffer')
        self.logger.setLevel(logging.DEBUG)
        self.log_capture = []

        # Capture logs for assertions
        self.log_capture_handler = logging.Handler()
        self.log_capture_handler.emit = lambda record: self.log_capture.append(record)
        self.logger.addHandler(self.log_capture_handler)

    def tearDown(self):
        self.logger.removeHandler(self.log_capture_handler)

    @patch('frigate_buffer.services.download.requests.post')
    @patch('frigate_buffer.services.download.requests.get')
    @patch('frigate_buffer.services.download.DownloadService.download_and_transcode_clip') # Mock fallback to avoid side effects
    @patch('time.sleep') # Speed up polling
    def test_export_success_false_logs_warning(self, mock_sleep, mock_fallback, mock_get, mock_post):
        # Scenario: Export API returns 200 OK but body has success: false

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"success": False, "message": "Something went wrong"}
        mock_response.text = '{"success": false, "message": "Something went wrong"}'

        mock_post.return_value = mock_response

        # Mock polling to return nothing, so it eventually falls back
        mock_list_resp = MagicMock()
        mock_list_resp.json.return_value = []
        mock_get.return_value = mock_list_resp

        mock_fallback.return_value = False

        # Call the method
        self.download_service.export_and_transcode_clip(
            event_id="evt1",
            folder_path="/tmp",
            camera="cam1",
            start_time=1000,
            end_time=1010,
            export_buffer_before=0,
            export_buffer_after=0
        )

        # Check logs
        found_warning = False
        for record in self.log_capture:
            if record.levelno == logging.WARNING and 'Export failed' in record.getMessage() and 'Something went wrong' in record.getMessage():
                found_warning = True
                break

        self.assertTrue(found_warning, "Should log WARNING with raw response when success: false")

    @patch('frigate_buffer.services.download.requests.get')
    def test_download_404_no_retry(self, mock_get):
        # Scenario: Download clip returns 404

        mock_response = MagicMock()
        mock_response.status_code = 404
        # Configure context manager
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None

        error = requests.exceptions.HTTPError("404 Client Error", response=mock_response)
        mock_response.raise_for_status.side_effect = error

        mock_get.return_value = mock_response

        # Call the method
        result = self.download_service.download_and_transcode_clip("evt_missing", "/tmp")

        self.assertFalse(result)

        # Check logs
        found_warning = False
        for record in self.log_capture:
            if record.levelno == logging.WARNING and "No recording available for event evt_missing" in record.getMessage():
                found_warning = True
                break

        self.assertTrue(found_warning, "Should log specific warning for 404")

        # Verify retries
        self.assertEqual(mock_get.call_count, 1, "Should not retry on 404")

    @patch('frigate_buffer.services.download.requests.get')
    @patch('time.sleep')
    def test_download_400_retries(self, mock_sleep, mock_get):
        # Scenario: Download clip returns 400 (Not Ready) then succeeds (or fails after retries)

        # Mock 400 response
        mock_response_400 = MagicMock()
        mock_response_400.status_code = 400
        # Configure context manager
        mock_response_400.__enter__.return_value = mock_response_400
        mock_response_400.__exit__.return_value = None

        mock_response_400.raise_for_status.side_effect = requests.exceptions.HTTPError("400 Client Error", response=mock_response_400)

        # Setup mock to return 400 three times (exhaust retries)
        mock_get.return_value = mock_response_400

        result = self.download_service.download_and_transcode_clip("evt_notready", "/tmp")

        self.assertFalse(result)
        self.assertEqual(mock_get.call_count, 3, "Should retry 3 times on 400")

    # ---- Export API (Frigate 0.17+): POST with JSON body, log non-200 response.text ----

    @patch('frigate_buffer.services.download.requests.get')
    @patch('frigate_buffer.services.download.requests.post')
    @patch('frigate_buffer.services.download.DownloadService.download_and_transcode_clip')
    @patch('time.sleep')
    def test_export_post_always_sends_json_body(self, mock_sleep, mock_fallback, mock_post, mock_get):
        """POST must be called with json= dict (Frigate 0.17+ requires body to exist)."""
        mock_post.return_value = MagicMock(status_code=200, headers={"content-type": "application/json"}, text='{}')
        mock_post.return_value.json.return_value = {}
        mock_post.return_value.ok = True
        mock_post.return_value.raise_for_status = MagicMock()
        mock_get.return_value.json.return_value = []
        mock_get.return_value.raise_for_status = MagicMock()
        self.download_service.export_and_transcode_clip(
            event_id="evt1", folder_path="/tmp", camera="cam1",
            start_time=1000, end_time=1010, export_buffer_before=0, export_buffer_after=0
        )
        self.assertGreaterEqual(mock_post.call_count, 1)
        call_kw = mock_post.call_args[1]
        self.assertIn('json', call_kw, "POST must be called with json= keyword")
        self.assertIsInstance(call_kw['json'], dict, "json= must be a dict")

    @patch('frigate_buffer.services.download.requests.post')
    @patch('frigate_buffer.services.download.DownloadService.download_and_transcode_clip')
    @patch('time.sleep')
    def test_export_422_logs_response_text(self, mock_sleep, mock_fallback, mock_post):
        """Non-200 (422) must log response.text so Pydantic validation errors are visible."""
        mock_post.return_value = MagicMock(
            status_code=422,
            ok=False,
            headers={"content-type": "application/json"},
            text='{"detail":[{"loc":["body","field"],"msg":"Field required","type":"value_error.missing"}]}',
        )
        mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError("422", response=mock_post.return_value)
        mock_fallback.return_value = False
        self.download_service.export_and_transcode_clip(
            event_id="evt1", folder_path="/tmp", camera="cam1",
            start_time=1000, end_time=1010, export_buffer_before=0, export_buffer_after=0
        )
        found = any(
            ("422" in record.getMessage() or "Field required" in record.getMessage())
            and ("response.text" in record.getMessage() or "detail" in record.getMessage() or "Field required" in record.getMessage())
            for record in self.log_capture
        )
        self.assertTrue(found, "Should log 422 and response.text or detail")

    @patch('frigate_buffer.services.download.requests.post')
    @patch('frigate_buffer.services.download.DownloadService.download_and_transcode_clip')
    @patch('time.sleep')
    def test_export_405_logs_response_text(self, mock_sleep, mock_fallback, mock_post):
        """405 Method Not Allowed: log response.text and fall back without crash."""
        mock_post.return_value = MagicMock(
            status_code=405,
            ok=False,
            text='{"detail":"Method Not Allowed"}',
        )
        mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError("405", response=mock_post.return_value)
        mock_fallback.return_value = False
        self.download_service.export_and_transcode_clip(
            event_id="evt1", folder_path="/tmp", camera="cam1",
            start_time=1000, end_time=1010, export_buffer_before=0, export_buffer_after=0
        )
        found = any("Method Not Allowed" in record.getMessage() or "405" in record.getMessage() for record in self.log_capture)
        self.assertTrue(found, "Should log 405 or Method Not Allowed")
        mock_fallback.assert_called_once()

    @patch('frigate_buffer.services.download.requests.post')
    @patch('frigate_buffer.services.download.DownloadService.download_and_transcode_clip')
    @patch('time.sleep')
    def test_export_500_logs_response_text(self, mock_sleep, mock_fallback, mock_post):
        """500: log error payload and fall back."""
        mock_post.return_value = MagicMock(status_code=500, ok=False, text='{"detail":"Internal Server Error"}')
        mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError("500", response=mock_post.return_value)
        mock_fallback.return_value = False
        self.download_service.export_and_transcode_clip(
            event_id="evt1", folder_path="/tmp", camera="cam1",
            start_time=1000, end_time=1010, export_buffer_before=0, export_buffer_after=0
        )
        found = any("500" in record.getMessage() or "Internal Server Error" in record.getMessage() for record in self.log_capture)
        self.assertTrue(found, "Should log 500 or error body")

    @patch('frigate_buffer.services.download.requests.post')
    @patch('frigate_buffer.services.download.DownloadService.download_and_transcode_clip')
    @patch('time.sleep')
    def test_export_non200_empty_body_logs_status(self, mock_sleep, mock_fallback, mock_post):
        """Non-200 with empty response.text: still log status, no crash."""
        mock_post.return_value = MagicMock(status_code=503, ok=False, text="")
        mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError("503", response=mock_post.return_value)
        mock_fallback.return_value = False
        self.download_service.export_and_transcode_clip(
            event_id="evt1", folder_path="/tmp", camera="cam1",
            start_time=1000, end_time=1010, export_buffer_before=0, export_buffer_after=0
        )
        found = any("503" in record.getMessage() or "non-200" in record.getMessage() for record in self.log_capture)
        self.assertTrue(found, "Should log status when body is empty")

    @patch('frigate_buffer.services.download.requests.post')
    @patch('frigate_buffer.services.download.DownloadService.download_and_transcode_clip')
    @patch('time.sleep')
    def test_export_timeout_no_response_does_not_crash(self, mock_sleep, mock_fallback, mock_post):
        """RequestException with no response (e.g. timeout): log exception, do not access e.response.text."""
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")
        mock_fallback.return_value = False
        self.download_service.export_and_transcode_clip(
            event_id="evt1", folder_path="/tmp", camera="cam1",
            start_time=1000, end_time=1010, export_buffer_before=0, export_buffer_after=0
        )
        mock_fallback.assert_called_once()
        # No AttributeError from e.response.text
        found = any("evt1" in record.getMessage() for record in self.log_capture)
        self.assertTrue(found, "Should have logged something for event")

    @patch('frigate_buffer.services.download.requests.post')
    @patch('frigate_buffer.services.download.DownloadService.download_and_transcode_clip')
    @patch('time.sleep')
    def test_export_url_uses_integer_timestamps(self, mock_sleep, mock_fallback, mock_post):
        """URL path must use integer start/end (Frigate 0.17 Pydantic rejects float/string)."""
        mock_post.return_value = MagicMock(status_code=200, headers={"content-type": "application/json"}, text='{}')
        mock_post.return_value.json.return_value = {}
        mock_post.return_value.ok = True
        mock_post.return_value.raise_for_status = MagicMock()
        mock_fallback.return_value = False
        with patch('frigate_buffer.services.download.requests.get') as mock_get:
            mock_get.return_value.json.return_value = []
            self.download_service.export_and_transcode_clip(
                event_id="evt1", folder_path="/tmp", camera="cam1",
                start_time=1000.7, end_time=1010.9, export_buffer_before=0, export_buffer_after=0
            )
        call_url = mock_post.call_args[0][0]
        self.assertIn("/start/1000/", call_url, "start_ts must be integer in URL")
        self.assertIn("/end/1010", call_url, "end_ts must be integer in URL")

    @patch('frigate_buffer.services.download.requests.post')
    @patch('frigate_buffer.services.download.DownloadService.download_and_transcode_clip')
    @patch('time.sleep')
    def test_export_long_event_id_name_truncated(self, mock_sleep, mock_fallback, mock_post):
        """Long event_id: name in body must be <= 256 chars, request still valid JSON."""
        mock_post.return_value = MagicMock(status_code=200, headers={"content-type": "application/json"}, text='{}')
        mock_post.return_value.json.return_value = {}
        mock_post.return_value.ok = True
        mock_post.return_value.raise_for_status = MagicMock()
        mock_fallback.return_value = False
        long_id = "x" * 300
        with patch('frigate_buffer.services.download.requests.get') as mock_get:
            mock_get.return_value.json.return_value = []
            self.download_service.export_and_transcode_clip(
                event_id=long_id, folder_path="/tmp", camera="cam1",
                start_time=1000, end_time=1010, export_buffer_before=0, export_buffer_after=0
            )
        call_kw = mock_post.call_args[1]
        name = call_kw.get("json", {}).get("name", "")
        self.assertLessEqual(len(name), 256, "name must be truncated to 256 chars")

if __name__ == "__main__":
    unittest.main()
