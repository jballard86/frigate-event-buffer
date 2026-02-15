
import unittest
from unittest.mock import MagicMock, patch, ANY
import logging
import sys
import os
import requests

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frigate_buffer.managers.file import FileManager

class TestFileManagerEnhancement(unittest.TestCase):
    def setUp(self):
        self.storage_path = "/tmp/test_storage"
        self.frigate_url = "http://mock-frigate:5000"
        self.mock_video_service = MagicMock()
        self.file_manager = FileManager(self.storage_path, self.frigate_url, 7, self.mock_video_service)

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

    @patch('frigate_buffer.managers.file.requests.post')
    @patch('frigate_buffer.managers.file.requests.get')
    @patch('frigate_buffer.managers.file.FileManager.download_and_transcode_clip') # Mock fallback to avoid side effects
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
        self.file_manager.export_and_transcode_clip(
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

    @patch('frigate_buffer.managers.file.requests.get')
    def test_download_404_no_retry(self, mock_get):
        # Scenario: Download clip returns 404

        mock_response = MagicMock()
        mock_response.status_code = 404
        error = requests.exceptions.HTTPError("404 Client Error", response=mock_response)
        mock_response.raise_for_status.side_effect = error

        mock_get.return_value = mock_response

        # Call the method
        result = self.file_manager.download_and_transcode_clip("evt_missing", "/tmp")

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

    @patch('frigate_buffer.managers.file.requests.get')
    @patch('time.sleep')
    def test_download_400_retries(self, mock_sleep, mock_get):
        # Scenario: Download clip returns 400 (Not Ready) then succeeds (or fails after retries)

        # Mock 400 response
        mock_response_400 = MagicMock()
        mock_response_400.status_code = 400
        mock_response_400.raise_for_status.side_effect = requests.exceptions.HTTPError("400 Client Error", response=mock_response_400)

        # Setup mock to return 400 three times (exhaust retries)
        mock_get.return_value = mock_response_400

        result = self.file_manager.download_and_transcode_clip("evt_notready", "/tmp")

        self.assertFalse(result)
        self.assertEqual(mock_get.call_count, 3, "Should retry 3 times on 400")

if __name__ == "__main__":
    unittest.main()
