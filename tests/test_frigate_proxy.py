"""
Tests for web frigate_proxy: proxy_snapshot and proxy_camera_latest
return correct (body, status) or Response for invalid/config/errors.
"""

import unittest
from unittest.mock import MagicMock, patch

import requests
from flask import Response

from frigate_buffer.web.frigate_proxy import proxy_camera_latest, proxy_snapshot


class TestProxySnapshot(unittest.TestCase):
    """proxy_snapshot returns 503 when Frigate URL empty, 502 on request failure."""

    def test_empty_frigate_url_returns_503(self):
        """When frigate_url is empty, returns 503 and message."""
        body, status = proxy_snapshot("", "evt123")
        self.assertEqual(status, 503)
        self.assertIn("not configured", body)

    def test_request_exception_returns_502(self):
        """When requests.get raises RequestException, returns 502 and message."""
        with patch("frigate_buffer.web.frigate_proxy.requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException("connection refused")
            result = proxy_snapshot("http://frigate:5000", "evt1")
        self.assertIsInstance(result, tuple)
        body, status = result
        self.assertEqual(status, 502)
        self.assertIn("unavailable", body)


class TestProxyCameraLatest(unittest.TestCase):
    """proxy_camera_latest validates name and allowed_cameras, returns 503/502 on errors."""

    def test_invalid_camera_name_returns_400(self):
        """Camera name with invalid characters returns 400."""
        result = proxy_camera_latest("http://frigate:5000", "cam/../x", [])
        self.assertIsInstance(result, tuple)
        body, status = result
        self.assertEqual(status, 400)
        self.assertIn("Invalid", body)

    def test_empty_camera_name_returns_400(self):
        """Empty camera name returns 400."""
        result = proxy_camera_latest("http://frigate:5000", "", [])
        self.assertIsInstance(result, tuple)
        body, status = result
        self.assertEqual(status, 400)

    def test_camera_not_in_allowed_returns_404(self):
        """When allowed_cameras is non-empty and camera not in list, returns 404."""
        result = proxy_camera_latest("http://frigate:5000", "front_door", ["back_door"])
        self.assertIsInstance(result, tuple)
        body, status = result
        self.assertEqual(status, 404)
        self.assertIn("not configured", body)

    def test_empty_frigate_url_returns_503(self):
        """When frigate_url is empty, returns 503."""
        result = proxy_camera_latest("", "front_door", [])
        self.assertIsInstance(result, tuple)
        body, status = result
        self.assertEqual(status, 503)

    def test_request_exception_returns_502(self):
        """When requests.get raises RequestException, returns 502."""
        with patch("frigate_buffer.web.frigate_proxy.requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException("timeout")
            result = proxy_camera_latest("http://frigate:5000", "front_door", [])
        self.assertIsInstance(result, tuple)
        body, status = result
        self.assertEqual(status, 502)

    def test_success_returns_response(self):
        """When Frigate returns 200, returns a Flask Response (streaming)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "image/jpeg"}
        mock_resp.iter_content = MagicMock(return_value=iter((b"\xff\xd8", b"\xff\xd9")))
        mock_resp.raise_for_status = MagicMock()
        with patch("frigate_buffer.web.frigate_proxy.requests.get", return_value=mock_resp):
            result = proxy_camera_latest("http://frigate:5000", "front_door", [])
        self.assertIsInstance(result, Response)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.content_type, "image/jpeg")
        list(result.response)
        mock_resp.iter_content.assert_called_once()
