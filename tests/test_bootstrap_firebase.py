"""Tests for bootstrap() Firebase initialization and error handling."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from frigate_buffer.main import bootstrap


def _minimal_config(
    mobile_app_enabled: bool = False, credentials_path: str = ""
) -> dict:
    """Minimal config dict that passes orchestrator and load_config requirements."""
    return {
        "MQTT_BROKER": "localhost",
        "MQTT_PORT": 1883,
        "FRIGATE_URL": "http://frigate",
        "BUFFER_IP": "localhost",
        "STORAGE_PATH": "/tmp/test_storage",
        "LOG_LEVEL": "INFO",
        "NOTIFICATIONS_MOBILE_APP_ENABLED": mobile_app_enabled,
        "MOBILE_APP_GOOGLE_APPLICATION_CREDENTIALS": credentials_path,
    }


class TestBootstrapFirebase(unittest.TestCase):
    """Firebase init only when mobile_app enabled; failures disable the provider."""

    @patch("frigate_buffer.main.StateAwareOrchestrator")
    @patch("frigate_buffer.main.ensure_detection_model_ready")
    @patch("frigate_buffer.main.log_gpu_status")
    @patch("frigate_buffer.main.setup_logging")
    @patch("frigate_buffer.main.load_config")
    def test_firebase_init_not_called_when_mobile_app_disabled(
        self,
        mock_load_config,
        mock_setup_logging,
        mock_log_gpu,
        mock_ensure_model,
        mock_orch,
    ):
        """When NOTIFICATIONS_MOBILE_APP_ENABLED is False, Firebase not initialized."""
        mock_load_config.return_value = _minimal_config(mobile_app_enabled=False)
        mock_orch.return_value = MagicMock()

        with patch.dict(os.environ, {"FRIGATE_BUFFER_SINGLE_WORKER": "1"}, clear=False):
            bootstrap()

        mock_load_config.assert_called_once()
        mock_orch.assert_called_once()
        config_passed = mock_orch.call_args[0][0]
        assert not config_passed["NOTIFICATIONS_MOBILE_APP_ENABLED"]

    @patch("frigate_buffer.main.StateAwareOrchestrator")
    @patch("frigate_buffer.main.ensure_detection_model_ready")
    @patch("frigate_buffer.main.log_gpu_status")
    @patch("frigate_buffer.main.setup_logging")
    @patch("frigate_buffer.main.load_config")
    def test_firebase_init_failure_disables_mobile_app(
        self,
        mock_load_config,
        mock_setup_logging,
        mock_log_gpu,
        mock_ensure_model,
        mock_orch,
    ):
        """When mobile_app enabled, Firebase init raises; config disabled and
        bootstrap succeeds.
        """
        mock_load_config.return_value = _minimal_config(
            mobile_app_enabled=True, credentials_path="/nonexistent.json"
        )
        mock_orch.return_value = MagicMock()

        mock_fa = MagicMock()
        mock_fa.initialize_app.side_effect = ValueError("Credentials file not found")

        with patch.dict(os.environ, {"FRIGATE_BUFFER_SINGLE_WORKER": "1"}, clear=False):
            with patch.dict(sys.modules, {"firebase_admin": mock_fa}):
                bootstrap()

        config_passed = mock_orch.call_args[0][0]
        assert not config_passed["NOTIFICATIONS_MOBILE_APP_ENABLED"]
        mock_fa.initialize_app.assert_called_once()

    @patch("frigate_buffer.main.StateAwareOrchestrator")
    @patch("frigate_buffer.main.ensure_detection_model_ready")
    @patch("frigate_buffer.main.log_gpu_status")
    @patch("frigate_buffer.main.setup_logging")
    @patch("frigate_buffer.main.load_config")
    def test_firebase_init_sets_env_from_config_path_when_env_unset(
        self,
        mock_load_config,
        mock_setup_logging,
        mock_log_gpu,
        mock_ensure_model,
        mock_orch,
    ):
        """mobile_app enabled, credentials_path set, env unset; env set before init."""
        mock_load_config.return_value = _minimal_config(
            mobile_app_enabled=True, credentials_path="/config/creds.json"
        )
        mock_orch.return_value = MagicMock()

        mock_fa = MagicMock()
        mock_fa.initialize_app.side_effect = ValueError("bad")

        with patch.dict(os.environ, {"FRIGATE_BUFFER_SINGLE_WORKER": "1"}, clear=False):
            prev = os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            try:
                with patch.dict(sys.modules, {"firebase_admin": mock_fa}):
                    bootstrap()
                assert os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") == (
                    "/config/creds.json"
                )
            finally:
                if prev is not None:
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = prev
                elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
                    del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
