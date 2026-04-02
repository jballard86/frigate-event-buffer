"""Pushover, mobile app, credentials env overrides."""

import unittest
from unittest.mock import mock_open, patch

from frigate_buffer.config import load_config


class TestConfigSchemaNotificationsProviders(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_pushover_block_valid(self, mock_yaml_load, mock_exists, mock_file):
        """Valid pushover block under notifications is stored in
        config['pushover']; validation passes."""
        valid_yaml = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "notifications": {
                "pushover": {
                    "enabled": True,
                    "pushover_user_key": "uk",
                    "pushover_api_token": "tok",
                    "device": "phone",
                    "default_sound": "pushover",
                    "html": 1,
                },
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        assert "pushover" in config
        po = config["pushover"]
        assert isinstance(po, dict)
        assert po.get("enabled")
        assert po.get("pushover_user_key") == "uk"
        assert po.get("pushover_api_token") == "tok"
        assert po.get("device") == "phone"
        assert po.get("default_sound") == "pushover"
        assert po.get("html") == 1

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_pushover_empty_normalized_to_dict(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """When notifications.pushover is present but blank (None),
        config['pushover'] is {} so .get() never raises."""
        valid_yaml = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "notifications": {
                "pushover": None,
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        assert "pushover" in config
        po = config["pushover"]
        assert isinstance(po, dict)
        # Must not raise AttributeError (env overrides may add
        # pushover_user_key/pushover_api_token).
        _ = po.get("enabled")
        _ = po.get("pushover_api_token")
        _ = po.get("pushover_user_key")

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_notifications_mobile_app_enabled_true(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """When notifications.mobile_app.enabled is True and credentials_path set,
        NOTIFICATIONS_MOBILE_APP_ENABLED is True and path is stored."""
        valid_yaml = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "notifications": {
                "mobile_app": {
                    "enabled": True,
                    "credentials_path": "/path/to/serviceAccount.json",
                },
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        assert config["NOTIFICATIONS_MOBILE_APP_ENABLED"]
        assert config["MOBILE_APP_GOOGLE_APPLICATION_CREDENTIALS"] == (
            "/path/to/serviceAccount.json"
        )

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_notifications_mobile_app_enabled_default_false(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """When notifications.mobile_app is absent or enabled omitted,
        NOTIFICATIONS_MOBILE_APP_ENABLED defaults to False."""
        valid_yaml = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        assert not config["NOTIFICATIONS_MOBILE_APP_ENABLED"]
        assert config["MOBILE_APP_GOOGLE_APPLICATION_CREDENTIALS"] == ""

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("os.getenv")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_google_application_credentials_env_override(
        self, mock_yaml_load, mock_getenv, mock_exists, mock_file
    ):
        """GOOGLE_APPLICATION_CREDENTIALS from env overrides config path."""

        def getenv(key, default=None):
            if key == "GOOGLE_APPLICATION_CREDENTIALS":
                return "/env/path/to/creds.json"
            return default

        mock_getenv.side_effect = getenv
        mock_exists.return_value = True
        mock_yaml_load.return_value = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "notifications": {
                "mobile_app": {
                    "enabled": False,
                    "credentials_path": "/config/path/creds.json",
                },
            },
        }

        config = load_config()
        assert config["MOBILE_APP_GOOGLE_APPLICATION_CREDENTIALS"] == (
            "/env/path/to/creds.json"
        )
