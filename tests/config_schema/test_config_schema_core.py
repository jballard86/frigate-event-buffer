"""Core config schema: merge, notifications, settings, validation errors."""

import os
import unittest
from unittest.mock import mock_open, patch

import pytest

from frigate_buffer.config import load_config


class TestConfigSchemaCore(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_valid_config(self, mock_yaml_load, mock_exists, mock_file):
        # Mock valid config
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
        assert config["ALLOWED_CAMERAS"] == ["cam1"]
        assert config["MQTT_BROKER"] == "localhost"

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_notifications_block_optional_default_false(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """When notifications block is absent, NOTIFICATIONS_HOME_ASSISTANT_ENABLED
        is False (opt-in default)."""
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
        assert not config["NOTIFICATIONS_HOME_ASSISTANT_ENABLED"]

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_notifications_home_assistant_enabled_false(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """When notifications.home_assistant.enabled is False,
        NOTIFICATIONS_HOME_ASSISTANT_ENABLED is False."""
        valid_yaml = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "notifications": {
                "home_assistant": {"enabled": False},
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        assert not config["NOTIFICATIONS_HOME_ASSISTANT_ENABLED"]

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_notifications_home_assistant_enabled_default_false(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """When notifications.home_assistant is present but enabled omitted,
        default is False."""
        valid_yaml = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "notifications": {
                "home_assistant": {},
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        assert not config["NOTIFICATIONS_HOME_ASSISTANT_ENABLED"]

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_notifications_home_assistant_enabled_string_false(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """When notifications.home_assistant.enabled is the string 'false',
        NOTIFICATIONS_HOME_ASSISTANT_ENABLED becomes False."""
        valid_yaml = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "notifications": {
                "home_assistant": {"enabled": "false"},
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        assert not config["NOTIFICATIONS_HOME_ASSISTANT_ENABLED"]

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    @patch.dict(
        os.environ,
        {"NOTIFICATIONS_HOME_ASSISTANT_ENABLED": "false"},
        clear=False,
    )
    def test_notifications_home_assistant_env_override_false(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """When NOTIFICATIONS_HOME_ASSISTANT_ENABLED env is 'false', HA notifications
        are disabled even if config file has enabled true or omits it."""
        valid_yaml = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "notifications": {
                "home_assistant": {"enabled": True},
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        assert not config["NOTIFICATIONS_HOME_ASSISTANT_ENABLED"]

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_minimum_event_seconds_config(self, mock_yaml_load, mock_exists, mock_file):
        """minimum_event_seconds from settings is merged into config as
        MINIMUM_EVENT_SECONDS."""
        valid_yaml = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "settings": {
                "minimum_event_seconds": 10,
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        assert config["MINIMUM_EVENT_SECONDS"] == 10

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_max_event_length_seconds_config(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """max_event_length_seconds from settings is merged into config as
        MAX_EVENT_LENGTH_SECONDS."""
        valid_yaml = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "settings": {
                "max_event_length_seconds": 300,
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        assert config["MAX_EVENT_LENGTH_SECONDS"] == 300

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_max_event_length_seconds_default(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """MAX_EVENT_LENGTH_SECONDS defaults to 120 when
        max_event_length_seconds omitted."""
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
        assert config["MAX_EVENT_LENGTH_SECONDS"] == 120

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_gemini_frames_per_hour_cap_config(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """gemini_frames_per_hour_cap from settings is merged;
        default 200 when omitted."""
        valid_yaml = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "settings": {
                "gemini_frames_per_hour_cap": 100,
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        assert config["GEMINI_FRAMES_PER_HOUR_CAP"] == 100

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_gemini_frames_per_hour_cap_default(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """When settings omit gemini_frames_per_hour_cap, default is 200."""
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
        assert config["GEMINI_FRAMES_PER_HOUR_CAP"] == 200

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_quick_title_config_from_settings(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """quick_title_delay_seconds and quick_title_enabled from settings
        are merged."""
        valid_yaml = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "settings": {
                "quick_title_delay_seconds": 5,
                "quick_title_enabled": False,
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        assert config["QUICK_TITLE_DELAY_SECONDS"] == 5
        assert not config["QUICK_TITLE_ENABLED"]

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_mqtt_auth_config(self, mock_yaml_load, mock_exists, mock_file):
        # Mock config with MQTT auth
        auth_yaml = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "mqtt_user": "testuser",
                "mqtt_password": "testpassword",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = auth_yaml

        config = load_config()
        assert config["MQTT_USER"] == "testuser"
        assert config["MQTT_PASSWORD"] == "testpassword"

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_missing_cameras(self, mock_yaml_load, mock_exists, mock_file):
        # Mock invalid config (missing cameras)
        invalid_yaml = {"network": {"mqtt_broker": "localhost"}}
        mock_exists.return_value = True
        mock_yaml_load.return_value = invalid_yaml

        # Expect SystemExit(1) due to schema validation failure
        with pytest.raises(SystemExit) as cm:
            load_config()
        assert cm.value.code == 1

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_invalid_camera_type(self, mock_yaml_load, mock_exists, mock_file):
        # Mock invalid config (cameras not list)
        invalid_yaml = {"cameras": "not a list"}
        mock_exists.return_value = True
        mock_yaml_load.return_value = invalid_yaml

        with pytest.raises(SystemExit) as cm:
            load_config()
        assert cm.value.code == 1

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_invalid_network_field_type(self, mock_yaml_load, mock_exists, mock_file):
        # Mock invalid config (mqtt_port as string)
        invalid_yaml = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_port": "invalid_port"  # String that is not int
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = invalid_yaml

        with pytest.raises(SystemExit) as cm:
            load_config()
        assert cm.value.code == 1

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_extra_field_allowed(self, mock_yaml_load, mock_exists, mock_file):
        # Mock config with extra field (should be allowed with ALLOW_EXTRA)
        valid_yaml = {
            "cameras": [{"name": "cam1"}],
            "extra_field": "something",
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        try:
            config = load_config()
        except SystemExit:
            self.fail("load_config raised SystemExit unexpectedly with extra fields")

        assert config["ALLOWED_CAMERAS"] == ["cam1"]
