"""multi_cam, gemini_proxy, and related validation."""

import unittest
from unittest.mock import mock_open, patch

import pytest

from frigate_buffer.config import load_config


class TestConfigSchemaMultiCam(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_multi_cam_and_gemini_proxy_config(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """Valid config with multi_cam and gemini_proxy passes and
        flattens correctly."""
        yaml_with_new = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "multi_cam": {
                "max_multi_cam_frames_min": 60,
                "max_multi_cam_frames_sec": 3,
                "crop_width": 1920,
                "crop_height": 1080,
                "multi_cam_system_prompt_file": "/path/to/prompt.txt",
                "detection_imgsz": 1280,
                "person_area_debug": True,
            },
            "gemini_proxy": {
                "url": "http://proxy:5050",
                "model": "gemini-2.0-flash",
                "temperature": 0.5,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1,
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = yaml_with_new

        config = load_config()
        assert config["MAX_MULTI_CAM_FRAMES_MIN"] == 60
        assert config["MAX_MULTI_CAM_FRAMES_SEC"] == 3
        assert config["CROP_WIDTH"] == 1920
        assert config["CROP_HEIGHT"] == 1080
        assert config["MULTI_CAM_SYSTEM_PROMPT_FILE"] == "/path/to/prompt.txt"
        assert config["DETECTION_IMGSZ"] == 1280
        assert config["PERSON_AREA_DEBUG"]
        assert config["GEMINI_PROXY_URL"] == "http://proxy:5050"
        assert config["GEMINI_PROXY_MODEL"] == "gemini-2.0-flash"
        assert config["GEMINI_PROXY_TEMPERATURE"] == 0.5
        assert config["GEMINI_PROXY_TOP_P"] == 0.9
        assert config["GEMINI_PROXY_FREQUENCY_PENALTY"] == 0.1
        assert config["GEMINI_PROXY_PRESENCE_PENALTY"] == 0.1

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_max_multi_cam_frames_sec_accepts_decimal(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """max_multi_cam_frames_sec accepts decimal values (e.g. 0.5, 1.5)
        and is stored as float."""
        yaml_with_decimal = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "multi_cam": {
                "max_multi_cam_frames_min": 45,
                "max_multi_cam_frames_sec": 1.5,
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = yaml_with_decimal
        config = load_config()
        assert config["MAX_MULTI_CAM_FRAMES_SEC"] == 1.5
        assert isinstance(config["MAX_MULTI_CAM_FRAMES_SEC"], float)

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_multi_cam_gemini_proxy_defaults_when_omitted(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """When multi_cam and gemini_proxy are omitted, flat keys use
        source defaults."""
        yaml_minimal = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = yaml_minimal

        config = load_config()
        assert config["MAX_MULTI_CAM_FRAMES_MIN"] == 45
        assert config["MAX_MULTI_CAM_FRAMES_SEC"] == 2
        assert config["CROP_WIDTH"] == 1280
        assert config["CROP_HEIGHT"] == 720
        assert config["MULTI_CAM_SYSTEM_PROMPT_FILE"] == ""
        assert not config["PERSON_AREA_DEBUG"]
        assert not config["DECODE_SECOND_CAMERA_CPU_ONLY"]
        assert config["GEMINI_PROXY_URL"] == ""
        assert config["GEMINI_PROXY_MODEL"] == "gemini-2.5-flash-lite"
        assert config["GEMINI_PROXY_TEMPERATURE"] == 0.3
        assert config["GEMINI_PROXY_TOP_P"] == 1
        assert config["GEMINI_PROXY_FREQUENCY_PENALTY"] == 0
        assert config["GEMINI_PROXY_PRESENCE_PENALTY"] == 0

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_gemini_proxy_from_gemini_when_gemini_proxy_absent(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """When only gemini is set (no gemini_proxy), GEMINI_PROXY_URL and
        MODEL come from gemini."""
        yaml_gemini_only = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "gemini": {
                "proxy_url": "http://gemini-only:5050",
                "api_key": "",
                "model": "gemini-special",
                "enabled": True,
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = yaml_gemini_only

        config = load_config()
        assert config["GEMINI_PROXY_URL"] == "http://gemini-only:5050"
        assert config["GEMINI_PROXY_MODEL"] == "gemini-special"
        assert config["GEMINI_PROXY_TEMPERATURE"] == 0.3
        assert config["GEMINI_PROXY_TOP_P"] == 1

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("os.getenv")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_gemini_proxy_url_env_override(
        self, mock_yaml_load, mock_getenv, mock_exists, mock_file
    ):
        """GEMINI_PROXY_URL from env overrides config."""

        def getenv(key, default=None):
            if key == "GEMINI_PROXY_URL":
                return "http://env-proxy:6060"
            return (
                default  # so other keys get their default and load_config still works
            )

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
            "gemini_proxy": {"url": "http://file-proxy:5050"},
        }

        config = load_config()
        assert config["GEMINI_PROXY_URL"] == "http://env-proxy:6060"

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_decode_second_camera_cpu_only_default_and_override(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """DECODE_SECOND_CAMERA_CPU_ONLY defaults to False;
        multi_cam.decode_second_camera_cpu_only: true overrides to True."""
        yaml_default = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "multi_cam": {
                "max_multi_cam_frames_min": 60,
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = yaml_default

        config = load_config()
        assert not config["DECODE_SECOND_CAMERA_CPU_ONLY"]

        yaml_override = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "multi_cam": {
                "max_multi_cam_frames_min": 60,
                "decode_second_camera_cpu_only": True,
            },
        }
        mock_yaml_load.return_value = yaml_override

        config = load_config()
        assert config["DECODE_SECOND_CAMERA_CPU_ONLY"]

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_invalid_multi_cam_field_type(self, mock_yaml_load, mock_exists, mock_file):
        """Invalid type for multi_cam field fails schema validation."""
        invalid_yaml = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "multi_cam": {
                "max_multi_cam_frames_min": "not_an_int",
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = invalid_yaml

        with pytest.raises(SystemExit) as cm:
            load_config()
        assert cm.value.code == 1
