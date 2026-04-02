"""GPU vendor, device index, Intel QSV options, effective_gpu_device_index."""

import os
import unittest
from unittest.mock import mock_open, patch

import pytest

from frigate_buffer.config import effective_gpu_device_index, load_config


class TestConfigSchemaGpu(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_gpu_vendor_and_device_defaults(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """Unset multi_cam GPU keys yield nvidia, index 0, and CUDA_* mirror."""
        mock_exists.return_value = True
        mock_yaml_load.return_value = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
        }
        config = load_config()
        assert config["GPU_VENDOR"] == "nvidia"
        assert config["GPU_DEVICE_INDEX"] == 0
        assert config["CUDA_DEVICE_INDEX"] == 0

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_yaml_cuda_device_index_legacy_maps_to_gpu_device_index(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """Deprecated multi_cam.cuda_device_index sets GPU_DEVICE_INDEX and mirror."""
        mock_exists.return_value = True
        mock_yaml_load.return_value = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "multi_cam": {"cuda_device_index": 2},
        }
        config = load_config()
        assert config["GPU_DEVICE_INDEX"] == 2
        assert config["CUDA_DEVICE_INDEX"] == 2

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_yaml_gpu_device_index_wins_over_cuda_device_index(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """gpu_device_index takes precedence over cuda_device_index when both set."""
        mock_exists.return_value = True
        mock_yaml_load.return_value = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "multi_cam": {"gpu_device_index": 1, "cuda_device_index": 9},
        }
        config = load_config()
        assert config["GPU_DEVICE_INDEX"] == 1
        assert config["CUDA_DEVICE_INDEX"] == 1

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    @patch.dict(os.environ, {"CUDA_DEVICE_INDEX": "3"}, clear=False)
    def test_env_cuda_device_index_overrides_yaml(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """CUDA_DEVICE_INDEX env still works (deprecated) and overrides YAML index."""
        mock_exists.return_value = True
        mock_yaml_load.return_value = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "multi_cam": {"gpu_device_index": 0},
        }
        config = load_config()
        assert config["GPU_DEVICE_INDEX"] == 3
        assert config["CUDA_DEVICE_INDEX"] == 3

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    @patch.dict(
        os.environ,
        {"GPU_DEVICE_INDEX": "2", "CUDA_DEVICE_INDEX": "9"},
        clear=False,
    )
    def test_env_gpu_device_index_wins_over_cuda_env(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """GPU_DEVICE_INDEX env wins when both env vars are set."""
        mock_exists.return_value = True
        mock_yaml_load.return_value = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
        }
        config = load_config()
        assert config["GPU_DEVICE_INDEX"] == 2
        assert config["CUDA_DEVICE_INDEX"] == 2

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_gpu_vendor_unsupported_raises(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """Unknown GPU_VENDOR values are rejected."""
        mock_exists.return_value = True
        mock_yaml_load.return_value = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "multi_cam": {"gpu_vendor": "matrox"},
        }
        with pytest.raises(ValueError, match="not supported"):
            load_config()

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_gpu_vendor_intel_allowed(self, mock_yaml_load, mock_exists, mock_file):
        """multi_cam.gpu_vendor intel is accepted (.so required when decoding)."""
        mock_exists.return_value = True
        mock_yaml_load.return_value = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "multi_cam": {"gpu_vendor": "intel"},
        }
        config = load_config()
        assert config["GPU_VENDOR"] == "intel"

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_gpu_vendor_amd_allowed(self, mock_yaml_load, mock_exists, mock_file):
        """multi_cam.gpu_vendor amd is accepted (native .so required when decoding)."""
        mock_exists.return_value = True
        mock_yaml_load.return_value = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "multi_cam": {"gpu_vendor": "amd"},
        }
        config = load_config()
        assert config["GPU_VENDOR"] == "amd"

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    def test_multi_cam_intel_qsv_encode_options_merge(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """multi_cam.intel qsv_encode_* map to INTEL_QSV_* flat keys."""
        mock_exists.return_value = True
        mock_yaml_load.return_value = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "multi_cam": {
                "intel": {
                    "qsv_encode_preset": "slow",
                    "qsv_encode_global_quality": 22,
                },
            },
        }
        config = load_config()
        assert config["INTEL_QSV_ENCODE_PRESET"] == "slow"
        assert config["INTEL_QSV_ENCODE_GLOBAL_QUALITY"] == 22

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists")
    @patch("frigate_buffer.config.yaml.safe_load")
    @patch.dict(
        os.environ,
        {
            "INTEL_QSV_ENCODE_PRESET": "veryfast",
            "INTEL_QSV_ENCODE_GLOBAL_QUALITY": "31",
        },
        clear=False,
    )
    def test_env_intel_qsv_options_override_yaml(
        self, mock_yaml_load, mock_exists, mock_file
    ):
        """INTEL_QSV_ENCODE_* env overrides YAML after merge."""
        mock_exists.return_value = True
        mock_yaml_load.return_value = {
            "cameras": [{"name": "cam1"}],
            "network": {
                "mqtt_broker": "localhost",
                "frigate_url": "http://frigate",
                "buffer_ip": "localhost",
                "storage_path": "/tmp",
            },
            "multi_cam": {
                "intel": {
                    "qsv_encode_preset": "slow",
                    "qsv_encode_global_quality": 22,
                },
            },
        }
        config = load_config()
        assert config["INTEL_QSV_ENCODE_PRESET"] == "veryfast"
        assert config["INTEL_QSV_ENCODE_GLOBAL_QUALITY"] == 31

    def test_effective_gpu_device_index_prefers_gpu_key(self) -> None:
        assert effective_gpu_device_index({"GPU_DEVICE_INDEX": 2}) == 2
        assert effective_gpu_device_index({"CUDA_DEVICE_INDEX": 5}) == 5
        assert (
            effective_gpu_device_index({"GPU_DEVICE_INDEX": 1, "CUDA_DEVICE_INDEX": 9})
            == 1
        )
