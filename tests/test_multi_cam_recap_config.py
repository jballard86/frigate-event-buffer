"""Tests that the standalone multi_cam_recap script uses the main package config."""

import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

# Minimal flat config as returned by frigate_buffer.config.load_config.
# Script only needs MQTT_*, STORAGE_PATH, multi_cam flat keys, gemini_proxy flat keys, GEMINI, SAVE_AI_*, CREATE_AI_*.
_MINIMAL_FLAT_CONFIG = {
    "MQTT_BROKER": "mqtt.example.com",
    "MQTT_PORT": 1883,
    "STORAGE_PATH": "/app/storage",
    "MAX_MULTI_CAM_FRAMES_MIN": 60,
    "MAX_MULTI_CAM_FRAMES_SEC": 3,
    "MOTION_THRESHOLD_PX": 80,
    "CROP_WIDTH": 1920,
    "CROP_HEIGHT": 1080,
    "MOTION_CROP_MIN_AREA_FRACTION": 0.002,
    "MOTION_CROP_MIN_PX": 600,
    "GEMINI_PROXY_URL": "http://proxy:5050",
    "GEMINI": {"api_key": "test-key", "proxy_url": "", "model": "", "enabled": False},
    "GEMINI_PROXY_MODEL": "gemini-2.0-flash",
    "GEMINI_PROXY_TEMPERATURE": 0.5,
    "GEMINI_PROXY_TOP_P": 0.9,
    "GEMINI_PROXY_FREQUENCY_PENALTY": 0.1,
    "GEMINI_PROXY_PRESENCE_PENALTY": 0.1,
    "SAVE_AI_FRAMES": True,
    "CREATE_AI_ANALYSIS_ZIP": False,
}


def _load_multi_cam_recap_module():
    """Load scripts/multi_cam_recap.py as a module after config is patched."""
    script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "multi_cam_recap.py")
    script_path = os.path.abspath(script_path)
    spec = importlib.util.spec_from_file_location("multi_cam_recap", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["multi_cam_recap"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestMultiCamRecapConfig(unittest.TestCase):
    """Verify multi_cam_recap builds CONF from main app flat config."""

    @patch("frigate_buffer.config.load_config")
    def test_conf_uses_main_config_flat_keys(self, mock_load_config):
        """CONF is built from the same flat keys as the main app (multi_cam / gemini_proxy)."""
        sys.modules.pop("multi_cam_recap", None)
        mock_load_config.return_value = _MINIMAL_FLAT_CONFIG.copy()
        mod = _load_multi_cam_recap_module()

        self.assertEqual(mod.CONF["max_multi_cam_frames_min"], 60)
        self.assertEqual(mod.CONF["max_multi_cam_frames_sec"], 3)
        self.assertEqual(mod.CONF["motion_threshold_px"], 80)
        self.assertEqual(mod.CONF["crop_width"], 1920)
        self.assertEqual(mod.CONF["crop_height"], 1080)
        self.assertEqual(mod.CONF["motion_crop_min_area_fraction"], 0.002)
        self.assertEqual(mod.CONF["motion_crop_min_px"], 600)
        self.assertEqual(mod.CONF["gemini_proxy_url"], "http://proxy:5050")
        self.assertEqual(mod.CONF["gemini_proxy_api_key"], "test-key")
        self.assertEqual(mod.CONF["gemini_proxy_model"], "gemini-2.0-flash")
        self.assertEqual(mod.CONF["gemini_proxy_temperature"], 0.5)
        self.assertEqual(mod.CONF["gemini_proxy_top_p"], 0.9)
        self.assertEqual(mod.CONF["gemini_proxy_frequency_penalty"], 0.1)
        self.assertEqual(mod.CONF["gemini_proxy_presence_penalty"], 0.1)
        self.assertTrue(mod.CONF["save_ai_frames"])
        self.assertFalse(mod.CONF["create_ai_analysis_zip"])

        self.assertEqual(mod.MQTT_BROKER, "mqtt.example.com")
        self.assertEqual(mod.MQTT_PORT, 1883)
        self.assertEqual(mod.STORAGE_PATH, "/app/storage")

    @patch("frigate_buffer.config.load_config")
    def test_conf_defaults_when_keys_omitted(self, mock_load_config):
        """When flat config omits optional keys, script uses same defaults as main app."""
        sys.modules.pop("multi_cam_recap", None)
        minimal = {
            "MQTT_BROKER": "localhost",
            "MQTT_PORT": 1883,
            "STORAGE_PATH": "/tmp",
            "MAX_MULTI_CAM_FRAMES_MIN": 45,
            "MAX_MULTI_CAM_FRAMES_SEC": 2,
            "MOTION_THRESHOLD_PX": 50,
            "CROP_WIDTH": 1280,
            "CROP_HEIGHT": 720,
            "GEMINI_PROXY_URL": "",
            "GEMINI": {},
        }
        mock_load_config.return_value = minimal
        mod = _load_multi_cam_recap_module()

        self.assertEqual(mod.CONF["motion_crop_min_area_fraction"], 0.001)
        self.assertEqual(mod.CONF["motion_crop_min_px"], 500)
        self.assertEqual(mod.CONF["gemini_proxy_model"], "gemini-2.5-flash-lite")
        self.assertTrue(mod.CONF["save_ai_frames"])
        self.assertTrue(mod.CONF["create_ai_analysis_zip"])
