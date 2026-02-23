"""Tests for HA state fetch and storage-stats cache (StorageStatsAndHaHelper)."""

import unittest
from unittest.mock import MagicMock, patch

from frigate_buffer.services.ha_storage_stats import (
    DEFAULT_STORAGE_STATS_MAX_AGE_SECONDS,
    StorageStatsAndHaHelper,
    fetch_ha_state,
)


class TestFetchHaState(unittest.TestCase):
    """Tests for module-level fetch_ha_state."""

    @patch("frigate_buffer.services.ha_storage_stats.requests.get")
    def test_returns_state_when_ok(self, mock_get):
        mock_get.return_value.ok = True
        mock_get.return_value.json.return_value = {"state": "42.5"}
        result = fetch_ha_state("http://ha:8123", "token", "sensor.foo")
        self.assertEqual(result, "42.5")
        mock_get.assert_called_once()
        call_kw = mock_get.call_args[1]
        self.assertEqual(call_kw["headers"]["Authorization"], "Bearer token")
        self.assertEqual(call_kw["timeout"], 5)

    @patch("frigate_buffer.services.ha_storage_stats.requests.get")
    def test_returns_none_when_not_ok(self, mock_get):
        mock_get.return_value.ok = False
        result = fetch_ha_state("http://ha:8123", "t", "sensor.foo")
        self.assertIsNone(result)

    @patch("frigate_buffer.services.ha_storage_stats.requests.get")
    def test_returns_none_on_request_exception(self, mock_get):
        import requests
        mock_get.side_effect = requests.RequestException("network error")
        result = fetch_ha_state("http://ha:8123", "t", "sensor.foo")
        self.assertIsNone(result)

    @patch("frigate_buffer.services.ha_storage_stats.requests.get")
    def test_builds_api_states_path_when_base_does_not_end_with_api(self, mock_get):
        mock_get.return_value.ok = True
        mock_get.return_value.json.return_value = {"state": "on"}
        fetch_ha_state("http://ha:8123", "t", "light.living")
        url = mock_get.call_args[0][0]
        self.assertIn("/api/states/", url)
        self.assertTrue(url.endswith("light.living") or "light.living" in url)

    @patch("frigate_buffer.services.ha_storage_stats.requests.get")
    def test_uses_states_path_when_base_ends_with_api(self, mock_get):
        mock_get.return_value.ok = True
        mock_get.return_value.json.return_value = {"state": "off"}
        fetch_ha_state("http://ha:8123/api", "t", "switch.x")
        url = mock_get.call_args[0][0]
        self.assertIn("/states/", url)


class TestStorageStatsAndHaHelper(unittest.TestCase):
    """Tests for StorageStatsAndHaHelper cache and HA delegation."""

    def setUp(self):
        self.config = {}

    def test_get_returns_default_cache_before_update(self):
        helper = StorageStatsAndHaHelper(self.config)
        out = helper.get()
        self.assertEqual(out["total"], 0)
        self.assertEqual(out["clips"], 0)
        self.assertEqual(out["snapshots"], 0)
        self.assertEqual(out["descriptions"], 0)
        self.assertEqual(out["by_camera"], {})

    def test_update_stores_result_from_file_manager(self):
        helper = StorageStatsAndHaHelper(self.config)
        fm = MagicMock()
        fm.compute_storage_stats.return_value = {
            "clips": 100,
            "snapshots": 200,
            "descriptions": 50,
            "total": 350,
            "by_camera": {"cam1": {"total": 350, "clips": 100, "snapshots": 200, "descriptions": 50}},
        }
        helper.update(fm)
        out = helper.get()
        self.assertEqual(out["clips"], 100)
        self.assertEqual(out["total"], 350)
        self.assertIn("cam1", out["by_camera"])
        fm.compute_storage_stats.assert_called_once()

    def test_update_called_twice_refreshes_cache(self):
        """When max_age is 0, every update() calls file_manager and overwrites cache."""
        self.config["STORAGE_STATS_MAX_AGE_SECONDS"] = 0
        helper = StorageStatsAndHaHelper(self.config)
        fm = MagicMock()
        fm.compute_storage_stats.side_effect = [
            {"clips": 1, "snapshots": 0, "descriptions": 0, "total": 1, "by_camera": {}},
            {"clips": 2, "snapshots": 0, "descriptions": 0, "total": 2, "by_camera": {}},
        ]
        helper.update(fm)
        self.assertEqual(helper.get()["total"], 1)
        helper.update(fm)
        self.assertEqual(helper.get()["total"], 2)
        self.assertEqual(fm.compute_storage_stats.call_count, 2)

    def test_fetch_ha_state_delegates_to_module_function(self):
        helper = StorageStatsAndHaHelper(self.config)
        with patch("frigate_buffer.services.ha_storage_stats.fetch_ha_state", return_value="99") as mock_fetch:
            result = helper.fetch_ha_state("http://ha", "tok", "sensor.x")
            self.assertEqual(result, "99")
            mock_fetch.assert_called_once_with("http://ha", "tok", "sensor.x")

    def test_max_age_from_config(self):
        self.config["STORAGE_STATS_MAX_AGE_SECONDS"] = 600
        helper = StorageStatsAndHaHelper(self.config)
        self.assertEqual(helper._max_age_seconds, 600)

    def test_max_age_default_when_not_in_config(self):
        helper = StorageStatsAndHaHelper(self.config)
        self.assertEqual(helper._max_age_seconds, DEFAULT_STORAGE_STATS_MAX_AGE_SECONDS)

    def test_update_skips_when_cache_fresh(self):
        """Second update within max_age does not call file_manager again."""
        self.config["STORAGE_STATS_MAX_AGE_SECONDS"] = 3600
        helper = StorageStatsAndHaHelper(self.config)
        fm = MagicMock()
        fm.compute_storage_stats.return_value = {
            "clips": 10, "snapshots": 0, "descriptions": 0, "total": 10, "by_camera": {}
        }
        helper.update(fm)
        helper.update(fm)
        fm.compute_storage_stats.assert_called_once()
