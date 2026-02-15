import os
import unittest
import sys
from unittest.mock import patch, mock_open

# Ensure project root is in path
sys.path.append(os.getcwd())

from frigate_buffer.config import load_config

class TestConfigSchema(unittest.TestCase):

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_valid_config(self, mock_yaml_load, mock_exists, mock_file):
        # Mock valid config
        valid_yaml = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_broker': 'localhost',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost',
                'storage_path': '/tmp'
            }
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        self.assertEqual(config['ALLOWED_CAMERAS'], ['cam1'])
        self.assertEqual(config['MQTT_BROKER'], 'localhost')

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_mqtt_auth_config(self, mock_yaml_load, mock_exists, mock_file):
        # Mock config with MQTT auth
        auth_yaml = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_broker': 'localhost',
                'mqtt_user': 'testuser',
                'mqtt_password': 'testpassword',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost'
            }
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = auth_yaml

        config = load_config()
        self.assertEqual(config['MQTT_USER'], 'testuser')
        self.assertEqual(config['MQTT_PASSWORD'], 'testpassword')

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_missing_cameras(self, mock_yaml_load, mock_exists, mock_file):
        # Mock invalid config (missing cameras)
        invalid_yaml = {
            'network': {'mqtt_broker': 'localhost'}
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = invalid_yaml

        # Expect SystemExit(1) due to schema validation failure
        with self.assertRaises(SystemExit) as cm:
            load_config()
        self.assertEqual(cm.exception.code, 1)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_invalid_camera_type(self, mock_yaml_load, mock_exists, mock_file):
        # Mock invalid config (cameras not list)
        invalid_yaml = {
            'cameras': "not a list"
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = invalid_yaml

        with self.assertRaises(SystemExit) as cm:
            load_config()
        self.assertEqual(cm.exception.code, 1)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_invalid_network_field_type(self, mock_yaml_load, mock_exists, mock_file):
        # Mock invalid config (mqtt_port as string)
        invalid_yaml = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_port': "invalid_port" # String that is not int
            }
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = invalid_yaml

        with self.assertRaises(SystemExit) as cm:
            load_config()
        self.assertEqual(cm.exception.code, 1)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_extra_field_allowed(self, mock_yaml_load, mock_exists, mock_file):
        # Mock config with extra field (should be allowed with ALLOW_EXTRA)
        valid_yaml = {
            'cameras': [{'name': 'cam1'}],
            'extra_field': 'something',
            'network': {
                'mqtt_broker': 'localhost',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost',
                'storage_path': '/tmp'
            }
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        try:
            config = load_config()
        except SystemExit:
            self.fail("load_config raised SystemExit unexpectedly with extra fields")

        self.assertEqual(config['ALLOWED_CAMERAS'], ['cam1'])

if __name__ == '__main__':
    unittest.main()
