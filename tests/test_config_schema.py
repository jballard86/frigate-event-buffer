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
    def test_notifications_block_optional_legacy_default(self, mock_yaml_load, mock_exists, mock_file):
        """When notifications block is absent, NOTIFICATIONS_HOME_ASSISTANT_ENABLED is True (legacy behavior)."""
        valid_yaml = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_broker': 'localhost',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost',
                'storage_path': '/tmp',
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        self.assertTrue(config['NOTIFICATIONS_HOME_ASSISTANT_ENABLED'])

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_notifications_home_assistant_enabled_false(self, mock_yaml_load, mock_exists, mock_file):
        """When notifications.home_assistant.enabled is False, NOTIFICATIONS_HOME_ASSISTANT_ENABLED is False."""
        valid_yaml = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_broker': 'localhost',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost',
                'storage_path': '/tmp',
            },
            'notifications': {
                'home_assistant': {'enabled': False},
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        self.assertFalse(config['NOTIFICATIONS_HOME_ASSISTANT_ENABLED'])

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_notifications_home_assistant_enabled_default_true(self, mock_yaml_load, mock_exists, mock_file):
        """When notifications.home_assistant is present but enabled omitted, default is True."""
        valid_yaml = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_broker': 'localhost',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost',
                'storage_path': '/tmp',
            },
            'notifications': {
                'home_assistant': {},
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        self.assertTrue(config['NOTIFICATIONS_HOME_ASSISTANT_ENABLED'])

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_minimum_event_seconds_config(self, mock_yaml_load, mock_exists, mock_file):
        """minimum_event_seconds from settings is merged into config as MINIMUM_EVENT_SECONDS."""
        valid_yaml = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_broker': 'localhost',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost',
                'storage_path': '/tmp',
            },
            'settings': {
                'minimum_event_seconds': 10,
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        self.assertEqual(config['MINIMUM_EVENT_SECONDS'], 10)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_max_event_length_seconds_config(self, mock_yaml_load, mock_exists, mock_file):
        """max_event_length_seconds from settings is merged into config as MAX_EVENT_LENGTH_SECONDS."""
        valid_yaml = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_broker': 'localhost',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost',
                'storage_path': '/tmp',
            },
            'settings': {
                'max_event_length_seconds': 300,
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        self.assertEqual(config['MAX_EVENT_LENGTH_SECONDS'], 300)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_max_event_length_seconds_default(self, mock_yaml_load, mock_exists, mock_file):
        """MAX_EVENT_LENGTH_SECONDS defaults to 120 when max_event_length_seconds omitted."""
        valid_yaml = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_broker': 'localhost',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost',
                'storage_path': '/tmp',
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        self.assertEqual(config['MAX_EVENT_LENGTH_SECONDS'], 120)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_gemini_frames_per_hour_cap_config(self, mock_yaml_load, mock_exists, mock_file):
        """gemini_frames_per_hour_cap from settings is merged; default 200 when omitted."""
        valid_yaml = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_broker': 'localhost',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost',
                'storage_path': '/tmp',
            },
            'settings': {
                'gemini_frames_per_hour_cap': 100,
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        self.assertEqual(config['GEMINI_FRAMES_PER_HOUR_CAP'], 100)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_gemini_frames_per_hour_cap_default(self, mock_yaml_load, mock_exists, mock_file):
        """When settings omit gemini_frames_per_hour_cap, default is 200."""
        valid_yaml = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_broker': 'localhost',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost',
                'storage_path': '/tmp',
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        self.assertEqual(config['GEMINI_FRAMES_PER_HOUR_CAP'], 200)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_quick_title_config_from_settings(self, mock_yaml_load, mock_exists, mock_file):
        """quick_title_delay_seconds and quick_title_enabled from settings are merged."""
        valid_yaml = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_broker': 'localhost',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost',
                'storage_path': '/tmp',
            },
            'settings': {
                'quick_title_delay_seconds': 5,
                'quick_title_enabled': False,
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        self.assertEqual(config['QUICK_TITLE_DELAY_SECONDS'], 5)
        self.assertFalse(config['QUICK_TITLE_ENABLED'])

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

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_multi_cam_and_gemini_proxy_config(self, mock_yaml_load, mock_exists, mock_file):
        """Valid config with multi_cam and gemini_proxy passes and flattens correctly."""
        yaml_with_new = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_broker': 'localhost',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost',
                'storage_path': '/tmp',
            },
            'multi_cam': {
                'max_multi_cam_frames_min': 60,
                'max_multi_cam_frames_sec': 3,
                'crop_width': 1920,
                'crop_height': 1080,
                'multi_cam_system_prompt_file': '/path/to/prompt.txt',
                'detection_imgsz': 1280,
                'person_area_debug': True,
            },
            'gemini_proxy': {
                'url': 'http://proxy:5050',
                'model': 'gemini-2.0-flash',
                'temperature': 0.5,
                'top_p': 0.9,
                'frequency_penalty': 0.1,
                'presence_penalty': 0.1,
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = yaml_with_new

        config = load_config()
        self.assertEqual(config['MAX_MULTI_CAM_FRAMES_MIN'], 60)
        self.assertEqual(config['MAX_MULTI_CAM_FRAMES_SEC'], 3)
        self.assertEqual(config['CROP_WIDTH'], 1920)
        self.assertEqual(config['CROP_HEIGHT'], 1080)
        self.assertEqual(config['MULTI_CAM_SYSTEM_PROMPT_FILE'], '/path/to/prompt.txt')
        self.assertEqual(config['DETECTION_IMGSZ'], 1280)
        self.assertTrue(config['PERSON_AREA_DEBUG'])
        self.assertEqual(config['GEMINI_PROXY_URL'], 'http://proxy:5050')
        self.assertEqual(config['GEMINI_PROXY_MODEL'], 'gemini-2.0-flash')
        self.assertEqual(config['GEMINI_PROXY_TEMPERATURE'], 0.5)
        self.assertEqual(config['GEMINI_PROXY_TOP_P'], 0.9)
        self.assertEqual(config['GEMINI_PROXY_FREQUENCY_PENALTY'], 0.1)
        self.assertEqual(config['GEMINI_PROXY_PRESENCE_PENALTY'], 0.1)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_max_multi_cam_frames_sec_accepts_decimal(self, mock_yaml_load, mock_exists, mock_file):
        """max_multi_cam_frames_sec accepts decimal values (e.g. 0.5, 1.5) and is stored as float."""
        yaml_with_decimal = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_broker': 'localhost',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost',
                'storage_path': '/tmp',
            },
            'multi_cam': {
                'max_multi_cam_frames_min': 45,
                'max_multi_cam_frames_sec': 1.5,
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = yaml_with_decimal
        config = load_config()
        self.assertEqual(config['MAX_MULTI_CAM_FRAMES_SEC'], 1.5)
        self.assertIsInstance(config['MAX_MULTI_CAM_FRAMES_SEC'], float)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_multi_cam_gemini_proxy_defaults_when_omitted(self, mock_yaml_load, mock_exists, mock_file):
        """When multi_cam and gemini_proxy are omitted, flat keys use source defaults."""
        yaml_minimal = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_broker': 'localhost',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost',
                'storage_path': '/tmp',
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = yaml_minimal

        config = load_config()
        self.assertEqual(config['MAX_MULTI_CAM_FRAMES_MIN'], 45)
        self.assertEqual(config['MAX_MULTI_CAM_FRAMES_SEC'], 2)
        self.assertEqual(config['CROP_WIDTH'], 1280)
        self.assertEqual(config['CROP_HEIGHT'], 720)
        self.assertEqual(config['MULTI_CAM_SYSTEM_PROMPT_FILE'], '')
        self.assertFalse(config['PERSON_AREA_DEBUG'])
        self.assertEqual(config['DECODE_SECOND_CAMERA_CPU_ONLY'], False)
        self.assertEqual(config['GEMINI_PROXY_URL'], '')
        self.assertEqual(config['GEMINI_PROXY_MODEL'], 'gemini-2.5-flash-lite')
        self.assertEqual(config['GEMINI_PROXY_TEMPERATURE'], 0.3)
        self.assertEqual(config['GEMINI_PROXY_TOP_P'], 1)
        self.assertEqual(config['GEMINI_PROXY_FREQUENCY_PENALTY'], 0)
        self.assertEqual(config['GEMINI_PROXY_PRESENCE_PENALTY'], 0)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_gemini_proxy_from_gemini_when_gemini_proxy_absent(self, mock_yaml_load, mock_exists, mock_file):
        """When only gemini is set (no gemini_proxy), GEMINI_PROXY_URL and MODEL come from gemini."""
        yaml_gemini_only = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_broker': 'localhost',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost',
                'storage_path': '/tmp',
            },
            'gemini': {
                'proxy_url': 'http://gemini-only:5050',
                'api_key': '',
                'model': 'gemini-special',
                'enabled': True,
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = yaml_gemini_only

        config = load_config()
        self.assertEqual(config['GEMINI_PROXY_URL'], 'http://gemini-only:5050')
        self.assertEqual(config['GEMINI_PROXY_MODEL'], 'gemini-special')
        self.assertEqual(config['GEMINI_PROXY_TEMPERATURE'], 0.3)
        self.assertEqual(config['GEMINI_PROXY_TOP_P'], 1)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('os.getenv')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_gemini_proxy_url_env_override(self, mock_yaml_load, mock_getenv, mock_exists, mock_file):
        """GEMINI_PROXY_URL from env overrides config."""
        def getenv(key, default=None):
            if key == 'GEMINI_PROXY_URL':
                return 'http://env-proxy:6060'
            return default  # so other keys get their default and load_config still works

        mock_getenv.side_effect = getenv
        mock_exists.return_value = True
        mock_yaml_load.return_value = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_broker': 'localhost',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost',
                'storage_path': '/tmp',
            },
            'gemini_proxy': {'url': 'http://file-proxy:5050'},
        }

        config = load_config()
        self.assertEqual(config['GEMINI_PROXY_URL'], 'http://env-proxy:6060')

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_decode_second_camera_cpu_only_default_and_override(self, mock_yaml_load, mock_exists, mock_file):
        """DECODE_SECOND_CAMERA_CPU_ONLY defaults to False; multi_cam.decode_second_camera_cpu_only: true overrides to True."""
        yaml_default = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_broker': 'localhost',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost',
                'storage_path': '/tmp',
            },
            'multi_cam': {
                'max_multi_cam_frames_min': 60,
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = yaml_default

        config = load_config()
        self.assertEqual(config['DECODE_SECOND_CAMERA_CPU_ONLY'], False)

        yaml_override = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_broker': 'localhost',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost',
                'storage_path': '/tmp',
            },
            'multi_cam': {
                'max_multi_cam_frames_min': 60,
                'decode_second_camera_cpu_only': True,
            },
        }
        mock_yaml_load.return_value = yaml_override

        config = load_config()
        self.assertEqual(config['DECODE_SECOND_CAMERA_CPU_ONLY'], True)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_invalid_multi_cam_field_type(self, mock_yaml_load, mock_exists, mock_file):
        """Invalid type for multi_cam field fails schema validation."""
        invalid_yaml = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_broker': 'localhost',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost',
                'storage_path': '/tmp',
            },
            'multi_cam': {
                'max_multi_cam_frames_min': 'not_an_int',
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = invalid_yaml

        with self.assertRaises(SystemExit) as cm:
            load_config()
        self.assertEqual(cm.exception.code, 1)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_pushover_block_valid(self, mock_yaml_load, mock_exists, mock_file):
        """Valid pushover block under notifications is stored in config['pushover']; validation passes."""
        valid_yaml = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_broker': 'localhost',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost',
                'storage_path': '/tmp',
            },
            'notifications': {
                'pushover': {
                    'enabled': True,
                    'pushover_user_key': 'uk',
                    'pushover_api_token': 'tok',
                    'device': 'phone',
                    'default_sound': 'pushover',
                    'html': 1,
                },
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        self.assertIn('pushover', config)
        po = config['pushover']
        self.assertIsInstance(po, dict)
        self.assertTrue(po.get('enabled'))
        self.assertEqual(po.get('pushover_user_key'), 'uk')
        self.assertEqual(po.get('pushover_api_token'), 'tok')
        self.assertEqual(po.get('device'), 'phone')
        self.assertEqual(po.get('default_sound'), 'pushover')
        self.assertEqual(po.get('html'), 1)

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    @patch('frigate_buffer.config.yaml.safe_load')
    def test_pushover_empty_normalized_to_dict(self, mock_yaml_load, mock_exists, mock_file):
        """When notifications.pushover is present but blank (None), config['pushover'] is {} so .get() never raises."""
        valid_yaml = {
            'cameras': [{'name': 'cam1'}],
            'network': {
                'mqtt_broker': 'localhost',
                'frigate_url': 'http://frigate',
                'buffer_ip': 'localhost',
                'storage_path': '/tmp',
            },
            'notifications': {
                'pushover': None,
            },
        }
        mock_exists.return_value = True
        mock_yaml_load.return_value = valid_yaml

        config = load_config()
        self.assertIn('pushover', config)
        po = config['pushover']
        self.assertIsInstance(po, dict)
        # Must not raise AttributeError (env overrides may add pushover_user_key/pushover_api_token).
        _ = po.get('enabled')
        _ = po.get('pushover_api_token')
        _ = po.get('pushover_user_key')

if __name__ == '__main__':
    unittest.main()
