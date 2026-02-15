import unittest
from unittest.mock import MagicMock, patch
from frigate_buffer.services.mqtt_client import MqttClientWrapper

class TestMqttAuth(unittest.TestCase):
    @patch('paho.mqtt.client.Client')
    def test_mqtt_auth_credentials_set(self, MockClient):
        mock_client_instance = MockClient.return_value

        wrapper = MqttClientWrapper(
            broker="localhost",
            port=1883,
            username="user",
            password="password"
        )

        mock_client_instance.username_pw_set.assert_called_with("user", "password")

    @patch('paho.mqtt.client.Client')
    def test_mqtt_auth_no_credentials(self, MockClient):
        mock_client_instance = MockClient.return_value

        wrapper = MqttClientWrapper(
            broker="localhost",
            port=1883
        )

        mock_client_instance.username_pw_set.assert_not_called()

if __name__ == '__main__':
    unittest.main()
