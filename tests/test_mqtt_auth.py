import ssl
import unittest
from unittest.mock import patch

from frigate_buffer.services.mqtt_client import MqttClientWrapper


class TestMqttAuth(unittest.TestCase):
    @patch("frigate_buffer.services.mqtt_client.mqtt.Client")
    def test_mqtt_auth_credentials_set(self, MockClient):
        mock_client_instance = MockClient.return_value

        MqttClientWrapper(
            broker="localhost", port=1883, username="user", password="password"
        )

        mock_client_instance.username_pw_set.assert_called_once_with("user", "password")

    @patch("frigate_buffer.services.mqtt_client.mqtt.Client")
    def test_mqtt_auth_no_credentials(self, MockClient):
        mock_client_instance = MockClient.return_value

        MqttClientWrapper(broker="localhost", port=1883)

        mock_client_instance.username_pw_set.assert_not_called()

    @patch("frigate_buffer.services.mqtt_client.mqtt.Client")
    def test_mqtt_port_8883_configures_tls(self, MockClient):
        """When port is 8883, tls_set is called with CERT_REQUIRED and TLS 1.2."""
        mock_client_instance = MockClient.return_value

        MqttClientWrapper(
            broker="mqtt.example.com",
            port=8883,
        )

        mock_client_instance.tls_set.assert_called_once_with(
            cert_reqs=ssl.CERT_REQUIRED,
            tls_version=ssl.PROTOCOL_TLSv1_2,
        )

    @patch("frigate_buffer.services.mqtt_client.mqtt.Client")
    def test_mqtt_port_1883_does_not_call_tls_set(self, MockClient):
        """When port is not 8883, tls_set is not called."""
        mock_client_instance = MockClient.return_value

        MqttClientWrapper(
            broker="localhost",
            port=1883,
        )

        mock_client_instance.tls_set.assert_not_called()


if __name__ == "__main__":
    unittest.main()
