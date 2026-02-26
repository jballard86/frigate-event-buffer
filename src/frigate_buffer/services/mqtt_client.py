"""MQTT client wrapper: connection lifecycle, subscriptions, and message
callback registration."""

import logging
import ssl
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import paho.mqtt.client as mqtt

if TYPE_CHECKING:
    from paho.mqtt.client import ConnectFlags, DisconnectFlags
    from paho.mqtt.reasoncodes import ReasonCode

logger = logging.getLogger("frigate-buffer")


class MqttClientWrapper:
    """Wraps Paho MQTT client: setup, on_connect/on_disconnect, start/stop loop.
    Message routing is delegated via callback."""

    DEFAULT_TOPICS: list[tuple[str, int]] = [
        ("frigate/events", 0),
        ("frigate/+/tracked_object_update", 0),
        ("frigate/reviews", 0),
    ]

    def __init__(
        self,
        broker: str,
        port: int,
        client_id: str = "frigate-event-buffer",
        topics: list[tuple[str, int]] | None = None,
        on_message_callback: Callable[..., Any] | None = None,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self._broker = broker
        self._port = port
        self._topics = topics or self.DEFAULT_TOPICS
        self._on_message_callback = on_message_callback
        self.mqtt_connected = False

        # paho-mqtt 2.x: callback_api_version required; type stubs may not
        # export CallbackAPIVersion
        callback_api_version = getattr(mqtt, "CallbackAPIVersion", None)
        if callback_api_version is not None:
            self._client = mqtt.Client(
                callback_api_version.VERSION2, client_id=client_id
            )
        else:
            self._client = mqtt.Client(client_id=client_id)
        if username:
            self._client.username_pw_set(username, password)
        if self._port == 8883:
            logger.info("Configuring MQTT connection with TLS/SSL")
            self._client.tls_set(
                cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLSv1_2
            )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.reconnect_delay_set(min_delay=1, max_delay=120)

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: "ConnectFlags",
        reason_code: "ReasonCode",
        properties: Any,
    ) -> None:
        """Handle MQTT connection."""
        rc = getattr(reason_code, "value", reason_code)
        if rc == 0:
            self.mqtt_connected = True
            logger.info(f"Connected to MQTT broker {self._broker}")

            for topic, qos in self._topics:
                client.subscribe(topic, qos)
                logger.info(f"Subscribed to: {topic}")
        else:
            logger.error(f"MQTT connection failed with code: {rc}")

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: "DisconnectFlags",
        reason_code: "ReasonCode",
        properties: Any,
    ) -> None:
        """Handle MQTT disconnection."""
        self.mqtt_connected = False
        rc = getattr(reason_code, "value", reason_code)
        if rc != 0:
            logger.warning(f"Unexpected MQTT disconnect (rc={rc}), reconnecting...")
        else:
            logger.info("MQTT disconnected")

    def _on_message(
        self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage
    ) -> None:
        """Forward messages to the registered callback."""
        if self._on_message_callback:
            self._on_message_callback(client, userdata, msg)

    @property
    def client(self) -> mqtt.Client:
        """Expose the underlying Paho client for publish/subscribe usage."""
        return self._client

    def start(self) -> None:
        """Connect and start the network loop."""
        try:
            self._client.connect_async(
                self._broker,
                self._port,
                keepalive=60,
            )
            self._client.loop_start()
        except Exception as e:
            logger.error(f"Failed to start MQTT client: {e}")

    def stop(self) -> None:
        """Stop the loop and disconnect."""
        self._client.loop_stop()
        self._client.disconnect()
