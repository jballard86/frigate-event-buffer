"""MQTT client wrapper: connection lifecycle, subscriptions, and message callback registration."""

import logging
from typing import Any, Callable, List, Optional, Tuple

import paho.mqtt.client as mqtt

logger = logging.getLogger('frigate-buffer')


class MqttClientWrapper:
    """Wraps Paho MQTT client: setup, on_connect/on_disconnect, start/stop loop. Message routing is delegated via callback."""

    DEFAULT_TOPICS: List[Tuple[str, int]] = [
        ("frigate/events", 0),
        ("frigate/+/tracked_object_update", 0),
        ("frigate/reviews", 0),
    ]

    def __init__(
        self,
        broker: str,
        port: int,
        client_id: str = "frigate-event-buffer",
        topics: Optional[List[Tuple[str, int]]] = None,
        on_message_callback: Optional[Callable[..., Any]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self._broker = broker
        self._port = port
        self._topics = topics or self.DEFAULT_TOPICS
        self._on_message_callback = on_message_callback
        self.mqtt_connected = False

        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )
        if username:
            self._client.username_pw_set(username, password)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.reconnect_delay_set(min_delay=1, max_delay=120)

    def _on_connect(self, client: mqtt.Client, userdata: Any, flags: dict, reason_code: int, properties: Any) -> None:
        """Handle MQTT connection."""
        if reason_code == 0:
            self.mqtt_connected = True
            logger.info(f"Connected to MQTT broker {self._broker}")

            for topic, qos in self._topics:
                client.subscribe(topic, qos)
                logger.info(f"Subscribed to: {topic}")
        else:
            logger.error(f"MQTT connection failed with code: {reason_code}")

    def _on_disconnect(self, client: mqtt.Client, userdata: Any, flags: dict, reason_code: int, properties: Any) -> None:
        """Handle MQTT disconnection."""
        self.mqtt_connected = False
        if reason_code != 0:
            logger.warning(f"Unexpected MQTT disconnect (rc={reason_code}), reconnecting...")
        else:
            logger.info("MQTT disconnected")

    def _on_message(self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
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
