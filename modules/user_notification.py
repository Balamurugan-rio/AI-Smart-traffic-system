"""
User Notification Module

- Provides simple interfaces to send notifications via WebSocket, MQTT, or email.
- Optional dependencies: paho-mqtt, websockets, twilio (for SMS).

Usage (websocket):
    notifier = NotificationCenter()
    notifier.broadcast({'type':'alert','message':'Heavy congestion'})

Usage (mqtt):
    notifier.send_mqtt(topic, payload)

Usage (email):
    notifier.send_email(to, subject, body)

This module provides safe stubs; integrate concrete transports in production.
"""

import json

class NotificationCenter:
    def __init__(self):
        # Placeholders for real transports
        self.websockets = []  # accept websocket connections to broadcast
        self.mqtt_client = None
        self.email_client = None

    # WebSocket broadcasting (integrate with your server's ws layer)
    def register_ws(self, ws_conn):
        self.websockets.append(ws_conn)

    def unregister_ws(self, ws_conn):
        if ws_conn in self.websockets:
            self.websockets.remove(ws_conn)

    def broadcast(self, message):
        """Broadcast message to all websocket clients. Message should be JSON-serializable."""
        data = json.dumps(message)
        for ws in list(self.websockets):
            try:
                # ws.send may be sync or async depending on implementation
                ws.send(data)
            except Exception:
                try:
                    self.websockets.remove(ws)
                except Exception:
                    pass

    # MQTT (placeholder)
    def init_mqtt(self, client):
        self.mqtt_client = client

    def send_mqtt(self, topic, payload):
        if not self.mqtt_client:
            raise RuntimeError('MQTT client not initialized')
        self.mqtt_client.publish(topic, json.dumps(payload))

    # Email (placeholder)
    def init_email(self, client):
        self.email_client = client

    def send_email(self, to_address, subject, body):
        if not self.email_client:
            raise RuntimeError('Email client not initialized')
        # Expected client implements sendmail or similar
        self.email_client.sendmail(to_address, subject, body)

if __name__ == '__main__':
    print('NotificationCenter loaded. Integrate WebSocket/MQTT/Email transports to enable notifications.')
