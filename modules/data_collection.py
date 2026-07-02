"""
Data Collection Module

- Provides interfaces to ingest real-time video streams and sensor data.
- Lightweight stubs that can be extended to connect to cameras (RTSP/HTTP), MQTT sensors, or files.

Usage:
    collector = DataCollector()
    collector.add_camera('rtsp://...')
    for frame, meta in collector.stream():
        # process frame

This module intentionally avoids hard external dependencies; add connectors as needed.
"""

import time
import threading
from collections import deque

class DataCollector:
    """Collects video frames and sensor events from multiple sources."""

    def __init__(self, max_buffer=100):
        self.cameras = {}  # id -> source_url
        self.sensors = {}  # id -> sensor_config
        self.frame_buffers = {}  # cam_id -> deque
        self.running = False
        self.max_buffer = max_buffer
        self.lock = threading.Lock()

    def add_camera(self, cam_id, source_url):
        """Register a camera source (RTSP, HTTP, or file path)."""
        self.cameras[cam_id] = source_url
        self.frame_buffers[cam_id] = deque(maxlen=self.max_buffer)

    def remove_camera(self, cam_id):
        self.cameras.pop(cam_id, None)
        self.frame_buffers.pop(cam_id, None)

    def add_sensor(self, sensor_id, config):
        """Register a sensor (MQTT topic, HTTP endpoint, etc.).
        `config` is a dict describing how to connect; this module leaves implementation to integrators.
        """
        self.sensors[sensor_id] = config

    def start(self):
        """Start background ingestion threads for registered sources."""
        self.running = True
        for cam_id, src in list(self.cameras.items()):
            t = threading.Thread(target=self._ingest_camera, args=(cam_id, src), daemon=True)
            t.start()

        # Sensor ingestion placeholder (implement per sensor type)

    def stop(self):
        self.running = False

    def _ingest_camera(self, cam_id, source_url):
        """Simple stub that pretends to ingest frames; replace with OpenCV `VideoCapture` or GStreamer pipeline."""
        while self.running:
            # In a real implementation, capture frames from `source_url`.
            fake_frame = None
            timestamp = time.time()
            with self.lock:
                buf = self.frame_buffers.get(cam_id)
                if buf is not None:
                    buf.append({'frame': fake_frame, 'timestamp': timestamp})
            time.sleep(0.1)

    def get_latest_frame(self, cam_id):
        """Return the most recent frame dict for a camera, or None."""
        with self.lock:
            buf = self.frame_buffers.get(cam_id)
            if not buf:
                return None
            return buf[-1]

    def stream(self, cam_id=None):
        """Generator that yields frames for a specific camera or all cameras.
        Yields (cam_id, frame_dict)
        """
        while self.running:
            with self.lock:
                if cam_id:
                    buf = self.frame_buffers.get(cam_id, [])
                    items = [(cam_id, buf.popleft())] if len(buf) else []
                else:
                    items = []
                    for cid, buf in self.frame_buffers.items():
                        if len(buf):
                            items.append((cid, buf.popleft()))
            for item in items:
                yield item
            time.sleep(0.05)


if __name__ == '__main__':
    # Simple demo
    collector = DataCollector()
    collector.add_camera('cam1', 'rtsp://example')
    collector.start()
    time.sleep(0.5)
    print('Started collector with cameras:', collector.cameras)
    collector.stop()
