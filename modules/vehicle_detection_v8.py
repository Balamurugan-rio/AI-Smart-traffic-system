"""
Vehicle Detection Module (YOLOv8 wrapper)

- Lightweight wrapper that uses `ultralytics` (YOLOv8) when available.
- Falls back to a helpful error message if the package is missing.

Usage:
    det = YOLOv8Detector(model='yolov8n.pt')
    results = det.detect_image(image)

Note: This is a wrapper skeleton. To enable full YOLOv8 capabilities, install:
    pip install ultralytics
"""

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    YOLO = None
    HAS_ULTRALYTICS = False

import numpy as np

class YOLOv8Detector:
    def __init__(self, model='yolov8n.pt', conf=0.25):
        if not HAS_ULTRALYTICS:
            raise RuntimeError('ultralytics package not found. Install with `pip install ultralytics` to use YOLOv8')
        self.model = YOLO(model)
        self.conf = conf

    def detect_image(self, image):
        """Run detection on a BGR or RGB image (numpy array).
        Returns a list of detections: [{ 'xyxy': [x1,y1,x2,y2], 'conf': , 'class': , 'name': }]
        """
        results = self.model.predict(source=image, conf=self.conf)
        out = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = self.model.names.get(cls, str(cls))
                out.append({'xyxy': xyxy, 'conf': conf, 'class': cls, 'name': name})
        return out

    def detect_file(self, path):
        return self.detect_image(path)

if __name__ == '__main__':
    print('YOLOv8 detector module loaded. Install ultralytics to use it.')
