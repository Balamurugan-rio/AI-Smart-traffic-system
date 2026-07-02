"""
Vehicle Detection Module
Uses YOLOv7 for detecting vehicles in images and videos
"""

import cv2
import numpy as np
from config_simple import Config
import os


class VehicleDetector:
    """Detects vehicles in images and videos using YOLO"""
    
    VEHICLE_CLASSES = {
        'car', 'motorcycle', 'bus', 'truck', 'bicycle',
        'bike', 'auto'  # Auto-rickshaw
    }
    
    def __init__(self):
        """Initialize YOLO detector"""
        self.net = None
        self.layer_names = None
        self.classes = []
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD
        self.nms_threshold = Config.NMS_THRESHOLD
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model and classes"""
        try:
            # Load YOLO model
            cfg_path = Config.YOLO_CONFIG
            weights_path = Config.YOLO_WEIGHTS
            
            if not os.path.exists(cfg_path) or not os.path.exists(weights_path):
                raise FileNotFoundError(
                    f"YOLO model files not found. "
                    f"Ensure {cfg_path} and {weights_path} exist."
                )
            
            self.net = cv2.dnn.readNet(weights_path, cfg_path)
            self.layer_names = self.net.getLayerNames()
            unconnected = self.net.getUnconnectedOutLayers()
            if hasattr(unconnected, 'flatten'):
                unconnected = unconnected.flatten()
            self.output_layers = [
                self.layer_names[i - 1]
                for i in unconnected
            ]
            
            # Load class names
            names_path = Config.YOLO_NAMES
            if not os.path.exists(names_path):
                raise FileNotFoundError(f"Class names file not found: {names_path}")
            
            with open(names_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            print(f"✓ YOLO model loaded successfully")
            print(f"✓ Loaded {len(self.classes)} classes")
            
        except Exception as e:
            print(f"✗ Error loading YOLO model: {e}")
            raise
    
    def detect_vehicles(self, image_path):
        """
        Detect vehicles in an image
        Returns: (image_with_boxes, detections)
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        height, width, _ = image.shape
        
        # Preprocess image for YOLO
        blob = cv2.dnn.blobFromImage(
            image, 
            0.00392, 
            (416, 416), 
            (0, 0, 0), 
            True, 
            crop=False
        )
        
        # Run inference
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        detections = []
        
        for layer_output in outs:
            for detection in layer_output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Get box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = center_x - w // 2
                    y = center_y - h // 2
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(
            boxes, 
            confidences, 
            self.confidence_threshold,
            self.nms_threshold
        )
        
        # Draw boxes and collect vehicle data
        vehicle_count = {
            'car': 0,
            'bike': 0,
            'truck': 0,
            'bus': 0,
            'auto': 0,
            'other': 0
        }
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                confidence = confidences[i]
                class_id = class_ids[i]
                class_name = self.classes[class_id]
                
                # Categorize vehicle
                vehicle_type = self._categorize_vehicle(class_name)
                vehicle_count[vehicle_type] += 1
                
                # Draw bounding box
                color = (0, 255, 0)  # Green
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                
                # Add label
                label = f"{vehicle_type}: {confidence:.2f}"
                cv2.putText(
                    image, 
                    label, 
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    color, 
                    2
                )
                
                detections.append({
                    'type': vehicle_type,
                    'confidence': confidence,
                    'bbox': [x, y, w, h]
                })
        
        return image, detections, vehicle_count
    
    def detect_vehicles_video(self, video_path, output_path=None, skip_frames=2):
        """
        Detect vehicles in a video
        Returns: (video_path, detections, vehicle_count)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            try:
                cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            except Exception:
                cap = None
        if cap is None or not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or width <= 0 or height <= 0:
            cap.release()
            raise ValueError(
                f"Could not read video properties: fps={fps}, width={width}, height={height}"
            )
        
        out = None
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                cap.release()
                raise ValueError(f"Could not create VideoWriter for: {output_path}")
        
        frame_count = 0
        vehicle_count_total = {
            'car': 0,
            'bike': 0,
            'truck': 0,
            'bus': 0,
            'auto': 0,
            'other': 0
        }
        all_detections = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame is None:
                    continue

                # Process every skip_frames-th frame
                if frame_count % skip_frames == 0:
                    # Preprocess frame for YOLO
                    blob = cv2.dnn.blobFromImage(
                        frame, 
                        0.00392, 
                        (416, 416), 
                        (0, 0, 0), 
                        True, 
                        crop=False
                    )
                    
                    self.net.setInput(blob)
                    outs = self.net.forward(self.output_layers)
                    
                    # Process detections
                    boxes = []
                    confidences = []
                    class_ids = []
                    
                    for layer_output in outs:
                        for detection in layer_output:
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
                            
                            if confidence > self.confidence_threshold:
                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                w = int(detection[2] * width)
                                h = int(detection[3] * height)
                                
                                x = center_x - w // 2
                                y = center_y - h // 2
                                
                                boxes.append([x, y, w, h])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)
                    
                    # Apply NMS
                    indices = cv2.dnn.NMSBoxes(
                        boxes, 
                        confidences, 
                        self.confidence_threshold,
                        self.nms_threshold
                    )
                    
                    # Draw boxes and count vehicles
                    if len(indices) > 0:
                        for i in indices.flatten():
                            x, y, w, h = boxes[i]
                            class_id = class_ids[i]
                            class_name = self.classes[class_id]
                            vehicle_type = self._categorize_vehicle(class_name)
                            
                            vehicle_count_total[vehicle_type] += 1
                            
                            # Draw box
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(
                                frame, 
                                vehicle_type, 
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (0, 255, 0), 
                                2
                            )
                            
                            # Store detection details
                            all_detections.append({
                                'frame': frame_count,
                                'type': vehicle_type,
                                'confidence': confidences[i],
                                'bbox': [x, y, w, h]
                            })

                # Write frame
                if output_path:
                    out.write(frame)

                frame_count += 1
        
        finally:
            cap.release()
            if out is not None:
                out.release()
        
        return output_path, all_detections, vehicle_count_total
    
    def _categorize_vehicle(self, class_name):
        """Categorize detected object into vehicle types"""
        class_name_lower = class_name.lower()
        
        if 'car' in class_name_lower or 'automobile' in class_name_lower:
            return 'car'
        elif 'motorcycle' in class_name_lower or 'bike' in class_name_lower or 'motorbike' in class_name_lower:
            return 'bike'
        elif 'truck' in class_name_lower:
            return 'truck'
        elif 'bus' in class_name_lower:
            return 'bus'
        elif 'bicycle' in class_name_lower:
            return 'bike'
        else:
            return 'other'
