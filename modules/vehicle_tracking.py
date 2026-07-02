"""
Enhanced Vehicle Tracking Module with Kalman Filtering & Speed Calculation

Features:
- DeepSORT integration for appearance-based tracking
- Kalman Filter for smooth trajectory prediction
- Centroid-based tracking fallback with optimal assignment
- Speed and direction calculation
- Trajectory smoothing and confidence scoring
- Re-identification (ReID) feature support

Install optional dependencies:
    pip install deep_sort_realtime scipy numpy
"""

import numpy as np
from collections import defaultdict, deque
import math

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    HAS_DEEPSORT = True
except Exception:
    DeepSort = None
    HAS_DEEPSORT = False

try:
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


class KalmanFilter:
    """Simplified Kalman Filter for 2D trajectory smoothing"""
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.state = None
        self.covariance = 1.0
        
    def predict(self):
        """Predict next state"""
        if self.state is None:
            return None
        self.covariance += self.process_variance
        return self.state
    
    def update(self, measurement):
        """Update with new measurement"""
        if self.state is None:
            self.state = measurement
            return measurement
        
        kalman_gain = self.covariance / (self.covariance + self.measurement_variance)
        self.state = self.state + kalman_gain * (measurement - self.state)
        self.covariance = (1 - kalman_gain) * self.covariance
        return self.state


class CentroidTracker:
    """Centroid-based tracker with optimal assignment (Hungarian algorithm fallback)"""
    def __init__(self, max_distance=50, max_age=30):
        self.max_distance = max_distance
        self.max_age = max_age
        self.next_id = 1
        self.tracks = {}  # track_id -> track_info
        self.frame_count = 0
        
    def _get_centroid(self, bbox):
        """Calculate centroid from bbox [x1, y1, x2, y2]"""
        x1, y1, x2, y2 = bbox[:4]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return np.array([cx, cy])
    
    def _compute_distance(self, centroid1, centroid2):
        """Euclidean distance between centroids"""
        return np.linalg.norm(centroid1 - centroid2)
    
    def update(self, detections):
        """Update tracks with new detections using optimal assignment"""
        self.frame_count += 1
        
        if not detections:
            # Age out old tracks
            self.tracks = {k: v for k, v in self.tracks.items() if self.frame_count - v['last_seen'] < self.max_age}
            return []
        
        # Get current track centroids
        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid]['centroid'] for tid in track_ids]
        detection_centroids = [self._get_centroid(det) for det in detections]
        
        # Compute distance matrix
        if track_centroids and detection_centroids:
            distances = cdist(track_centroids, detection_centroids)
        else:
            distances = np.empty((len(track_centroids), len(detection_centroids)))
        
        # Hungarian algorithm for optimal assignment
        if HAS_SCIPY and distances.size > 0:
            row_ind, col_ind = linear_sum_assignment(distances)
        else:
            # Fallback: greedy matching
            row_ind, col_ind = [], []
            used_cols = set()
            for i, row in enumerate(distances):
                if len(row) > 0:
                    j = np.argmin(row)
                    if j not in used_cols and distances[i, j] < self.max_distance:
                        row_ind.append(i)
                        col_ind.append(j)
                        used_cols.add(j)
        
        # Update matched tracks
        matched_detection_indices = set()
        for row, col in zip(row_ind, col_ind):
            if distances[row, col] < self.max_distance:
                track_id = track_ids[row]
                detection = detections[col]
                centroid = detection_centroids[col]
                
                # Calculate speed
                prev_centroid = self.tracks[track_id]['centroid']
                speed = self._compute_distance(prev_centroid, centroid)
                direction = np.arctan2(centroid[1] - prev_centroid[1], centroid[0] - prev_centroid[0])
                
                # Update track
                self.tracks[track_id].update({
                    'centroid': centroid,
                    'bbox': detection[:4],
                    'speed': speed,
                    'direction': direction,
                    'last_seen': self.frame_count,
                    'confidence': detection[4] if len(detection) > 4 else 1.0,
                    'class_id': detection[5] if len(detection) > 5 else None,
                    'age': self.tracks[track_id]['age'] + 1
                })
                matched_detection_indices.add(col)
        
        # Create new tracks for unmatched detections
        for col, detection in enumerate(detections):
            if col not in matched_detection_indices:
                track_id = self.next_id
                self.next_id += 1
                centroid = detection_centroids[col]
                self.tracks[track_id] = {
                    'track_id': track_id,
                    'centroid': centroid,
                    'bbox': detection[:4],
                    'speed': 0,
                    'direction': 0,
                    'last_seen': self.frame_count,
                    'confidence': detection[4] if len(detection) > 4 else 1.0,
                    'class_id': detection[5] if len(detection) > 5 else None,
                    'age': 1
                }
        
        # Remove aged-out tracks
        self.tracks = {k: v for k, v in self.tracks.items() if self.frame_count - v['last_seen'] < self.max_age}
        
        return list(self.tracks.values())
    
    def reset(self):
        """Reset tracker state"""
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0


class VehicleTracker:
    """Enhanced vehicle tracker with multiple algorithms and accuracy improvements"""
    
    def __init__(self, max_age=30, use_deepsort=True, use_kalman=True):
        self.max_age = max_age
        self.use_kalman = use_kalman
        self.deepsort_tracker = None
        self.centroid_tracker = CentroidTracker(max_age=max_age)
        self.kalman_filters = {}  # track_id -> KalmanFilter for each dimension
        
        if use_deepsort and HAS_DEEPSORT:
            self.deepsort_tracker = DeepSort(max_age=max_age)
        
        self.frame_count = 0
    
    def update(self, detections, frame=None):
        """
        Update tracker with detections.
        detections: list of [x1, y1, x2, y2, confidence, class_id]
        frame: optional frame for appearance-based tracking
        Returns: list of tracks with enhanced info including speed/direction
        """
        self.frame_count += 1
        
        # Try DeepSORT first if available
        if self.deepsort_tracker is not None and frame is not None:
            return self._update_deepsort(detections, frame)
        
        # Fallback to centroid-based tracking with Kalman filtering
        return self._update_centroid(detections)
    
    def _update_deepsort(self, detections, frame):
        """Update using DeepSORT"""
        try:
            objs = []
            for det in detections:
                x1, y1, x2, y2 = det[:4]
                conf = det[4] if len(det) > 4 else 1.0
                class_id = det[5] if len(det) > 5 else None
                objs.append(([x1, y1, x2 - x1, y2 - y1], conf, class_id))
            
            tracks = self.deepsort_tracker.update_tracks(objs, frame=frame)
            out = []
            for t in tracks:
                if not t.is_confirmed():
                    continue
                bbox = t.to_ltwh()
                track_info = {
                    'track_id': t.track_id,
                    'bbox': [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
                    'class_id': getattr(t, 'class_id', None),
                    'score': getattr(t, 'score', 0.9),
                    'age': t.hits,
                    'speed': 0,  # DeepSORT doesn't provide speed by default
                    'direction': 0
                }
                out.append(track_info)
            return out
        except Exception as e:
            print(f"DeepSORT error: {e}. Falling back to centroid tracking.")
            return self._update_centroid(detections)
    
    def _update_centroid(self, detections):
        """Update using enhanced centroid-based tracker with Kalman filtering"""
        tracks = self.centroid_tracker.update(detections)
        
        # Apply Kalman filtering if enabled
        if self.use_kalman:
            smoothed_tracks = []
            for track in tracks:
                track_id = track['track_id']
                bbox = track['bbox']
                
                # Initialize Kalman filters for this track if needed
                if track_id not in self.kalman_filters:
                    self.kalman_filters[track_id] = {
                        'x': KalmanFilter(0.01, 0.1),
                        'y': KalmanFilter(0.01, 0.1)
                    }
                
                # Get center coordinates
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                
                # Apply Kalman filter
                kf_x = self.kalman_filters[track_id]['x']
                kf_y = self.kalman_filters[track_id]['y']
                
                smooth_cx = kf_x.update(cx)
                smooth_cy = kf_y.update(cy)
                
                # Update bbox with smoothed center
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                track['bbox'] = [
                    smooth_cx - width / 2,
                    smooth_cy - height / 2,
                    smooth_cx + width / 2,
                    smooth_cy + height / 2
                ]
                
                smoothed_tracks.append(track)
            
            # Clean up old Kalman filters
            active_ids = {t['track_id'] for t in smoothed_tracks}
            self.kalman_filters = {k: v for k, v in self.kalman_filters.items() if k in active_ids}
            
            return smoothed_tracks
        
        return tracks
    
    def reset(self):
        """Reset tracker state"""
        self.centroid_tracker.reset()
        self.kalman_filters = {}
        if self.deepsort_tracker:
            self.deepsort_tracker = DeepSort(max_age=self.max_age)
        self.frame_count = 0


if __name__ == '__main__':
    print('Enhanced Vehicle Tracking module loaded.')
    print('Features: DeepSORT, Kalman filtering, centroid tracking, speed calculation')
    print('Install deep_sort_realtime for DeepSORT integration: pip install deep_sort_realtime')
