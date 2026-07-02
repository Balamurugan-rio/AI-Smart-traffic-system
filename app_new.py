"""
AI Traffic Management System
Simplified Flask Application
"""

from flask import Flask, render_template, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from config_simple import config, Config
from utils import VehicleDetector, TrafficAnalyzer, ReportGenerator
import os
import cv2
import base64
import sqlite3
from datetime import datetime
import json
import numpy as np

# Optional advanced modules
try:
    from modules.vehicle_tracking import VehicleTracker
    HAS_TRACKING = True
except Exception as e:
    HAS_TRACKING = False
    print(f"⚠ Vehicle Tracking not available: {e}")

try:
    from modules.vehicle_detection_v8 import YOLOv8Detector
    HAS_YOLOV8 = True
except Exception as e:
    HAS_YOLOV8 = False
    print(f"⚠ YOLOv8 Detector not available: {e}")

try:
    from modules.traffic_prediction import TrafficPredictor, prepare_sequences
    HAS_PREDICTION = True
except Exception as e:
    HAS_PREDICTION = False
    print(f"⚠ Traffic Prediction not available: {e}")

try:
    from modules.smart_signal_control import SmartSignalAgent
    HAS_SIGNAL_CONTROL = True
except Exception as e:
    HAS_SIGNAL_CONTROL = False
    print(f"⚠ Smart Signal Control not available: {e}")

try:
    from modules.user_notification import NotificationCenter
    HAS_NOTIFICATIONS = True
except Exception as e:
    HAS_NOTIFICATIONS = False
    print(f"⚠ User Notifications not available: {e}")


app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Configuration
app.config.from_object(config['development'])
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_UPLOAD_SIZE
CORS(app)

# Create necessary directories
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.PROCESSED_FOLDER, exist_ok=True)
os.makedirs(Config.REPORTS_FOLDER, exist_ok=True)
os.makedirs(Config.DATABASE_FOLDER, exist_ok=True)
os.makedirs(Config.MODEL_FOLDER, exist_ok=True)

# Initialize components
# These are set to None first so endpoint validation can provide clear errors

detector = None
analyzer = None
report_gen = None

try:
    detector = VehicleDetector()
    analyzer = TrafficAnalyzer()
    report_gen = ReportGenerator()
    print("✓ All components initialized successfully")
except Exception as e:
    detector = None
    analyzer = None
    report_gen = None
    print(f"✗ Error initializing components: {e}")
    print("  Make sure YOLO model files are present and dependencies are installed")

# Optional components
optional_tracker = None
optional_predictor = None
optional_signal_agent = None
notification_center = None

if HAS_TRACKING:
    try:
        optional_tracker = VehicleTracker()
        print("✓ Vehicle Tracker initialized")
    except Exception as e:
        print(f"✗ Vehicle Tracker init failed: {e}")

if HAS_PREDICTION:
    try:
        optional_predictor = TrafficPredictor(
            model_path=Config.PREDICTION_MODEL_PATH,
            seq_len=Config.PREDICTION_SEQ_LEN,
            forecast_steps=Config.PREDICTION_FORECAST_STEPS,
            architecture=Config.PREDICTION_ARCHITECTURE
        )
        if optional_predictor.model is not None or optional_predictor.ensemble_models:
            print("✓ Traffic Predictor loaded from model")
        else:
            print("✓ Traffic Predictor initialized (no trained model loaded yet)")
    except Exception as e:
        optional_predictor = None
        print(f"✗ Traffic Predictor init failed: {e}")

if HAS_SIGNAL_CONTROL:
    try:
        optional_signal_agent = SmartSignalAgent()
        print("✓ Smart Signal Agent initialized")
    except Exception as e:
        print(f"✗ Smart Signal Agent init failed: {e}")

if HAS_NOTIFICATIONS:
    try:
        notification_center = NotificationCenter()
        print("✓ Notification Center initialized")
    except Exception as e:
        print(f"✗ Notification Center init failed: {e}")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def _ensure_core_components():
    if detector is None or analyzer is None or report_gen is None:
        raise RuntimeError(
            'Core AI components are not initialized. '
            'Check YOLO model files and report generator dependencies.'
        )


def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect(Config.DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            total_vehicles INTEGER,
            traffic_level TEXT,
            traffic_density REAL,
            signal_time INTEGER,
            ai_decision TEXT,
            processed_image_path TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER,
            report_type TEXT,
            file_path TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (analysis_id) REFERENCES analysis_history(id)
        )
    ''')
    
    conn.commit()
    conn.close()


def save_to_database(analysis_data, original_filename, file_type, processed_image_path):
    """Save analysis to database"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_history 
            (filename, file_type, total_vehicles, traffic_level, traffic_density, signal_time, ai_decision, processed_image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            original_filename,
            file_type,
            analysis_data['total_vehicles'],
            analysis_data['traffic_level'],
            analysis_data['traffic_density_percentage'],
            analysis_data['signal_time_recommendation'],
            analysis_data['ai_decision'],
            processed_image_path
        ))
        
        conn.commit()
        analysis_id = cursor.lastrowid
        conn.close()
        
        return analysis_id
    except Exception as e:
        print(f"Error saving to database: {e}")
        return None


def encode_image_to_base64(image_path):
    """Convert image to base64 for JSON response"""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index_new.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        saved_filename = timestamp + filename
        upload_path = os.path.join(Config.UPLOAD_FOLDER, saved_filename)
        file.save(upload_path)
        
        # Determine file type
        file_ext = filename.rsplit('.', 1)[1].lower()
        file_type = 'image' if file_ext in {'jpg', 'jpeg', 'png'} else 'video'
        
        return jsonify({
            'success': True,
            'filename': saved_filename,
            'original_name': filename,
            'file_type': file_type,
            'upload_path': upload_path
        }), 200
        
    except Exception as e:
        print(f"Error uploading file: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_traffic():
    """Analyze uploaded file for traffic"""
    try:
        _ensure_core_components()

        data = request.json or {}
        upload_path = data.get('upload_path')
        filename = data.get('filename')
        file_type = data.get('file_type')
        
        if not upload_path or not os.path.exists(upload_path):
            return jsonify({'error': 'File not found'}), 400
        
        # Perform vehicle detection
        if file_type == 'image':
            # Process image
            processed_image, detections, vehicle_count = detector.detect_vehicles(upload_path)
            
            # Save processed image
            processed_filename = f"processed_{filename}"
            processed_path = os.path.join(Config.PROCESSED_FOLDER, processed_filename)
            cv2.imwrite(processed_path, processed_image)
            
        else:
            # Process video using the detector's video pipeline
            # Ensure processed filename keeps original name but prefixed and has .mp4
            base_name, ext = os.path.splitext(filename)
            out_filename = f"processed_{base_name}.mp4"
            processed_path = os.path.join(Config.PROCESSED_FOLDER, out_filename)

            try:
                video_output_path, detections, vehicle_count = detector.detect_vehicles_video(
                    upload_path,
                    output_path=processed_path,
                    skip_frames=2
                )
            except Exception as video_error:
                print(f"Video detection error: {video_error}")
                return jsonify({'error': f"Video processing failed: {video_error}"}), 500

            # Determine which video file to read for thumbnail (prefer the processed output)
            thumb_source = video_output_path if video_output_path and os.path.exists(video_output_path) else upload_path

            # Create a thumbnail for display: try first frame, then middle frame
            thumbnail_path = os.path.join(Config.PROCESSED_FOLDER, f"thumb_{base_name}.jpg")
            image_base64 = None
            try:
                cap = cv2.VideoCapture(thumb_source)
                if cap is not None and cap.isOpened():
                    # try reading first frame
                    ret, frame = cap.read()
                    if not ret:
                        # try middle frame
                        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                        if total > 0:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
                            ret, frame = cap.read()
                    if ret and frame is not None:
                        cv2.imwrite(thumbnail_path, frame)
                        image_base64 = encode_image_to_base64(thumbnail_path)
                if cap:
                    cap.release()
            except Exception as thumb_err:
                print(f"Thumbnail generation failed: {thumb_err}")
                image_base64 = None

            # Ensure detections structure
            detections = detections or []
            video_summary = vehicle_count
        
        # Analyze traffic
        analysis_data = analyzer.analyze_traffic(vehicle_count)
        
        # Save to database
        analysis_id = save_to_database(
            analysis_data, 
            filename, 
            file_type, 
            processed_path
        )
        
        # Encode image to base64 for display
        if file_type == 'image':
            image_base64 = encode_image_to_base64(processed_path)
        else:
            # Use thumbnail for video instead of the MP4 file
            image_base64 = locals().get('image_base64', None)
        
        # Return results
        response = {
            'success': True,
            'analysis_id': analysis_id,
            'processed_image': f"data:image/jpeg;base64,{image_base64}" if image_base64 else None,
            'analysis': analysis_data
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error analyzing traffic: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/report/pdf', methods=['POST'])
def generate_pdf():
    """Generate PDF report"""
    try:
        _ensure_core_components()

        data = request.json or {}
        analysis_id = data.get('analysis_id')
        analysis_data = data.get('analysis')
        processed_image_path = data.get('processed_image_path')
        
        if not analysis_id:
            return jsonify({'error': 'Missing analysis ID'}), 400
        if not analysis_data:
            return jsonify({'error': 'Missing analysis payload'}), 400
        
        # Generate filename
        report_filename = f"report_{analysis_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_path = os.path.join(Config.REPORTS_FOLDER, report_filename)
        
        # Generate PDF (report_gen validated earlier)
        try:
            report_gen.generate_pdf_report(analysis_data, processed_image_path, report_path)
        except Exception as e:
            print(f"PDF generation failed: {e}")
            return jsonify({'error': f"PDF generation failed: {e}"}), 500
        
        # Save to database
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO reports (analysis_id, report_type, file_path)
            VALUES (?, ?, ?)
        ''', (analysis_id, 'pdf', report_path))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'report_filename': report_filename,
            'report_path': f"/api/download/{report_filename}"
        }), 200
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/report/csv', methods=['POST'])
def generate_csv():
    """Generate CSV report"""
    try:
        _ensure_core_components()

        data = request.json or {}
        analysis_id = data.get('analysis_id')
        analysis_data = data.get('analysis')
        
        if not analysis_id:
            return jsonify({'error': 'Missing analysis ID'}), 400
        if not analysis_data:
            return jsonify({'error': 'Missing analysis payload'}), 400
        
        # Generate filename
        report_filename = f"report_{analysis_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        report_path = os.path.join(Config.REPORTS_FOLDER, report_filename)
        
        # Generate CSV
        try:
            report_gen.generate_csv_report(analysis_data, None, report_path)
        except Exception as e:
            print(f"CSV generation failed: {e}")
            return jsonify({'error': f"CSV generation failed: {e}"}), 500
        
        # Save to database
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO reports (analysis_id, report_type, file_path)
            VALUES (?, ?, ?)
        ''', (analysis_id, 'csv', report_path))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'report_filename': report_filename,
            'report_path': f"/api/download/{report_filename}"
        }), 200
        
    except Exception as e:
        print(f"Error generating CSV: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<filename>')
def download_file(filename):
    """Download a generated report"""
    try:
        file_path = os.path.join(Config.REPORTS_FOLDER, secure_filename(filename))
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        print(f"Error downloading file: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/history')
def get_history():
    """Get analysis history"""
    try:
        conn = sqlite3.connect(Config.DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, file_type, total_vehicles, traffic_level, 
                   traffic_density, signal_time, timestamp
            FROM analysis_history
            ORDER BY timestamp DESC
            LIMIT 20
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                'id': row[0],
                'filename': row[1],
                'file_type': row[2],
                'total_vehicles': row[3],
                'traffic_level': row[4],
                'traffic_density': row[5],
                'signal_time': row[6],
                'timestamp': row[7]
            })
        
        return jsonify({'history': history}), 200
        
    except Exception as e:
        print(f"Error fetching history: {e}")
        return jsonify({'error': str(e)}), 500


# ========== OPTIONAL ADVANCED MODULE ENDPOINTS ==========

@app.route('/api/detect/yolov8', methods=['POST'])
def detect_yolov8():
    """Optional endpoint: detect vehicles using YOLOv8 (requires ultralytics)"""
    if not HAS_YOLOV8:
        return jsonify({
            'error': 'YOLOv8 detector not available',
            'message': 'Install ultralytics: pip install ultralytics'
        }), 503
    
    try:
        data = request.json
        upload_path = data.get('upload_path')
        
        if not upload_path or not os.path.exists(upload_path):
            return jsonify({'error': 'File not found'}), 400
        
        detector_v8 = YOLOv8Detector()
        image, detections = detector_v8.detect(upload_path)
        
        # Save processed image
        filename = os.path.basename(upload_path)
        processed_filename = f"yolov8_{filename}"
        processed_path = os.path.join(Config.PROCESSED_FOLDER, processed_filename)
        cv2.imwrite(processed_path, image)
        
        image_base64 = encode_image_to_base64(processed_path)
        
        return jsonify({
            'success': True,
            'detector': 'YOLOv8',
            'detections': detections,
            'processed_image': f"data:image/jpeg;base64,{image_base64}" if image_base64 else None
        }), 200
        
    except Exception as e:
        print(f"Error in YOLOv8 detection: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/track/vehicles', methods=['POST'])
def track_vehicles():
    """Optional endpoint: track vehicle movements (requires deep_sort_realtime or uses fallback)"""
    if not HAS_TRACKING:
        return jsonify({
            'error': 'Vehicle Tracker not available',
            'message': 'Install deep_sort_realtime: pip install deep_sort_realtime'
        }), 503
    
    try:
        data = request.json or {}
        upload_path = data.get('upload_path')
        file_type = data.get('file_type', 'image')
        
        if not upload_path or not os.path.exists(upload_path):
            return jsonify({'error': 'File not found'}), 400
        
        tracker = VehicleTracker()
        
        if file_type == 'image':
            # Single frame tracking
            image = cv2.imread(upload_path)
            if image is None:
                return jsonify({'error': 'Could not read image for tracking'}), 400
            detections = [
                {'bbox': [10, 20, 100, 150], 'conf': 0.95, 'class_id': 2},
            ]
            
            tracks = tracker.update(detections, frame=image)
            
            # Save annotated image
            tracked_image = image.copy()
            for track in tracks:
                if 'track_id' in track:
                    x1, y1, x2, y2 = map(int, track.get('bbox', [0, 0, 0, 0]))
                    cv2.rectangle(tracked_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(tracked_image, f"ID: {track['track_id']}", (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            tracked_filename = f"tracked_{os.path.basename(upload_path)}"
            tracked_path = os.path.join(Config.PROCESSED_FOLDER, tracked_filename)
            cv2.imwrite(tracked_path, tracked_image)
            
            image_base64 = encode_image_to_base64(tracked_path)
            
            return jsonify({
                'success': True,
                'tracks': tracks,
                'tracked_image': f"data:image/jpeg;base64,{image_base64}" if image_base64 else None
            }), 200
        else:
            return jsonify({'error': 'Video tracking requires video input stream'}), 400
        
    except Exception as e:
        print(f"Error in vehicle tracking: {e}")
        return jsonify({'error': str(e)}), 500


def load_vehicle_history(history_length):
    """Load recent vehicle counts from database."""
    conn = sqlite3.connect(Config.DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT total_vehicles FROM analysis_history
        ORDER BY timestamp DESC LIMIT ?
    ''', (history_length,))
    records = cursor.fetchall()
    conn.close()
    return [r[0] for r in reversed(records)]


@app.route('/api/train/prediction', methods=['POST'])
def train_prediction_model():
    """Train and persist the traffic prediction model."""
    if not HAS_PREDICTION:
        return jsonify({
            'error': 'Traffic Predictor not available',
            'message': 'Install tensorflow: pip install tensorflow'
        }), 503
    
    try:
        data = request.json or {}
        architecture = data.get('architecture', Config.PREDICTION_ARCHITECTURE)
        seq_len = int(data.get('seq_len', Config.PREDICTION_SEQ_LEN))
        forecast_steps = int(data.get('forecast_steps', Config.PREDICTION_FORECAST_STEPS))
        epochs = int(data.get('epochs', 20))
        batch_size = int(data.get('batch_size', 16))
        validation_split = float(data.get('validation_split', 0.1))
        use_ensemble = bool(data.get('use_ensemble', False))
        history_length = int(data.get('history_length', seq_len + forecast_steps + 5))
        
        sequence = load_vehicle_history(history_length)
        if len(sequence) < seq_len + forecast_steps + 1:
            return jsonify({
                'error': 'Not enough history for training',
                'required': seq_len + forecast_steps + 1,
                'available': len(sequence)
            }), 400
        
        X, y = prepare_sequences(sequence, seq_len=seq_len, forecast_steps=forecast_steps)
        
        predictor = TrafficPredictor(
            model_path=Config.PREDICTION_MODEL_PATH,
            seq_len=seq_len,
            forecast_steps=forecast_steps,
            architecture=architecture
        )
        
        if architecture == 'ensemble' or use_ensemble:
            predictor.build_ensemble((seq_len, 1), num_models=3)
        else:
            predictor.build_model((seq_len, 1))
        
        history = predictor.train(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        predictor.save(Config.PREDICTION_MODEL_PATH)
        
        global optional_predictor
        optional_predictor = predictor
        
        return jsonify({
            'success': True,
            'message': 'Traffic prediction model trained and saved',
            'architecture': architecture,
            'seq_len': seq_len,
            'forecast_steps': forecast_steps,
            'epochs': epochs,
            'batch_size': batch_size,
            'history_size': len(sequence)
        }), 200
    except Exception as e:
        print(f"Error training prediction model: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/congestion', methods=['POST'])
def predict_congestion():
    """Optional endpoint: predict future congestion using a persisted LSTM model."""
    if not HAS_PREDICTION:
        return jsonify({
            'error': 'Traffic Predictor not available',
            'message': 'Install tensorflow: pip install tensorflow'
        }), 503
    
    try:
        data = request.json or {}
        history_length = int(data.get('history_length', Config.PREDICTION_SEQ_LEN))
        use_ensemble = bool(data.get('use_ensemble', False))
        
        if optional_predictor is None:
            return jsonify({
                'error': 'Prediction model not initialized',
                'message': 'Train the model using /api/train/prediction'
            }), 503
        
        sequence = load_vehicle_history(history_length)
        if len(sequence) < Config.PREDICTION_SEQ_LEN:
            return jsonify({
                'error': 'Not enough history for prediction',
                'required': Config.PREDICTION_SEQ_LEN,
                'available': len(sequence)
            }), 400
        
        predictions = optional_predictor.predict(sequence, use_ensemble=use_ensemble)
        
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()
        elif not isinstance(predictions, list):
            predictions = [predictions]
        
        return jsonify({
            'success': True,
            'input_sequence': sequence,
            'predictions': predictions,
            'forecast_steps': len(predictions),
            'model_loaded': optional_predictor.model is not None or bool(optional_predictor.ensemble_models)
        }), 200
    except Exception as e:
        print(f"Error in congestion prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/signal/optimize', methods=['POST'])
def optimize_signal_timing():
    """Optional endpoint: optimize signal timing using DQN (requires stable-baselines3)"""
    if not HAS_SIGNAL_CONTROL:
        return jsonify({
            'error': 'Smart Signal Control not available',
            'message': 'Install stable-baselines3 and gym: pip install stable-baselines3 gym'
        }), 503
    
    try:
        data = request.json
        vehicle_count = data.get('vehicle_count', 10)
        current_signal_time = data.get('current_signal_time', 30)
        
        agent = SmartSignalAgent()
        
        # Simple observation: [vehicle_count, current_signal_time]
        observation = np.array([vehicle_count, current_signal_time], dtype=np.float32)
        
        # Predict optimal action
        action = agent.predict(observation)
        
        # Convert action to signal time recommendation
        action_to_time = {0: 20, 1: 30, 2: 45, 3: 60}
        optimized_time = action_to_time.get(action, current_signal_time)
        
        return jsonify({
            'success': True,
            'current_signal_time': current_signal_time,
            'vehicle_count': vehicle_count,
            'rl_action': int(action),
            'optimized_signal_time': optimized_time,
            'change_seconds': optimized_time - current_signal_time
        }), 200
        
    except Exception as e:
        print(f"Error in signal optimization: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/notify/alert', methods=['POST'])
def send_alert():
    """Optional endpoint: send traffic alerts via notification center"""
    if not HAS_NOTIFICATIONS:
        return jsonify({
            'error': 'Notifications not available',
            'message': 'Notification center not initialized'
        }), 503
    
    try:
        data = request.json
        alert_type = data.get('type', 'info')
        message = data.get('message', '')
        severity = data.get('severity', 'normal')
        
        alert_payload = {
            'type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        
        # Broadcast to connected clients
        notification_center.broadcast(alert_payload)
        
        return jsonify({
            'success': True,
            'alert': alert_payload,
            'message': 'Alert broadcasted to all connected clients'
        }), 200
        
    except Exception as e:
        print(f"Error sending alert: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/modules/status')
def get_modules_status():
    """Get status of all optional modules"""
    return jsonify({
        'modules': {
            'vehicle_tracking': {'available': HAS_TRACKING, 'endpoint': '/api/track/vehicles'},
            'yolov8_detection': {'available': HAS_YOLOV8, 'endpoint': '/api/detect/yolov8'},
            'traffic_prediction': {'available': HAS_PREDICTION, 'endpoint': '/api/predict/congestion'},
            'smart_signal_control': {'available': HAS_SIGNAL_CONTROL, 'endpoint': '/api/signal/optimize'},
            'user_notifications': {'available': HAS_NOTIFICATIONS, 'endpoint': '/api/notify/alert'}
        }
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Server error'}), 500


if __name__ == '__main__':
    # Initialize database
    init_database()
    app.config['MAX_CONTENT_LENGTH'] = Config.MAX_UPLOAD_SIZE
    
    # Run Flask app
    print("=" * 60)
    print("🚗 AI TRAFFIC MANAGEMENT SYSTEM")
    print("=" * 60)
    print("Starting Flask application...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
