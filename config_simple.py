import os
from datetime import datetime

class Config:
    """Base configuration for Traffic Management System"""
    
    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'traffic-ai-secret-key'
    DEBUG = False
    
    # Upload settings
    MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov', 'mkv'}
    
    # Paths
    UPLOAD_FOLDER = 'static/uploads'
    PROCESSED_FOLDER = 'static/processed'
    REPORTS_FOLDER = 'static/reports'
    DATABASE_FOLDER = 'database'
    MODEL_FOLDER = 'models'
    
    # Database
    DATABASE_PATH = os.path.join(DATABASE_FOLDER, 'traffic.db')
    
    # Prediction Model settings
    PREDICTION_MODEL_PATH = os.path.join(MODEL_FOLDER, 'traffic_predictor.h5')
    PREDICTION_SEQ_LEN = 10
    PREDICTION_FORECAST_STEPS = 3
    PREDICTION_ARCHITECTURE = 'bidirectional'
    
    # YOLO Model settings
    YOLO_CONFIG = 'yolov7.cfg'
    YOLO_WEIGHTS = 'yolov7.weights'
    YOLO_NAMES = 'coco.names'
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    
    # Traffic Analysis settings
    MAX_ROAD_CAPACITY = 50  # Maximum vehicles for density calculation
    LOW_TRAFFIC_THRESHOLD = 30  # 0-30% = Low
    MEDIUM_TRAFFIC_THRESHOLD = 70  # 30-70% = Medium, 70-100% = Heavy
    
    # Signal Timing (in seconds)
    SIGNAL_TIMINGS = {
        'low': 20,
        'medium': 40,
        'heavy': 60
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
