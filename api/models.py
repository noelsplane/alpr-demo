from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime


Base = declarative_base()

class PlateDetection(Base):
    __tablename__ = 'detections'

    id = Column(Integer, primary_key=True)
    plate_text = Column(String)
    confidence = Column(Float)
    image_name = Column(String)
    x1 = Column(Integer)
    y1 = Column(Integer)
    x2 = Column(Integer)
    y2 = Column(Integer)
    plate_image_base64 = Column(Text)
    state = Column(String, nullable=True)
    state_confidence = Column(Float, default=0.0)
    camera_id = Column(String, nullable=True, default='default_camera')
    # New vehicle attribute fields
    vehicle_type = Column(String, nullable=True)
    vehicle_type_confidence = Column(Float, default=0.0)
    vehicle_make = Column(String, nullable=True)
    vehicle_make_confidence = Column(Float, default=0.0)
    vehicle_model = Column(String, nullable=True)
    vehicle_model_confidence = Column(Float, default=0.0)
    vehicle_color = Column(String, nullable=True)
    vehicle_color_confidence = Column(Float, default=0.0)
    vehicle_year = Column(String, nullable=True)
    vehicle_year_confidence = Column(Float, default=0.0)
    
    timestamp = Column(DateTime, default=datetime.utcnow)

class SurveillanceSession(Base):
    __tablename__ = 'surveillance_sessions'
    
    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    status = Column(String, default='active')  # active, completed, interrupted
    total_detections = Column(Integer, default=0)
    total_vehicles = Column(Integer, default=0)
    total_alerts = Column(Integer, default=0)
    session_notes = Column(Text, nullable=True)
    
class SessionDetection(Base):
    __tablename__ = 'session_detections'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, nullable=False)
    plate_text = Column(String)
    confidence = Column(Float)
    state = Column(String, nullable=True)
    state_confidence = Column(Float, default=0.0)
    detection_time = Column(DateTime, default=datetime.utcnow)
    frame_id = Column(String, nullable=True)
    plate_image_base64 = Column(Text)
    camera_id = Column(String, nullable=True)  # Added camera_id field
    
    # Vehicle attributes if detected
    vehicle_type = Column(String, nullable=True)
    vehicle_color = Column(String, nullable=True)
    vehicle_make = Column(String, nullable=True)
    vehicle_model = Column(String, nullable=True)
    
class SessionAlert(Base):
    __tablename__ = 'session_alerts'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, nullable=False)
    alert_type = Column(String)  # LOITERING, RAPID_REAPPEARANCE, etc.
    severity = Column(String)  # low, medium, high, critical
    plate_text = Column(String, nullable=True)
    message = Column(Text)
    details = Column(Text, nullable=True)  # JSON string
    alert_time = Column(DateTime, default=datetime.utcnow)

# Add these to models.py after the existing SessionAlert class:

# Add these classes to models.py after the SessionAlert class:

class VehicleTrack(Base):
    __tablename__ = 'vehicle_tracks'
    
    id = Column(Integer, primary_key=True)
    track_id = Column(String, unique=True, nullable=False)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    
    # Vehicle attributes
    vehicle_type = Column(String, nullable=True)
    vehicle_make = Column(String, nullable=True)
    vehicle_model = Column(String, nullable=True)
    vehicle_color = Column(String, nullable=True)
    vehicle_year = Column(String, nullable=True)
    
    # Confidence scores
    type_confidence = Column(Float, default=0.0)
    make_confidence = Column(Float, default=0.0)
    model_confidence = Column(Float, default=0.0)
    color_confidence = Column(Float, default=0.0)
    year_confidence = Column(Float, default=0.0)
    
    # Tracking data
    total_appearances = Column(Integer, default=0)
    is_suspicious = Column(Boolean, default=False)
    has_no_plate = Column(Boolean, default=False)
    anomaly_count = Column(Integer, default=0)
    
class TrackPlateAssociation(Base):
    __tablename__ = 'track_plate_associations'
    
    id = Column(Integer, primary_key=True)
    track_id = Column(String, nullable=False)
    plate_text = Column(String, nullable=False)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    appearance_count = Column(Integer, default=1)
    
class VehicleAnomaly(Base):
    __tablename__ = 'vehicle_anomalies'
    
    id = Column(Integer, primary_key=True)
    track_id = Column(String, nullable=False)
    anomaly_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    detected_time = Column(DateTime, default=datetime.utcnow)
    plate_text = Column(String, nullable=True)
    message = Column(Text)
    details = Column(Text, nullable=True)  # JSON string
    image_data = Column(Text, nullable=True)  # Base64 encoded image
    session_id = Column(Integer, nullable=True)