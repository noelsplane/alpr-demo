"""
Enhanced database models optimized for TimescaleDB
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import json

Base = declarative_base()

class PlateDetection(Base):
    """Time-series optimized plate detection model"""
    __tablename__ = 'detections'
    
    # Use timestamp as part of primary key for TimescaleDB
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Plate information
    plate_text = Column(String(20), index=True)
    confidence = Column(Float)
    state = Column(String(2), index=True)
    state_confidence = Column(Float, default=0.0)
    
    # Location information
    camera_id = Column(String(50), index=True)
    image_name = Column(String(255))
    x1 = Column(Integer)
    y1 = Column(Integer)
    x2 = Column(Integer)
    y2 = Column(Integer)
    
    # Vehicle attributes (denormalized for query performance)
    vehicle_type = Column(String(50), index=True)
    vehicle_type_confidence = Column(Float, default=0.0)
    vehicle_make = Column(String(50), index=True)
    vehicle_make_confidence = Column(Float, default=0.0)
    vehicle_model = Column(String(50))
    vehicle_model_confidence = Column(Float, default=0.0)
    vehicle_color = Column(String(30), index=True)
    vehicle_color_confidence = Column(Float, default=0.0)
    vehicle_year = Column(String(4))
    vehicle_year_confidence = Column(Float, default=0.0)
    
    # Image data (consider external storage for production)
    plate_image_base64 = Column(Text)
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_plate_timestamp', 'plate_text', 'timestamp'),
        Index('idx_camera_timestamp', 'camera_id', 'timestamp'),
        Index('idx_vehicle_attrs', 'vehicle_make', 'vehicle_model', 'vehicle_color'),
        Index('idx_timestamp_desc', timestamp.desc()),
    )


class VehicleTrack(Base):
    """Aggregated vehicle tracking data"""
    __tablename__ = 'vehicle_tracks'
    
    id = Column(Integer, primary_key=True)
    track_id = Column(String(100), unique=True, nullable=False, index=True)
    first_seen = Column(DateTime, default=datetime.utcnow, index=True)
    last_seen = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Aggregated vehicle attributes
    vehicle_type = Column(String(50))
    vehicle_make = Column(String(50), index=True)
    vehicle_model = Column(String(50))
    vehicle_color = Column(String(30))
    vehicle_year = Column(String(4))
    
    # Confidence scores
    type_confidence = Column(Float, default=0.0)
    make_confidence = Column(Float, default=0.0)
    model_confidence = Column(Float, default=0.0)
    color_confidence = Column(Float, default=0.0)
    year_confidence = Column(Float, default=0.0)
    
    # Statistics
    total_appearances = Column(Integer, default=0)
    is_suspicious = Column(Boolean, default=False, index=True)
    has_no_plate = Column(Boolean, default=False, index=True)
    anomaly_count = Column(Integer, default=0, index=True)
    
    # Relationships
    plate_associations = relationship("TrackPlateAssociation", back_populates="track", cascade="all, delete-orphan")
    anomalies = relationship("VehicleAnomaly", back_populates="track", cascade="all, delete-orphan")


class TrackPlateAssociation(Base):
    """Many-to-many relationship between tracks and plates"""
    __tablename__ = 'track_plate_associations'
    
    id = Column(Integer, primary_key=True)
    track_id = Column(String(100), ForeignKey('vehicle_tracks.track_id'), nullable=False, index=True)
    plate_text = Column(String(20), nullable=False, index=True)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    appearance_count = Column(Integer, default=1)
    
    # Relationship
    track = relationship("VehicleTrack", back_populates="plate_associations")
    
    __table_args__ = (
        Index('idx_track_plate', 'track_id', 'plate_text'),
    )


class VehicleAnomaly(Base):
    """Time-series anomaly events"""
    __tablename__ = 'vehicle_anomalies'
    
    id = Column(Integer, primary_key=True)
    track_id = Column(String(100), ForeignKey('vehicle_tracks.track_id'), nullable=False, index=True)
    anomaly_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    detected_time = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Anomaly details
    plate_text = Column(String(20), index=True)
    camera_id = Column(String(50), index=True)
    message = Column(Text)
    details = Column(Text)  # JSON string
    image_data = Column(Text)  # Base64 encoded
    session_id = Column(Integer)
    
    # Relationship
    track = relationship("VehicleTrack", back_populates="anomalies")
    
    __table_args__ = (
        Index('idx_anomaly_time_type', 'detected_time', 'anomaly_type'),
        Index('idx_severity_time', 'severity', 'detected_time'),
    )


class SurveillanceSession(Base):
    """Surveillance session tracking"""
    __tablename__ = 'surveillance_sessions'
    
    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    end_time = Column(DateTime, index=True)
    status = Column(String(20), default='active', index=True)
    
    # Statistics
    total_detections = Column(Integer, default=0)
    total_vehicles = Column(Integer, default=0)
    total_alerts = Column(Integer, default=0)
    
    # Metadata
    camera_ids = Column(Text)  # JSON array of camera IDs
    session_config = Column(Text)  # JSON configuration
    session_notes = Column(Text)
    
    # Relationships
    session_detections = relationship("SessionDetection", back_populates="session", cascade="all, delete-orphan")
    session_alerts = relationship("SessionAlert", back_populates="session", cascade="all, delete-orphan")


class SessionDetection(Base):
    """Individual detections within a session"""
    __tablename__ = 'session_detections'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('surveillance_sessions.id'), nullable=False, index=True)
    detection_time = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Detection data
    plate_text = Column(String(20), index=True)
    confidence = Column(Float)
    state = Column(String(2))
    state_confidence = Column(Float, default=0.0)
    
    # Reference data
    frame_id = Column(String(50))
    camera_id = Column(String(50), index=True)
    plate_image_base64 = Column(Text)
    
    # Vehicle attributes
    vehicle_type = Column(String(50))
    vehicle_color = Column(String(30))
    vehicle_make = Column(String(50))
    vehicle_model = Column(String(50))
    
    # Relationship
    session = relationship("SurveillanceSession", back_populates="session_detections")
    
    __table_args__ = (
        Index('idx_session_time', 'session_id', 'detection_time'),
    )


class SessionAlert(Base):
    """Alerts generated during sessions"""
    __tablename__ = 'session_alerts'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('surveillance_sessions.id'), nullable=False, index=True)
    alert_time = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Alert data
    alert_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    plate_text = Column(String(20))
    camera_id = Column(String(50))
    message = Column(Text)
    details = Column(Text)  # JSON string
    
    # Relationship
    session = relationship("SurveillanceSession", back_populates="session_alerts")
    
    __table_args__ = (
        Index('idx_alert_session_time', 'session_id', 'alert_time'),
        Index('idx_alert_severity', 'severity', 'alert_time'),
    )


# Analytics materialized views (updated periodically)
class HourlyStats(Base):
    """Hourly aggregated statistics for dashboard"""
    __tablename__ = 'hourly_stats'
    
    hour_timestamp = Column(DateTime, primary_key=True)
    camera_id = Column(String(50), primary_key=True)
    
    # Counts
    detection_count = Column(Integer, default=0)
    unique_vehicles = Column(Integer, default=0)
    anomaly_count = Column(Integer, default=0)
    no_plate_count = Column(Integer, default=0)
    
    # Top vehicles (JSON)
    top_vehicles = Column(Text)  # JSON array
    
    __table_args__ = (
        Index('idx_hourly_camera', 'camera_id', 'hour_timestamp'),
    )


class DailyStats(Base):
    """Daily aggregated statistics"""
    __tablename__ = 'daily_stats'
    
    date = Column(DateTime, primary_key=True)
    camera_id = Column(String(50), primary_key=True)
    
    # Counts
    total_detections = Column(Integer, default=0)
    unique_vehicles = Column(Integer, default=0)
    total_anomalies = Column(Integer, default=0)
    
    # Breakdown by type (JSON)
    vehicle_type_breakdown = Column(Text)
    state_breakdown = Column(Text)
    anomaly_breakdown = Column(Text)
    peak_hours = Column(Text)  # JSON with hourly activity
    
    __table_args__ = (
        Index('idx_daily_camera', 'camera_id', 'date'),
    )