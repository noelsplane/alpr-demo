from sqlalchemy import Column, Integer, String, Float, DateTime, Text
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