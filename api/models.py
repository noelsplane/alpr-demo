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
    plate_image_base64 = Column(Text)  # Store base64 encoded cropped plate image
    state = Column(String, nullable=True)  # State code (e.g., 'CA', 'NY')
    state_confidence = Column(Float, default=0.0)  # Confidence in state detection
    timestamp = Column(DateTime, default=datetime.utcnow)
