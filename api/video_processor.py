# api/video_processor.py
import asyncio
import cv2
from typing import Optional
from datetime import datetime
import numpy as np

class VideoStreamProcessor:
    def __init__(self, model, ocr_reader, frame_skip=5):
        self.model = model
        self.ocr_reader = ocr_reader
        self.frame_skip = frame_skip
        self.is_processing = False
        self.vehicle_tracker = VehicleTracker()
        
    async def process_stream(self, video_source: str | int, websocket_manager):
        """Process video stream continuously"""
        cap = cv2.VideoCapture(video_source)
        frame_count = 0
        
        self.is_processing = True
        
        try:
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames for efficiency
                if frame_count % self.frame_skip != 0:
                    continue
                
                # Process frame
                detections = await self.process_frame(frame)
                
                # Track vehicles and detect anomalies
                anomalies = self.vehicle_tracker.process_detection(detections)
                
                # Broadcast updates
                await websocket_manager.broadcast({
                    'type': 'detection',
                    'timestamp': datetime.now().isoformat(),
                    'detections': detections,
                    'anomalies': anomalies
                })
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
                
        finally:
            cap.release()
            self.is_processing = False