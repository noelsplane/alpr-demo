"""
Real-time video processing for surveillance and anomaly detection.
Fixed implementation with proper threading and frame streaming.
"""

import asyncio
import cv2
import numpy as np
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import base64
from io import BytesIO
from PIL import Image
import logging
from collections import defaultdict, deque
import threading
import queue
import time
import json

logger = logging.getLogger(__name__)

class FrameProcessor:
    """Processes frames in a separate thread."""
    
    def __init__(self, model, ocr_reader, plate_recognizer_func=None):
        self.model = model
        self.ocr_reader = ocr_reader
        self.plate_recognizer_func = plate_recognizer_func
        self.processing_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue()
        self.is_running = False
        self.process_thread = None
        
    def start(self):
        """Start the processing thread."""
        if not self.is_running:
            self.is_running = True
            self.process_thread = threading.Thread(target=self._process_loop)
            self.process_thread.daemon = True
            self.process_thread.start()
            logger.info("Frame processor started")
    
    def stop(self):
        """Stop the processing thread."""
        self.is_running = False
        if self.process_thread:
            self.process_thread.join(timeout=2)
        logger.info("Frame processor stopped")
    
    def add_frame(self, frame: np.ndarray, frame_id: int) -> bool:
        """Add frame to processing queue. Returns False if queue is full."""
        try:
            self.processing_queue.put((frame, frame_id), block=False)
            return True
        except queue.Full:
            return False
    
    def get_results(self) -> List[Dict]:
        """Get all available results."""
        results = []
        try:
            while True:
                result = self.result_queue.get_nowait()
                results.append(result)
        except queue.Empty:
            pass
        return results
    
    def _process_loop(self):
        """Main processing loop running in separate thread."""
        while self.is_running:
            try:
                # Get frame from queue with timeout
                frame_data = self.processing_queue.get(timeout=0.1)
                if frame_data is None:
                    continue
                    
                frame, frame_id = frame_data
                
                # Process the frame
                detections = self._process_frame(frame)
                
                if detections:
                    self.result_queue.put({
                        'frame_id': frame_id,
                        'timestamp': datetime.now().isoformat(),
                        'detections': detections
                    })
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
    
    def _process_frame(self, frame: np.ndarray) -> List[Dict]:
        """Process a single frame for license plates."""
        try:
            # Run YOLO detection
            results = self.model.predict(frame, conf=0.25, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
            
            if len(boxes) == 0:
                return []
            
            detections = []
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                
                # Add padding
                padding = 10
                y1 = max(0, y1 - padding)
                y2 = min(frame.shape[0], y2 + padding)
                x1 = max(0, x1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                
                # Crop plate region
                plate_crop = frame[y1:y2, x1:x2]
                
                if plate_crop.shape[0] < 20 or plate_crop.shape[1] < 50:
                    continue
                
                # Try PlateRecognizer first if available
                plate_text = None
                confidence = 0.0
                state = None
                state_confidence = 0.0
                
                if self.plate_recognizer_func:
                    try:
                        # Save crop temporarily for API
                        temp_path = f"/tmp/plate_{datetime.now().timestamp()}.jpg"
                        cv2.imwrite(temp_path, plate_crop)
                        
                        pr_results = self.plate_recognizer_func(temp_path)
                        if pr_results and len(pr_results) > 0:
                            pr_result = pr_results[0]
                            plate_text = pr_result.get('plate', '')
                            confidence = 0.9  # PR is generally accurate
                            state = pr_result.get('state')
                            state_confidence = pr_result.get('state_confidence', 0)
                        
                        # Clean up temp file
                        import os
                        os.remove(temp_path)
                        
                    except Exception as e:
                        logger.error(f"PlateRecognizer error: {e}")
                
                # Fallback to OCR if no PR result
                if not plate_text:
                    try:
                        ocr_results = self.ocr_reader.readtext(plate_crop, width_ths=0.7, height_ths=0.7)
                        
                        if ocr_results:
                            # Extract best text
                            best_text = ""
                            best_conf = 0
                            
                            for bbox, text, conf in ocr_results:
                                if conf > best_conf and len(text) >= 3:
                                    best_text = text
                                    best_conf = conf
                            
                            if best_text:
                                plate_text = best_text.upper()
                                confidence = float(best_conf)
                    except Exception as e:
                        logger.error(f"OCR error: {e}")
                
                if plate_text:
                    # Convert crop to base64
                    plate_img = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
                    buffer = BytesIO()
                    plate_img.save(buffer, format="JPEG", quality=85)
                    plate_base64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    detection = {
                        'plate_text': plate_text,
                        'confidence': confidence,
                        'box': [x1, y1, x2, y2],
                        'plate_image_base64': plate_base64,
                        'state': state,
                        'state_confidence': state_confidence
                    }
                    
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return []


class RealtimeVideoProcessor:
    """Main video processor that handles capture and streaming."""
    
    def __init__(self, model, ocr_reader, websocket_manager, plate_recognizer_func=None):
        self.model = model
        self.ocr_reader = ocr_reader
        self.websocket_manager = websocket_manager
        self.plate_recognizer_func = plate_recognizer_func
        
        self.is_processing = False
        self.capture_thread = None
        self.frame_processor = FrameProcessor(model, ocr_reader, plate_recognizer_func)
        self.vehicle_tracker = VehicleTracker()
        
        # Frame management
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.frame_counter = 0
        self.fps_counter = FPSCounter()
        
        # Processing control
        self.frame_skip = 5  # Process every 5th frame
        self.last_process_time = 0
        self.min_process_interval = 0.5  # Minimum seconds between processing
    
    async def start_processing(self, video_source: Any = 0):
        """Start processing video from camera or file."""
        if self.is_processing:
            logger.warning("Video processing already running")
            return
        
        self.is_processing = True
        self.frame_processor.start()
        
        # Start capture thread
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(video_source,)
        )
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Start async processing loop
        asyncio.create_task(self._async_processing_loop())
        
        logger.info(f"Started video processing from source: {video_source}")
    
    def stop_processing(self):
        """Stop video processing."""
        self.is_processing = False
        self.frame_processor.stop()
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        
        logger.info("Stopped video processing")
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame for streaming."""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def _capture_loop(self, video_source):
        """Capture frames in a separate thread."""
        cap = cv2.VideoCapture(video_source)
        
        # Set camera properties for better performance
        if isinstance(video_source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            # USB camera optimizations
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        elif isinstance(video_source, str) and video_source.startswith('rtsp://'):
            # Network camera optimizations
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_TIMEOUT, 5000)  # 5 second timeout
        
        try:
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(video_source, str):
                        # Loop video file
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        # Camera disconnected
                        logger.error("Camera disconnected")
                        break
                
                # Update current frame
                with self.frame_lock:
                    self.current_frame = frame
                    self.frame_counter += 1
                
                # Update FPS
                self.fps_counter.update()
                
                # Add frame to processing queue based on skip rate
                current_time = time.time()
                if (self.frame_counter % self.frame_skip == 0 and 
                    current_time - self.last_process_time >= self.min_process_interval):
                    
                    if self.frame_processor.add_frame(frame.copy(), self.frame_counter):
                        self.last_process_time = current_time
                
                # Small delay to control capture rate
                time.sleep(0.01)
                
        finally:
            cap.release()
            logger.info("Video capture released")
    
    async def _async_processing_loop(self):
        """Process results and send updates via WebSocket."""
        while self.is_processing:
            try:
                # Get processing results
                results = self.frame_processor.get_results()
                
                for result in results:
                    detections = result['detections']
                    
                    # Track vehicles and detect anomalies
                    anomalies = self.vehicle_tracker.process_detections(detections)
                    
                    # Send detection update
                    await self.websocket_manager.send_detection_update({
                        'detections': detections,
                        'anomalies': anomalies,
                        'frame_id': result['frame_id'],
                        'fps': self.fps_counter.get_fps(),
                        'tracking_stats': self.vehicle_tracker.get_tracking_stats()
                    })
                    
                    # Send alerts for high-severity anomalies
                    for anomaly in anomalies:
                        if anomaly.get('severity') in ['high', 'critical']:
                            await self.websocket_manager.send_anomaly_alert(anomaly)
                
                # Send periodic stats update
                if self.frame_counter % 30 == 0:
                    await self.websocket_manager.send_detection_update({
                        'type': 'stats_update',
                        'fps': self.fps_counter.get_fps(),
                        'frames_processed': self.frame_counter,
                        'tracking_stats': self.vehicle_tracker.get_tracking_stats()
                    })
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in async processing loop: {e}")
                await asyncio.sleep(0.5)
    
    async def get_frame_stream(self):
        """Generate JPEG frames for streaming."""
        while self.is_processing:
            frame = self.get_current_frame()
            if frame is not None:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield buffer.tobytes()
            await asyncio.sleep(0.033)  # ~30 FPS


class VehicleTracker:
    """Track vehicles and detect anomalies in real-time."""
    
    def __init__(self):
        self.recent_vehicles = defaultdict(lambda: deque(maxlen=100))
        self.alert_cooldown = defaultdict(datetime)
        self.cooldown_minutes = 5
        self.stats = {
            'total_vehicles': 0,
            'total_detections': 0,
            'active_vehicles': 0
        }
    
    def process_detections(self, detections: List[Dict]) -> List[Dict]:
        """Process detections and return anomalies."""
        anomalies = []
        current_time = datetime.now()
        
        for detection in detections:
            plate = detection.get('plate_text', '')
            if not plate:
                continue
            
            # Add to recent vehicles
            self.recent_vehicles[plate].append({
                'timestamp': current_time,
                'detection': detection
            })
            
            # Update stats
            self.stats['total_detections'] += 1
            
            # Check for anomalies
            anomaly = self._check_for_anomalies(plate, detection)
            if anomaly:
                anomalies.append(anomaly)
        
        # Update active vehicles count
        self.stats['active_vehicles'] = len([
            p for p, appearances in self.recent_vehicles.items()
            if appearances and (current_time - appearances[-1]['timestamp']) < timedelta(minutes=5)
        ])
        
        return anomalies
    
    def _check_for_anomalies(self, plate: str, detection: Dict) -> Optional[Dict]:
        """Check for various types of anomalies."""
        current_time = datetime.now()
        appearances = list(self.recent_vehicles[plate])
        
        # Check cooldown
        last_alert = self.alert_cooldown.get(plate)
        if last_alert and (current_time - last_alert) < timedelta(minutes=self.cooldown_minutes):
            return None
        
        # Loitering detection
        recent_appearances = [
            a for a in appearances
            if (current_time - a['timestamp']) < timedelta(minutes=30)
        ]
        
        if len(recent_appearances) >= 5:
            time_span = recent_appearances[-1]['timestamp'] - recent_appearances[0]['timestamp']
            if time_span.total_seconds() > 60:  # At least 1 minute span
                self.alert_cooldown[plate] = current_time
                return {
                    'type': 'LOITERING',
                    'severity': 'medium',
                    'plate': plate,
                    'message': f'Vehicle {plate} detected {len(recent_appearances)} times in {time_span.total_seconds() / 60:.1f} minutes',
                    'details': {
                        'appearance_count': len(recent_appearances),
                        'time_span_minutes': time_span.total_seconds() / 60,
                        'detection': detection
                    }
                }
        
        # Rapid reappearance
        if len(appearances) >= 2:
            last_appearance = appearances[-2]['timestamp']
            time_since_last = current_time - last_appearance
            
            if timedelta(seconds=5) < time_since_last < timedelta(seconds=30):
                self.alert_cooldown[plate] = current_time
                return {
                    'type': 'RAPID_REAPPEARANCE',
                    'severity': 'low',
                    'plate': plate,
                    'message': f'Vehicle {plate} reappeared after {time_since_last.total_seconds():.1f} seconds',
                    'details': {
                        'seconds_between': time_since_last.total_seconds(),
                        'detection': detection
                    }
                }
        
        return None
    
    def get_tracking_stats(self) -> Dict:
        """Get current tracking statistics."""
        total_vehicles = len(self.recent_vehicles)
        total_appearances = sum(len(v) for v in self.recent_vehicles.values())
        
        frequent_vehicles = [
            (plate, len(appearances))
            for plate, appearances in self.recent_vehicles.items()
            if len(appearances) >= 3
        ]
        frequent_vehicles.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'total_vehicles_tracked': total_vehicles,
            'total_appearances': total_appearances,
            'active_vehicles': self.stats['active_vehicles'],
            'frequent_vehicles': frequent_vehicles[:10]
        }


class FPSCounter:
    """Simple FPS counter."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)
    
    def update(self):
        """Add current timestamp."""
        self.timestamps.append(time.time())
    
    def get_fps(self) -> float:
        """Calculate current FPS."""
        if len(self.timestamps) < 2:
            return 0.0
        
        time_span = self.timestamps[-1] - self.timestamps[0]
        if time_span > 0:
            return (len(self.timestamps) - 1) / time_span
        return 0.0