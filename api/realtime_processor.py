"""
Real-time video processing for surveillance and anomaly detection.
"""

import os
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
from anomaly_detector import EnhancedAnomalyDetector
from cross_camera_tracker import cross_camera_tracker
from camera_detector import camera_detector, CameraInfo
from state_filter import clean_plate_text, is_valid_plate_text

logger = logging.getLogger(__name__)

class FrameProcessor:
    """Processes frames in a separate thread."""
    
    def __init__(self, model, ocr_reader, plate_recognizer_func=None, camera_id=None):
        self.model = model
        self.ocr_reader = ocr_reader
        self.plate_recognizer_func = plate_recognizer_func
        self.camera_id = camera_id
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
        try:
            # Run YOLO detection
            results = self.model.predict(frame, conf=0.25, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
            
            logger.info(f"YOLO detected {len(boxes)} objects")
            
            if len(boxes) == 0:
                return []
            
            detections = []
            
            # Process each detection as a potential plate
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                
                # Add padding around detection
                padding = 10
                y1 = max(0, y1 - padding)
                y2 = min(frame.shape[0], y2 + padding)
                x1 = max(0, x1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                
                # Crop the detected region
                crop = frame[y1:y2, x1:x2]
                
                # Skip if crop is too small
                if crop.shape[0] < 20 or crop.shape[1] < 50:
                    logger.debug(f"Skipping small crop: {crop.shape}")
                    continue
                
                logger.info(f"Processing crop {i+1}: size {crop.shape}")
                
                # Try PlateRecognizer first if available
                plate_text = None
                confidence = 0.0
                state = None
                state_confidence = 0.0
                vehicle_info = {}
                
                if self.plate_recognizer_func:
                    try:
                        # Save crop temporarily for API
                        temp_path = f"/tmp/plate_{datetime.now().timestamp()}.jpg"
                        cv2.imwrite(temp_path, crop)
                        
                        pr_results = self.plate_recognizer_func(temp_path)
                        if pr_results and len(pr_results) > 0:
                            pr_result = pr_results[0]
                            plate_text = pr_result.get('plate', '')
                            confidence = 0.9  # PR is generally accurate
                            state = pr_result.get('state')
                            state_confidence = pr_result.get('state_confidence', 0)
                            
                            # Get vehicle info if available
                            if 'vehicle' in pr_result:
                                v_info = pr_result['vehicle']
                                vehicle_info = {
                                    'vehicle_type': v_info.get('type'),
                                    'vehicle_type_confidence': v_info.get('type_confidence', 0),
                                    'vehicle_make': v_info.get('make'),
                                    'vehicle_make_confidence': v_info.get('make_confidence', 0),
                                    'vehicle_model': v_info.get('model'),
                                    'vehicle_model_confidence': v_info.get('model_confidence', 0),
                                    'vehicle_color': v_info.get('color'),
                                    'vehicle_color_confidence': v_info.get('color_confidence', 0),
                                    'vehicle_year': v_info.get('year'),
                                    'vehicle_year_confidence': v_info.get('year_confidence', 0)
                                }
                            
                            logger.info(f"PlateRecognizer found: {plate_text}")
                        else:
                            logger.info("PlateRecognizer returned no results")
                        
                        os.remove(temp_path)
                        
                    except Exception as e:
                        logger.error(f"PlateRecognizer error: {e}")
                
                # Fallback to OCR if no PR result
                if not plate_text:
                    try:
                        # Try OCR
                        ocr_results = self.ocr_reader.readtext(crop, width_ths=0.7, height_ths=0.7)
                        logger.info(f"OCR found {len(ocr_results)} text regions")
                        
                        if ocr_results:
                            # Extract best text
                            best_text = ""
                            best_conf = 0
                            
                            for bbox, text, conf in ocr_results:
                                logger.debug(f"OCR text: '{text}' (conf: {conf})")
                                if conf > best_conf and len(text) >= 3:
                                    best_text = text
                                    best_conf = conf
                            
                            if best_text:
                                plate_text = best_text.upper()
                                confidence = float(best_conf)
                                logger.info(f"OCR best text: {plate_text} (conf: {confidence})")
                    except Exception as e:
                        logger.error(f"OCR error: {e}")
                
                # If we found text, validate and filter it
                if plate_text:
                    # Clean and validate the plate text
                    original_text = plate_text
                    cleaned_text = clean_plate_text(plate_text)
                    
                    # Skip if the text is filtered out (state name, invalid format, etc.)
                    if not cleaned_text:
                        logger.info(f"Filtered out invalid plate text: '{original_text}'")
                        continue
                    
                    # Additional validation with confidence
                    if not is_valid_plate_text(cleaned_text, confidence):
                        logger.info(f"Filtered out low-quality plate text: '{original_text}' -> '{cleaned_text}' (conf: {confidence})")
                        continue
                    
                    # Use the cleaned text
                    plate_text = cleaned_text
                    # Convert crop to base64
                    plate_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    buffer = BytesIO()
                    plate_img.save(buffer, format="JPEG", quality=85)
                    plate_base64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    detection = {
                        'plate_text': plate_text,
                        'confidence': confidence,
                        'box': [x1, y1, x2, y2],
                        'plate_image_base64': plate_base64,
                        'state': state,
                        'state_confidence': state_confidence,
                        'is_vehicle_without_plate': False,
                        'camera_id': self.camera_id,
                        **vehicle_info  # Include vehicle info if available
                    }
                    
                    detections.append(detection)
                    logger.info(f"Added detection: {plate_text}")
                else:
                    logger.info(f"No text found in crop {i+1}")
            
            logger.info(f"Returning {len(detections)} detections")
            return detections
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            return []


class RealtimeVideoProcessor:
    """Main video processor that handles capture and streaming."""
    
    def __init__(self, model, ocr_reader, websocket_manager, plate_recognizer_func=None, camera_id=None):
        self.model = model
        self.ocr_reader = ocr_reader
        self.websocket_manager = websocket_manager
        self.plate_recognizer_func = plate_recognizer_func
        self.camera_id = camera_id or "default_camera"
        
        self.is_processing = False
        self.capture_thread = None
        self.frame_processor = FrameProcessor(model, ocr_reader, plate_recognizer_func, camera_id)
        self.anomaly_detector = EnhancedAnomalyDetector()
        self.vehicle_tracker = VehicleTracker()  # Initialize vehicle tracker
        
        # Frame management
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.frame_counter = 0
        self.fps_counter = FPSCounter()
        
        # Processing control
        self.frame_skip = 5
        self.last_process_time = 0
        self.min_process_interval = 0.5
    
    async def start_processing(self, video_source: Any = 0):
        """Start processing video from camera or file."""
        if self.is_processing:
            logger.warning("Video processing already running")
            return
        
        self.is_processing = True
        self.frame_processor.start()
        
        # For browser stream, we don't start camera capture
        if video_source != "browser_stream":
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
        
        # Save anomaly detector data if database session exists
        if hasattr(self, 'db_session') and self.db_session:
            self.anomaly_detector.save_tracks_to_db(self.db_session)
        
        logger.info("Stopped video processing")
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame for streaming."""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def _capture_loop(self, video_source):
        """Capture frames in a separate thread."""
        # If video_source is an integer, try to get camera info for better initialization
        if isinstance(video_source, int):
            camera_info = camera_detector.get_camera_by_index(video_source)
            if camera_info and not camera_info.is_working:
                logger.warning(f"Camera {video_source} was detected but marked as non-working")
        
        cap = cv2.VideoCapture(video_source)
        
        if isinstance(video_source, int):
            # Set optimal camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Try to set additional properties for better USB camera support
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time processing
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # MJPEG format
        
        try:
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(video_source, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        logger.error("Camera disconnected")
                        break
                
                with self.frame_lock:
                    self.current_frame = frame
                    self.frame_counter += 1
                
                self.fps_counter.update()
                
                current_time = time.time()
                if (self.frame_counter % self.frame_skip == 0 and 
                    current_time - self.last_process_time >= self.min_process_interval):
                    
                    if self.frame_processor.add_frame(frame.copy(), self.frame_counter):
                        self.last_process_time = current_time
                
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
                    
                    if not detections:
                        continue
                    
                    # Track vehicles and detect anomalies
                    anomalies = self.vehicle_tracker.process_detections(detections)
                    
                    # Process with enhanced anomaly detector
                    enhanced_anomalies = self.anomaly_detector.process_frame_detections(
                        detections, 
                        {'timestamp': datetime.now(), 'camera_id': self.camera_id}
                    )
                    
                    # Combine anomalies
                    all_anomalies = anomalies + enhanced_anomalies
                    
                    # Process for cross-camera tracking
                    cross_camera_data = []
                    for detection in detections:
                        tracking_result = cross_camera_tracker.process_detection(
                            detection, 
                            self.camera_id,
                            datetime.now()
                        )
                        
                        if tracking_result.get('anomalies'):
                            all_anomalies.extend(tracking_result['anomalies'])
                        
                        cross_camera_data.append({
                            'global_vehicle_id': tracking_result.get('global_vehicle_id'),
                            'journey_info': tracking_result.get('journey_info', {}),
                            'vehicle_info': tracking_result.get('vehicle_info', {})
                        })
                    
                    # Send detection update
                    update_data = {
                        'detections': detections,
                        'anomalies': all_anomalies,
                        'frame_id': result['frame_id'],
                        'fps': self.fps_counter.get_fps(),
                        'tracking_stats': self.vehicle_tracker.get_tracking_stats()
                    }
                    
                    if cross_camera_data:
                        update_data['cross_camera_tracking'] = cross_camera_data
                        update_data['cross_camera_stats'] = cross_camera_tracker.get_tracking_statistics()
                    
                    await self.websocket_manager.send_detection_update(update_data)
                    
                    # Send alerts for high-severity anomalies
                    for anomaly in all_anomalies:
                        if anomaly.get('severity') in ['high', 'critical']:
                            await self.websocket_manager.send_anomaly_alert(anomaly)
                
                # Send periodic stats update
                if self.frame_counter % 30 == 0:
                    stats_data = {
                        'type': 'stats_update',
                        'fps': self.fps_counter.get_fps(),
                        'frames_processed': self.frame_counter,
                        'tracking_stats': self.vehicle_tracker.get_tracking_stats()
                    }
                    
                    if self.camera_id:
                        stats_data['cross_camera_stats'] = cross_camera_tracker.get_tracking_statistics()
                    
                    await self.websocket_manager.send_detection_update(stats_data)
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in async processing loop: {e}")
                await asyncio.sleep(0.5)
    
    async def get_frame_stream(self):
        """Generate JPEG frames for streaming."""
        while self.is_processing:
            frame = self.get_current_frame()
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield buffer.tobytes()
            await asyncio.sleep(0.033)


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
            
            # Handle no-plate vehicles
            if detection.get('is_vehicle_without_plate', False):
                anomaly = self._check_for_anomalies('NO_PLATE', detection)
                if anomaly:
                    anomalies.append(anomaly)
                self.stats['total_detections'] += 1
                continue
            
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
        
        # Check if this is a no-plate vehicle
        if detection.get('is_vehicle_without_plate', False):
            return {
                'type': 'NO_PLATE_VEHICLE',
                'severity': 'high',
                'plate': 'NO_PLATE',
                'message': 'Vehicle detected without visible license plate',
                'timestamp': current_time.isoformat(),
                'details': {
                    'detection': detection,
                    'timestamp': current_time.isoformat(),
                    'vehicle_image': detection.get('plate_image_base64')
                }
            }
        
        if plate == 'NO_PLATE':
            return None
            
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
            if time_span.total_seconds() > 60:
                self.alert_cooldown[plate] = current_time
                return {
                    'type': 'LOITERING',
                    'severity': 'medium',
                    'plate': plate,
                    'message': f'Vehicle {plate} detected {len(recent_appearances)} times in {time_span.total_seconds() / 60:.1f} minutes',
                    'timestamp': current_time.isoformat(),
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
                    'timestamp': current_time.isoformat(),
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