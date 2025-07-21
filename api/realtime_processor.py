
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
        try:
            # Run YOLO detection
            results = self.model.predict(frame, conf=0.25, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
            
            if len(boxes) == 0:
                return []
            
            detections = []
            vehicle_boxes = []
            plate_boxes = []
            
            # First pass: categorize detections by size and aspect ratio
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                
                # Calculate box area and aspect ratio
                width = x2 - x1
                height = y2 - y1
                area = width * height
                aspect_ratio = width / height if height > 0 else 0
                
                # Get confidence
                conf = float(results.boxes.conf[i].cpu().numpy()) if results.boxes.conf is not None else 0.5
                
                # Heuristic: Large boxes are likely vehicles, smaller ones are plates
                # Vehicles are typically wider and have larger area
                if area > 50000 and aspect_ratio > 1.2:  # Adjust thresholds as needed
                    vehicle_boxes.append({
                        'box': [x1, y1, x2, y2],
                        'area': area,
                        'confidence': conf
                    })
                else:
                    plate_boxes.append({
                        'box': [x1, y1, x2, y2],
                        'area': area,
                        'confidence': conf
                    })
            
            # Process vehicles and check for associated plates
            for vehicle in vehicle_boxes:
                vx1, vy1, vx2, vy2 = vehicle['box']
                has_plate = False
                associated_plate = None
                
                # Check if any plate box is within this vehicle box
                for plate in plate_boxes:
                    px1, py1, px2, py2 = plate['box']
                    # Check if plate center is within vehicle bounds
                    plate_center_x = (px1 + px2) / 2
                    plate_center_y = (py1 + py2) / 2
                    
                    if (vx1 <= plate_center_x <= vx2 and 
                        vy1 <= plate_center_y <= vy2):
                        has_plate = True
                        associated_plate = plate
                        break
                
                if not has_plate:
                    # This is a vehicle without a visible plate
                    vehicle_crop = frame[vy1:vy2, vx1:vx2]
                    
                    # Try to get vehicle attributes using PlateRecognizer
                    vehicle_attrs = {}
                    if self.plate_recognizer_func:
                        try:
                            temp_path = f"/tmp/vehicle_{datetime.now().timestamp()}.jpg"
                            cv2.imwrite(temp_path, vehicle_crop)
                            # Add camera ID to detection
                            detection['camera_id'] = getattr(self, 'camera_id', 'default')
                            pr_results = self.plate_recognizer_func(temp_path)
                            if pr_results and len(pr_results) > 0:
                                vehicle_info = pr_results[0].get('vehicle', {})
                                vehicle_attrs = {
                                    'vehicle_type': vehicle_info.get('type'),
                                    'vehicle_type_confidence': vehicle_info.get('type_confidence', 0),
                                    'vehicle_make': vehicle_info.get('make'),
                                    'vehicle_make_confidence': vehicle_info.get('make_confidence', 0),
                                    'vehicle_model': vehicle_info.get('model'),
                                    'vehicle_model_confidence': vehicle_info.get('model_confidence', 0),
                                    'vehicle_color': vehicle_info.get('color'),
                                    'vehicle_color_confidence': vehicle_info.get('color_confidence', 0),
                                    'vehicle_year': vehicle_info.get('year'),
                                    'vehicle_year_confidence': vehicle_info.get('year_confidence', 0)
                                }
                            
                            os.remove(temp_path)
                        except Exception as e:
                            logger.error(f"Error getting vehicle attributes: {e}")
                    
                    # Convert crop to base64
                    vehicle_img = Image.fromarray(cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2RGB))
                    buffer = BytesIO()
                    vehicle_img.save(buffer, format="JPEG", quality=85)
                    vehicle_base64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    detection = {
                        'plate_text': 'NO_PLATE_DETECTED',
                        'confidence': 0.0,
                        'box': vehicle['box'],
                        'plate_image_base64': vehicle_base64,
                        'state': None,
                        'state_confidence': 0,
                        'is_vehicle_without_plate': True,
                        'anomaly_type': 'NO_PLATE_VEHICLE',
                        **vehicle_attrs
                    }
                    
                    detections.append(detection)
                else:
                    # Process the associated plate
                    px1, py1, px2, py2 = associated_plate['box']
                    plate_crop = frame[py1:py2, px1:px2]
                    
                    # Process plate text (existing logic)
                    plate_text, confidence, state, state_confidence = self._process_plate_ocr(plate_crop)
                    
                    if plate_text:
                        # Get vehicle attributes
                        vehicle_attrs = self._get_vehicle_attributes_from_crop(frame[vy1:vy2, vx1:vx2])
                        
                        # Convert plate crop to base64
                        plate_img = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
                        buffer = BytesIO()
                        plate_img.save(buffer, format="JPEG", quality=85)
                        plate_base64 = base64.b64encode(buffer.getvalue()).decode()
                        
                        detection = {
                            'plate_text': plate_text,
                            'confidence': confidence,
                            'box': [px1, py1, px2, py2],
                            'vehicle_box': vehicle['box'],
                            'plate_image_base64': plate_base64,
                            'state': state,
                            'state_confidence': state_confidence,
                            'is_vehicle_without_plate': False,
                            **vehicle_attrs
                        }
                        
                        detections.append(detection)
            
            # Also process standalone plates (might be motorcycles or partial views)
            for plate in plate_boxes:
                # Check if this plate was already processed with a vehicle
                already_processed = False
                px1, py1, px2, py2 = plate['box']
                plate_center_x = (px1 + px2) / 2
                plate_center_y = (py1 + py2) / 2
                
                for vehicle in vehicle_boxes:
                    vx1, vy1, vx2, vy2 = vehicle['box']
                    if (vx1 <= plate_center_x <= vx2 and vy1 <= plate_center_y <= vy2):
                        already_processed = True
                        break
                
                if not already_processed:
                    # Process standalone plate
                    plate_crop = frame[py1:py2, px1:px2]
                    plate_text, confidence, state, state_confidence = self._process_plate_ocr(plate_crop)
                    
                    if plate_text:
                        # Convert plate crop to base64
                        plate_img = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
                        buffer = BytesIO()
                        plate_img.save(buffer, format="JPEG", quality=85)
                        plate_base64 = base64.b64encode(buffer.getvalue()).decode()
                        
                        detection = {
                            'plate_text': plate_text,
                            'confidence': confidence,
                            'box': [px1, py1, px2, py2],
                            'plate_image_base64': plate_base64,
                            'state': state,
                            'state_confidence': state_confidence,
                            'is_vehicle_without_plate': False
                        }
                        
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return []

def _process_plate_ocr(self, plate_crop):
    """Extract plate text using OCR."""
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
    
    return plate_text, confidence, state, state_confidence

def _get_vehicle_attributes_from_crop(self, vehicle_crop):
    """Try to get vehicle attributes from crop."""
    attrs = {}
    
    if self.plate_recognizer_func:
        try:
            temp_path = f"/tmp/vehicle_attr_{datetime.now().timestamp()}.jpg"
            cv2.imwrite(temp_path, vehicle_crop)
            
            pr_results = self.plate_recognizer_func(temp_path)
            if pr_results and len(pr_results) > 0:
                vehicle_info = pr_results[0].get('vehicle', {})
                attrs = {
                    'vehicle_type': vehicle_info.get('type'),
                    'vehicle_type_confidence': vehicle_info.get('type_confidence', 0),
                    'vehicle_make': vehicle_info.get('make'),
                    'vehicle_make_confidence': vehicle_info.get('make_confidence', 0),
                    'vehicle_model': vehicle_info.get('model'),
                    'vehicle_model_confidence': vehicle_info.get('model_confidence', 0),
                    'vehicle_color': vehicle_info.get('color'),
                    'vehicle_color_confidence': vehicle_info.get('color_confidence', 0),
                    'vehicle_year': vehicle_info.get('year'),
                    'vehicle_year_confidence': vehicle_info.get('year_confidence', 0)
                }
            
            os.remove(temp_path)
        except Exception as e:
            logger.error(f"Error getting vehicle attributes: {e}")
    
    return attrs

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
        self.frame_processor = FrameProcessor(model, ocr_reader, plate_recognizer_func)
        self.anomaly_detector = EnhancedAnomalyDetector()
        
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
                
                # Process for cross-camera tracking
                cross_camera_data = []
                if self.camera_id:
                    for detection in detections:
                        tracking_result = cross_camera_tracker.process_detection(
                            detection, 
                            self.camera_id,
                            datetime.now()
                        )
                        
                        # Include cross-camera anomalies
                        if tracking_result.get('anomalies'):
                            anomalies.extend(tracking_result['anomalies'])
                        
                        # Store cross-camera data for WebSocket update
                        cross_camera_data.append({
                            'global_vehicle_id': tracking_result.get('global_vehicle_id'),
                            'journey_info': tracking_result.get('journey_info', {}),
                            'vehicle_info': tracking_result.get('vehicle_info', {})
                        })
                
                # Send detection update
                update_data = {
                    'detections': detections,
                    'anomalies': anomalies,
                    'frame_id': result['frame_id'],
                    'fps': self.fps_counter.get_fps(),
                    'tracking_stats': self.vehicle_tracker.get_tracking_stats()
                }
                
                # Add cross-camera tracking data if available
                if cross_camera_data:
                    update_data['cross_camera_tracking'] = cross_camera_data
                    update_data['cross_camera_stats'] = cross_camera_tracker.get_tracking_statistics()
                
                await self.websocket_manager.send_detection_update(update_data)
                
                # Send alerts for high-severity anomalies
                for anomaly in anomalies:
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
                
                # Include cross-camera stats if available
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
                'details': {
                    'detection': detection,
                    'timestamp': current_time.isoformat(),
                    'vehicle_image': detection.get('plate_image_base64')
                }
            }
        
        # Rest of the existing anomaly checks...
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