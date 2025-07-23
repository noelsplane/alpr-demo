from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import os
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import base64
from io import BytesIO
from PIL import Image, ImageEnhance
from datetime import datetime, timedelta
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, func, and_, or_
from sqlalchemy.orm import sessionmaker
from database_config import db_config
from models import (
    Base,
    PlateDetection,
    SurveillanceSession,
    SessionDetection,
    SessionAlert,
    VehicleTrack,
    TrackPlateAssociation,
    VehicleAnomaly
)
from state_model import get_state_classifier
from plate_filter_utils import extract_plate_number, detect_state_from_context
import logging
import requests
from dotenv import load_dotenv
from platerecognizer_manager import pr_manager
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, Any
import asyncio
from websocket_manager import manager
from realtime_processor import RealtimeVideoProcessor
from fastapi import Query
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
PLATERECOGNIZER_TOKEN = os.getenv("PLATERECOGNIZER_TOKEN")

# Database setup
engine = db_config.init_engine()
SessionLocal = db_config.SessionLocal
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI()

# Static files with no-cache headers
class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response

app.mount("/ui", NoCacheStaticFiles(directory="static", html=True), name="static")

# Load models
license_plate_model_path = "models/license_plate_yolov8.pt"
if os.path.exists(license_plate_model_path):
    logger.info("Loading specialized license plate detection model")
    model = YOLO(license_plate_model_path)
    logger.info("License plate model loaded successfully!")
else:
    logger.warning("License plate model not found, using default")
    model = YOLO("yolov8n.pt")

# Initialize OCR and state classifier
logger.info("Initializing EasyOCR...")
ocr = easyocr.Reader(['en'], gpu=False)
state_classifier = get_state_classifier()
logger.info("Initialization complete!")

# Configuration
UPLOAD_DIR = "../data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global realtime processor and session tracking
realtime_processor = None
current_session_id = None
def encode_image_to_base64(img_array):
    """Convert numpy array to base64 string."""
    img = Image.fromarray(img_array)
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        pass


def enhance_plate_image(img):
    """Enhance plate image for better OCR accuracy."""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Resize if too small
    width, height = pil_img.size
    if width < 200:
        scale = 200 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Enhance contrast and sharpness
    pil_img = ImageEnhance.Contrast(pil_img).enhance(2.0)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(2.0)
    
    return np.array(pil_img)


def get_platerecognizer_results(image_path):
    """Get plate, state, and vehicle info from PlateRecognizer API."""
    
    def _api_call(path):
        """Inner function that makes the actual API call."""
        try:
            url = "https://api.platerecognizer.com/v1/plate-reader/"
            headers = {"Authorization": f"Token {PLATERECOGNIZER_TOKEN}"}
            
            # Updated data to request vehicle attributes
            data = {
                'regions': ['us'],
                'mmc': 'true',  # Enable Make, Model, Color detection
                'config': json.dumps({
                    'mmc': True,
                    'mode': 'fast'
                })
            }
            
            with open(path, 'rb') as fp:
                response = requests.post(
                    url, 
                    files=dict(upload=fp), 
                    data=data, 
                    headers=headers, 
                    timeout=10
                )
            
            logger.info(f"PlateRecognizer response status: {response.status_code}")
            
            if response.status_code in [200, 201]:
                result = response.json()
                logger.info(f"PlateRecognizer raw response: {json.dumps(result, indent=2)}")
                
                # Return ALL plates found by PlateRecognizer
                all_plates = []
                
                for plate_result in result.get('results', []):
                    # Extract plate info
                    pr_text = plate_result.get('plate', '')
                    pr_state = None
                    pr_state_conf = 0
                    
                    if plate_result.get('region'):
                        region_code = plate_result['region'].get('code', '')
                        if region_code.startswith('us-'):
                            pr_state = region_code.split('-')[1].upper()
                            pr_state_conf = plate_result['region'].get('score', 0)
                    
                    # Extract vehicle attributes
                    vehicle_info = plate_result.get('vehicle', {})
                    
                    vehicle_data = {
                        'type': vehicle_info.get('type'),
                        'type_confidence': vehicle_info.get('score', 0),
                        'make': vehicle_info.get('make'),
                        'make_confidence': vehicle_info.get('make_score', 0),
                        'model': vehicle_info.get('model'),
                        'model_confidence': vehicle_info.get('model_score', 0),
                        'color': vehicle_info.get('color'),
                        'color_confidence': vehicle_info.get('color_score', 0),
                        'year': vehicle_info.get('year'),
                        'year_confidence': vehicle_info.get('year_score', 0),
                    }
                    
                    all_plates.append({
                        'plate': pr_text,
                        'state': pr_state,
                        'state_confidence': pr_state_conf,
                        'vehicle': vehicle_data,
                        'box': plate_result.get('box', {})
                    })
                
                return all_plates if all_plates else None
            else:
                logger.error(f"PlateRecognizer API error: {response.status_code} - {response.text}")
                    
        except Exception as e:
            logger.error(f"PlateRecognizer API error: {e}")
        
        return None
    
    # Use the manager to handle caching and rate limiting
    return pr_manager.process_image(image_path, _api_call)
    

def save_detections_to_db(image_name, plates_detected, vehicle_info=None):
    """Save detection results to database."""
    db = get_db()
    try:
        for plate in plates_detected:
            detection = PlateDetection(
                plate_text=plate["text"],
                confidence=plate["confidence"],
                image_name=image_name,
                x1=plate["box"][0],
                y1=plate["box"][1],
                x2=plate["box"][2],
                y2=plate["box"][3],
                plate_image_base64=plate["plate_image_base64"],
                state=plate.get("state"),
                state_confidence=plate.get("state_confidence", 0.0)
            )
            
            # Add vehicle attributes if available
            if vehicle_info:
                detection.vehicle_type = vehicle_info.get('type')
                detection.vehicle_type_confidence = vehicle_info.get('type_confidence', 0.0)
                detection.vehicle_make = vehicle_info.get('make')
                detection.vehicle_make_confidence = vehicle_info.get('make_confidence', 0.0)
                detection.vehicle_model = vehicle_info.get('model')
                detection.vehicle_model_confidence = vehicle_info.get('model_confidence', 0.0)
                detection.vehicle_color = vehicle_info.get('color')
                detection.vehicle_color_confidence = vehicle_info.get('color_confidence', 0.0)
                detection.vehicle_year = vehicle_info.get('year')
                detection.vehicle_year_confidence = vehicle_info.get('year_confidence', 0.0)
            
            db.add(detection)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving to database: {e}")
    finally:
        db.close()

# API Endpoints
@app.get("/")
def root():
    return {"message": "ALPR API is running"}


@app.post("/api/v1/sighting")
async def create_sighting(file: UploadFile = File(...)):
    """Process uploaded image for license plate detection."""
    logger.info(f"Processing image: {file.filename}")
    
    # Save uploaded file
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Get PlateRecognizer results
    pr_all_results = get_platerecognizer_results(file_location)
    pr_plates = pr_all_results if pr_all_results else []

    plate_texts = []
    
    if pr_plates:
        logger.info(f"PlateRecognizer found {len(pr_plates)} plates")
        
        # Load image for cropping plate regions
        img_bgr = cv2.imread(file_location)
        
        for idx, pr_plate in enumerate(pr_plates):
            logger.info(f"PR Plate {idx+1}: {pr_plate['plate']} - {pr_plate.get('state', 'Unknown')} (conf: {pr_plate.get('state_confidence', 0):.3f})")
            
            # Get the bounding box from PlateRecognizer
            box = pr_plate.get('box', {})
            if box:
                x1 = box.get('xmin', 0)
                y1 = box.get('ymin', 0)
                x2 = box.get('xmax', 100)
                y2 = box.get('ymax', 100)
                
                # Crop the plate region
                cropped = img_bgr[y1:y2, x1:x2]
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                
                plate_data = {
                    "text": pr_plate['plate'].upper(),
                    "confidence": 0.9,  # PlateRecognizer is generally very accurate
                    "box": [x1, y1, x2, y2],
                    "plate_image_base64": encode_image_to_base64(cropped_rgb),
                    "state": pr_plate.get('state'),
                    "state_confidence": pr_plate.get('state_confidence', 0),
                    "detection_method": "platerecognizer"
                }
                
                # Add vehicle info if available
                vehicle_info = pr_plate.get('vehicle', {})
                if vehicle_info and vehicle_info.get('type'):
                    plate_data["vehicle_type"] = vehicle_info.get('type')
                    plate_data["vehicle_type_confidence"] = vehicle_info.get('type_confidence', 0)
                
                plate_texts.append(plate_data)
                
                logger.info(f"=== Plate {idx+1} Final Result ===")
                logger.info(f"  Text: '{plate_data['text']}' | State: {plate_data['state'] or 'Unknown'} | Method: platerecognizer")
    else:
        # Fallback to local detection if PlateRecognizer fails
        logger.info("No PlateRecognizer results, falling back to local detection")
        
        img_bgr = cv2.imread(file_location)
        results = model.predict(img_bgr, conf=0.25)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        
        logger.info(f"Detected {len(boxes)} potential license plates locally")

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # Add padding around detection
            padding = 10
            y1 = max(0, y1 - padding)
            y2 = min(img_bgr.shape[0], y2 + padding)
            x1 = max(0, x1 - padding)
            x2 = min(img_bgr.shape[1], x2 + padding)
            
            # Crop plate region
            cropped = img_bgr[y1:y2, x1:x2]
            if cropped.shape[0] < 20 or cropped.shape[1] < 50:
                continue
            
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            
            # Run OCR
            ocr_results = ocr.readtext(cropped_rgb, width_ths=0.7, height_ths=0.7)
            enhanced = enhance_plate_image(cropped)
            enhanced_results = ocr.readtext(enhanced, width_ths=0.7, height_ths=0.7)
            all_results = ocr_results + enhanced_results
            
            # Extract plate text
            plate_text, confidence = extract_plate_number(all_results)
            
            if plate_text:
                # Try pattern matching for state
                state = None
                state_conf = 0
                pattern_state, pattern_conf = state_classifier.classify_from_text(plate_text)
                if pattern_state and pattern_conf > 0.6:
                    state = pattern_state
                    state_conf = pattern_conf
                
                plate_data = {
                    "text": plate_text,
                    "confidence": round(float(confidence), 2),
                    "box": [x1, y1, x2, y2],
                    "plate_image_base64": encode_image_to_base64(cropped_rgb),
                    "state": state,
                    "state_confidence": round(state_conf, 2) if state_conf else 0.0,
                    "detection_method": "local"
                }
                
                plate_texts.append(plate_data)
                logger.info(f"=== Plate {i+1} Final Result ===")
                logger.info(f"  Text: '{plate_text}' | State: {state or 'Unknown'} | Method: local")

    # Get vehicle info from first PlateRecognizer result for database
    pr_vehicle_info = pr_plates[0].get('vehicle', {}) if pr_plates else None

    # Save and return results
    save_detections_to_db(file.filename, plate_texts, pr_vehicle_info)
    
    return JSONResponse(content={
        "image": file.filename,
        "timestamp": datetime.now().isoformat(),
        "plates_detected": plate_texts,
        "vehicle_info": pr_vehicle_info if pr_vehicle_info else None
    })


@app.get("/api/v1/detections")
def get_all_detections():
    """Get all detection history."""
    db = get_db()
    try:
        detections = db.query(PlateDetection).order_by(PlateDetection.timestamp.desc()).all()
        
        # Group by image/timestamp
        grouped_detections = {}
        for detection in detections:
            key = f"{detection.image_name}_{detection.timestamp.isoformat()}"
            if key not in grouped_detections:
                grouped_detections[key] = {
                    "image": detection.image_name,
                    "timestamp": detection.timestamp.isoformat(),
                    "plates_detected": []
                }
            
            grouped_detections[key]["plates_detected"].append({
                "text": detection.plate_text,
                "confidence": detection.confidence,
                "box": [detection.x1, detection.y1, detection.x2, detection.y2],
                "plate_image_base64": detection.plate_image_base64,
                "state": detection.state,
                "state_confidence": detection.state_confidence
            })
        
        return JSONResponse(content=list(grouped_detections.values()))
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        db.close()


@app.delete("/api/v1/detections")
def clear_all_detections():
    """Clear all detection history."""
    db = get_db()
    try:
        db.query(PlateDetection).delete()
        db.commit()
        return JSONResponse(content={"message": "All detection history cleared successfully"})
    except Exception as e:
        db.rollback()
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        db.close()


@app.get("/api/v1/state-analytics")
def get_state_analytics():
    """Get state recognition analytics."""
    db = get_db()
    try:
        detections = db.query(PlateDetection).all()
        
        # Calculate analytics
        total_detections = len(detections)
        states_identified = sum(1 for d in detections if d.state and d.state != 'UNKNOWN')
        
        # State distribution
        state_counts = {}
        confidence_sum = 0
        confidence_count = 0
        method_counts = {'platerecognizer': 0, 'pattern': 0, 'context': 0, 'none': 0}
        
        for detection in detections:
            if detection.state and detection.state != 'UNKNOWN':
                state_counts[detection.state] = state_counts.get(detection.state, 0) + 1
                if detection.state_confidence:
                    confidence_sum += detection.state_confidence
                    confidence_count += 1
        
        analytics = {
            'summary': {
                'total_detections': total_detections,
                'states_identified': states_identified,
                'identification_rate': (states_identified / total_detections * 100) if total_detections > 0 else 0,
                'average_confidence': (confidence_sum / confidence_count * 100) if confidence_count > 0 else 0,
                'unique_states': len(state_counts)
            },
            'state_distribution': state_counts,
            'recent_detections': []
        }
        
        # Add recent detections
        recent = db.query(PlateDetection).order_by(PlateDetection.timestamp.desc()).limit(20).all()
        for det in recent:
            analytics['recent_detections'].append({
                'timestamp': det.timestamp.isoformat(),
                'plate_text': det.plate_text,
                'state': det.state,
                'state_confidence': det.state_confidence,
                'image_name': det.image_name
            })
        
        return JSONResponse(content=analytics)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        db.close()


@app.get("/api/v1/platerecognizer/usage")
def get_api_usage():
    """Get PlateRecognizer API usage statistics."""
    return pr_manager.get_usage_report()


@app.get("/api/v1/vehicle-profiles")
def get_vehicle_profiles(min_confidence: float = 0.0, time_window_hours: Optional[int] = None):
    """Get all vehicle profiles with optional filtering."""
    from vehicle_profiles import VehicleProfileAggregator
    
    try:
        aggregator = VehicleProfileAggregator()
        aggregator.build_profiles(time_window_hours)
        
        profiles = aggregator.get_all_profiles(min_confidence)
        stats = aggregator.get_summary_stats()
        
        return JSONResponse(content={
            "profiles": profiles,
            "stats": stats,
            "generated_at": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error generating vehicle profiles: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/v1/vehicle-profiles/{plate_text}")
def get_vehicle_profile_by_plate(plate_text: str):
    """Get vehicle profile for a specific plate number."""
    from vehicle_profiles import get_vehicle_profile
    
    try:
        profile = get_vehicle_profile(plate_text)
        
        if profile:
            return JSONResponse(content=profile)
        else:
            return JSONResponse(
                content={"error": f"No profile found for plate: {plate_text}"}, 
                status_code=404
            )
    except Exception as e:
        logger.error(f"Error retrieving vehicle profile: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/v1/vehicle-profiles/suspicious/list")
def get_suspicious_vehicles():
    """Get list of vehicles with suspicious activity or anomalies."""
    from vehicle_profiles import VehicleProfileAggregator
    
    try:
        aggregator = VehicleProfileAggregator()
        aggregator.build_profiles()
        
        suspicious_profiles = aggregator.get_suspicious_profiles()
        suspicious_data = [p.to_dict() for p in suspicious_profiles]
        
        return JSONResponse(content={
            "suspicious_vehicles": suspicious_data,
            "total_count": len(suspicious_data),
            "generated_at": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error finding suspicious vehicles: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/v1/vehicle-image/{image_name}")
async def get_vehicle_image(image_name: str):
    """Get full vehicle image by filename."""
    # Check both upload directories
    upload_paths = [
        os.path.join(UPLOAD_DIR, image_name),
        os.path.join("../data/uploads", image_name),
        os.path.join("data/uploads", image_name)
    ]
    
    for path in upload_paths:
        if os.path.exists(path):
            return FileResponse(
                path,
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=3600"}
            )
    
    return JSONResponse(
        content={"error": f"Image not found: {image_name}"}, 
        status_code=404
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket, "general")
    try:
        while True:
            # Keep connection alive and handle messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/ws/surveillance")
async def surveillance_websocket(websocket: WebSocket):
    """WebSocket endpoint for surveillance feed."""
    await manager.connect(websocket, "surveillance")
    try:
        while True:
            data = await websocket.receive_text()
            # Handle commands
            if data == "get_stats":
                if realtime_processor:
                    stats = realtime_processor.vehicle_tracker.get_tracking_stats()
                    await websocket.send_json({"type": "stats", "data": stats})
            elif data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Surveillance endpoints
@app.post("/api/v1/surveillance/start")
async def start_surveillance(request: dict = None):
    """Start real-time video surveillance."""
    global realtime_processor, current_session_id
    
    if request is None:
        request = {}
    
    video_source = request.get('video_source', '0')
    
    # Create new surveillance session
    db = get_db()
    try:
        session = SurveillanceSession(
            start_time=datetime.now(),
            status='active'
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        current_session_id = session.id
        logger.info(f"Started surveillance session {current_session_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating session: {e}")
    finally:
        db.close()
    
    if realtime_processor is None:
        # Create processor with PlateRecognizer integration
        camera_id = f"camera_{video_source}" if video_source != "browser_stream" else "browser_camera"
        realtime_processor = RealtimeVideoProcessor(
            model, 
            ocr, 
            manager,
            plate_recognizer_func=get_platerecognizer_results,
            camera_id=camera_id
        )
    
    try:
        # For browser stream, we don't start camera capture
        if video_source == "browser_stream":
            realtime_processor.is_processing = True
            realtime_processor.frame_processor.start()
            
            # Start the async processing loop without capture
            asyncio.create_task(realtime_processor._async_processing_loop())
            
            logger.info("Started browser-based surveillance processing")
        else:
            # Original camera-based processing
            source = int(video_source) if video_source.isdigit() else video_source
            await realtime_processor.start_processing(source)
        
        return JSONResponse(content={
            "status": "started",
            "source": video_source,
            "session_id": current_session_id,
            "message": "Surveillance started successfully"
        })
    except Exception as e:
        logger.error(f"Error starting surveillance: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@app.post("/api/v1/surveillance/stop")
async def stop_surveillance():
    """Stop real-time video surveillance."""
    global realtime_processor, current_session_id
    
    if realtime_processor:
        realtime_processor.stop_processing()
        
        # Update session status
        if current_session_id:
            db = get_db()
            try:
                session = db.query(SurveillanceSession).filter(
                    SurveillanceSession.id == current_session_id
                ).first()
                
                if session:
                    session.end_time = datetime.now()
                    session.status = 'completed'
                    
                    # Get session stats
                    detection_count = db.query(SessionDetection).filter(
                        SessionDetection.session_id == current_session_id
                    ).count()
                    
                    unique_plates = db.query(SessionDetection.plate_text).filter(
                        SessionDetection.session_id == current_session_id
                    ).distinct().count()
                    
                    alert_count = db.query(SessionAlert).filter(
                        SessionAlert.session_id == current_session_id
                    ).count()
                    
                    session.total_detections = detection_count
                    session.total_vehicles = unique_plates
                    session.total_alerts = alert_count
                    
                    db.commit()
                    logger.info(f"Completed surveillance session {current_session_id}")
            except Exception as e:
                db.rollback()
                logger.error(f"Error updating session: {e}")
            finally:
                db.close()
                current_session_id = None
        
        return JSONResponse(content={
            "status": "stopped",
            "message": "Surveillance stopped successfully"
        })
    else:
        return JSONResponse(content={
            "status": "not_running",
            "message": "Surveillance is not currently running"
        })


@app.get("/api/v1/surveillance/status")
async def get_surveillance_status():
    """Get current surveillance status."""
    global realtime_processor
    
    if realtime_processor and realtime_processor.is_processing:
        stats = realtime_processor.vehicle_tracker.get_tracking_stats()
        return JSONResponse(content={
            "status": "running",
            "connection_stats": manager.get_connection_stats(),
            "tracking_stats": stats,
            "fps": realtime_processor.fps_counter.get_fps() if hasattr(realtime_processor, 'fps_counter') else 0
        })
    else:
        return JSONResponse(content={
            "status": "stopped",
            "connection_stats": manager.get_connection_stats()
        })


@app.get("/api/v1/surveillance/video-feed")
async def video_feed():
    """Stream video feed as MJPEG."""
    global realtime_processor
    
    if not realtime_processor or not realtime_processor.is_processing:
        return JSONResponse(
            content={"error": "Surveillance not running"}, 
            status_code=400
        )
    
    async def generate():
        async for frame in realtime_processor.get_frame_stream():
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/v1/surveillance/process-frame")
async def process_browser_frame(
    frame: UploadFile = File(...),
    frame_id: str = Form(...)
):
    """Process a frame sent from the browser."""
    global realtime_processor, current_session_id
    
    if not realtime_processor or not realtime_processor.is_processing:
        return JSONResponse(
            content={"error": "Surveillance not running"}, 
            status_code=400
        )
    
    try:
        # Read the frame data
        frame_data = await frame.read()
        nparr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(
                content={"error": "Invalid image data"}, 
                status_code=400
            )
        
        # Process the frame directly
        detections = realtime_processor.frame_processor._process_frame(img)
        
        if detections and current_session_id:
            # Save detections to database
            db = get_db()
            try:
                for detection in detections:
                    session_detection = SessionDetection(
                        session_id=current_session_id,
                        plate_text=detection.get('plate_text'),
                        confidence=detection.get('confidence', 0),
                        state=detection.get('state'),
                        state_confidence=detection.get('state_confidence', 0),
                        frame_id=frame_id,
                        plate_image_base64=detection.get('plate_image_base64'),
                        detection_time=datetime.now()
                    )
                    db.add(session_detection)
                db.commit()
            except Exception as e:
                db.rollback()
                logger.error(f"Error saving session detections: {e}")
            finally:
                db.close()
        
        if detections:
            # Track vehicles and detect anomalies
            anomalies = realtime_processor.vehicle_tracker.process_detections(detections)
            
            # Save alerts to database
            if anomalies and current_session_id:
                db = get_db()
                try:
                    for anomaly in anomalies:
                        alert = SessionAlert(
                            session_id=current_session_id,
                            alert_type=anomaly.get('type'),
                            severity=anomaly.get('severity'),
                            plate_text=anomaly.get('plate'),
                            message=anomaly.get('message'),
                            details=json.dumps(anomaly.get('details', {})),
                            alert_time=datetime.now()
                        )
                        db.add(alert)
                    db.commit()
                except Exception as e:
                    db.rollback()
                    logger.error(f"Error saving alerts: {e}")
                finally:
                    db.close()
            
            # Send updates via WebSocket
            await manager.send_detection_update({
                'detections': detections,
                'anomalies': anomalies,
                'frame_id': frame_id,
                'tracking_stats': realtime_processor.vehicle_tracker.get_tracking_stats()
            })
            
            # Send alerts for high-severity anomalies
            for anomaly in anomalies:
                if anomaly.get('severity') in ['high', 'critical']:
                    await manager.send_anomaly_alert(anomaly)
        
        return JSONResponse(content={
            "status": "processed",
            "frame_id": frame_id,
            "detections": detections,
            "detection_count": len(detections)
        })
        
    except Exception as e:
        logger.error(f"Error processing browser frame: {e}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )


@app.get("/api/v1/surveillance/sessions")
def get_surveillance_sessions(limit: int = 50):
    """Get list of surveillance sessions."""
    db = get_db()
    try:
        sessions = db.query(SurveillanceSession).order_by(
            SurveillanceSession.start_time.desc()
        ).limit(limit).all()
        
        session_list = []
        for session in sessions:
            session_dict = {
                'id': session.id,
                'start_time': session.start_time.isoformat(),
                'end_time': session.end_time.isoformat() if session.end_time else None,
                'status': session.status,
                'duration_minutes': None,
                'total_detections': session.total_detections,
                'total_vehicles': session.total_vehicles,
                'total_alerts': session.total_alerts
            }
            
            if session.end_time and session.start_time:
                duration = session.end_time - session.start_time
                session_dict['duration_minutes'] = round(duration.total_seconds() / 60, 1)
            
            session_list.append(session_dict)
        
        return JSONResponse(content={'sessions': session_list})
    except Exception as e:
        logger.error(f"Error fetching sessions: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        db.close()


@app.get("/api/v1/surveillance/sessions/{session_id}")
def get_session_details(session_id: int):
    """Get detailed information about a specific surveillance session."""
    db = get_db()
    try:
        session = db.query(SurveillanceSession).filter(
            SurveillanceSession.id == session_id
        ).first()
        
        if not session:
            return JSONResponse(content={"error": "Session not found"}, status_code=404)
        
        # Get all detections
        detections = db.query(SessionDetection).filter(
            SessionDetection.session_id == session_id
        ).order_by(SessionDetection.detection_time).all()
        
        # Get all alerts
        alerts = db.query(SessionAlert).filter(
            SessionAlert.session_id == session_id
        ).order_by(SessionAlert.alert_time).all()
        
        # Get unique plates
        unique_plates = db.query(SessionDetection.plate_text).filter(
            SessionDetection.session_id == session_id
        ).distinct().all()
        
        session_data = {
            'session': {
                'id': session.id,
                'start_time': session.start_time.isoformat(),
                'end_time': session.end_time.isoformat() if session.end_time else None,
                'status': session.status,
                'total_detections': session.total_detections,
                'total_vehicles': session.total_vehicles,
                'total_alerts': session.total_alerts
            },
            'unique_plates': [p[0] for p in unique_plates if p[0]],
            'detections': [{
                'id': d.id,
                'plate_text': d.plate_text,
                'confidence': d.confidence,
                'state': d.state,
                'state_confidence': d.state_confidence,
                'detection_time': d.detection_time.isoformat(),
                'plate_image': d.plate_image_base64
            } for d in detections],
            'alerts': [{
                'id': a.id,
                'type': a.alert_type,
                'severity': a.severity,
                'plate': a.plate_text,
                'message': a.message,
                'time': a.alert_time.isoformat()
            } for a in alerts]
        }
        
        return JSONResponse(content=session_data)
    except Exception as e:
        logger.error(f"Error fetching session details: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        db.close()


@app.get("/api/v1/test-camera")
async def test_camera():
    """Simple endpoint to test camera availability."""
    return FileResponse("static/camera_test.html")


# Add these endpoints to main.py:

@app.get("/api/v1/anomalies/no-plate-vehicles")
def get_no_plate_vehicles(time_window_hours: Optional[int] = 24):
    """Get all vehicles detected without license plates."""
    db = get_db()
    try:
        query = db.query(VehicleAnomaly).filter(
            VehicleAnomaly.anomaly_type.in_(['NO_PLATE_VEHICLE', 'SUSPICIOUS_NO_PLATE'])
        ).order_by(VehicleAnomaly.detected_time.desc())
        
        if time_window_hours:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            query = query.filter(VehicleAnomaly.detected_time >= cutoff_time)
        
        anomalies = query.all()
        
        # Group by track_id
        vehicles = {}
        for anomaly in anomalies:
            track_id = anomaly.track_id
            if track_id not in vehicles:
                # Get track details
                track = db.query(VehicleTrack).filter(
                    VehicleTrack.track_id == track_id
                ).first()
                
                vehicles[track_id] = {
                    'track_id': track_id,
                    'first_seen': track.first_seen.isoformat() if track else None,
                    'last_seen': track.last_seen.isoformat() if track else None,
                    'vehicle_description': '',
                    'total_appearances': track.total_appearances if track else 0,
                    'anomalies': []
                }
                
                if track:
                    desc_parts = []
                    if track.vehicle_color:
                        desc_parts.append(track.vehicle_color)
                    if track.vehicle_year:
                        desc_parts.append(track.vehicle_year)
                    if track.vehicle_make:
                        desc_parts.append(track.vehicle_make)
                    if track.vehicle_model:
                        desc_parts.append(track.vehicle_model)
                    if track.vehicle_type:
                        desc_parts.append(f"({track.vehicle_type})")
                    
                    vehicles[track_id]['vehicle_description'] = ' '.join(desc_parts)
            
            vehicles[track_id]['anomalies'].append({
                'id': anomaly.id,
                'type': anomaly.anomaly_type,
                'severity': anomaly.severity,
                'detected_time': anomaly.detected_time.isoformat(),
                'message': anomaly.message,
                'image': anomaly.image_data
            })
        
        return JSONResponse(content={
            'no_plate_vehicles': list(vehicles.values()),
            'total_count': len(vehicles),
            'time_window_hours': time_window_hours,
            'generated_at': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error fetching no-plate vehicles: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        db.close()


@app.get("/api/v1/anomalies/all")
def get_all_anomalies(
    time_window_hours: Optional[int] = 24,
    severity: Optional[str] = None,
    anomaly_type: Optional[str] = None
):
    """Get all detected anomalies with filtering options."""
    db = get_db()
    try:
        query = db.query(VehicleAnomaly).order_by(VehicleAnomaly.detected_time.desc())
        
        if time_window_hours:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            query = query.filter(VehicleAnomaly.detected_time >= cutoff_time)
        
        if severity:
            query = query.filter(VehicleAnomaly.severity == severity)
        
        if anomaly_type:
            query = query.filter(VehicleAnomaly.anomaly_type == anomaly_type)
        
        anomalies = query.limit(200).all()
        
        # Format results
        results = []
        for anomaly in anomalies:
            # Get associated track info
            track = db.query(VehicleTrack).filter(
                VehicleTrack.track_id == anomaly.track_id
            ).first()
            
            results.append({
                'id': anomaly.id,
                'track_id': anomaly.track_id,
                'type': anomaly.anomaly_type,
                'severity': anomaly.severity,
                'detected_time': anomaly.detected_time.isoformat(),
                'plate_text': anomaly.plate_text,
                'message': anomaly.message,
                'vehicle_info': {
                    'type': track.vehicle_type if track else None,
                    'make': track.vehicle_make if track else None,
                    'model': track.vehicle_model if track else None,
                    'color': track.vehicle_color if track else None,
                    'year': track.vehicle_year if track else None
                } if track else None,
                'image': anomaly.image_data
            })
        
        # Get summary statistics
        summary = {
            'total_anomalies': len(results),
            'by_type': {},
            'by_severity': {}
        }
        
        for anomaly in results:
            atype = anomaly['type']
            severity = anomaly['severity']
            
            summary['by_type'][atype] = summary['by_type'].get(atype, 0) + 1
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
        
        return JSONResponse(content={
            'anomalies': results,
            'summary': summary,
            'filters': {
                'time_window_hours': time_window_hours,
                'severity': severity,
                'anomaly_type': anomaly_type
            },
            'generated_at': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error fetching anomalies: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        db.close()


@app.get("/api/v1/anomalies/vehicle-tracks/{track_id}")
def get_vehicle_track_details(track_id: str):
    """Get detailed information about a specific vehicle track."""
    db = get_db()
    try:
        # Get track info
        track = db.query(VehicleTrack).filter(
            VehicleTrack.track_id == track_id
        ).first()
        
        if not track:
            return JSONResponse(content={"error": "Track not found"}, status_code=404)
        
        # Get all plates associated with this track
        plates = db.query(TrackPlateAssociation).filter(
            TrackPlateAssociation.track_id == track_id
        ).all()
        
        # Get all anomalies for this track
        anomalies = db.query(VehicleAnomaly).filter(
            VehicleAnomaly.track_id == track_id
        ).order_by(VehicleAnomaly.detected_time).all()
        
        # Get recent detections
        recent_detections = []
        for plate_assoc in plates:
            detections = db.query(PlateDetection).filter(
                PlateDetection.plate_text == plate_assoc.plate_text
            ).order_by(PlateDetection.timestamp.desc()).limit(10).all()
            
            for detection in detections:
                recent_detections.append({
                    'plate_text': detection.plate_text,
                    'timestamp': detection.timestamp.isoformat(),
                    'confidence': detection.confidence,
                    'state': detection.state,
                    'image_name': detection.image_name
                })
        
        # Sort recent detections by timestamp
        recent_detections.sort(key=lambda x: x['timestamp'], reverse=True)
        
        result = {
            'track_id': track_id,
            'vehicle_info': {
                'type': track.vehicle_type,
                'make': track.vehicle_make,
                'model': track.vehicle_model,
                'color': track.vehicle_color,
                'year': track.vehicle_year,
                'type_confidence': track.type_confidence,
                'make_confidence': track.make_confidence,
                'model_confidence': track.model_confidence,
                'color_confidence': track.color_confidence,
                'year_confidence': track.year_confidence
            },
            'tracking_info': {
                'first_seen': track.first_seen.isoformat(),
                'last_seen': track.last_seen.isoformat(),
                'total_appearances': track.total_appearances,
                'is_suspicious': track.is_suspicious,
                'has_no_plate': track.has_no_plate,
                'anomaly_count': track.anomaly_count
            },
            'associated_plates': [
                {
                    'plate_text': p.plate_text,
                    'first_seen': p.first_seen.isoformat(),
                    'last_seen': p.last_seen.isoformat(),
                    'appearance_count': p.appearance_count
                } for p in plates
            ],
            'anomalies': [
                {
                    'id': a.id,
                    'type': a.anomaly_type,
                    'severity': a.severity,
                    'detected_time': a.detected_time.isoformat(),
                    'message': a.message,
                    'plate_text': a.plate_text
                } for a in anomalies
            ],
            'recent_detections': recent_detections[:20]
        }
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error fetching track details: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        db.close()


@app.post("/api/v1/anomalies/report")
async def generate_anomaly_report(
    time_window_hours: int = 24,
    include_images: bool = False
):
    """Generate a comprehensive anomaly report."""
    try:
        # Get the current anomaly detector instance
        if not realtime_processor or not hasattr(realtime_processor, 'anomaly_detector'):
            return JSONResponse(
                content={"error": "Anomaly detector not initialized"}, 
                status_code=400
            )
        
        report = realtime_processor.anomaly_detector.export_suspicious_vehicles_report()
        
        # Add database statistics
        db = get_db()
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            # Count anomalies by type
            anomaly_counts = db.query(
                VehicleAnomaly.anomaly_type,
                func.count(VehicleAnomaly.id).label('count')
            ).filter(
                VehicleAnomaly.detected_time >= cutoff_time
            ).group_by(VehicleAnomaly.anomaly_type).all()
            
            report['database_stats'] = {
                'time_window_hours': time_window_hours,
                'anomaly_counts': {ac[0]: ac[1] for ac in anomaly_counts}
            }
            
            if not include_images:
                # Remove image data to reduce response size
                for category in ['no_plate_vehicles', 'plate_switchers', 'loiterers', 'all_anomalies']:
                    if category in report:
                        for item in report[category]:
                            if 'details' in item and 'image' in item['details']:
                                del item['details']['image']
        finally:
            db.close()
        
        return JSONResponse(content=report)
    except Exception as e:
        logger.error(f"Error generating anomaly report: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
# Cross-camera tracking endpoints
@app.post("/api/v1/cameras/register")
async def register_camera(camera_info: dict):
    """Register a new camera in the tracking system."""
    from cross_camera_tracker import CameraInfo, cross_camera_tracker
    
    try:
        camera = CameraInfo(
            camera_id=camera_info['camera_id'],
            location_name=camera_info['location_name'],
            latitude=camera_info['latitude'],
            longitude=camera_info['longitude'],
            direction=camera_info.get('direction'),
            coverage_radius_meters=camera_info.get('coverage_radius_meters', 50.0)
        )
        
        cross_camera_tracker.register_camera(camera)
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Camera {camera.camera_id} registered successfully"
        })
    except Exception as e:
        logger.error(f"Error registering camera: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/v1/tracking/vehicle/{global_id}")
async def get_vehicle_journey(global_id: str):
    """Get complete journey information for a specific vehicle."""
    from cross_camera_tracker import cross_camera_tracker
    
    journey = cross_camera_tracker.get_vehicle_journey(global_id)
    
    if journey:
        return JSONResponse(content=journey)
    else:
        return JSONResponse(
            content={"error": f"No journey found for vehicle {global_id}"}, 
            status_code=404
        )


@app.get("/api/v1/tracking/between-cameras")
async def find_vehicles_between_cameras(
    camera_a: str,
    camera_b: str,
    time_window_hours: int = 24
):
    """Find vehicles that traveled between two specific cameras."""
    from cross_camera_tracker import cross_camera_tracker
    
    try:
        vehicles = cross_camera_tracker.find_vehicles_between_cameras(
            camera_a, camera_b, time_window_hours
        )
        
        return JSONResponse(content={
            "camera_a": camera_a,
            "camera_b": camera_b,
            "time_window_hours": time_window_hours,
            "vehicles_found": len(vehicles),
            "vehicles": vehicles
        })
    except Exception as e:
        logger.error(f"Error finding vehicles between cameras: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/v1/tracking/statistics")
async def get_tracking_statistics():
    """Get cross-camera tracking system statistics."""
    from cross_camera_tracker import cross_camera_tracker
    
    stats = cross_camera_tracker.get_tracking_statistics()
    return JSONResponse(content=stats)


@app.post("/api/v1/tracking/process")
async def process_cross_camera_detection(request: dict):
    """Process a detection for cross-camera tracking."""
    from cross_camera_tracker import cross_camera_tracker
    
    try:
        detection = request.get('detection', {})
        camera_id = request.get('camera_id', 'unknown')
        timestamp = request.get('timestamp')
        
        if timestamp:
            timestamp = datetime.fromisoformat(timestamp)
        
        result = cross_camera_tracker.process_detection(
            detection, camera_id, timestamp
        )
        
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error processing cross-camera detection: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.get("/api/v1/search/vehicles")
async def search_vehicles(
    plate: Optional[str] = Query(None, description="Plate number to search for"),
    timeRange: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d, custom"),
    startDate: Optional[str] = Query(None, description="Start date for custom range"),
    endDate: Optional[str] = Query(None, description="End date for custom range"),
    vehicleTypes: Optional[str] = Query(None, description="Comma-separated vehicle types"),
    alerts: Optional[str] = Query(None, description="Comma-separated alert types"),
    cameras: Optional[str] = Query(None, description="Comma-separated camera IDs"),
    limit: int = Query(50, description="Maximum results to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """
    Search for vehicles based on various criteria.
    """
    db = get_db()
    try:
        # Apply time range filter
        end_time = datetime.now()
        if timeRange == "1h":
            start_time = end_time - timedelta(hours=1)
        elif timeRange == "24h":
            start_time = end_time - timedelta(hours=24)
        elif timeRange == "7d":
            start_time = end_time - timedelta(days=7)
        elif timeRange == "30d":
            start_time = end_time - timedelta(days=30)
        elif timeRange == "custom" and startDate and endDate:
            start_time = datetime.fromisoformat(startDate)
            end_time = datetime.fromisoformat(endDate)
        else:
            start_time = end_time - timedelta(hours=24)
        
        # Query regular detections
        regular_query = db.query(PlateDetection)
        
        # Apply plate filter
        if plate:
            regular_query = regular_query.filter(PlateDetection.plate_text.ilike(f"%{plate}%"))
        
        regular_query = regular_query.filter(PlateDetection.timestamp.between(start_time, end_time))
        
        # Apply vehicle type filter
        if vehicleTypes:
            types = vehicleTypes.split(',')
            regular_query = regular_query.filter(PlateDetection.vehicle_type.in_(types))
        
        # Apply camera filter
        if cameras:
            camera_list = cameras.split(',')
            regular_query = regular_query.filter(PlateDetection.camera_id.in_(camera_list))
        
        # Query session detections
        session_query = db.query(SessionDetection)
        
        if plate:
            session_query = session_query.filter(SessionDetection.plate_text.ilike(f"%{plate}%"))
        
        session_query = session_query.filter(SessionDetection.detection_time.between(start_time, end_time))
        
        if vehicleTypes:
            types = vehicleTypes.split(',')
            session_query = session_query.filter(SessionDetection.vehicle_type.in_(types))
        
        # Get total count from both sources
        regular_count = regular_query.count()
        session_count = session_query.count()
        total_count = regular_count + session_count
        
        # Apply pagination to both queries
        # Split limit between both sources proportionally
        if regular_count > 0 and session_count > 0:
            regular_ratio = regular_count / total_count
            regular_limit = max(1, int(limit * regular_ratio))
            session_limit = limit - regular_limit
        elif regular_count > 0:
            regular_limit = limit
            session_limit = 0
        else:
            regular_limit = 0
            session_limit = limit
        
        # Get regular detections
        regular_detections = regular_query.order_by(PlateDetection.timestamp.desc())\
                                        .offset(offset)\
                                        .limit(regular_limit)\
                                        .all()
        
        # Get session detections
        session_detections = session_query.order_by(SessionDetection.detection_time.desc())\
                                        .offset(max(0, offset - regular_count))\
                                        .limit(session_limit)\
                                        .all()
        
        # Merge and format results
        results = []
        seen_plates_timestamps = set()  # To avoid duplicates
        
        # Process regular detections
        for detection in regular_detections:
            key = f"{detection.plate_text}_{detection.timestamp.isoformat()}"
            if key not in seen_plates_timestamps:
                seen_plates_timestamps.add(key)
                result = {
                    'id': detection.id,
                    'plate_text': detection.plate_text,
                    'confidence': detection.confidence,
                    'timestamp': detection.timestamp.isoformat(),
                    'camera_id': detection.camera_id,
                    'image_name': detection.image_name,
                    'plate_image': detection.plate_image_base64,
                    'vehicle': {
                        'type': detection.vehicle_type,
                        'make': detection.vehicle_make,
                        'model': detection.vehicle_model,
                        'color': detection.vehicle_color,
                        'year': detection.vehicle_year
                    },
                    'state': detection.state,
                    'state_confidence': detection.state_confidence,
                    'detection_source': 'regular',
                    'alerts': []
                }
                results.append(result)
        
        # Process session detections
        for detection in session_detections:
            key = f"{detection.plate_text}_{detection.detection_time.isoformat()}"
            if key not in seen_plates_timestamps:
                seen_plates_timestamps.add(key)
                
                # Get session info
                session_info = db.query(SurveillanceSession).filter(
                    SurveillanceSession.id == detection.session_id
                ).first()
                
                result = {
                    'id': f"session_{detection.id}",
                    'plate_text': detection.plate_text,
                    'confidence': detection.confidence,
                    'timestamp': detection.detection_time.isoformat(),
                    'camera_id': 'surveillance_camera',  # Default camera ID for sessions
                    'session_id': detection.session_id,
                    'session_status': session_info.status if session_info else 'unknown',
                    'frame_id': detection.frame_id,
                    'plate_image': detection.plate_image_base64,
                    'vehicle': {
                        'type': detection.vehicle_type,
                        'make': detection.vehicle_make,
                        'model': detection.vehicle_model,
                        'color': detection.vehicle_color,
                        'year': None  # Session detection doesn't have year field
                    },
                    'state': detection.state,
                    'state_confidence': detection.state_confidence,
                    'detection_source': 'session',
                    'alerts': []
                }
                results.append(result)
        
        # Sort all results by timestamp
        results.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Check for anomalies if alerts filter is specified
        if alerts and results:
            alert_types = alerts.split(',')
            plate_texts = [r['plate_text'] for r in results if r['plate_text']]
            
            # Query anomalies for these vehicles
            vehicle_anomalies = db.query(VehicleAnomaly).filter(
                VehicleAnomaly.plate_text.in_(plate_texts),
                VehicleAnomaly.anomaly_type.in_(alert_types),
                VehicleAnomaly.detected_time.between(start_time, end_time)
            ).all()
            
            # Group anomalies by plate
            anomaly_map = defaultdict(list)
            for anomaly in vehicle_anomalies:
                anomaly_map[anomaly.plate_text].append({
                    'type': anomaly.anomaly_type,
                    'severity': anomaly.severity,
                    'timestamp': anomaly.detected_time.isoformat(),
                    'message': anomaly.message
                })
            
            # Add anomalies to results
            for result in results:
                if result['plate_text'] in anomaly_map:
                    result['alerts'] = anomaly_map[result['plate_text']]
        
        # Also check for session alerts
        if alerts and results:
            session_ids = [r.get('session_id') for r in results if r.get('session_id')]
            if session_ids:
                session_alerts = db.query(SessionAlert).filter(
                    SessionAlert.session_id.in_(session_ids),
                    SessionAlert.alert_type.in_(alert_types),
                    SessionAlert.alert_time.between(start_time, end_time)
                ).all()
                
                # Group alerts by session
                session_alert_map = defaultdict(list)
                for alert in session_alerts:
                    session_alert_map[alert.session_id].append({
                        'type': alert.alert_type,
                        'severity': alert.severity,
                        'timestamp': alert.alert_time.isoformat(),
                        'message': alert.message
                    })
                
                # Add session alerts to results
                for result in results:
                    if result.get('session_id') in session_alert_map:
                        result['alerts'].extend(session_alert_map[result['session_id']])
        
        return JSONResponse(content={
            'results': results[:limit],  # Ensure we don't exceed limit
            'total': total_count,
            'limit': limit,
            'offset': offset,
            'filters': {
                'plate': plate,
                'timeRange': timeRange,
                'vehicleTypes': vehicleTypes,
                'alerts': alerts,
                'cameras': cameras
            },
            'source_counts': {
                'regular_detections': regular_count,
                'session_detections': session_count
            }
        })
        
    except Exception as e:
        logger.error(f"Error searching vehicles: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        db.close()

@app.get("/api/v1/search/analytics")
async def get_search_analytics(
    timeRange: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    vehicleTypes: Optional[str] = Query(None, description="Comma-separated vehicle types"),
    cameras: Optional[str] = Query(None, description="Comma-separated camera IDs")
):
    """
    Get analytics for the search page including both regular and session detections.
    """
    db = get_db()
    try:
        # Determine time range
        end_time = datetime.now()
        if timeRange == "1h":
            start_time = end_time - timedelta(hours=1)
        elif timeRange == "24h":
            start_time = end_time - timedelta(hours=24)
        elif timeRange == "7d":
            start_time = end_time - timedelta(days=7)
        elif timeRange == "30d":
            start_time = end_time - timedelta(days=30)
        else:
            start_time = end_time - timedelta(hours=24)
        
        # 1. Suspicious Vehicles (vehicles with anomalies)
        suspicious_plates = set()
        
        # Get plates with anomalies
        anomaly_query = db.query(VehicleAnomaly.plate_text).filter(
            VehicleAnomaly.detected_time.between(start_time, end_time),
            VehicleAnomaly.plate_text.isnot(None)
        ).distinct()
        
        for plate in anomaly_query:
            suspicious_plates.add(plate[0])
        
        # Get plates from session alerts
        session_alert_query = db.query(SessionAlert.plate_text).filter(
            SessionAlert.alert_time.between(start_time, end_time),
            SessionAlert.plate_text.isnot(None)
        ).distinct()
        
        for plate in session_alert_query:
            suspicious_plates.add(plate[0])
        
        suspicious_count = len(suspicious_plates)
        
        # 2. Frequent Visitors (plates appearing 5+ times)
        frequent_plates = set()
        
        # Count from regular detections
        regular_freq = db.query(
            PlateDetection.plate_text,
            func.count(PlateDetection.id).label('count')
        ).filter(
            PlateDetection.timestamp.between(start_time, end_time),
            PlateDetection.plate_text.isnot(None)
        )
        
        if vehicleTypes:
            types = vehicleTypes.split(',')
            regular_freq = regular_freq.filter(PlateDetection.vehicle_type.in_(types))
        
        if cameras:
            camera_list = cameras.split(',')
            regular_freq = regular_freq.filter(PlateDetection.camera_id.in_(camera_list))
        
        regular_freq = regular_freq.group_by(PlateDetection.plate_text).having(
            func.count(PlateDetection.id) >= 5
        ).all()
        
        for plate, count in regular_freq:
            frequent_plates.add(plate)
        
        # Count from session detections
        session_freq = db.query(
            SessionDetection.plate_text,
            func.count(SessionDetection.id).label('count')
        ).filter(
            SessionDetection.detection_time.between(start_time, end_time),
            SessionDetection.plate_text.isnot(None)
        )
        
        if vehicleTypes:
            types = vehicleTypes.split(',')
            session_freq = session_freq.filter(SessionDetection.vehicle_type.in_(types))
        
        session_freq = session_freq.group_by(SessionDetection.plate_text).having(
            func.count(SessionDetection.id) >= 5
        ).all()
        
        for plate, count in session_freq:
            frequent_plates.add(plate)
        
        # Also check combined counts
        all_plates = defaultdict(int)
        
        # Get all regular detection counts
        all_regular = db.query(
            PlateDetection.plate_text,
            func.count(PlateDetection.id).label('count')
        ).filter(
            PlateDetection.timestamp.between(start_time, end_time),
            PlateDetection.plate_text.isnot(None)
        ).group_by(PlateDetection.plate_text).all()
        
        for plate, count in all_regular:
            all_plates[plate] += count
        
        # Get all session detection counts
        all_session = db.query(
            SessionDetection.plate_text,
            func.count(SessionDetection.id).label('count')
        ).filter(
            SessionDetection.detection_time.between(start_time, end_time),
            SessionDetection.plate_text.isnot(None)
        ).group_by(SessionDetection.plate_text).all()
        
        for plate, count in all_session:
            all_plates[plate] += count
        
        # Add plates with combined count >= 5
        for plate, total_count in all_plates.items():
            if total_count >= 5:
                frequent_plates.add(plate)
        
        frequent_count = len(frequent_plates)
        
        # 3. Recent Detections (last hour)
        recent_time = end_time - timedelta(hours=1)
        
        regular_recent = db.query(func.count(PlateDetection.id)).filter(
            PlateDetection.timestamp >= recent_time
        )
        
        if vehicleTypes:
            types = vehicleTypes.split(',')
            regular_recent = regular_recent.filter(PlateDetection.vehicle_type.in_(types))
        
        if cameras:
            camera_list = cameras.split(',')
            regular_recent = regular_recent.filter(PlateDetection.camera_id.in_(camera_list))
        
        regular_recent_count = regular_recent.scalar() or 0
        
        session_recent = db.query(func.count(SessionDetection.id)).filter(
            SessionDetection.detection_time >= recent_time
        )
        
        if vehicleTypes:
            types = vehicleTypes.split(',')
            session_recent = session_recent.filter(SessionDetection.vehicle_type.in_(types))
        
        session_recent_count = session_recent.scalar() or 0
        
        recent_count = regular_recent_count + session_recent_count
        
        # 4. Active Anomalies (last hour)
        active_anomaly_count = db.query(func.count(VehicleAnomaly.id)).filter(
            VehicleAnomaly.detected_time >= recent_time
        ).scalar() or 0
        
        active_session_alerts = db.query(func.count(SessionAlert.id)).filter(
            SessionAlert.alert_time >= recent_time
        ).scalar() or 0
        
        active_anomalies = active_anomaly_count + active_session_alerts
        
        return JSONResponse(content={
            'analytics': {
                'suspicious_vehicles': suspicious_count,
                'frequent_visitors': frequent_count,
                'recent_detections': recent_count,
                'active_anomalies': active_anomalies
            },
            'time_range': timeRange,
            'filters': {
                'vehicleTypes': vehicleTypes,
                'cameras': cameras
            }
        })
        
    except Exception as e:
        logger.error(f"Error calculating search analytics: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        db.close()

@app.get("/api/v1/cameras")
async def get_cameras():
    """Get list of all registered cameras with their status."""
    from cross_camera_tracker import cross_camera_tracker
    
    db = get_db()
    try:
        cameras = []
        
        # Get registered cameras from cross-camera tracker
        for cam_id, cam_info in cross_camera_tracker.cameras.items():
            # Get detection stats for this camera
            recent_detections = db.query(func.count(PlateDetection.id)).filter(
                PlateDetection.camera_id == cam_id,
                PlateDetection.timestamp >= datetime.now() - timedelta(hours=1)
            ).scalar()
            
            cameras.append({
                'id': cam_id,
                'name': cam_info.location_name,
                'latitude': cam_info.latitude,
                'longitude': cam_info.longitude,
                'direction': cam_info.direction,
                'status': 'active' if cam_id in ['cam_01', 'cam_02', 'cam_03'] else 'offline',
                'detections_per_hour': recent_detections,
                'last_detection': None  # You can add logic to get last detection time
            })
        
        return JSONResponse(content={'cameras': cameras})
        
    except Exception as e:
        logger.error(f"Error getting cameras: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        db.close()


@app.post("/api/v1/cameras/{camera_id}/stream")
async def start_camera_stream(camera_id: str, request: dict):
    """Start streaming from a specific camera."""
    global realtime_processor
    
    try:
        video_source = request.get('source', '0')
        
        # Create a new processor for this camera if needed
        if realtime_processor is None:
            realtime_processor = RealtimeVideoProcessor(
                model, 
                ocr, 
                manager,
                plate_recognizer_func=get_platerecognizer_results,
                camera_id=camera_id
            )
        
        # Start processing
        await realtime_processor.start_processing(video_source)
        
        return JSONResponse(content={
            'status': 'streaming',
            'camera_id': camera_id,
            'message': f'Started streaming from camera {camera_id}'
        })
        
    except Exception as e:
        logger.error(f"Error starting camera stream: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/v1/search/journey/{plate_text}")
async def get_vehicle_journey(plate_text: str):
    """Get the complete journey of a vehicle across all cameras."""
    from cross_camera_tracker import cross_camera_tracker
    
    db = get_db()
    try:
        # Get all detections for this plate
        detections = db.query(PlateDetection).filter(
            PlateDetection.plate_text == plate_text
        ).order_by(PlateDetection.timestamp).all()
        
        if not detections:
            return JSONResponse(
                content={"error": f"No records found for plate {plate_text}"}, 
                status_code=404
            )
        
        # Build journey timeline
        journey = []
        cameras_visited = set()
        
        for detection in detections:
            cameras_visited.add(detection.camera_id)
            journey.append({
                'timestamp': detection.timestamp.isoformat(),
                'camera_id': detection.camera_id,
                'camera_name': cross_camera_tracker.cameras.get(
                    detection.camera_id, 
                    {'location_name': 'Unknown'}
                ).location_name if detection.camera_id in cross_camera_tracker.cameras else 'Unknown',
                'confidence': detection.confidence,
                'state': detection.state,
                'image': detection.plate_image_base64
            })
        
        # Calculate journey statistics
        first_seen = detections[0].timestamp
        last_seen = detections[-1].timestamp
        duration = last_seen - first_seen
        
        return JSONResponse(content={
            'plate_text': plate_text,
            'vehicle_info': {
                'type': detections[-1].vehicle_type,
                'make': detections[-1].vehicle_make,
                'model': detections[-1].vehicle_model,
                'color': detections[-1].vehicle_color,
                'year': detections[-1].vehicle_year
            },
            'journey_stats': {
                'first_seen': first_seen.isoformat(),
                'last_seen': last_seen.isoformat(),
                'duration_minutes': duration.total_seconds() / 60,
                'total_sightings': len(detections),
                'cameras_visited': len(cameras_visited),
                'unique_cameras': list(cameras_visited)
            },
            'timeline': journey
        })
        
    except Exception as e:
        logger.error(f"Error getting vehicle journey: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        db.close()


@app.get("/api/v1/search/frequent-vehicles")
async def get_frequent_vehicles(
    time_window_hours: int = Query(24, description="Time window in hours"),
    min_appearances: int = Query(5, description="Minimum number of appearances"),
    camera_id: Optional[str] = Query(None, description="Filter by camera ID")
):
    """Get frequently appearing vehicles."""
    db = get_db()
    try:
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        # Build query
        query = db.query(
            PlateDetection.plate_text,
            func.count(PlateDetection.id).label('appearance_count'),
            func.max(PlateDetection.timestamp).label('last_seen'),
            func.min(PlateDetection.timestamp).label('first_seen')
        ).filter(
            PlateDetection.timestamp >= cutoff_time,
            PlateDetection.plate_text.isnot(None)
        )
        
        if camera_id:
            query = query.filter(PlateDetection.camera_id == camera_id)
        
        # Group by plate and filter by minimum appearances
        frequent_vehicles = query.group_by(PlateDetection.plate_text)\
                                .having(func.count(PlateDetection.id) >= min_appearances)\
                                .order_by(func.count(PlateDetection.id).desc())\
                                .all()
        
        results = []
        for plate, count, last_seen, first_seen in frequent_vehicles:
            # Get vehicle details from most recent detection
            latest_detection = db.query(PlateDetection).filter(
                PlateDetection.plate_text == plate
            ).order_by(PlateDetection.timestamp.desc()).first()
            
            results.append({
                'plate_text': plate,
                'appearance_count': count,
                'first_seen': first_seen.isoformat(),
                'last_seen': last_seen.isoformat(),
                'frequency_per_hour': count / time_window_hours,
                'vehicle_info': {
                    'type': latest_detection.vehicle_type,
                    'make': latest_detection.vehicle_make,
                    'model': latest_detection.vehicle_model,
                    'color': latest_detection.vehicle_color
                } if latest_detection else None
            })
        
        return JSONResponse(content={
            'frequent_vehicles': results,
            'time_window_hours': time_window_hours,
            'min_appearances': min_appearances,
            'total_found': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error getting frequent vehicles: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        db.close()


@app.websocket("/ws/camera/{camera_id}")
async def camera_websocket(websocket: WebSocket, camera_id: str):
    """WebSocket endpoint for individual camera feeds."""
    await manager.connect(websocket, f"camera_{camera_id}")
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)