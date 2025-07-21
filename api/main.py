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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
PLATERECOGNIZER_TOKEN = os.getenv("PLATERECOGNIZER_TOKEN")

# Database setup
engine = create_engine('sqlite:///detections.db')
SessionLocal = sessionmaker(bind=engine)
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

# Helper functions
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


# WebSocket endpoints
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
        realtime_processor = RealtimeVideoProcessor(
            model, 
            ocr, 
            manager,
            plate_recognizer_func=get_platerecognizer_results
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