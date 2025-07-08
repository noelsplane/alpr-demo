from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime
from fastapi.staticfiles import StaticFiles
from starlette.staticfiles import StaticFiles as StarletteStaticFiles  
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, PlateDetection
from state_model import get_state_classifier
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

engine = create_engine('sqlite:///detections.db')
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)


def encode_image_to_base64(img_array):
    img = Image.fromarray(img_array)
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded_string

def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # Don't close here, we'll close manually


app = FastAPI()
class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response
    
app.mount("/ui", NoCacheStaticFiles(directory="static", html=True), name="static")

# Load the specialized license plate detection model
license_plate_model_path = "../models/license_plate_yolov8.pt"
if os.path.exists(license_plate_model_path):
    logger.info(f"Loading specialized license plate detection model from {license_plate_model_path}")
    model = YOLO(license_plate_model_path)
    logger.info("License plate model loaded successfully!")
else:
    logger.warning(f"License plate model not found at {license_plate_model_path}")
    model = YOLO("../models/yolov8n.pt")

# Initialize PaddleOCR
logger.info("Initializing PaddleOCR...")
ocr = PaddleOCR(
    use_angle_cls=True,  # Detect text angle
    lang='en',           # English
    use_gpu=False,       # CPU mode
    show_log=False,      # Reduce verbosity
    det_db_thresh=0.3,   # Text detection threshold
    rec_batch_num=6,     # Batch size for recognition
    max_text_length=25,  # Max text length
    use_space_char=True  # Recognize spaces
)
logger.info("PaddleOCR initialized successfully!")

state_classifier = get_state_classifier()

LOG_DIR = "../logs"
os.makedirs(LOG_DIR, exist_ok=True)

UPLOAD_DIR = "../data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message": "ALPR API is running with PaddleOCR"}

def save_detections_to_db(image_name, plates_detected):
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
            db.add(detection)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving to database: {e}")
    finally:
        db.close()

@app.get("/api/v1/detections")
def get_all_detections():
    db = get_db()
    try:
        detections = db.query(PlateDetection).order_by(PlateDetection.timestamp.desc()).all()
        
        # Group detections by image and timestamp
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
        return JSONResponse(content={"error": f"Failed to retrieve detections: {str(e)}"}, status_code=500)
    finally:
        db.close()

@app.delete("/api/v1/detections")
def clear_all_detections():
    db = get_db()
    try:
        db.query(PlateDetection).delete()
        db.commit()
        return JSONResponse(content={"message": "All detection history cleared successfully"})
    except Exception as e:
        db.rollback()
        return JSONResponse(content={"error": f"Failed to clear history: {str(e)}"}, status_code=500)
    finally:
        db.close()

def preprocess_for_paddle(img):
    """Preprocess image for better OCR results with PaddleOCR."""
    # Resize if too small
    height, width = img.shape[:2]
    if width < 150:
        scale = 150 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    return img

def clean_plate_text(text):
    """Clean and format license plate text."""
    # Convert to uppercase
    text = text.upper()
    
    # Common OCR corrections for license plates
    corrections = {
        '|': 'I',
        '!': 'I',
        '1': 'I',  # Often 'I' is read as '1' in plates
        '0': 'O',  # Often 'O' is read as '0'
        '[': 'I',
        ']': 'I',
        '{': 'I',
        '}': 'I',
        '(': 'I',
        ')': 'I',
        '.': '',
        ',': '',
        '-': '',
        '_': '',
    }
    
    for old, new in corrections.items():
        text = text.replace(old, new)
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    # Common license plate patterns - remove spaces in typical formats
    # e.g., "ABC 123" -> "ABC123", "12 ABC 34" -> "12ABC34"
    text = re.sub(r'([A-Z]+)\s+([0-9]+)', r'\1\2', text)
    text = re.sub(r'([0-9]+)\s+([A-Z]+)\s+([0-9]+)', r'\1\2\3', text)
    text = re.sub(r'([A-Z]+)\s+([0-9]+)\s+([A-Z]+)', r'\1\2\3', text)
    
    return text

@app.post("/api/v1/sighting")
async def create_sighting(file: UploadFile = File(...)):
    logger.info(f"Processing image: {file.filename}")
    
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())

    img_bgr = cv2.imread(file_location)
    
    # Run license plate detection
    results = model.predict(img_bgr, conf=0.25)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    
    logger.info(f"Detected {len(boxes)} potential license plates")

    plate_texts = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        
        # Add padding around the detected box
        padding = 5
        y1 = max(0, y1 - padding)
        y2 = min(img_bgr.shape[0], y2 + padding)
        x1 = max(0, x1 - padding)
        x2 = min(img_bgr.shape[1], x2 + padding)
        
        cropped = img_bgr[y1:y2, x1:x2]
        
        # Skip if crop is too small
        if cropped.shape[0] < 20 or cropped.shape[1] < 40:
            logger.warning(f"Skipping small detection: {cropped.shape}")
            continue
        
        # Preprocess for OCR
        cropped_processed = preprocess_for_paddle(cropped)
        
        # Run PaddleOCR
        try:
            ocr_result = ocr.ocr(cropped_processed, cls=True)
            
            if ocr_result and ocr_result[0]:
                # Combine all text found in the plate
                all_texts = []
                total_conf = 0
                count = 0
                
                for line in ocr_result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]
                        conf = line[1][1]
                        all_texts.append(text)
                        total_conf += conf
                        count += 1
                
                if all_texts:
                    # Join all text parts
                    full_text = ' '.join(all_texts)
                    avg_conf = total_conf / count if count > 0 else 0
                    
                    # Clean the text
                    cleaned_text = clean_plate_text(full_text)
                    
                    logger.info(f"Plate {i+1}: Raw='{full_text}', Cleaned='{cleaned_text}', Conf={avg_conf:.2f}")
                    
                    # Only keep if we have meaningful text
                    if len(cleaned_text) >= 3 and any(c.isalnum() for c in cleaned_text):
                        # Detect state
                        state, state_conf = state_classifier.classify_from_text(cleaned_text)
                        
                        # Convert for base64 encoding
                        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                        
                        plate_texts.append({
                            "text": cleaned_text,
                            "confidence": round(float(avg_conf), 2),
                            "box": [x1, y1, x2, y2],
                            "plate_image_base64": encode_image_to_base64(cropped_rgb),
                            "state": state,
                            "state_confidence": round(state_conf, 2) if state_conf else 0.0
                        })
            else:
                logger.warning(f"No text found in plate {i+1}")
                
        except Exception as e:
            logger.error(f"OCR error on plate {i+1}: {str(e)}")
            continue

    # Save to database
    save_detections_to_db(file.filename, plate_texts)
    
    response = {
        "image": file.filename,
        "timestamp": datetime.now().isoformat(),
        "plates_detected": plate_texts
    }

    return JSONResponse(content=response)
