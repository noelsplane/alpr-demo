from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import base64
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter
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
        pass


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
    logger.info(f"Loading specialized license plate detection model")
    model = YOLO(license_plate_model_path)
    logger.info("License plate model loaded successfully!")
else:
    logger.warning("License plate model not found")
    model = YOLO("../models/yolov8n.pt")

# Initialize EasyOCR with better settings
logger.info("Initializing EasyOCR...")
ocr = easyocr.Reader(['en'], gpu=False)
logger.info("EasyOCR initialized!")

state_classifier = get_state_classifier()

LOG_DIR = "../logs"
os.makedirs(LOG_DIR, exist_ok=True)

UPLOAD_DIR = "../data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Create debug directory
DEBUG_DIR = "debug_images"
os.makedirs(DEBUG_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message": "ALPR API is running with improved OCR"}

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

def enhance_plate_image(img):
    """Apply multiple enhancement techniques to improve OCR accuracy."""
    # Convert to PIL Image for enhancement
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Resize if too small
    width, height = pil_img.size
    if width < 200:
        scale = 200 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(2.0)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(2.0)
    
    # Convert back to numpy array
    enhanced = np.array(pil_img)
    
    return enhanced

def preprocess_variants(img):
    """Generate multiple preprocessing variants for OCR."""
    variants = []
    
    # Original
    variants.append(img)
    
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants.append(gray)
    
    # Binary threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(binary)
    
    # Inverted binary
    inv_binary = cv2.bitwise_not(binary)
    variants.append(inv_binary)
    
    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
    variants.append(adaptive)
    
    # Denoised
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    variants.append(denoised)
    
    # Edge preserved smoothing
    smooth = cv2.bilateralFilter(gray, 11, 17, 17)
    _, smooth_binary = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(smooth_binary)
    
    return variants

def clean_ocr_text(text):
    """Clean and correct common OCR errors in license plates."""
    # Convert to uppercase
    text = text.upper().strip()
    
    # Remove special characters except hyphen and space
    text = re.sub(r'[^A-Z0-9\s-]', '', text)
    
    # Common OCR corrections
    corrections = {
        'O': '0', '0': 'O',  # Try both ways
        'I': '1', '1': 'I',  # Try both ways
        'S': '5', '5': 'S',  # Try both ways
        'B': '8', '8': 'B',  # Try both ways
        'G': '6', '6': 'G',  # Try both ways
    }
    
    # For license plates, we often know the pattern
    # Try to identify if it's letters or numbers based on position
    parts = text.split()
    
    # Remove excess spaces
    text = ' '.join(parts)
    
    return text

def extract_text_with_voting(img, ocr, save_debug=False, debug_name="plate"):
    """Try multiple preprocessing methods and use voting for best result."""
    # Enhance the image first
    enhanced = enhance_plate_image(img)
    
    # Generate preprocessing variants
    variants = preprocess_variants(enhanced)
    
    # Save debug images
    if save_debug:
        for i, variant in enumerate(variants):
            debug_path = os.path.join(DEBUG_DIR, f"{debug_name}_variant_{i}.jpg")
            if len(variant.shape) == 2:  # Grayscale
                cv2.imwrite(debug_path, variant)
            else:  # Color
                cv2.imwrite(debug_path, cv2.cvtColor(variant, cv2.COLOR_RGB2BGR))
    
    # Run OCR on all variants
    all_results = {}
    
    for i, variant in enumerate(variants):
        try:
            results = ocr.readtext(variant)
            for bbox, text, conf in results:
                cleaned = clean_ocr_text(text)
                if len(cleaned) >= 3:  # Minimum plate length
                    if cleaned not in all_results:
                        all_results[cleaned] = []
                    all_results[cleaned].append(conf)
                    logger.info(f"Variant {i}: '{cleaned}' (conf: {conf:.2f})")
        except Exception as e:
            logger.error(f"OCR error on variant {i}: {e}")
            continue
    
    # Vote for best result (highest average confidence)
    best_text = None
    best_conf = 0
    
    for text, confidences in all_results.items():
        avg_conf = np.mean(confidences)
        occurrences = len(confidences)
        # Weight by both confidence and frequency
        weighted_score = avg_conf * (1 + 0.1 * occurrences)
        
        if weighted_score > best_conf:
            best_conf = avg_conf
            best_text = text
    
    return best_text, best_conf

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
        
        # Add padding
        padding = 10
        y1 = max(0, y1 - padding)
        y2 = min(img_bgr.shape[0], y2 + padding)
        x1 = max(0, x1 - padding)
        x2 = min(img_bgr.shape[1], x2 + padding)
        
        cropped = img_bgr[y1:y2, x1:x2]
        
        if cropped.shape[0] < 20 or cropped.shape[1] < 50:
            continue
        
        # Save original crop for debugging
        debug_name = f"{file.filename}_{i}"
        debug_path = os.path.join(DEBUG_DIR, f"{debug_name}_original.jpg")
        cv2.imwrite(debug_path, cropped)
        
        # Extract text with voting
        text, confidence = extract_text_with_voting(cropped, ocr, save_debug=True, debug_name=debug_name)
        
        if text and confidence > 0.3:
            # Detect state
            state, state_conf = state_classifier.classify_from_text(text)
            
            logger.info(f"Plate {i+1}: Text='{text}', Confidence={confidence:.2f}, State={state}")
            
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            
            plate_texts.append({
                "text": text,
                "confidence": round(float(confidence), 2),
                "box": [x1, y1, x2, y2],
                "plate_image_base64": encode_image_to_base64(cropped_rgb),
                "state": state,
                "state_confidence": round(state_conf, 2) if state_conf else 0.0
            })

    # Save to database
    save_detections_to_db(file.filename, plate_texts)
    
    response = {
        "image": file.filename,
        "timestamp": datetime.now().isoformat(),
        "plates_detected": plate_texts
    }

    return JSONResponse(content=response)
