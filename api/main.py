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
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, PlateDetection
from state_model import get_state_classifier
from plate_filter_utils import extract_plate_number, detect_state_from_context
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


def extract_state_from_plate_region(img, ocr, debug_name=""):
    """Try to extract state by focusing on specific regions of the plate."""
    height, width = img.shape[:2]
    states_found = []
    
    # Define regions to search for state text
    regions = [
        ("top", img[0:int(height*0.35), :]),  # Top 35% - where state names often appear
        ("middle", img[int(height*0.3):int(height*0.7), :]),  # Middle region
        ("bottom", img[int(height*0.65):, :]),  # Bottom 35% - DMV URLs
        ("full", img)  # Full image as fallback
    ]
    
    for region_name, region in regions:
        try:
            # Skip if region is too small
            if region.shape[0] < 10 or region.shape[1] < 10:
                continue
                
            # Convert to RGB for OCR
            if len(region.shape) == 2:
                region_rgb = cv2.cvtColor(region, cv2.COLOR_GRAY2RGB)
            else:
                region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            
            # Try with different OCR parameters for better text detection
            results = ocr.readtext(region_rgb, width_ths=0.5, height_ths=0.5, 
                                  paragraph=True, decoder='beamsearch')
            
            # Log what we found in this region
            if results:
                logger.info(f"Region {region_name} OCR results:")
                for result in results:
                    if len(result) >= 2:
                        logger.info(f"  - Text: '{result[1]}'")
            
            # Check for state indicators
            state, conf = detect_state_from_context(results)
            if state:
                states_found.append((state, conf, region_name))
                
            # Also try enhanced version
            try:
                # Apply different preprocessing for state text
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
                # Try inverted (sometimes state text is light on dark)
                inverted = cv2.bitwise_not(gray)
                results_inv = ocr.readtext(inverted, width_ths=0.5, height_ths=0.5)
                state_inv, conf_inv = detect_state_from_context(results_inv)
                if state_inv and conf_inv > 0:
                    states_found.append((state_inv, conf_inv, f"{region_name}_inverted"))
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error in region {region_name}: {e}")
            continue
    
    # Return best state found
    if states_found:
        # Sort by confidence
        states_found.sort(key=lambda x: x[1], reverse=True)
        best_state = states_found[0]
        logger.info(f"Best state found: {best_state[0]} in {best_state[2]} region (conf: {best_state[1]})")
        return best_state[0], best_state[1]
    
    return None, 0


def enhance_for_state_text(img):
    """Special preprocessing for reading state names which are often in decorative fonts."""
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Try multiple thresholding methods
    variants = []
    
    # Standard binary
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(binary)
    
    # Inverted binary
    variants.append(cv2.bitwise_not(binary))
    
    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    variants.append(adaptive)
    
    return variants


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
        
        # Convert to RGB for OCR
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        
        # Try OCR with better text grouping parameters
        ocr_results = ocr.readtext(cropped_rgb, width_ths=0.7, height_ths=0.7)
        
        # Also try on enhanced image
        enhanced = enhance_plate_image(cropped)
        enhanced_results = ocr.readtext(enhanced, width_ths=0.7, height_ths=0.7)
        
        # Combine results
        all_results = ocr_results + enhanced_results
        
        # Debug logging
        logger.info(f"=== Plate {i+1} OCR Results ===")
        logger.info(f"Number of OCR detections: {len(all_results)}")
        for idx, result in enumerate(all_results):
            if len(result) >= 2:
                logger.info(f"  Detection {idx}: Text='{result[1]}', Conf={result[2] if len(result) > 2 else 'N/A'}")
        
        # Extract just the license plate number using our filter
        plate_text, confidence = extract_plate_number(all_results)
        logger.info(f"Extracted plate text: '{plate_text}' (conf: {confidence})")
        
        # Try multiple methods for state detection
        state = None
        state_conf = 0
        
        # Method 1: Try to detect state from the current OCR results
        state, state_conf = detect_state_from_context(all_results)
        if state:
            logger.info(f"State '{state}' detected from plate OCR (conf: {state_conf})")
        
        # Method 2: Try focused region detection if no state found
        if not state:
            logger.info("Trying focused region state detection...")
            state, state_conf = extract_state_from_plate_region(cropped, ocr, debug_name)
        
        # Method 3: Try larger context area
        if not state:
            logger.info("Trying expanded context for state detection...")
            try:
                # Expand search area significantly for state text
                context_padding = 150  # Increased padding
                context_y1 = max(0, y1 - context_padding)
                context_y2 = min(img_bgr.shape[0], y2 + context_padding)
                context_x1 = max(0, x1 - context_padding)
                context_x2 = min(img_bgr.shape[1], x2 + context_padding)
                
                context_area = img_bgr[context_y1:context_y2, context_x1:context_x2]
                context_rgb = cv2.cvtColor(context_area, cv2.COLOR_BGR2RGB)
                
                # Try with different OCR parameters
                context_results = ocr.readtext(context_rgb, paragraph=True, width_ths=0.5)
                
                # Also try preprocessing variants
                for variant in enhance_for_state_text(context_area):
                    variant_results = ocr.readtext(variant)
                    context_results.extend(variant_results)
                
                state, state_conf = detect_state_from_context(context_results)
                
                if state:
                    logger.info(f"State '{state}' found in expanded context (conf: {state_conf})")
                    
            except Exception as e:
                logger.error(f"Context state detection failed: {e}")
        
        # Method 4: Try pattern matching on the plate number
        if not state and plate_text:
            pattern_state, pattern_conf = state_classifier.classify_from_text(plate_text)
            if pattern_state:
                state = pattern_state
                state_conf = pattern_conf
                logger.info(f"State '{state}' detected from plate pattern (conf: {state_conf})")
        
        # Only save if we found a valid plate number
        if plate_text:
            logger.info(f"=== Plate {i+1} Final Result ===")
            logger.info(f"  Plate Text: '{plate_text}'")
            logger.info(f"  OCR Confidence: {confidence:.2f}")
            logger.info(f"  State: {state or 'Unknown'}")
            logger.info(f"  State Confidence: {state_conf:.2f}")
            
            plate_texts.append({
                "text": plate_text,
                "confidence": round(float(confidence), 2),
                "box": [x1, y1, x2, y2],
                "plate_image_base64": encode_image_to_base64(cropped_rgb),
                "state": state,
                "state_confidence": round(state_conf, 2) if state_conf else 0.0
            })
        else:
            logger.warning(f"No valid plate number found in detection {i+1}")

    # Save to database
    save_detections_to_db(file.filename, plate_texts)
    
    response = {
        "image": file.filename,
        "timestamp": datetime.now().isoformat(),
        "plates_detected": plate_texts
    }

    return JSONResponse(content=response)


@app.get("/api/v1/state-analytics")
def get_state_analytics():
    """Get state recognition analytics data."""
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
        
        for detection in detections:
            if detection.state and detection.state != 'UNKNOWN':
                state_counts[detection.state] = state_counts.get(detection.state, 0) + 1
                if detection.state_confidence:
                    confidence_sum += detection.state_confidence
                    confidence_count += 1
        
        # Method distribution (simplified for now)
        method_counts = {
            'pattern': states_identified * 0.7,
            'context': states_identified * 0.2,
            'visual': states_identified * 0.1
        }
        
        analytics = {
            'summary': {
                'total_detections': total_detections,
                'states_identified': states_identified,
                'identification_rate': (states_identified / total_detections * 100) if total_detections > 0 else 0,
                'average_confidence': (confidence_sum / confidence_count * 100) if confidence_count > 0 else 0,
                'unique_states': len(state_counts)
            },
            'state_distribution': state_counts,
            'method_distribution': method_counts,
            'recent_detections': []
        }
        
        # Add recent detections with state info
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
        return JSONResponse(content={"error": f"Failed to get analytics: {str(e)}"}, status_code=500)
    finally:
        db.close()