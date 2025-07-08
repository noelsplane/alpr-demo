# improved_plate_detector.py
"""
Enhanced plate detection module with better false positive filtering
and improved state recognition preprocessing.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from typing import List, Dict, Tuple, Optional
import re
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedPlateDetector:
    """
    Improved license plate detector with better filtering and preprocessing.
    """
    
    def __init__(self, model_path: str = "../models/yolov8n.pt", 
                 confidence_threshold: float = 0.45,  # Lowered from 0.6
                 iou_threshold: float = 0.4):
        """
        Initialize the improved plate detector.
        
        Args:
            model_path: Path to YOLOv8 model
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IOU threshold for NMS
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.reader = easyocr.Reader(['en'], gpu=True)
        
    def validate_plate_region(self, img: np.ndarray, box: List[int]) -> bool:
        """
        Validate if a detected region is likely to be a license plate.
        
        Args:
            img: Original image
            box: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            True if region passes validation
        """
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Check aspect ratio (US plates typically 2:1 to 4:1, but being more lenient)
        aspect_ratio = width / height if height > 0 else 0
        if not (1.5 <= aspect_ratio <= 5.0):  # Widened from 1.8-4.5
            logger.debug(f"Rejected: aspect ratio {aspect_ratio:.2f}")
            return False
        
        # Check minimum size
        img_area = img.shape[0] * img.shape[1]
        box_area = width * height
        area_ratio = box_area / img_area
        
        # Plate should be at least 0.05% of image but not more than 30%
        if not (0.0005 <= area_ratio <= 0.35):  # More lenient size requirements
            logger.debug(f"Rejected: area ratio {area_ratio:.4f}")
            return False
        
        # Check if region has sufficient edge content (plates have text)
        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            edges = cv2.Canny(gray_roi, 50, 200)  # Lowered first threshold
            edge_density = np.sum(edges > 0) / edges.size
            
            if edge_density < 0.003:  # Lowered from 0.005
                logger.debug(f"Rejected: low edge density {edge_density:.3f}")
                return False
                
        return True
    
    def preprocess_plate_for_ocr(self, plate_img: np.ndarray) -> List[np.ndarray]:
        """
        Preprocess plate image with multiple techniques for better OCR.
        
        Args:
            plate_img: Cropped plate image
            
        Returns:
            List of preprocessed images to try
        """
        processed_images = []
        
        # Ensure we have a copy to work with
        img = plate_img.copy()
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # 1. Original image (color if available)
        processed_images.append(plate_img)
        
        # 2. High contrast version
        # Resize to standard height
        target_height = 150
        scale = target_height / gray.shape[0]
        new_width = int(gray.shape[1] * scale)
        resized = cv2.resize(gray, (new_width, target_height))
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)
        processed_images.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))
        
        # 3. Binary threshold version
        # Denoise first
        denoised = cv2.fastNlMeansDenoising(resized, None, 10, 7, 21)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        processed_images.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
        
        # 4. Inverted binary (sometimes helps with dark plates)
        binary_inv = cv2.bitwise_not(binary)
        processed_images.append(cv2.cvtColor(binary_inv, cv2.COLOR_GRAY2BGR))
        
        # 5. Sharpened version
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        processed_images.append(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR))
        
        return processed_images
    
    def extract_text_from_plate(self, plate_img: np.ndarray) -> Tuple[str, float, List[str]]:
        """
        Extract text from plate using multiple preprocessing methods.
        
        Args:
            plate_img: Cropped plate image
            
        Returns:
            Tuple of (best_text, confidence, all_texts)
        """
        preprocessed_images = self.preprocess_plate_for_ocr(plate_img)
        
        all_results = []
        all_texts = []
        
        # Try OCR on each preprocessed version
        for idx, processed_img in enumerate(preprocessed_images):
            try:
                results = self.reader.readtext(processed_img, detail=1)
                for bbox, text, conf in results:
                    # Filter out very short detections
                    if len(text.strip()) >= 3:
                        all_results.append((text, conf, idx))
                        all_texts.append(text)
            except Exception as e:
                logger.error(f"OCR error on variant {idx}: {e}")
                continue
        
        if not all_results:
            return "", 0.0, []
        
        # Sort by confidence
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        # Get the best result
        best_text, best_conf, best_idx = all_results[0]
        
        # Clean the text
        cleaned_text = self.clean_plate_text(best_text)
        
        logger.info(f"Best OCR result: '{cleaned_text}' (conf: {best_conf:.2f}, variant: {best_idx})")
        
        return cleaned_text, best_conf, all_texts
    
    def clean_plate_text(self, text: str) -> str:
        """
        Clean and normalize plate text.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        # Convert to uppercase and strip
        cleaned = text.upper().strip()
        
        # Remove common OCR artifacts
        # Replace similar looking characters
        replacements = {
            'O': '0',  # O -> 0 in positions that should be numbers
            'I': '1',  # I -> 1
            'S': '5',  # S -> 5
            'B': '8',  # B -> 8
            'G': '6',  # G -> 6
            'Z': '2',  # Z -> 2
        }
        
        # Smart replacement based on position
        chars = list(cleaned)
        for i, char in enumerate(chars):
            # If it looks like it should be a number position
            if i > 0 and i < len(chars) - 1:
                if chars[i-1].isdigit() or chars[i+1].isdigit():
                    if char in replacements:
                        chars[i] = replacements[char]
        
        cleaned = ''.join(chars)
        
        # Remove special characters except spaces and hyphens
        cleaned = re.sub(r'[^A-Z0-9\s\-]', '', cleaned)
        
        # Normalize spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned
    
    def detect_plates(self, image_path: str) -> List[Dict]:
        """
        Detect and validate license plates in image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of validated detections
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image: {image_path}")
            return []
        
        # Run YOLO detection with higher confidence threshold
        results = self.model.predict(
            img, 
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        validated_detections = []
        
        for r in results:
            if r.boxes is None:
                continue
                
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                
                # Validate the detection
                if not self.validate_plate_region(img, [x1, y1, x2, y2]):
                    continue
                
                # Extract plate region with padding
                padding = 10
                y1_pad = max(0, y1 - padding)
                y2_pad = min(img.shape[0], y2 + padding)
                x1_pad = max(0, x1 - padding)
                x2_pad = min(img.shape[1], x2 + padding)
                
                plate_img = img[y1_pad:y2_pad, x1_pad:x2_pad]
                
                # Extract text
                text, ocr_conf, all_texts = self.extract_text_from_plate(plate_img)
                
                # Additional validation: plate should have reasonable text
                if len(text) < 4 or len(text) > 10:
                    logger.debug(f"Rejected: invalid text length '{text}'")
                    continue
                
                # Check if text looks like a plate (has mix of letters and numbers)
                has_letters = any(c.isalpha() for c in text)
                has_numbers = any(c.isdigit() for c in text)
                
                if not (has_letters and has_numbers):
                    logger.debug(f"Rejected: no letter/number mix '{text}'")
                    continue
                
                detection = {
                    'box': [x1, y1, x2, y2],
                    'confidence': conf,
                    'text': text,
                    'ocr_confidence': ocr_conf,
                    'all_ocr_texts': all_texts,
                    'plate_image': plate_img
                }
                
                validated_detections.append(detection)
                logger.info(f"Valid plate detected: '{text}' (YOLO conf: {conf:.2f}, OCR conf: {ocr_conf:.2f})")
        
        return validated_detections


# Integration function for use with existing code
def process_image_with_improved_detection(image_path: str, 
                                        detector: Optional[ImprovedPlateDetector] = None) -> List[Dict]:
    """
    Process image with improved detection for easy integration.
    
    Args:
        image_path: Path to image
        detector: Optional pre-initialized detector
        
    Returns:
        List of detection dictionaries compatible with existing code
    """
    if detector is None:
        detector = ImprovedPlateDetector(confidence_threshold=0.65)
    
    detections = detector.detect_plates(image_path)
    
    # Format for compatibility with existing code
    formatted_detections = []
    for det in detections:
        formatted_det = {
            'text': det['text'],
            'confidence': det['ocr_confidence'],
            'box': det['box'],
            'plate_image': det['plate_image'],
            'detection_confidence': det['confidence']
        }
        formatted_detections.append(formatted_det)
    
    return formatted_detections


if __name__ == "__main__":
    # Test the improved detector
    detector = ImprovedPlateDetector(
        model_path="../models/yolov8n.pt",
        confidence_threshold=0.65  # Higher threshold to reduce false positives
    )
    
    # Test on an image
    test_image = "test_plate.jpg"
    detections = detector.detect_plates(test_image)
    
    print(f"\nFound {len(detections)} valid plates:")
    for i, det in enumerate(detections):
        print(f"\nPlate {i+1}:")
        print(f"  Text: {det['text']}")
        print(f"  Detection confidence: {det['confidence']:.2f}")
        print(f"  OCR confidence: {det['ocr_confidence']:.2f}")
        print(f"  All OCR attempts: {det['all_ocr_texts']}")
