import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging

logger = logging.getLogger(__name__)

def super_resolution_plate(img, scale=2):
    """Apply super resolution to make text clearer."""
    height, width = img.shape[:2]
    new_width = width * scale
    new_height = height * scale
    
    # Use INTER_CUBIC for upscaling
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply unsharp mask
    gaussian = cv2.GaussianBlur(resized, (0, 0), 2.0)
    unsharp = cv2.addWeighted(resized, 1.5, gaussian, -0.5, 0)
    
    return unsharp

def correct_skew(image):
    """Correct skew in license plate image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    
    if lines is not None:
        # Calculate the average angle
        angles = []
        for rho, theta in lines[:, 0]:
            angle = (theta * 180 / np.pi) - 90
            if -45 < angle < 45:  # Filter out extreme angles
                angles.append(angle)
        
        if angles:
            median_angle = np.median(angles)
            
            # Rotate image to correct skew
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return corrected
    
    return image

def preprocess_for_ocr_advanced(img):
    """Advanced preprocessing specifically for license plates."""
    variants = []
    
    # 1. Super resolution
    super_res = super_resolution_plate(img, scale=2)
    
    # 2. Correct skew
    corrected = correct_skew(super_res)
    
    # Convert to grayscale if needed
    if len(corrected.shape) == 3:
        gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    else:
        gray = corrected
    
    # 3. Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # 4. CLAHE for contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(morph)
    variants.append(enhanced)
    
    # 5. Binary with different thresholds
    _, binary1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(binary1)
    
    # 6. Adaptive threshold
    adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    variants.append(adaptive)
    
    # 7. Inverse for dark plates
    inv_binary = cv2.bitwise_not(binary1)
    variants.append(inv_binary)
    
    return variants

def validate_plate_text(text):
    """Validate if the text looks like a real license plate."""
    # Remove spaces for validation
    compact = text.replace(" ", "").replace("-", "")
    
    # Check minimum length
    if len(compact) < 4 or len(compact) > 10:
        return False
    
    # Must have both letters and numbers
    has_letter = any(c.isalpha() for c in compact)
    has_digit = any(c.isdigit() for c in compact)
    
    return has_letter and has_digit

def fix_common_ocr_errors(text):
    """Fix common OCR errors specific to license plates."""
    # Common substitutions in license plates
    fixes = {
        # Letters often misread as numbers
        '0': 'O', 'O': '0',  # Context-dependent
        '1': 'I', 'I': '1',  # Context-dependent
        '5': 'S', 'S': '5',  # Context-dependent
        '8': 'B', 'B': '8',  # Context-dependent
        '6': 'G', 'G': '6',  # Context-dependent
        '2': 'Z', 'Z': '2',  # Context-dependent
        
        # Common errors
        '|': 'I',
        '!': 'I',
        '[': 'I',
        ']': 'I',
        '(': 'C',
        ')': 'D',
        '{': 'C',
        '}': 'D',
    }
    
    # Split into segments to apply context-aware fixes
    parts = text.split()
    fixed_parts = []
    
    for part in parts:
        # If it looks like it should be all letters (beginning of plate)
        if len(fixed_parts) == 0 and len(part) <= 3:
            # Likely state code - favor letters
            fixed = part
            for num, letter in [('0', 'O'), ('1', 'I'), ('5', 'S'), ('8', 'B')]:
                fixed = fixed.replace(num, letter)
        # If it looks like it should be numbers
        elif all(c.isdigit() or c in '0158' for c in part):
            # Favor keeping as numbers
            fixed = part
        else:
            # Mixed - use context
            fixed = part
            
        fixed_parts.append(fixed)
    
    return ' '.join(fixed_parts)
