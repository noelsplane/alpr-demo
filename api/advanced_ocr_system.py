# advanced_ocr_system.py
"""
Advanced OCR system that uses multiple OCR engines and preprocessing techniques
for improved license plate text recognition.
"""

import cv2
import numpy as np
import easyocr
import pytesseract
from PIL import Image
import re
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Container for OCR results."""
    text: str
    confidence: float
    method: str
    preprocessing: str


class AdvancedOCR:
    """
    Advanced OCR system that combines multiple OCR engines and preprocessing methods.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize the advanced OCR system.
        
        Args:
            use_gpu: Whether to use GPU for EasyOCR
        """
        # Initialize OCR engines
        self.easy_reader = easyocr.Reader(['en'], gpu=use_gpu)
        
        # Check if Tesseract is available
        self.tesseract_available = self._check_tesseract()
        
        # Common OCR mistakes mapping
        self.ocr_corrections = {
            # Letter to number corrections
            'O': '0', 'o': '0',
            'I': '1', 'i': '1', 'l': '1',
            'S': '5', 's': '5',
            'B': '8', 'b': '8',
            'G': '6', 'g': '6',
            'Z': '2', 'z': '2',
            # Number to letter corrections (context-dependent)
            '0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G', '2': 'Z'
        }
        
        # License plate patterns (for validation)
        self.plate_patterns = [
            r'^[A-Z]{1,3}[0-9]{1,4}$',
            r'^[0-9]{1,4}[A-Z]{1,3}$',
            r'^[A-Z]{1,3}[0-9]{1,4}[A-Z]{1,3}$',
            r'^[A-Z]{1,3}[\s\-]?[0-9]{1,4}$',
            r'^[0-9]{1,4}[\s\-]?[A-Z]{1,3}$',
        ]
    
    def _check_tesseract(self) -> bool:
        """Check if Tesseract is installed and available."""
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR is available")
            return True
        except Exception:
            logger.warning("Tesseract OCR not available. Install with: sudo apt-get install tesseract-ocr")
            return False
    
    def preprocess_image(self, image: np.ndarray, method: str = "standard") -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: Input image
            method: Preprocessing method to use
            
        Returns:
            Preprocessed image
        """
        # Ensure we're working with a copy
        img = image.copy()
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        if method == "standard":
            # Standard preprocessing
            # Resize to optimal height
            height = gray.shape[0]
            if height < 50:
                scale = 50 / height
                width = int(gray.shape[1] * scale)
                gray = cv2.resize(gray, (width, 50), interpolation=cv2.INTER_CUBIC)
            
            # Apply bilateral filter for noise reduction
            denoised = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            return enhanced
        
        elif method == "binary":
            # Binary threshold preprocessing
            # Resize first
            if gray.shape[0] < 50:
                scale = 50 / gray.shape[0]
                width = int(gray.shape[1] * scale)
                gray = cv2.resize(gray, (width, 50), interpolation=cv2.INTER_CUBIC)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive threshold
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            return binary
        
        elif method == "morph":
            # Morphological preprocessing
            # Resize
            if gray.shape[0] < 50:
                scale = 50 / gray.shape[0]
                width = int(gray.shape[1] * scale)
                gray = cv2.resize(gray, (width, 50), interpolation=cv2.INTER_CUBIC)
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Enhance contrast
            enhanced = cv2.equalizeHist(morph)
            
            return enhanced
        
        elif method == "super_resolution":
            # Simple super-resolution using interpolation
            # Scale up by 2x
            height, width = gray.shape
            upscaled = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
            
            # Apply sharpening
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            sharpened = cv2.filter2D(upscaled, -1, kernel)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(sharpened, None, 10, 7, 21)
            
            return denoised
        
        else:
            return gray
    
    def apply_easyocr(self, image: np.ndarray, preprocess_method: str) -> List[OCRResult]:
        """Apply EasyOCR to the image."""
        results = []
        
        try:
            # Ensure image is in RGB format for EasyOCR
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image
            
            # Run EasyOCR
            ocr_results = self.easy_reader.readtext(image_rgb, detail=1)
            
            for bbox, text, conf in ocr_results:
                if conf > 0.3 and len(text.strip()) >= 3:
                    results.append(OCRResult(
                        text=text.strip(),
                        confidence=conf,
                        method="easyocr",
                        preprocessing=preprocess_method
                    ))
                    
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
        
        return results
    
    def apply_tesseract(self, image: np.ndarray, preprocess_method: str) -> List[OCRResult]:
        """Apply Tesseract OCR to the image."""
        if not self.tesseract_available:
            return []
        
        results = []
        
        try:
            # Configure Tesseract for license plates
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            
            # Run Tesseract with confidence scores
            data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Extract text with confidence
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                if conf > 30 and len(text) >= 3:
                    results.append(OCRResult(
                        text=text,
                        confidence=conf / 100.0,
                        method="tesseract",
                        preprocessing=preprocess_method
                    ))
            
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
        
        return results
    
    def smart_text_correction(self, text: str, position_hints: Optional[List[str]] = None) -> str:
        """
        Apply smart corrections based on license plate patterns.
        
        Args:
            text: Input text
            position_hints: Hints about character positions (e.g., ['letter', 'number', 'number'])
            
        Returns:
            Corrected text
        """
        if not text:
            return text
        
        # If we have position hints, use them
        if position_hints and len(position_hints) == len(text):
            corrected = []
            for char, hint in zip(text, position_hints):
                if hint == 'letter' and char in '0123456789':
                    # Convert number to likely letter
                    corrected.append(self.ocr_corrections.get(char, char))
                elif hint == 'number' and char in 'OISBGZoisgbz':
                    # Convert letter to likely number
                    corrected.append(self.ocr_corrections.get(char.upper(), char))
                else:
                    corrected.append(char)
            return ''.join(corrected)
        
        # Otherwise, try to detect pattern automatically
        # Check if text matches common patterns
        text_upper = text.upper()
        
        # Try different correction strategies
        candidates = [text_upper]
        
        # Strategy 1: Assume first part is letters, second part is numbers
        for i in range(1, len(text_upper)):
            candidate = text_upper[:i] + text_upper[i:]
            # Correct first part as letters
            first_part = ''.join([self.ocr_corrections.get(c, c) if c in '0156' else c for c in candidate[:i]])
            # Correct second part as numbers
            second_part = ''.join([self.ocr_corrections.get(c, c) if c in 'OISBGZ' else c for c in candidate[i:]])
            candidates.append(first_part + second_part)
        
        # Find the candidate that best matches known patterns
        for candidate in candidates:
            for pattern in self.plate_patterns:
                if re.match(pattern, candidate):
                    return candidate
        
        # If no pattern matches, return the original uppercase text
        return text_upper
    
    def merge_ocr_results(self, results: List[OCRResult]) -> Tuple[str, float]:
        """
        Merge multiple OCR results to get the best text.
        
        Args:
            results: List of OCR results
            
        Returns:
            Tuple of (best_text, confidence)
        """
        if not results:
            return "", 0.0
        
        # Group similar texts
        text_groups = {}
        for result in results:
            # Normalize text for grouping
            normalized = result.text.upper().replace(' ', '').replace('-', '')
            
            if normalized not in text_groups:
                text_groups[normalized] = []
            text_groups[normalized].append(result)
        
        # Find the most common text with highest average confidence
        best_text = ""
        best_score = 0.0
        
        for normalized_text, group in text_groups.items():
            # Calculate score based on frequency and average confidence
            avg_confidence = sum(r.confidence for r in group) / len(group)
            frequency_bonus = min(len(group) * 0.1, 0.5)  # Up to 0.5 bonus for frequency
            score = avg_confidence + frequency_bonus
            
            if score > best_score:
                best_score = score
                # Use the highest confidence version of this text
                best_text = max(group, key=lambda r: r.confidence).text
        
        return best_text, min(best_score, 1.0)
    
    def extract_plate_text(self, image: np.ndarray, debug: bool = False) -> Dict:
        """
        Extract license plate text using multiple OCR methods.
        
        Args:
            image: License plate image
            debug: Whether to return debug information
            
        Returns:
            Dictionary with results
        """
        all_results = []
        debug_info = []
        
        # Preprocessing methods to try
        preprocess_methods = ["standard", "binary", "morph", "super_resolution"]
        
        for method in preprocess_methods:
            # Preprocess image
            processed = self.preprocess_image(image, method)
            
            # Apply EasyOCR
            easy_results = self.apply_easyocr(processed, method)
            all_results.extend(easy_results)
            
            # Apply Tesseract
            tess_results = self.apply_tesseract(processed, method)
            all_results.extend(tess_results)
            
            if debug:
                debug_info.append({
                    "method": method,
                    "easyocr_results": [(r.text, r.confidence) for r in easy_results],
                    "tesseract_results": [(r.text, r.confidence) for r in tess_results]
                })
        
        # Merge results
        best_text, confidence = self.merge_ocr_results(all_results)
        
        # Apply smart corrections
        corrected_text = self.smart_text_correction(best_text)
        
        # Extract all unique texts for state recognition
        all_texts = list(set([r.text.upper() for r in all_results]))
        
        result = {
            "text": corrected_text,
            "confidence": confidence,
            "all_texts": all_texts,
            "total_attempts": len(all_results)
        }
        
        if debug:
            result["debug_info"] = debug_info
            result["all_results"] = [(r.text, r.confidence, r.method, r.preprocessing) for r in all_results]
        
        return result


class EnhancedStateRecognizer:
    """
    Enhanced state recognizer that works with the advanced OCR system.
    """
    
    def __init__(self):
        # State patterns and names from original
        self.state_patterns = self._load_state_patterns()
        self.state_names = self._load_state_names()
        self.state_keywords = self._load_state_keywords()
    
    def _load_state_patterns(self) -> Dict[str, List[str]]:
        """Load state-specific license plate patterns."""
        # This is a subset - you should include all patterns
        return {
            'CA': [r'^[0-9][A-Z]{3}[0-9]{3}$', r'^[A-Z]{3}[0-9]{4}$'],
            'TX': [r'^[A-Z]{3}[0-9]{4}$', r'^[A-Z]{2}[0-9]{5}$'],
            'NY': [r'^[A-Z]{3}[0-9]{4}$', r'^[A-Z]{3}[\s\-]?[0-9]{4}$'],
            'FL': [r'^[A-Z]{3}[0-9]{3}$', r'^[A-Z]{4}[0-9]{2}$'],
            # Add more states...
        }
    
    def _load_state_names(self) -> Dict[str, str]:
        """Load state code to name mapping."""
        return {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
            'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
            'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
            'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
            'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
            'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
            'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
            'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
            'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
            'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
            'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'Washington D.C.'
        }
    
    def _load_state_keywords(self) -> Dict[str, List[str]]:
        """Load keywords that might appear on state plates."""
        return {
            'FL': ['SUNSHINE', 'FLORIDA'],
            'NY': ['EMPIRE', 'NEW YORK', 'EXCELSIOR'],
            'CA': ['CALIFORNIA', 'GOLDEN STATE'],
            'TX': ['TEXAS', 'LONE STAR'],
            'GA': ['GEORGIA', 'PEACH STATE'],
            # Add more keywords...
        }
    
    def recognize_state_from_ocr_results(self, ocr_result: Dict) -> Optional[Dict[str, str]]:
        """
        Recognize state from advanced OCR results.
        
        Args:
            ocr_result: Result from AdvancedOCR.extract_plate_text()
            
        Returns:
            State information or None
        """
        # Check main text against patterns
        main_text = ocr_result.get('text', '')
        for state_code, patterns in self.state_patterns.items():
            for pattern in patterns:
                if re.match(pattern, main_text):
                    return {
                        'code': state_code,
                        'name': self.state_names[state_code],
                        'confidence': 'high',
                        'method': 'pattern'
                    }
        
        # Check all texts for state keywords
        all_texts = ocr_result.get('all_texts', [])
        combined_text = ' '.join(all_texts).upper()
        
        # Look for state codes
        for state_code in self.state_names:
            if re.search(r'\b' + state_code + r'\b', combined_text):
                return {
                    'code': state_code,
                    'name': self.state_names[state_code],
                    'confidence': 'high',
                    'method': 'code'
                }
        
        # Look for state keywords
        for state_code, keywords in self.state_keywords.items():
            for keyword in keywords:
                if keyword in combined_text:
                    return {
                        'code': state_code,
                        'name': self.state_names[state_code],
                        'confidence': 'medium',
                        'method': 'keyword'
                    }
        
        return None


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python advanced_ocr_system.py <image_path>")
        sys.exit(1)
    
    # Initialize system
    ocr_system = AdvancedOCR(use_gpu=True)
    state_recognizer = EnhancedStateRecognizer()
    
    # Read image
    image = cv2.imread(sys.argv[1])
    if image is None:
        print(f"Error: Could not read image {sys.argv[1]}")
        sys.exit(1)
    
    # Extract text
    print("Extracting text from license plate...")
    result = ocr_system.extract_plate_text(image, debug=True)
    
    print(f"\nBest text: {result['text']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Total OCR attempts: {result['total_attempts']}")
    print(f"\nAll unique texts found: {result['all_texts']}")
    
    # Recognize state
    state_info = state_recognizer.recognize_state_from_ocr_results(result)
    if state_info:
        print(f"\nState detected: {state_info['name']} ({state_info['code']})")
        print(f"Detection method: {state_info['method']}")
        print(f"Confidence: {state_info['confidence']}")
    else:
        print("\nNo state detected")
    
    # Show debug info
    if 'debug_info' in result:
        print("\n=== Debug Information ===")
        for info in result['debug_info']:
            print(f"\nPreprocessing: {info['method']}")
            print(f"  EasyOCR: {info['easyocr_results']}")
            print(f"  Tesseract: {info['tesseract_results']}")