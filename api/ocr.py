from paddleocr import PaddleOCR
import pytesseract
import numpy as np
import cv2
from typing import Tuple

_paddle = PaddleOCR(lang='en', use_angle_cls=False, show_log=False)

def read_plate(crop: np.ndarray) -> Tuple[str, float]:
    """
    Read the plate text using PaddleOCR and Tesseract OCR.
    
    Args:
        crop: Cropped image of the plate.
        
    Returns:
        Tuple containing the detected plate text and confidence score.
    """
    # Use PaddleOCR for initial detection
    result = _paddle.predict(crop, det=False, use_angle_cls=False)
    
    if result and result[0]:
        text = result[0][0][0]
        conf = float(result[0][0][1])
        return text, conf
    

    text = pytesseract.image_to_string(crop, config="--psm 7").strip()
    return text, 0.40
