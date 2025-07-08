def preprocess_for_ocr(plate_img):
    """Enhanced preprocessing for license plate OCR."""
    import cv2
    import numpy as np
    
    # Make a copy
    img = plate_img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize for better OCR (license plates are usually wide)
    height, width = gray.shape
    if width < 300:  # Increased from 200
        scale = 300 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply morphological operations to clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # Apply bilateral filter to reduce noise while keeping edges
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Try multiple threshold methods
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
    
    # Return both for OCR to try
    return gray, thresh1, thresh2


def extract_text_with_multiple_methods(cropped_rgb, ocr):
    """Try multiple OCR methods to get the best result."""
    import cv2
    import numpy as np
    
    all_results = []
    
    # Method 1: Direct OCR on RGB
    results1 = ocr.readtext(cropped_rgb)
    all_results.extend(results1)
    
    # Method 2: OCR on grayscale
    gray = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2GRAY)
    results2 = ocr.readtext(gray)
    all_results.extend(results2)
    
    # Method 3: OCR with preprocessing
    gray, thresh1, thresh2 = preprocess_for_ocr(cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR))
    
    results3 = ocr.readtext(gray)
    all_results.extend(results3)
    
    results4 = ocr.readtext(thresh1)
    all_results.extend(results4)
    
    results5 = ocr.readtext(thresh2)
    all_results.extend(results5)
    
    # Method 4: Try inverted image
    inverted = cv2.bitwise_not(gray)
    results6 = ocr.readtext(inverted)
    all_results.extend(results6)
    
    # Deduplicate and get best results
    seen_texts = {}
    for bbox, text, conf in all_results:
        text_clean = text.strip().upper()
        if text_clean:
            if text_clean not in seen_texts or conf > seen_texts[text_clean]:
                seen_texts[text_clean] = conf
    
    # Return the text with highest confidence
    if seen_texts:
        best_text = max(seen_texts.items(), key=lambda x: x[1])
        return best_text[0], best_text[1]
    
    return None, 0.0