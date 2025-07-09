"""
test_full_pipeline.py
Test the complete ALPR pipeline with state recognition
"""

import cv2
import easyocr
from unified_state_recognition import UnifiedStateRecognizer
from plate_filter_utils import extract_plate_number
import sys
import os

def test_alpr_pipeline(image_path):
    """Test the full ALPR pipeline on an image."""
    
    print(f"\n{'='*60}")
    print(f"Testing Full Pipeline: {os.path.basename(image_path)}")
    print(f"{'='*60}\n")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Cannot load image")
        return
    
    # Initialize components
    print("Initializing components...")
    ocr = easyocr.Reader(['en'], gpu=False)
    state_recognizer = UnifiedStateRecognizer()
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Step 1: Run OCR
    print("\n1. Running OCR...")
    ocr_results = ocr.readtext(img_rgb, width_ths=0.7, height_ths=0.7)
    
    print(f"   Found {len(ocr_results)} text regions")
    for i, (bbox, text, conf) in enumerate(ocr_results):
        print(f"   {i+1}. '{text}' (conf: {conf:.2f})")
    
    # Step 2: Extract plate number
    print("\n2. Extracting plate number...")
    plate_text, plate_conf = extract_plate_number(ocr_results)
    print(f"   Plate: '{plate_text}' (conf: {plate_conf:.2f})")
    
    if not plate_text:
        print("   ERROR: No plate text extracted!")
        return
    
    # Step 3: Recognize state
    print("\n3. Running state recognition...")
    
    # Get extended context (simulate what main.py does)
    context_results = ocr.readtext(img_rgb, width_ths=0.5, height_ths=0.5, paragraph=True)
    all_ocr_results = ocr_results + context_results
    
    state_result = state_recognizer.recognize_state(
        plate_text=plate_text,
        ocr_results=all_ocr_results,
        plate_image=img,
        context_image=img
    )
    
    print(f"\n4. Results:")
    print(f"   Plate: {plate_text}")
    print(f"   State: {state_result.get('state_code', 'Unknown')}")
    print(f"   State Name: {state_result.get('state_name', 'Unknown')}")
    print(f"   Confidence: {state_result.get('confidence', 0):.2%}")
    print(f"   Method: {state_result.get('method', 'none')}")
    
    if state_result.get('confidence_breakdown'):
        print(f"\n   Confidence Breakdown:")
        for method, details in state_result['confidence_breakdown'].items():
            if details:
                print(f"     {method}: {details}")
    
    # Final summary
    print(f"\n5. Summary:")
    if state_result.get('state_code'):
        print(f"   ✓ Successfully identified: {plate_text} from {state_result['state_name']}")
    else:
        print(f"   ✗ Could not identify state for: {plate_text}")
        print(f"   Suggestion: Add pattern for format: ", end="")
        for c in plate_text:
            print("L" if c.isalpha() else "N" if c.isdigit() else "?", end="")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_alpr_pipeline(sys.argv[1])
    else:
        # Test on your sample image
        test_path = "../data/test/car3.jpg"
        if os.path.exists(test_path):
            test_alpr_pipeline(test_path)
        else:
            print("Usage: python test_full_pipeline.py <image_path>")