"""
test_state_recognition.py
Tool for testing and debugging state recognition
"""

import cv2
import easyocr
import sys
import os
from unified_state_recognition import UnifiedStateRecognizer
from plate_filter_utils import extract_plate_number
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_state_recognition(image_path: str):
    """Test state recognition on a single image."""
    
    # Initialize components
    ocr = easyocr.Reader(['en'], gpu=False)
    recognizer = UnifiedStateRecognizer()
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(image_path)}")
    print(f"{'='*60}\n")
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run OCR on full image
    print("Running OCR on full image...")
    results = ocr.readtext(img_rgb, width_ths=0.5, height_ths=0.5, paragraph=True)
    
    print("\nAll OCR detections:")
    for i, (bbox, text, conf) in enumerate(results):
        print(f"  {i+1}. '{text}' (conf: {conf:.2f})")
    
    # Extract plate number
    plate_text, plate_conf = extract_plate_number(results)
    print(f"\nExtracted plate: '{plate_text}' (conf: {plate_conf:.2f})")
    
    # Test state recognition
    print("\nTesting state recognition...")
    state_result = recognizer.recognize_state(
        plate_text=plate_text,
        ocr_results=results,
        plate_image=img,
        context_image=img
    )
    
    # Display results
    print("\n" + recognizer.get_debug_info(state_result))
    
    # Additional debugging
    print("\n--- Additional Context ---")
    
    # Check for state keywords in all text
    all_text = ' '.join([r[1] for r in results]).upper()
    print(f"\nAll text combined: {all_text[:200]}...")
    
    # Known state indicators
    state_indicators = [
        'CALIFORNIA', 'TEXAS', 'NEW YORK', 'FLORIDA', 'ILLINOIS',
        'GARDEN STATE', 'EMPIRE STATE', 'GOLDEN STATE', 'SUNSHINE STATE',
        'DMV', 'GOV', 'STATE'
    ]
    
    print("\nState indicators found:")
    for indicator in state_indicators:
        if indicator in all_text:
            print(f"  - {indicator}")
    
    # Pattern analysis
    print(f"\nPlate pattern analysis:")
    print(f"  Length: {len(plate_text)}")
    print(f"  Letters: {sum(1 for c in plate_text if c.isalpha())}")
    print(f"  Digits: {sum(1 for c in plate_text if c.isdigit())}")
    
    # Suggest improvements
    print("\n--- Suggestions ---")
    if state_result['confidence'] < 0.5:
        print("Low confidence. Suggestions:")
        print("  1. Check if plate text was extracted correctly")
        print("  2. Look for state text in dealer frames or stickers")
        print("  3. Consider the plate format pattern")
        print("  4. Check for partial OCR reads that might contain state info")


def test_directory(directory: str):
    """Test all images in a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Results summary
    results_summary = {
        'total': 0,
        'detected': 0,
        'by_state': {},
        'by_method': {},
        'failed': []
    }
    
    recognizer = UnifiedStateRecognizer()
    ocr = easyocr.Reader(['en'], gpu=False)
    
    # Process each image
    for filename in os.listdir(directory):
        if os.path.splitext(filename)[1].lower() in image_extensions:
            results_summary['total'] += 1
            
            try:
                # Load and process image
                img_path = os.path.join(directory, filename)
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Run OCR
                ocr_results = ocr.readtext(img_rgb, width_ths=0.5, height_ths=0.5)
                
                # Extract plate
                plate_text, _ = extract_plate_number(ocr_results)
                
                if plate_text:
                    # Recognize state
                    state_result = recognizer.recognize_state(
                        plate_text=plate_text,
                        ocr_results=ocr_results,
                        plate_image=img
                    )
                    
                    state_code = state_result.get('state_code')
                    method = state_result.get('method', 'none')
                    
                    if state_code:
                        results_summary['detected'] += 1
                        results_summary['by_state'][state_code] = results_summary['by_state'].get(state_code, 0) + 1
                        results_summary['by_method'][method] = results_summary['by_method'].get(method, 0) + 1
                        print(f"✓ {filename}: {state_code} ({state_result['confidence']:.0%}) via {method}")
                    else:
                        results_summary['failed'].append(filename)
                        print(f"✗ {filename}: No state detected")
                else:
                    results_summary['failed'].append(filename)
                    print(f"✗ {filename}: No plate text extracted")
                    
            except Exception as e:
                results_summary['failed'].append(filename)
                print(f"✗ {filename}: Error - {str(e)}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total images: {results_summary['total']}")
    print(f"States detected: {results_summary['detected']} ({results_summary['detected']/results_summary['total']*100:.1f}%)")
    
    print("\nBy State:")
    for state, count in sorted(results_summary['by_state'].items()):
        print(f"  {state}: {count}")
    
    print("\nBy Method:")
    for method, count in sorted(results_summary['by_method'].items()):
        print(f"  {method}: {count}")
    
    if results_summary['failed']:
        print(f"\nFailed ({len(results_summary['failed'])}):")
        for filename in results_summary['failed'][:10]:  # Show first 10
            print(f"  - {filename}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_state_recognition.py <image_path>  # Test single image")
        print("  python test_state_recognition.py <directory>   # Test all images in directory")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if os.path.isfile(path):
        test_state_recognition(path)
    elif os.path.isdir(path):
        test_directory(path)
    else:
        print(f"Error: {path} is not a valid file or directory")