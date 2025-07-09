"""
test_ocr_simple.py
Simple test to see what OCR is detecting in your images
"""

import cv2
import easyocr
import sys
import os

def test_image(image_path):
    """Test what OCR can see in an image."""
    
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(image_path)}")
    print(f"{'='*60}\n")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load image")
        return
    
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
    
    # Initialize OCR
    print("Initializing EasyOCR...")
    ocr = easyocr.Reader(['en'], gpu=False)
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run OCR with different settings
    print("\n1. Running OCR with standard settings...")
    results1 = ocr.readtext(img_rgb)
    
    print(f"\nFound {len(results1)} text regions:")
    for i, (bbox, text, conf) in enumerate(results1):
        print(f"  {i+1}. '{text}' (confidence: {conf:.2f})")
    
    # Try with different parameters
    print("\n2. Running OCR with lower thresholds...")
    results2 = ocr.readtext(img_rgb, width_ths=0.3, height_ths=0.3, text_threshold=0.5)
    
    new_texts = []
    for bbox, text, conf in results2:
        if text not in [r[1] for r in results1]:
            new_texts.append((text, conf))
    
    if new_texts:
        print(f"\nAdditional texts found with lower thresholds:")
        for text, conf in new_texts:
            print(f"  - '{text}' (confidence: {conf:.2f})")
    else:
        print("\nNo additional texts found with lower thresholds")
    
    # Combine all texts and look for state indicators
    all_texts = [r[1].upper() for r in results1] + [r[1].upper() for r in results2]
    combined_text = ' '.join(all_texts)
    
    print(f"\n3. Looking for state indicators...")
    print(f"Combined text: {combined_text[:200]}...")
    
    # Simple state keywords
    states_found = []
    state_words = ['CALIFORNIA', 'TEXAS', 'FLORIDA', 'NEW YORK', 'NEW JERSEY', 
                   'ILLINOIS', 'PENNSYLVANIA', 'OHIO', 'GEORGIA', 'MICHIGAN',
                   'VIRGINIA', 'MASSACHUSETTS', 'ARIZONA', 'WASHINGTON', 'TENNESSEE',
                   'INDIANA', 'MISSOURI', 'MARYLAND', 'WISCONSIN', 'COLORADO',
                   'MINNESOTA', 'SOUTH CAROLINA', 'ALABAMA', 'LOUISIANA', 'KENTUCKY',
                   'OREGON', 'OKLAHOMA', 'CONNECTICUT', 'UTAH', 'IOWA', 'NEVADA',
                   'ARKANSAS', 'MISSISSIPPI', 'KANSAS', 'NEW MEXICO', 'NEBRASKA',
                   'WEST VIRGINIA', 'IDAHO', 'HAWAII', 'NEW HAMPSHIRE', 'MAINE',
                   'MONTANA', 'RHODE ISLAND', 'DELAWARE', 'SOUTH DAKOTA', 'NORTH DAKOTA',
                   'ALASKA', 'VERMONT', 'WYOMING']
    
    for state in state_words:
        if state in combined_text:
            states_found.append(state)
    
    # Also check for state codes
    state_codes = ['CA', 'TX', 'FL', 'NY', 'NJ', 'IL', 'PA', 'OH', 'GA', 'MI',
                   'VA', 'MA', 'AZ', 'WA', 'TN', 'IN', 'MO', 'MD', 'WI', 'CO',
                   'MN', 'SC', 'AL', 'LA', 'KY', 'OR', 'OK', 'CT', 'UT', 'IA',
                   'NV', 'AR', 'MS', 'KS', 'NM', 'NE', 'WV', 'ID', 'HI', 'NH',
                   'ME', 'MT', 'RI', 'DE', 'SD', 'ND', 'AK', 'VT', 'WY']
    
    for code in state_codes:
        # Look for standalone state codes
        if f' {code} ' in f' {combined_text} ' or combined_text.startswith(code + ' ') or combined_text.endswith(' ' + code):
            states_found.append(f"{code} (code)")
    
    if states_found:
        print(f"\nStates found: {', '.join(states_found)}")
    else:
        print("\nNo state indicators found")
    
    # Look for license plate patterns
    print(f"\n4. Looking for license plate patterns...")
    plate_candidates = []
    
    for bbox, text, conf in results1 + results2:
        # Clean text
        clean = text.upper().strip()
        clean = ''.join(c for c in clean if c.isalnum() or c in ' -')
        
        # Check if it looks like a plate
        if 3 <= len(clean) <= 10:
            # Count letters and numbers
            letters = sum(1 for c in clean if c.isalpha())
            numbers = sum(1 for c in clean if c.isdigit())
            
            if letters > 0 and numbers > 0:
                plate_candidates.append((clean, conf))
            elif len(clean) >= 6 and (letters >= 3 or numbers >= 3):
                plate_candidates.append((clean, conf))
    
    if plate_candidates:
        print("\nPotential license plates:")
        for plate, conf in sorted(plate_candidates, key=lambda x: x[1], reverse=True):
            print(f"  - '{plate}' (confidence: {conf:.2f})")
            # Analyze format
            format_str = ''.join('L' if c.isalpha() else 'N' if c.isdigit() else '?' for c in plate.replace(' ', ''))
            print(f"    Format: {format_str}")
    else:
        print("\nNo potential license plates found")
    
    # Save visualization
    print(f"\n5. Saving visualization...")
    vis_path = image_path.replace('.jpg', '_ocr_vis.jpg').replace('.png', '_ocr_vis.png')
    
    # Draw bounding boxes
    img_vis = img.copy()
    for bbox, text, conf in results1:
        # Draw rectangle
        pts = [(int(p[0]), int(p[1])) for p in bbox]
        cv2.polylines(img_vis, [np.array(pts)], True, (0, 255, 0), 2)
        # Add text
        cv2.putText(img_vis, f"{text} ({conf:.2f})", pts[0], 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(vis_path, img_vis)
    print(f"Saved visualization to: {vis_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_ocr_simple.py <image_path>")
        sys.exit(1)
    
    import numpy as np
    test_image(sys.argv[1])