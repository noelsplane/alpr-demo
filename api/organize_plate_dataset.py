# organize_plate_dataset.py
"""
Organize downloaded license plate images into proper structure.
This script processes raw images and extracts plate crops.
"""

import os
import cv2
import shutil
from pathlib import Path
from ultralytics import YOLO
import easyocr
from state_patterns import StatePatternMatcher

class PlateDatasetOrganizer:
    def __init__(self, source_dir="downloaded_plates", output_dir="plate_dataset"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.plate_model = self._load_plate_model()
        self.ocr = easyocr.Reader(['en'], gpu=False)
        self.state_matcher = StatePatternMatcher()
        
        # Create state directories
        self.states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
                      'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                      'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                      'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                      'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 
                      'UNKNOWN']
        
        for state in self.states:
            (self.output_dir / state).mkdir(exist_ok=True)
    
    def _load_plate_model(self):
        """Load the license plate detection model."""
        model_path = "../models/license_plate_yolov8.pt"
        if os.path.exists(model_path):
            return YOLO(model_path)
        else:
            print("Using general YOLO model - results may vary")
            return YOLO("yolov8n.pt")
    
    def process_image(self, image_path):
        """Process a single image to extract license plates."""
        img = cv2.imread(str(image_path))
        if img is None:
            return []
        
        # Detect plates
        results = self.plate_model.predict(img, conf=0.25)
        plates = []
        
        for r in results:
            if r.boxes is None:
                continue
                
            boxes = r.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Add padding
                padding = 10
                y1 = max(0, y1 - padding)
                y2 = min(img.shape[0], y2 + padding)
                x1 = max(0, x1 - padding)
                x2 = min(img.shape[1], x2 + padding)
                
                # Crop plate
                plate_img = img[y1:y2, x1:x2]
                
                if plate_img.shape[0] < 20 or plate_img.shape[1] < 50:
                    continue
                
                # OCR the plate
                plate_text = self._ocr_plate(plate_img)
                
                # Detect state
                state = 'UNKNOWN'
                if plate_text:
                    detected_state, confidence = self.state_matcher.extract_state_from_text(plate_text)
                    if detected_state and confidence > 0.5:
                        state = detected_state
                
                plates.append({
                    'image': plate_img,
                    'text': plate_text,
                    'state': state,
                    'source': image_path.name
                })
        
        return plates
    
    def _ocr_plate(self, plate_img):
        """Extract text from plate image."""
        try:
            results = self.ocr.readtext(plate_img)
            if results:
                # Get the text with highest confidence
                best_text = max(results, key=lambda x: x[2] if len(x) > 2 else 0)
                return best_text[1] if len(best_text) > 1 else ""
        except:
            pass
        return ""
    
    def organize_dataset(self):
        """Process all images and organize by state."""
        image_files = list(self.source_dir.glob("*.jpg")) + \
                     list(self.source_dir.glob("*.png")) + \
                     list(self.source_dir.glob("*.jpeg"))
        
        print(f"Found {len(image_files)} images to process")
        
        stats = {state: 0 for state in self.states}
        total_plates = 0
        
        for i, img_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {img_path.name}")
            
            plates = self.process_image(img_path)
            
            for j, plate in enumerate(plates):
                # Save plate image
                state = plate['state']
                filename = f"{state}_{plate['text']}_{i}_{j}.jpg"
                output_path = self.output_dir / state / filename
                
                cv2.imwrite(str(output_path), plate['image'])
                stats[state] += 1
                total_plates += 1
        
        # Print statistics
        print("\n" + "="*50)
        print(f"Dataset Organization Complete!")
        print(f"Total plates extracted: {total_plates}")
        print("\nPlates per state:")
        for state, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  {state}: {count} plates")
        print("="*50)
    
    def import_from_existing_db(self):
        """Import plates from your existing database."""
        import sqlite3
        import base64
        from io import BytesIO
        import numpy as np
        
        db_path = "detections.db"
        if not os.path.exists(db_path):
            print(f"Database {db_path} not found")
            return
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        query = """
        SELECT plate_text, state, plate_image_base64 
        FROM detections 
        WHERE plate_image_base64 IS NOT NULL
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        print(f"Importing {len(rows)} plates from database...")
        
        for i, (text, state, img_base64) in enumerate(rows):
            try:
                # Decode base64 image
                img_data = base64.b64decode(img_base64)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    continue
                
                # Determine state
                if not state or state == 'null':
                    state = 'UNKNOWN'
                
                # Save image
                filename = f"{state}_{text}_{i}.jpg"
                output_path = self.output_dir / state / filename
                cv2.imwrite(str(output_path), img)
                
            except Exception as e:
                print(f"Error processing row {i}: {e}")
        
        conn.close()
        print("Database import complete!")

def download_sample_dataset():
    """Download a sample dataset to get started."""
    import urllib.request
    import zipfile
    
    print("Downloading sample license plate dataset...")
    
    # Example: Download from a public source
    urls = [
        ("https://github.com/winter2897/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano/raw/main/doc/dataset.zip", "sample_plates.zip"),
    ]
    
    for url, filename in urls:
        try:
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            
            # Extract
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall("downloaded_plates")
            
            os.remove(filename)
            print(f"Extracted to downloaded_plates/")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

if __name__ == "__main__":
    print("License Plate Dataset Organizer")
    print("="*50)
    
    # Create organizer
    organizer = PlateDatasetOrganizer()

    
    # Option 2: Process downloaded images
    if organizer.source_dir.exists():
        print(f"\n2. Processing images from {organizer.source_dir}...")
        organizer.organize_dataset()
    else:
        print(f"\n2. No images found in {organizer.source_dir}")
        print("   Download some images first or use option 3")
    
    # Option 3: Download sample dataset
    response = input("\nDownload sample dataset? (y/n): ")
    if response.lower() == 'y':
        download_sample_dataset()
        organizer.organize_dataset()
    
    print(f"\nDataset ready in: {organizer.output_dir}")
    print("Run collect_plate_crops.py to create training dataset")