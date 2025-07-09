# collect_plate_crops.py
"""
Script to collect and organize labeled license plate crops for training.
Run this after processing images to extract and save plate crops with labels.
"""

import os
import cv2
import json
import shutil
from datetime import datetime
from pathlib import Path
import sqlite3
import base64
from typing import Dict, List

class PlateDataCollector:
    def __init__(self, output_dir: str = "plate_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each state
        self.states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
                      'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                      'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                      'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                      'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 
                      'UNKNOWN']
        
        for state in self.states:
            (self.output_dir / state).mkdir(exist_ok=True)
        
        # Metadata file
        self.metadata_file = self.output_dir / "dataset_metadata.json"
        self.metadata = self.load_metadata()
    
    def load_metadata(self) -> Dict:
        """Load existing metadata or create new."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'total_images': 0,
            'state_counts': {state: 0 for state in self.states},
            'collection_dates': [],
            'images': []
        }
    
    def save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def extract_from_database(self, db_path: str = "detections.db"):
        """Extract plate crops from the SQLite database."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all detections with state information
        query = """
        SELECT plate_text, confidence, state, state_confidence, 
               plate_image_base64, image_name, timestamp
        FROM detections
        WHERE plate_image_base64 IS NOT NULL
        ORDER BY timestamp DESC
        """
        
        cursor.execute(query)
        detections = cursor.fetchall()
        
        print(f"Found {len(detections)} detections in database")
        
        for idx, detection in enumerate(detections):
            plate_text, conf, state, state_conf, image_b64, source_image, timestamp = detection
            
            # Skip if no valid state
            if not state or state == 'null':
                state = 'UNKNOWN'
            
            # Decode image
            try:
                image_data = base64.b64decode(image_b64)
                # Save to temporary file to read with OpenCV
                temp_path = f"temp_{idx}.jpg"
                with open(temp_path, 'wb') as f:
                    f.write(image_data)
                
                img = cv2.imread(temp_path)
                os.remove(temp_path)
                
                if img is None:
                    continue
                
                # Generate filename
                clean_text = plate_text.replace(' ', '_').replace('-', '_')
                filename = f"{state}_{clean_text}_{int(conf*100)}_{idx}.jpg"
                output_path = self.output_dir / state / filename
                
                # Save image
                cv2.imwrite(str(output_path), img)
                
                # Update metadata
                self.metadata['total_images'] += 1
                self.metadata['state_counts'][state] += 1
                self.metadata['images'].append({
                    'filename': str(output_path),
                    'plate_text': plate_text,
                    'state': state,
                    'confidence': conf,
                    'state_confidence': state_conf,
                    'source_image': source_image,
                    'timestamp': timestamp
                })
                
                print(f"Saved: {filename} (State: {state}, Conf: {conf:.2f})")
                
            except Exception as e:
                print(f"Error processing detection {idx}: {e}")
                continue
        
        conn.close()
        self.save_metadata()
        print(f"\nDataset collection complete!")
        print(f"Total images: {self.metadata['total_images']}")
        print("\nState distribution:")
        for state, count in sorted(self.metadata['state_counts'].items()):
            if count > 0:
                print(f"  {state}: {count} images")
    
    def create_train_test_split(self, test_ratio: float = 0.2):
        """Create train/test split maintaining state distribution."""
        train_dir = self.output_dir / "train"
        test_dir = self.output_dir / "test"
        train_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        
        split_info = {
            'train': {'total': 0, 'states': {}},
            'test': {'total': 0, 'states': {}}
        }
        
        for state in self.states:
            state_dir = self.output_dir / state
            if not state_dir.exists():
                continue
                
            # Create state subdirs in train/test
            (train_dir / state).mkdir(exist_ok=True)
            (test_dir / state).mkdir(exist_ok=True)
            
            # Get all images for this state
            images = list(state_dir.glob("*.jpg"))
            if not images:
                continue
            
            # Shuffle and split
            import random
            random.shuffle(images)
            split_idx = int(len(images) * (1 - test_ratio))
            
            train_images = images[:split_idx]
            test_images = images[split_idx:]
            
            # Copy to train/test directories
            for img in train_images:
                shutil.copy2(img, train_dir / state / img.name)
                split_info['train']['total'] += 1
                split_info['train']['states'][state] = split_info['train']['states'].get(state, 0) + 1
            
            for img in test_images:
                shutil.copy2(img, test_dir / state / img.name)
                split_info['test']['total'] += 1
                split_info['test']['states'][state] = split_info['test']['states'].get(state, 0) + 1
        
        # Save split info
        with open(self.output_dir / "train_test_split.json", 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"\nTrain/Test split complete!")
        print(f"Train: {split_info['train']['total']} images")
        print(f"Test: {split_info['test']['total']} images")
    
    def create_yolo_dataset(self):
        """Create YOLO format dataset for training."""
        yolo_dir = self.output_dir / "yolo_format"
        yolo_dir.mkdir(exist_ok=True)
        
        # Create images and labels directories
        (yolo_dir / "images").mkdir(exist_ok=True)
        (yolo_dir / "labels").mkdir(exist_ok=True)
        
        # Create data.yaml for YOLO
        data_yaml = {
            'path': str(yolo_dir.absolute()),
            'train': 'images',
            'val': 'images',  # You'd split this properly in practice
            'nc': len(self.states) - 1,  # Exclude UNKNOWN
            'names': [s for s in self.states if s != 'UNKNOWN']
        }
        
        with open(yolo_dir / "data.yaml", 'w') as f:
            import yaml
            yaml.dump(data_yaml, f)
        
        # Copy images and create labels
        class_mapping = {state: idx for idx, state in enumerate(data_yaml['names'])}
        
        for state in self.states:
            if state == 'UNKNOWN':
                continue
                
            state_dir = self.output_dir / state
            if not state_dir.exists():
                continue
            
            for img_path in state_dir.glob("*.jpg"):
                # Copy image
                shutil.copy2(img_path, yolo_dir / "images" / img_path.name)
                
                # Create label (using full image as bounding box for now)
                img = cv2.imread(str(img_path))
                h, w = img.shape[:2]
                
                # YOLO format: class x_center y_center width height (all normalized)
                label_content = f"{class_mapping[state]} 0.5 0.5 1.0 1.0\n"
                
                label_path = yolo_dir / "labels" / img_path.with_suffix('.txt').name
                with open(label_path, 'w') as f:
                    f.write(label_content)
        
        print(f"\nYOLO dataset created at: {yolo_dir}")

def main():
    """Main function to run the plate collection process."""
    collector = PlateDataCollector()
    
    print("=== License Plate Dataset Collection ===\n")
    
    # Extract from database
    collector.extract_from_database()
    
    # Create train/test split
    print("\nCreating train/test split...")
    collector.create_train_test_split()
    
    # Create YOLO format dataset
    print("\nCreating YOLO format dataset...")
    collector.create_yolo_dataset()
    
    # Print summary
    print("\n=== Collection Summary ===")
    print(f"Total images collected: {collector.metadata['total_images']}")
    print(f"Dataset location: {collector.output_dir}")
    
    # Check if we have enough for training
    if collector.metadata['total_images'] < 200:
        print(f"\n⚠️  Warning: You have {collector.metadata['total_images']} images.")
        print("   Target is 200+ labeled plates for good training results.")
        print("   Continue collecting more data!")
    else:
        print(f"\n✅ Great! You have {collector.metadata['total_images']} labeled plates.")
        print("   Ready for training!")

if __name__ == "__main__":
    main()