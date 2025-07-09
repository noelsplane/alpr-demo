"""
collect_state_data.py
Tool for collecting and organizing state training data
"""

import os
import cv2
import json
import shutil
from datetime import datetime
from typing import Dict, List
import hashlib

class StateDataCollector:
    """Collect and organize license plate data by state."""
    
    def __init__(self, base_dir: str = "data/state_training"):
        self.base_dir = base_dir
        self.metadata_file = os.path.join(base_dir, "metadata.json")
        self.stats_file = os.path.join(base_dir, "stats.json")
        
        # Create directory structure
        self._setup_directories()
        
        # Load existing metadata
        self.metadata = self._load_metadata()
        
    def _setup_directories(self):
        """Create directory structure for all states."""
        states = [
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
        ]
        
        os.makedirs(self.base_dir, exist_ok=True)
        
        for state in states:
            state_dir = os.path.join(self.base_dir, state)
            os.makedirs(state_dir, exist_ok=True)
            
            # Create train/val/test splits
            for split in ['train', 'val', 'test']:
                os.makedirs(os.path.join(state_dir, split), exist_ok=True)
    
    def _load_metadata(self) -> Dict:
        """Load existing metadata."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def add_plate(self, image_path: str, plate_text: str, state_code: str, 
                  confidence: float = 1.0, source: str = "manual"):
        """
        Add a plate to the dataset.
        
        Args:
            image_path: Path to the plate image
            plate_text: OCR text of the plate
            state_code: Two-letter state code
            confidence: Confidence in the label (0-1)
            source: Source of the annotation
        """
        # Validate state code
        if not os.path.exists(os.path.join(self.base_dir, state_code)):
            print(f"Error: Invalid state code {state_code}")
            return False
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image {image_path}")
            return False
        
        # Generate unique ID
        img_hash = hashlib.md5(img.tobytes()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = f"{state_code}_{timestamp}_{img_hash}"
        
        # Determine split (80% train, 10% val, 10% test)
        import random
        rand = random.random()
        if rand < 0.8:
            split = 'train'
        elif rand < 0.9:
            split = 'val'
        else:
            split = 'test'
        
        # Save image
        output_filename = f"{unique_id}.jpg"
        output_path = os.path.join(self.base_dir, state_code, split, output_filename)
        cv2.imwrite(output_path, img)
        
        # Update metadata
        self.metadata[unique_id] = {
            'state': state_code,
            'plate_text': plate_text,
            'confidence': confidence,
            'source': source,
            'split': split,
            'timestamp': timestamp,
            'original_path': image_path,
            'filename': output_filename
        }
        
        self._save_metadata()
        print(f"Added {unique_id} to {state_code}/{split}")
        return True
    
    def import_from_detections(self, db_path: str = "detections.db"):
        """Import plates from existing detection database."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from models import PlateDetection
        
        engine = create_engine(f'sqlite:///{db_path}')
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            # Query all detections with state information
            detections = session.query(PlateDetection).filter(
                PlateDetection.state != None,
                PlateDetection.state_confidence > 0.7
            ).all()
            
            print(f"Found {len(detections)} high-confidence state detections")
            
            for det in detections:
                # Decode plate image
                import base64
                from io import BytesIO
                from PIL import Image
                import numpy as np
                
                # Decode base64 image
                img_data = base64.b64decode(det.plate_image_base64)
                img_pil = Image.open(BytesIO(img_data))
                img_np = np.array(img_pil)
                
                # Convert RGB to BGR for OpenCV
                if len(img_np.shape) == 3:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                # Save to temporary file
                temp_path = f"/tmp/plate_{det.id}.jpg"
                cv2.imwrite(temp_path, img_np)
                
                # Add to dataset
                self.add_plate(
                    temp_path,
                    det.plate_text,
                    det.state,
                    det.state_confidence,
                    source="database"
                )
                
                # Clean up
                os.remove(temp_path)
                
        finally:
            session.close()
    
    def generate_stats(self):
        """Generate statistics about the dataset."""
        stats = {
            'total_images': len(self.metadata),
            'by_state': {},
            'by_split': {'train': 0, 'val': 0, 'test': 0},
            'by_source': {},
            'avg_confidence': 0.0
        }
        
        total_conf = 0.0
        
        for item_id, item_data in self.metadata.items():
            state = item_data['state']
            split = item_data['split']
            source = item_data.get('source', 'unknown')
            confidence = item_data.get('confidence', 1.0)
            
            # Update counts
            stats['by_state'][state] = stats['by_state'].get(state, 0) + 1
            stats['by_split'][split] += 1
            stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
            total_conf += confidence
        
        if stats['total_images'] > 0:
            stats['avg_confidence'] = total_conf / stats['total_images']
        
        # Save stats
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print summary
        print("\n=== Dataset Statistics ===")
        print(f"Total images: {stats['total_images']}")
        print(f"Average confidence: {stats['avg_confidence']:.2%}")
        
        print("\nBy State (top 10):")
        sorted_states = sorted(stats['by_state'].items(), key=lambda x: x[1], reverse=True)
        for state, count in sorted_states[:10]:
            print(f"  {state}: {count}")
        
        print("\nBy Split:")
        for split, count in stats['by_split'].items():
            print(f"  {split}: {count}")
        
        print("\nStates needing more data (<50 samples):")
        for state, count in sorted(stats['by_state'].items()):
            if count < 50:
                print(f"  {state}: {count}")
    
    def create_yolo_dataset(self, output_dir: str = "data/yolo_state_dataset"):
        """Create a YOLO-format dataset for state classification."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create class mapping
        states = sorted(set(item['state'] for item in self.metadata.values()))
        class_mapping = {state: idx for idx, state in enumerate(states)}
        
        # Save class names
        with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
            for state in states:
                f.write(f"{state}\n")
        
        # Create dataset yaml
        yaml_content = f"""
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test: images/test

nc: {len(states)}
names: {states}
"""
        with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
            f.write(yaml_content)
        
        # Copy images and create labels
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
        
        for item_id, item_data in self.metadata.items():
            state = item_data['state']
            split = item_data['split']
            filename = item_data['filename']
            
            # Copy image
            src_path = os.path.join(self.base_dir, state, split, filename)
            dst_path = os.path.join(output_dir, 'images', split, filename)
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                
                # Create label file (classification format)
                label_path = os.path.join(output_dir, 'labels', split, 
                                        filename.replace('.jpg', '.txt'))
                with open(label_path, 'w') as f:
                    f.write(f"{class_mapping[state]}\n")
        
        print(f"Created YOLO dataset at {output_dir}")


def main():
    """Example usage."""
    collector = StateDataCollector()
    
    # Import from existing database
    print("Importing from database...")
    collector.import_from_detections()
    
    # Generate statistics
    collector.generate_stats()
    
    # Create YOLO dataset
    # collector.create_yolo_dataset()


if __name__ == "__main__":
    main()