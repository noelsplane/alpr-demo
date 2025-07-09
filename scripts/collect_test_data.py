"""
collect_test_data.py
Quickly collect test data from your existing database detections
"""

import os
import base64
from io import BytesIO
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import PlateDetection
import json

def collect_from_database(output_dir="test_data", min_confidence=0.7):
    """Extract test images from existing database."""
    
    # Connect to database
    engine = create_engine('sqlite:///detections.db')
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics
    stats = {
        'total_processed': 0,
        'by_state': {},
        'no_state': 0
    }
    
    try:
        # Query all detections with states
        detections = session.query(PlateDetection).filter(
            PlateDetection.state != None,
            PlateDetection.state_confidence >= min_confidence
        ).all()
        
        print(f"Found {len(detections)} detections with state confidence >= {min_confidence}")
        
        for det in detections:
            if not det.state or not det.plate_image_base64:
                continue
            
            # Create state directory
            state_dir = os.path.join(output_dir, det.state)
            os.makedirs(state_dir, exist_ok=True)
            
            # Decode and save image
            try:
                img_data = base64.b64decode(det.plate_image_base64)
                img = Image.open(BytesIO(img_data))
                
                # Generate filename
                filename = f"{det.state}_{det.plate_text}_{det.id}.jpg"
                # Clean filename
                filename = "".join(c if c.isalnum() or c in '_-' else '_' for c in filename)
                
                filepath = os.path.join(state_dir, filename)
                img.save(filepath)
                
                stats['total_processed'] += 1
                stats['by_state'][det.state] = stats['by_state'].get(det.state, 0) + 1
                
                print(f"Saved: {filepath}")
                
            except Exception as e:
                print(f"Error processing detection {det.id}: {e}")
        
        # Also get detections without states for testing
        no_state = session.query(PlateDetection).filter(
            (PlateDetection.state == None) | (PlateDetection.state == '')
        ).limit(50).all()
        
        if no_state:
            unknown_dir = os.path.join(output_dir, 'UNKNOWN')
            os.makedirs(unknown_dir, exist_ok=True)
            
            for det in no_state:
                if det.plate_image_base64:
                    try:
                        img_data = base64.b64decode(det.plate_image_base64)
                        img = Image.open(BytesIO(img_data))
                        
                        filename = f"unknown_{det.plate_text}_{det.id}.jpg"
                        filename = "".join(c if c.isalnum() or c in '_-' else '_' for c in filename)
                        
                        filepath = os.path.join(unknown_dir, filename)
                        img.save(filepath)
                        
                        stats['no_state'] += 1
                        
                    except Exception as e:
                        print(f"Error: {e}")
        
    finally:
        session.close()
    
    # Print summary
    print(f"\n=== Collection Summary ===")
    print(f"Total images collected: {stats['total_processed']}")
    print(f"Unknown state images: {stats['no_state']}")
    print(f"\nBy State:")
    for state, count in sorted(stats['by_state'].items()):
        print(f"  {state}: {count}")
    
    # Save stats
    with open(os.path.join(output_dir, 'collection_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def download_test_images():
    """Download sample license plate images from public datasets."""
    print("\nTo get more test data, you can:")
    print("1. Take photos of license plates you see (with permission)")
    print("2. Use public datasets:")
    print("   - https://github.com/openalpr/benchmarks")
    print("   - https://www.kaggle.com/datasets/andrewmvd/car-plate-detection")
    print("3. Search for 'license plate dataset' on Kaggle")
    print("\nMake sure to organize them by state in the test_data folder")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download_test_images()
    else:
        # Collect from your database
        stats = collect_from_database()
        
        if stats['total_processed'] < 100:
            print(f"\nYou have {stats['total_processed']} test images.")
            print("Consider collecting more data for better accuracy measurement.")
            print("Run 'python collect_test_data.py download' for dataset sources.")