# batch_process_plates.py
"""
Batch process extracted license plates through the ALPR API
to detect states and build a labeled dataset.
"""

import os
import requests
from pathlib import Path
import json
from typing import Dict, List
import time
from datetime import datetime

class BatchPlateProcessor:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = []
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'states_detected': 0,
            'by_state': {},
            'errors': []
        }
    
    def process_plate_image(self, image_path: Path) -> Dict:
        """Process a single plate image through the API."""
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f, 'image/jpeg')}
                response = requests.post(
                    f"{self.api_url}/api/v1/sighting",
                    files=files,
                    timeout=30
                )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {'error': f'API returned status {response.status_code}'}
                
        except Exception as e:
            return {'error': str(e)}
    
    def process_directory(self, directory: Path, limit: int = None):
        """Process all images in a directory."""
        image_files = list(directory.glob('*.jpg')) + list(directory.glob('*.png'))
        
        if limit:
            image_files = image_files[:limit]
        
        total_files = len(image_files)
        print(f"\nProcessing {total_files} images...")
        print("This may take a while. Progress will be shown every 10 images.\n")
        
        start_time = time.time()
        
        for idx, img_path in enumerate(image_files):
            # Show progress
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                remaining = (total_files - idx - 1) / rate
                print(f"Progress: {idx + 1}/{total_files} "
                      f"({(idx + 1) / total_files * 100:.1f}%) - "
                      f"Est. time remaining: {remaining / 60:.1f} minutes")
            
            # Process the image
            result = self.process_plate_image(img_path)
            
            # Update statistics
            self.stats['total_processed'] += 1
            
            if 'error' in result:
                self.stats['failed'] += 1
                self.stats['errors'].append({
                    'file': img_path.name,
                    'error': result['error']
                })
            else:
                self.stats['successful'] += 1
                
                # Extract plate and state info
                if 'plates_detected' in result:
                    for plate in result['plates_detected']:
                        plate_info = {
                            'filename': img_path.name,
                            'plate_text': plate.get('text', ''),
                            'confidence': plate.get('confidence', 0),
                            'state': plate.get('state'),
                            'state_confidence': plate.get('state_confidence', 0)
                        }
                        
                        self.results.append(plate_info)
                        
                        # Update state statistics
                        if plate.get('state') and plate['state'] != 'UNKNOWN':
                            self.stats['states_detected'] += 1
                            state = plate['state']
                            self.stats['by_state'][state] = self.stats['by_state'].get(state, 0) + 1
            
            # Small delay to not overwhelm the API
            time.sleep(0.1)
        
        total_time = time.time() - start_time
        print(f"\n✓ Processing complete in {total_time / 60:.1f} minutes")
    
    def save_results(self):
        """Save processing results to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_file = f"batch_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'stats': self.stats,
                'results': self.results
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Save state-organized results for easy review
        states_file = f"plates_by_state_{timestamp}.json"
        plates_by_state = {}
        
        for result in self.results:
            state = result.get('state', 'UNKNOWN')
            if state not in plates_by_state:
                plates_by_state[state] = []
            plates_by_state[state].append(result)
        
        with open(states_file, 'w') as f:
            json.dump(plates_by_state, f, indent=2)
        
        print(f"Plates by state saved to: {states_file}")
        
        # Create a CSV for easy review/editing
        csv_file = f"plates_for_review_{timestamp}.csv"
        with open(csv_file, 'w') as f:
            f.write("filename,plate_text,detected_state,confidence,correct_state\n")
            for result in self.results:
                f.write(f"{result['filename']},{result['plate_text']},"
                       f"{result.get('state', 'UNKNOWN')},{result.get('state_confidence', 0):.2f},\n")
        
        print(f"CSV for review saved to: {csv_file}")
    
    def print_summary(self):
        """Print processing summary."""
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        
        print(f"\nTotal images processed: {self.stats['total_processed']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"States detected: {self.stats['states_detected']}")
        
        if self.stats['by_state']:
            print("\nDetections by state:")
            for state, count in sorted(self.stats['by_state'].items(), 
                                      key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {state}: {count}")
        
        if self.stats['successful'] > 0:
            detection_rate = self.stats['states_detected'] / self.stats['successful'] * 100
            print(f"\nState detection rate: {detection_rate:.1f}%")
        
        if self.stats['errors']:
            print(f"\nErrors encountered: {len(self.stats['errors'])}")
            print("First few errors:")
            for err in self.stats['errors'][:3]:
                print(f"  {err['file']}: {err['error']}")


def organize_results_by_state(results_file: str, output_dir: str = "organized_plates"):
    """Organize processed plates into state directories."""
    import shutil
    
    print(f"\nOrganizing plates by state...")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create state directories
    states = set()
    for result in data['results']:
        state = result.get('state', 'UNKNOWN')
        if state is None:
            state = 'UNKNOWN'
        states.add(state)
    
    for state in states:
        (output_path / state).mkdir(exist_ok=True)
    
    # Copy plates to appropriate directories
    extracted_dir = Path("extracted_plates")
    organized_count = 0
    
    for result in data['results']:
        state = result.get('state', 'UNKNOWN')
        if state is None:
            state = 'UNKNOWN'
        filename = result['filename']
        source = extracted_dir / filename
        
        if source.exists():
            dest = output_path / state / filename
            shutil.copy2(source, dest)
            organized_count += 1
    
    print(f"✓ Organized {organized_count} plates into {len(states)} state directories")
    print(f"Location: {output_path}")


if __name__ == "__main__":
    print("=== Batch License Plate Processor ===\n")
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:8000")
        if response.status_code != 200:
            print("⚠️  Error: ALPR API is not responding properly")
            print("Make sure the server is running with:")
            print("  python -m uvicorn main:app --reload")
            exit(1)
    except:
        print("⚠️  Error: Cannot connect to ALPR API at http://localhost:8000")
        print("Make sure the server is running with:")
        print("  python -m uvicorn main:app --reload")
        exit(1)
    
    print("✓ API is running\n")
    
    # Initialize processor
    processor = BatchPlateProcessor()
    
    # Process options
    print("Processing options:")
    print("1. Process all plates (1083 images - may take ~20 minutes)")
    print("2. Process a sample (first 100 plates)")
    print("3. Process custom number")
    
    choice = input("\nSelect option (1-3): ")
    
    extracted_dir = Path("extracted_plates")
    
    if choice == '1':
        processor.process_directory(extracted_dir)
    elif choice == '2':
        processor.process_directory(extracted_dir, limit=100)
    elif choice == '3':
        num = int(input("How many plates to process? "))
        processor.process_directory(extracted_dir, limit=num)
    else:
        print("Invalid choice")
        exit(1)
    
    # Save results
    processor.save_results()
    processor.print_summary()
    
    # Ask if user wants to organize by state
    if processor.stats['states_detected'] > 0:
        organize = input("\nOrganize plates by detected state? (y/n): ")
        if organize.lower() == 'y':
            # Find the most recent results file
            results_files = list(Path('.').glob('batch_results_*.json'))
            if results_files:
                latest_results = max(results_files, key=lambda p: p.stat().st_mtime)
                organize_results_by_state(str(latest_results))