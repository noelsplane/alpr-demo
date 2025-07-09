"""
download_datasets.py
Download and prepare large license plate datasets for testing
"""

import os
import requests
import zipfile
import tarfile
import json
import shutil
from pathlib import Path
import gdown
import xml.etree.ElementTree as ET

class DatasetDownloader:
    def __init__(self, base_dir="datasets"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
    def download_ufpr_alpr(self):
        """
        Download UFPR-ALPR dataset (Brazilian, but good for testing detection)
        ~4,500 images
        """
        print("\n=== Downloading UFPR-ALPR Dataset ===")
        print("This dataset contains ~4,500 Brazilian plate images")
        
        dataset_dir = os.path.join(self.base_dir, "ufpr-alpr")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # The dataset is available on GitHub
        print("Please download manually from:")
        print("https://github.com/raysonlaroca/ufpr-alpr-dataset")
        print(f"Extract to: {dataset_dir}")
        
        return dataset_dir
    
    def download_openalpr_benchmarks(self):
        """
        Download OpenALPR benchmark dataset
        Contains US plates from various states
        """
        print("\n=== Downloading OpenALPR Benchmarks ===")
        print("This dataset contains US plates with state labels")
        
        dataset_dir = os.path.join(self.base_dir, "openalpr-benchmarks")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Clone the benchmarks repo
        if not os.path.exists(os.path.join(dataset_dir, "endtoend")):
            os.system(f"git clone https://github.com/openalpr/benchmarks.git {dataset_dir}")
        
        # Organize by state
        self._organize_openalpr_by_state(dataset_dir)
        
        return dataset_dir
    
    def download_cars_dataset(self):
        """
        Download Stanford Cars dataset (has some visible plates)
        """
        print("\n=== Stanford Cars Dataset ===")
        print("Large dataset with some visible license plates")
        print("Download from: http://ai.stanford.edu/~jkrause/cars/car_dataset.html")
        
    def download_ccpd_dataset(self):
        """
        Chinese City Parking Dataset - huge dataset but Chinese plates
        Good for testing detection accuracy
        """
        print("\n=== CCPD (Chinese City Parking Dataset) ===")
        print("290k+ images, but Chinese plates")
        print("Good for testing detection, not US state recognition")
        print("Download from: https://github.com/detectRecog/CCPD")
        
    def download_kaggle_plates(self):
        """
        Download from Kaggle (requires Kaggle API)
        """
        print("\n=== Kaggle License Plate Datasets ===")
        
        dataset_dir = os.path.join(self.base_dir, "kaggle-plates")
        os.makedirs(dataset_dir, exist_ok=True)
        
        print("Popular Kaggle datasets:")
        print("1. Car License Plate Detection:")
        print("   kaggle datasets download -d andrewmvd/car-plate-detection")
        print("2. License Plate Recognition:")
        print("   kaggle datasets download -d skull21/licence-plate-recognization")
        print("\nInstall Kaggle API: pip install kaggle")
        print("Set up credentials: https://github.com/Kaggle/kaggle-api")
        
        return dataset_dir
    
    def _organize_openalpr_by_state(self, dataset_dir):
        """Organize OpenALPR images by state."""
        print("Organizing images by state...")
        
        # Path to the benchmark data
        benchmark_dir = os.path.join(dataset_dir, "endtoend")
        if not os.path.exists(benchmark_dir):
            print(f"Benchmark directory not found: {benchmark_dir}")
            return
            
        us_dir = os.path.join(benchmark_dir, "us")
        if not os.path.exists(us_dir):
            print(f"US directory not found: {us_dir}")
            return
            
        # Create organized directory
        organized_dir = os.path.join(self.base_dir, "us-plates-by-state")
        os.makedirs(organized_dir, exist_ok=True)
        
        # Read the XML file with ground truth
        results_file = os.path.join(us_dir, "results.txt")
        if os.path.exists(results_file):
            self._parse_openalpr_results(us_dir, organized_dir, results_file)
        else:
            print("Results file not found, organizing by filename patterns...")
            self._organize_by_patterns(us_dir, organized_dir)
    
    def _parse_openalpr_results(self, source_dir, dest_dir, results_file):
        """Parse OpenALPR results file to get state information."""
        # OpenALPR results format varies, adjust as needed
        # For now, organize by filename patterns
        self._organize_by_patterns(source_dir, dest_dir)
    
    def _organize_by_patterns(self, source_dir, dest_dir):
        """Organize images by common state patterns."""
        import cv2
        from unified_state_recognition import UnifiedStateRecognizer
        import easyocr
        
        print("Analyzing images and organizing by detected state...")
        
        recognizer = UnifiedStateRecognizer()
        reader = easyocr.Reader(['en'], gpu=False)
        
        image_files = list(Path(source_dir).glob("*.jpg")) + \
                     list(Path(source_dir).glob("*.png"))
        
        state_counts = {}
        unknown_dir = os.path.join(dest_dir, "UNKNOWN")
        os.makedirs(unknown_dir, exist_ok=True)
        
        for i, img_path in enumerate(image_files[:100]):  # Process first 100 for testing
            if i % 10 == 0:
                print(f"Processing {i}/{len(image_files)}...")
            
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Run OCR
                results = reader.readtext(img)
                
                # Extract plate text (simplified)
                texts = [r[1] for r in results if len(r[1]) >= 4]
                
                if texts:
                    # Try to recognize state
                    plate_text = max(texts, key=len)  # Use longest text
                    state_result = recognizer.recognize_state(
                        plate_text=plate_text,
                        ocr_results=results,
                        plate_image=img
                    )
                    
                    state = state_result.get('state_code', 'UNKNOWN')
                    
                    # Create state directory
                    state_dir = os.path.join(dest_dir, state)
                    os.makedirs(state_dir, exist_ok=True)
                    
                    # Copy image
                    dest_path = os.path.join(state_dir, os.path.basename(img_path))
                    shutil.copy2(img_path, dest_path)
                    
                    state_counts[state] = state_counts.get(state, 0) + 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        print("\nOrganization complete!")
        print("State distribution:")
        for state, count in sorted(state_counts.items()):
            print(f"  {state}: {count}")


def download_synthetic_dataset():
    """Generate synthetic US license plates for testing."""
    print("\n=== Synthetic License Plate Generator ===")
    print("You can generate synthetic plates for testing:")
    print("1. https://github.com/Belval/TextRecognitionDataGenerator")
    print("2. https://github.com/Mahyar24/PlateGen")
    
    # Here's a simple synthetic generator
    create_synthetic_plates()


def create_synthetic_plates():
    """Create synthetic license plates for each state."""
    from PIL import Image, ImageDraw, ImageFont
    import random
    
    output_dir = "datasets/synthetic-plates"
    os.makedirs(output_dir, exist_ok=True)
    
    # State patterns for synthetic generation
    state_patterns = {
        'CA': lambda: f"{random.randint(1,9)}{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))}{random.randint(100,999)}",
        'TX': lambda: f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))}{random.randint(1000,9999)}",
        'NY': lambda: f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))}{random.randint(1000,9999)}",
        'FL': lambda: f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))}{random.randint(10,99)}",
        'NJ': lambda: f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(10,99)}{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))}",
        'MA': lambda: f"{random.randint(100,999)}{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))}{random.randint(0,9)}",
    }
    
    print("Generating synthetic plates...")
    
    for state, pattern_func in state_patterns.items():
        state_dir = os.path.join(output_dir, state)
        os.makedirs(state_dir, exist_ok=True)
        
        # Generate 10 plates per state
        for i in range(10):
            # Create plate image
            img = Image.new('RGB', (300, 100), color='white')
            draw = ImageDraw.Draw(img)
            
            # Generate plate text
            plate_text = pattern_func()
            
            # Try to use a font, fallback to default
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf", 60)
            except:
                font = ImageFont.load_default()
            
            # Draw text
            draw.text((10, 20), plate_text, fill='black', font=font)
            
            # Add state name
            try:
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 20)
            except:
                small_font = ImageFont.load_default()
            
            draw.text((10, 5), state, fill='blue', font=small_font)
            
            # Save
            filename = f"{state}_{plate_text}_{i}.png"
            img.save(os.path.join(state_dir, filename))
    
    print(f"Generated synthetic plates in {output_dir}")


def main():
    downloader = DatasetDownloader()
    
    print("=== License Plate Dataset Downloader ===")
    print("\nAvailable datasets:")
    print("1. OpenALPR Benchmarks (US plates with states)")
    print("2. UFPR-ALPR (Brazilian plates)")
    print("3. Kaggle datasets (various)")
    print("4. Generate synthetic US plates")
    print("5. Download all available")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1":
        downloader.download_openalpr_benchmarks()
    elif choice == "2":
        downloader.download_ufpr_alpr()
    elif choice == "3":
        downloader.download_kaggle_plates()
    elif choice == "4":
        create_synthetic_plates()
    elif choice == "5":
        downloader.download_openalpr_benchmarks()
        create_synthetic_plates()
        downloader.download_kaggle_plates()
    else:
        print("Invalid choice")
        return
    
    print("\n=== Next Steps ===")
    print("1. Process downloaded images through your system:")
    print("   python batch_process_images.py datasets/")
    print("2. Run accuracy tests:")
    print("   python measure_accuracy.py")


if __name__ == "__main__":
    main()