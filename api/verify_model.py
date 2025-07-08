from ultralytics import YOLO
import os

print("=== Model Verification ===\n")

# Check what models exist
models_dir = "../models"
print(f"Models in {models_dir}:")
for file in os.listdir(models_dir):
    if file.endswith('.pt'):
        size = os.path.getsize(os.path.join(models_dir, file)) / (1024*1024)
        print(f"  - {file}: {size:.2f} MB")

# Load and check the license plate model
lp_model_path = "../models/license_plate_yolov8.pt"
if os.path.exists(lp_model_path):
    print(f"\nLoading model from: {lp_model_path}")
    model = YOLO(lp_model_path)
    
    # Check model details
    print(f"Model type: {model.model}")
    print(f"Number of classes: {model.model.nc if hasattr(model.model, 'nc') else 'Unknown'}")
    
    # Get class names if available
    if hasattr(model.model, 'names'):
        print(f"Classes detected: {model.model.names}")
    elif hasattr(model, 'names'):
        print(f"Classes detected: {model.names}")
    
    # Test prediction to see what it detects
    print("\nTesting model on a dummy image...")
    import numpy as np
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model(dummy_img, verbose=False)
    print(f"Model loaded and working!")
    
else:
    print(f"\nERROR: Model not found at {lp_model_path}")

# Compare with generic model
generic_path = "../models/yolov8n.pt"
if os.path.exists(generic_path):
    print(f"\n--- Generic Model Comparison ---")
    generic_model = YOLO(generic_path)
    if hasattr(generic_model.model, 'names'):
        print(f"Generic model classes: {list(generic_model.model.names.values())[:10]}...")  # First 10 classes
