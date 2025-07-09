# This script updates main.py to add better model verification

import re

with open('main.py', 'r') as f:
    content = f.read()

# Find the model loading section
new_model_section = '''# Load the specialized license plate detection model
license_plate_model_path = "../models/license_plate_yolov8.pt"
if os.path.exists(license_plate_model_path):
    logger.info(f"Loading specialized license plate detection model from {license_plate_model_path}")
    model = YOLO(license_plate_model_path)
    
    # Verify model details
    model_size = os.path.getsize(license_plate_model_path) / (1024*1024)
    logger.info(f"Model size: {model_size:.2f} MB")
    
    # Check if it's actually a license plate model
    if hasattr(model.model, 'nc'):
        logger.info(f"Model classes: {model.model.nc}")
    if hasattr(model.model, 'names'):
        logger.info(f"Model detects: {model.model.names}")
    
    logger.info("License plate model loaded successfully!")
else:
    logger.warning(f"License plate model not found at {license_plate_model_path}")
    logger.warning("Falling back to general model - this will detect cars, not plates!")
    model = YOLO("../models/yolov8n.pt")'''

# Replace the model loading section
pattern = r'# Load the specialized license plate detection model.*?model = YOLO\([^)]+\)'
content = re.sub(pattern, new_model_section, content, flags=re.DOTALL)

with open('main.py', 'w') as f:
    f.write(content)

print("Updated main.py with better model verification")
