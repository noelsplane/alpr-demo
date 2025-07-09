import cv2
import os
from datetime import datetime

# Create debug directory
os.makedirs("debug_images", exist_ok=True)

# Add this function to your main.py
def save_debug_image(img, name, step=""):
    """Save debug images to see what's being processed"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debug_images/{timestamp}_{name}_{step}.jpg"
    cv2.imwrite(filename, img)
    return filename
