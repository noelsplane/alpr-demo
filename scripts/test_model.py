
from ultralytics import YOLO
import sys

# Load the model
model = YOLO('./models/license_plate_model/weights/best.pt')
print(f"Model loaded! Classes: {model.names}")
print("Ready to detect license plates!")

if len(sys.argv) > 1:
    results = model.predict(sys.argv[1], conf=0.25)
    print(f"Found {len(results[0].boxes) if results[0].boxes else 0} plates")
