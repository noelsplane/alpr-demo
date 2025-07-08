# ALPR Demo - Automatic License Plate Recognition

A comprehensive automatic license plate recognition (ALPR) system built with FastAPI, YOLOv8, and EasyOCR, featuring advanced state recognition and enhanced OCR preprocessing.

## Features

- **License Plate Detection**: Uses specialized YOLOv8 model for accurate license plate detection
- **Advanced OCR**: EasyOCR with multiple preprocessing techniques for enhanced accuracy
- **State Recognition**: Intelligent state identification from plate patterns and visual context
- **Web Interface**: Clean, responsive web UI for uploading images and viewing results
- **History Tracking**: SQLite database to store detection history with image thumbnails
- **Real-time Processing**: Fast processing with detailed confidence scoring
- **Enhanced Preprocessing**: Multiple image enhancement techniques for better OCR accuracy
- **Context-Aware Detection**: Multi-strategy approach for state and plate text recognition

## Project Structure

```
alpr-demo/
├── api/
│   ├── main.py                 # Main FastAPI application
│   ├── models.py               # Database models
│   ├── requirements.txt        # Python dependencies
│   ├── state_model.py          # State recognition logic
│   ├── plate_filter_utils.py   # OCR filtering utilities
│   ├── models/                 # ML model files
│   │   └── license_plate_yolov8.pt
│   ├── static/                 # Web interface files
│   │   ├── index.html
│   │   ├── live.html
│   │   └── history.html
│   └── utils/                  # Utility modules
│       └── state_recognition.py
├── data/
│   ├── test/                   # Test images
│   └── uploads/                # Uploaded images
├── notebooks/
│   └── day1_plate_detect.ipynb # Development notebook
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/noelsplane/alpr-demo.git
cd alpr-demo
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
cd api
pip install -r requirements.txt
```

## Usage

1. Start the FastAPI server:
```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. Access the web interface at `http://localhost:8000/ui/`

3. Upload images via the API or web interface to detect license plates

## API Endpoints

- `GET /` - Health check
- `POST /api/v1/sighting` - Upload image for license plate detection
- `GET /api/v1/detections` - Retrieve all detection history
- `DELETE /api/v1/detections` - Clear detection history

## Database Schema

The SQLite database stores detections with the following fields:
- `id`: Primary key
- `plate_text`: Detected license plate text
- `confidence`: OCR confidence score (0-1)
- `image_name`: Original uploaded image filename
- `x1, y1, x2, y2`: Bounding box coordinates
- `plate_image_base64`: Base64 encoded cropped plate image
- `timestamp`: Detection timestamp

## Dependencies

- FastAPI - Web framework
- Ultralytics - YOLOv8 model
- EasyOCR - Text recognition
- OpenCV - Image processing
- SQLAlchemy - Database ORM
- Pillow - Image handling

## License

This project is for demonstration purposes.
