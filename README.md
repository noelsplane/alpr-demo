# ALPR Demo - Automatic License Plate Recognition

A FastAPI-based Automatic License Plate Recognition (ALPR) system that uses YOLOv8 for vehicle detection and EasyOCR for license plate text recognition.

## Features

- **License Plate Detection**: Uses YOLOv8 model to detect license plates in images
- **Text Recognition**: EasyOCR extracts text from detected license plates
- **SQLite Database**: Stores detection history with confidence scores and bounding boxes
- **REST API**: FastAPI endpoints for uploading images and retrieving detection history
- **Web Interface**: Static HTML files for easy interaction
- **Base64 Image Storage**: Cropped license plate images stored as base64 in database

## Project Structure

```
alpr-demo/
├── api/
│   ├── main.py              # FastAPI application
│   ├── models.py            # SQLAlchemy database models
│   ├── schemas.py           # Pydantic schemas
│   ├── requirements.txt     # Python dependencies
│   └── static/             # Web interface files
│       ├── index.html
│       ├── live.html
│       └── history.html
├── data/
│   ├── test/               # Test images
│   └── uploads/            # Uploaded images
├── models/
│   └── yolov8n.pt          # YOLOv8 model weights
└── notebooks/
    └── day1_plate_detect.ipynb
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
