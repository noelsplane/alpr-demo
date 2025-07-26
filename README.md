# ALPR Surveillance System

A comprehensive Automatic License Plate Recognition (ALPR) system with real-time video surveillance, multi-camera tracking, and advanced vehicle analytics.

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview

This ALPR system provides enterprise-grade vehicle surveillance capabilities including:
- Real-time license plate detection and recognition
- Multi-camera vehicle tracking across locations  
- Anomaly detection (no plates, plate switching, loitering)
- State identification for all 50 US states
- Vehicle attribute recognition (make, model, color)
- WebSocket-based live monitoring
- Comprehensive search and analytics

Built during a summer internship demonstrating full-stack development, computer vision, and system architecture skills.

## Key Features

### Core Capabilities
- **Multi-Model Plate Detection**: YOLO + PlateRecognizer API integration with fallback to EasyOCR
- **State Recognition**: Pattern matching for all 50 US states with 90%+ accuracy
- **Real-time Processing**: WebSocket streaming with queue-based frame processing
- **Cross-Camera Tracking**: Track vehicles across multiple camera locations
- **Anomaly Detection**: 
  - Vehicles without plates
  - License plate switching
  - Loitering detection
  - Rapid reappearance alerts

### Technical Features  
- **Scalable Architecture**: Async processing with FastAPI
- **Smart Caching**: API rate limit management
- **Database**: SQLite/PostgreSQL with TimescaleDB support
- **Browser Streaming**: Direct browser camera integration
- **RESTful API**: Full CRUD operations with search

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Camera Feed    │────▶│  Video Process  │────▶│ Frame Queue     │
│  (Browser/IP)   │     │  (Threading)    │     │ (Async)         │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   WebSocket     │◀────│ Anomaly Detect  │◀────│  YOLO/OCR       │
│   Real-time     │     │ Vehicle Track   │     │  Detection      │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                          │
                        ┌─────────────────┐               ▼
                        │                 │     ┌─────────────────┐
                        │   Web UI        │     │                 │
                        │   Dashboard     │◀────│   Database      │
                        │                 │     │   (SQLite/PG)   │
                        └─────────────────┘     └─────────────────┘
```

## Technology Stack

- **Backend**: Python 3.10+, FastAPI, SQLAlchemy
- **Computer Vision**: YOLOv8, OpenCV, EasyOCR, PlateRecognizer API
- **Database**: SQLite (dev) / PostgreSQL + TimescaleDB (production)
- **Real-time**: WebSockets, asyncio, threading
- **Frontend**: HTML5, JavaScript, Chart.js
- **ML Models**: Custom trained YOLOv8 for license plates

## Prerequisites

- Python 3.10 or higher
- Webcam or IP camera (optional for live detection)
- PlateRecognizer API key (optional, falls back to local OCR)

## Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/noelsplane/alpr-demo.git
cd alpr-demo
```

2. **Set up Python environment**
```bash
cd api
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
# Create a .env file in the api directory
echo "PLATERECOGNIZER_TOKEN=your_api_key_here" > .env
# system will use local OCR if not provided
```

4. **Initialize the database**
```bash
python -c "from models import Base; from database_config import db_config; Base.metadata.create_all(bind=db_config.init_engine())"
```

5. **Verify YOLO models**
```bash
# Models are located in api/models/
# - license_plate_detector.pt
# - license_plate_yolov8.pt
# If not present, the system will use standard YOLOv8
```

6. **Start the API server**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

7. **Access the application**
- Dashboard: http://localhost:8000/static/navigation.html
- API Docs: http://localhost:8000/docs
- Upload: http://localhost:8000/static/
- Live Surveillance: http://localhost:8000/static/surveillance.html

## Usage Examples

### Upload an Image
```python
import requests

with open('car_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/sighting',
        files={'file': f}
    )
print(response.json())
```

### Search for Vehicles
```python
# Search for all vehicles in the last 24 hours
response = requests.get(
    'http://localhost:8000/api/v1/search/vehicles',
    params={
        'timeRange': '24h',
        'vehicleTypes': 'Sedan,SUV'
    }
)
```

### WebSocket Monitoring
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/surveillance');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'detection') {
        console.log('New vehicle detected:', data.data);
    }
};
```

## Features in Detail

### License Plate Detection
- Primary: YOLOv8 trained on license plate dataset
- Secondary: PlateRecognizer API (99% accuracy)
- Fallback: EasyOCR for offline processing
- Preprocessing: Contrast enhancement, perspective correction

### State Recognition
- Pattern matching for all 50 US states
- Format validation (e.g., CA: 1ABC234, NJ: A12BCD)
- Context clues from plate frames
- Confidence scoring system

### Vehicle Tracking
- Appearance-based matching using vehicle attributes
- Cross-camera trajectory analysis
- Time-based feasibility checking
- Persistent vehicle profiles

### Anomaly Detection
- **No License Plate**: Vehicles detected without visible plates
- **Plate Switching**: Same vehicle with multiple plates
- **Loitering**: Vehicles appearing frequently in short time
- **Rapid Reappearance**: Quick returns after leaving

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/sighting` | POST | Upload image for detection |
| `/api/v1/detections` | GET | Retrieve detection history |
| `/api/v1/search/vehicles` | GET | Search with filters |
| `/api/v1/surveillance/start` | POST | Start live monitoring |
| `/api/v1/anomalies/all` | GET | Get detected anomalies |
| `/api/v1/vehicle-profiles` | GET | Get vehicle profiles |
| `/ws/surveillance` | WebSocket | Real-time updates |

Full API documentation available at `/docs` when server is running.

## Configuration

### Environment Variables
```env
# PlateRecognizer API (optional)
PLATERECOGNIZER_TOKEN=your_api_key_here

# Database (PostgreSQL)
DB_USER=alpr_user
DB_PASSWORD=alpr_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=alpr_surveillance
```

### Camera Configuration
Edit `cross_camera_tracker.py` to add camera locations:
```python
CameraInfo("cam_01", "Main Entrance", latitude, longitude)
```

## Performance

- **Processing Speed**: 10-15 FPS real-time processing
- **Detection Accuracy**: 95%+ with PlateRecognizer, 85%+ with local OCR
- **Concurrent Cameras**: Tested with up to 4 simultaneous streams
- **API Rate Limiting**: Intelligent caching to stay within limits

## Troubleshooting

### Common Issues

1. **Camera not detected**
   - Ensure webcam permissions are granted
   - Try different camera index (0, 1, 2)

2. **Low detection accuracy**
   - Ensure good lighting conditions
   - Camera should capture plates at 100+ pixels width
   - Clean camera lens

3. **Database errors**
   - Run migration: `python add_camera_id_direct.py`
   - Check database permissions

## Acknowledgments

- YOLOv8 by Ultralytics
- PlateRecognizer for high-accuracy API
- EasyOCR for offline text recognition
- FastAPI for the amazing web framework

## Contact

Jayden Brown - [jnbrown116@gmail.com](mailto:jnbrown116@gmail.com)


---

**Built during Summer 2024 Internship**