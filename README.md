
# ALPR Demo - Automatic License Plate Recognition

A modern ALPR system using FastAPI, YOLOv8, EasyOCR, and the PlateRecognizer API for robust license plate and state recognition. Features unified state recognition, secure API token management, and a web UI for uploads and history.

## Features

- **YOLOv8 License Plate Detection**: Fast and accurate plate localization.
- **EasyOCR for Local OCR**: Extracts plate text with enhanced preprocessing.
- **PlateRecognizer API Integration**: Improves state recognition and accuracy.
- **Unified State Recognition**: Combines API, pattern matching, and context detection.
- **FastAPI Backend**: REST API for uploads, results, and analytics.
- **SQLite Database**: Stores detection history and analytics.
- **Web UI**: Upload images, view results, and browse detection history.
- **Secure API Token**: PlateRecognizer token loaded from `.env` (not in source control).

## Project Structure

```
alpr-demo/
├── api/
│   ├── main.py                 # FastAPI app (core logic)
│   ├── models.py               # SQLAlchemy models
│   ├── state_model.py          # Unified state recognition
│   ├── state_patterns.py       # Pattern matching for states
│   ├── plate_filter_utils.py   # Plate text extraction utilities
│   ├── enhanced_ocr_utils.py   # OCR preprocessing
│   ├── requirements.txt        # Python dependencies
│   ├── .env                    # API tokens (not tracked)
│   ├── static/                 # Web UI files
│   ├── detections.db           # SQLite database
│   └── ...                     # Other supporting modules
├── data/
│   ├── test/                   # Test images
│   └── uploads/                # Uploaded images
├── testing_results/            # Latest test results
├── utilities/
│   └── baseline_metrics.py     # Baseline metrics and comparison
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/noelsplane/alpr-demo.git
   cd alpr-demo
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   cd api
   pip install -r requirements.txt
   ```

4. Add your PlateRecognizer API token to `.env`:
   ```
   PLATERECOGNIZER_TOKEN=your_token_here
   ```

## Usage

1. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. Access the web UI at [http://localhost:8000/ui/](http://localhost:8000/ui/)

3. Use the API endpoints for uploads and analytics.

## API Endpoints

- `GET /` — Health check
- `POST /api/v1/sighting` — Upload image for plate detection
- `GET /api/v1/detections` — Get detection history
- `DELETE /api/v1/detections` — Clear detection history
- `GET /api/v1/state-analytics` — State recognition analytics

## Database Schema

Detections are stored with:
- `id`, `plate_text`, `confidence`, `image_name`, `x1`, `y1`, `x2`, `y2`, `plate_image_base64`, `state`, `state_confidence`, `timestamp`

## Security

- **API tokens** are loaded from `.env` and never committed to source control.
- `.env` is included in `.gitignore`.

## Dependencies

- FastAPI, Ultralytics (YOLOv8), EasyOCR, OpenCV, SQLAlchemy, Pillow, python-dotenv

## License

This project is for demonstration and research purposes.
