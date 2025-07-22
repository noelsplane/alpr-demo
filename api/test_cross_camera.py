#!/usr/bin/env python3
"""Test cross-camera tracking functionality"""

import requests
import json
from datetime import datetime, timedelta

API_BASE = "http://localhost:8000"

def test_cross_camera_tracking():
    print("Testing Cross-Camera Tracking System")
    print("="*50)
    
    # Test 1: Register additional cameras
    print("\n1. Registering test cameras...")
    cameras = [
        {"camera_id": "cam_05", "location_name": "North Plaza", "latitude": 40.7140, "longitude": -74.0045},
        {"camera_id": "cam_06", "location_name": "South Plaza", "latitude": 40.7120, "longitude": -74.0070}
    ]
    
    for cam in cameras:
        response = requests.post(f"{API_BASE}/api/v1/cameras/register", json=cam)
        print(f"   Camera {cam['camera_id']}: {response.json()}")
    
    # Test 2: Process some mock detections
    print("\n2. Processing mock detections...")
    
    # Same vehicle at different cameras
    detections = [
        {
            "detection": {
                "plate_text": "ABC123",
                "confidence": 0.95,
                "vehicle_type": "Sedan",
                "vehicle_make": "Toyota",
                "vehicle_model": "Camry",
                "vehicle_color": "Silver"
            },
            "camera_id": "cam_01",
            "timestamp": datetime.now().isoformat()
        },
        {
            "detection": {
                "plate_text": "ABC123",
                "confidence": 0.92,
                "vehicle_type": "Sedan",
                "vehicle_make": "Toyota",
                "vehicle_model": "Camry",
                "vehicle_color": "Silver"
            },
            "camera_id": "cam_02",
            "timestamp": (datetime.now() + timedelta(minutes=5)).isoformat()
        },
        {
            "detection": {
                "plate_text": "ABC123",
                "confidence": 0.90,
                "vehicle_type": "Sedan",
                "vehicle_make": "Toyota",
                "vehicle_model": "Camry",
                "vehicle_color": "Silver"
            },
            "camera_id": "cam_03",
            "timestamp": (datetime.now() + timedelta(minutes=15)).isoformat()
        }
    ]
    
    global_ids = []
    for det in detections:
        response = requests.post(f"{API_BASE}/api/v1/tracking/process", json=det)
        result = response.json()
        print(f"   Processed at {det['camera_id']}: Vehicle ID = {result.get('global_vehicle_id')}")
        if result.get('anomalies'):
            print(f"   Warning: Anomalies: {result['anomalies']}")
        global_ids.append(result.get('global_vehicle_id'))
    
    # Test 3: Get vehicle journey
    if global_ids:
        print(f"\n3. Getting journey for vehicle {global_ids[0]}...")
        response = requests.get(f"{API_BASE}/api/v1/tracking/vehicle/{global_ids[0]}")
        if response.status_code == 200:
            journey = response.json()
            print(f"   Total sightings: {journey['journey_info']['total_sightings']}")
            print(f"   Cameras visited: {journey['journey_info']['cameras_visited']}")
    
    # Test 4: Find vehicles between cameras
    print("\n4. Finding vehicles between cam_01 and cam_02...")
    response = requests.get(f"{API_BASE}/api/v1/tracking/between-cameras?camera_a=cam_01&camera_b=cam_02")
    data = response.json()
    print(f"   Found {data['vehicles_found']} vehicles")
    
    # Test 5: Get statistics
    print("\n5. Getting tracking statistics...")
    response = requests.get(f"{API_BASE}/api/v1/tracking/statistics")
    stats = response.json()
    print(f"   Total vehicles tracked: {stats['total_vehicles_tracked']}")
    print(f"   Active vehicles: {stats['active_vehicles_last_hour']}")
    print(f"   Cameras registered: {stats['cameras_registered']}")

if __name__ == "__main__":
    test_cross_camera_tracking()