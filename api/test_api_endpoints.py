#!/usr/bin/env python3
"""
Test the anomaly detection API endpoints.        if response.status_code == 200:
            data = response.json()
            print(f"   Surveillance status: {data['status']}")
            if data['status'] == 'running':
                print("   Tracking stats:")
                stats = data.get('tracking_stats', {})
                print(f"     - Total vehicles: {stats.get('total_vehicles_tracked', 0)}")
                print(f"     - Active vehicles: {stats.get('active_vehicles', 0)}")
        else:
            print(f"   Failed with status {response.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\nAPI endpoint tests completed!") API server is running before running this script.
"""

import requests
import json
from datetime import datetime

API_BASE = "http://localhost:8000"

def test_api_endpoints():
    print("Testing Anomaly Detection API Endpoints")
    print("="*50)
    
    # Test 1: Get no-plate vehicles
    print("\n1. Testing /api/v1/anomalies/no-plate-vehicles")
    try:
        response = requests.get(f"{API_BASE}/api/v1/anomalies/no-plate-vehicles")
        if response.status_code == 200:
            data = response.json()
            print(f"   Success! Found {data['total_count']} no-plate vehicles")
            if data['no_plate_vehicles']:
                print("   First vehicle:")
                vehicle = data['no_plate_vehicles'][0]
                print(f"     - Track ID: {vehicle['track_id']}")
                print(f"     - Description: {vehicle['vehicle_description']}")
        else:
            print(f"   Failed with status {response.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Get all anomalies
    print("\n2. Testing /api/v1/anomalies/all")
    try:
        response = requests.get(f"{API_BASE}/api/v1/anomalies/all?time_window_hours=24")
        if response.status_code == 200:
            data = response.json()
            print(f"   Success! Found {data['summary']['total_anomalies']} anomalies")
            print("   Anomaly types:")
            for atype, count in data['summary']['by_type'].items():
                print(f"     - {atype}: {count}")
        else:
            print(f"   Failed with status {response.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Generate anomaly report
    print("\n3. Testing /api/v1/anomalies/report")
    try:
        response = requests.post(f"{API_BASE}/api/v1/anomalies/report", 
                               json={"time_window_hours": 24, "include_images": False})
        if response.status_code == 200:
            data = response.json()
            print("   Success! Report generated")
            print(f"     - No-plate vehicles: {len(data.get('no_plate_vehicles', []))}")
            print(f"     - Plate switchers: {len(data.get('plate_switchers', []))}")
            print(f"     - Loiterers: {len(data.get('loiterers', []))}")
        else:
            print(f"   Failed with status {response.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 4: Get surveillance status
    print("\n4. Testing /api/v1/surveillance/status")
    try:
        response = requests.get(f"{API_BASE}/api/v1/surveillance/status")
        if response.status_code == 200:
            data = response.json()
            print(f"   Surveillance status: {data['status']}")
            if data['status'] == 'running':
                print("   Tracking stats:")
                stats = data.get('tracking_stats', {})
                print(f"     - Total vehicles: {stats.get('total_vehicles_tracked', 0)}")
                print(f"     - Active vehicles: {stats.get('active_vehicles', 0)}")
        else:
            print(f"   Failed with status {response.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\nAPI endpoint tests completed!")

if __name__ == "__main__":
    print("Make sure the API server is running (uvicorn main:app --reload)")
    input("Press Enter to continue...")
    test_api_endpoints()