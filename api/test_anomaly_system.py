#!/usr/bin/env python3
"""
Verbose test for debugging plate switch detection
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))

from anomaly_detector import create_anomaly_detector
import time
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

def test_plate_switch_verbose():
    print("="*60)
    print("VERBOSE PLATE SWITCH DETECTION TEST")
    print("="*60)
    
    detector = create_anomaly_detector()
    
    # Lower the threshold to make matching easier
    detector.config['vehicle_match_threshold'] = 0.7
    
    # Step 1: First vehicle
    print("\n[STEP 1] First detection - Silver Toyota Camry with ABC123")
    detection1 = {
        'plate_text': 'ABC123',
        'confidence': 0.95,
        'vehicle_type': 'Sedan',
        'vehicle_make': 'Toyota',
        'vehicle_model': 'Camry',
        'vehicle_color': 'Silver',
        'vehicle_type_confidence': 0.9,
        'vehicle_make_confidence': 0.9,
        'vehicle_model_confidence': 0.9,
        'vehicle_color_confidence': 0.9
    }
    
    anomalies1 = detector.process_frame_detections([detection1])
    print(f"\nAnomalies detected: {len(anomalies1)}")
    
    print("\nTracks after step 1:")
    for track_id, track in detector.vehicle_tracks.items():
        attrs = track['attributes']
        print(f"  Track ID: {track_id}")
        print(f"    Plates: {list(track['plates'])}")
        print(f"    Vehicle: {attrs.get('color')} {attrs.get('make')} {attrs.get('model')}")
        print(f"    Appearances: {len(track['appearances'])}")
    
    # Step 2: Different vehicle to show contrast
    print("\n[STEP 2] Different vehicle - Red Honda Civic with DEF456")
    time.sleep(1)
    
    detection2 = {
        'plate_text': 'DEF456',
        'confidence': 0.95,
        'vehicle_type': 'Sedan',
        'vehicle_make': 'Honda',
        'vehicle_model': 'Civic',
        'vehicle_color': 'Red',
        'vehicle_type_confidence': 0.9,
        'vehicle_make_confidence': 0.9,
        'vehicle_model_confidence': 0.9,
        'vehicle_color_confidence': 0.9
    }
    
    anomalies2 = detector.process_frame_detections([detection2])
    print(f"\nAnomalies detected: {len(anomalies2)}")
    
    print("\nTracks after step 2:")
    for track_id, track in detector.vehicle_tracks.items():
        attrs = track['attributes']
        print(f"  Track ID: {track_id}")
        print(f"    Plates: {list(track['plates'])}")
        print(f"    Vehicle: {attrs.get('color')} {attrs.get('make')} {attrs.get('model')}")
    
    # Step 3: SAME Toyota Camry but with different plate
    print("\n[STEP 3] SAME Silver Toyota Camry but with plate XYZ789!")
    time.sleep(1)
    
    detection3 = {
        'plate_text': 'XYZ789',  # Different plate!
        'confidence': 0.95,
        'vehicle_type': 'Sedan',
        'vehicle_make': 'Toyota',  # Same
        'vehicle_model': 'Camry',   # Same
        'vehicle_color': 'Silver',  # Same
        'vehicle_type_confidence': 0.9,
        'vehicle_make_confidence': 0.9,
        'vehicle_model_confidence': 0.9,
        'vehicle_color_confidence': 0.9
    }
    
    print("\nProcessing detection 3...")
    anomalies3 = detector.process_frame_detections([detection3])
    
    print(f"\nAnomalies detected: {len(anomalies3)}")
    for a in anomalies3:
        print(f"\n  ANOMALY TYPE: {a['type']}")
        print(f"  Message: {a['message']}")
        print(f"  Severity: {a['severity']}")
        if 'details' in a:
            print(f"  Details:")
            for key, value in a['details'].items():
                print(f"    - {key}: {value}")
    
    print("\nFinal tracks:")
    for track_id, track in detector.vehicle_tracks.items():
        attrs = track['attributes']
        print(f"\n  Track ID: {track_id}")
        print(f"    Plates: {list(track['plates'])}")
        print(f"    Vehicle: {attrs.get('color')} {attrs.get('make')} {attrs.get('model')}")
        print(f"    Appearances: {len(track['appearances'])}")
        print(f"    Anomalies: {[a['type'] for a in track['anomalies']]}")
    
    # Final check
    stats = detector.get_tracking_stats()
    print(f"\nPlate switches detected: {stats['plate_switches']}")
    
    success = stats['plate_switches'] > 0
    if success:
        print("\n✓ SUCCESS: Plate switch detected!")
    else:
        print("\n✗ FAILED: Plate switch not detected")
        print("\nDebugging info:")
        print(f"  - Total tracks created: {len(detector.vehicle_tracks)}")
        print(f"  - Vehicle match threshold: {detector.config['vehicle_match_threshold']}")
    
    return success

if __name__ == "__main__":
    success = test_plate_switch_verbose()
    sys.exit(0 if success else 1)