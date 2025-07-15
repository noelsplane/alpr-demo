# api/anomaly_detector.py
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np

class AnomalyDetector:
    def __init__(self):
        self.vehicle_history = defaultdict(list)
        self.plate_history = defaultdict(list)
        self.no_plate_vehicles = []
        
    def detect_anomalies(self, detection, vehicle_features=None):
        anomalies = []
        timestamp = datetime.now()
        
        # Check for no plates
        if not detection.get('plates_detected'):
            anomalies.append({
                'type': 'NO_PLATE_VEHICLE',
                'severity': 'medium',
                'timestamp': timestamp,
                'image': detection.get('image')
            })
            self.no_plate_vehicles.append({
                'timestamp': timestamp,
                'features': vehicle_features
            })
            return anomalies
        
        # Check each detected plate
        for plate in detection['plates_detected']:
            plate_text = plate['text']
            
            # Check for plate switching
            if vehicle_features:
                vehicle_id = self._generate_vehicle_id(vehicle_features)
                prev_plates = self.vehicle_history[vehicle_id]
                
                if prev_plates and plate_text not in prev_plates:
                    anomalies.append({
                        'type': 'PLATE_SWITCH',
                        'severity': 'high',
                        'timestamp': timestamp,
                        'old_plates': prev_plates,
                        'new_plate': plate_text,
                        'message': f'Vehicle switched plates from {prev_plates} to {plate_text}'
                    })
                
                self.vehicle_history[vehicle_id].append(plate_text)
            
            # Check for loitering
            self.plate_history[plate_text].append(timestamp)
            recent_appearances = [t for t in self.plate_history[plate_text] 
                                if timestamp - t < timedelta(minutes=30)]
            
            if len(recent_appearances) > 10:
                anomalies.append({
                    'type': 'LOITERING',
                    'severity': 'low',
                    'timestamp': timestamp,
                    'plate': plate_text,
                    'appearances': len(recent_appearances),
                    'message': f'Vehicle {plate_text} seen {len(recent_appearances)} times in 30 minutes'
                })
        
        return anomalies
    
    def _generate_vehicle_id(self, features):
        """Generate unique ID based on vehicle features (color, make, model)"""
        return f"{features.get('color', 'unknown')}_{features.get('make', 'unknown')}_{features.get('model', 'unknown')}"