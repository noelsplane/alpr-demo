"""
Enhanced anomaly detection system that uses vehicle attributes
to detect suspicious behavior including vehicles without plates.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
from difflib import SequenceMatcher
import numpy as np
from vehicle_matcher import VehicleMatcher
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker
from models import PlateDetection, SessionAlert, VehicleTrack, TrackPlateAssociation, VehicleAnomaly
import json

logger = logging.getLogger(__name__)

class EnhancedAnomalyDetector:
    """
    Advanced anomaly detection using vehicle attributes and behavioral patterns.
    """
    
    def __init__(self, db_session=None):
        self.vehicle_matcher = VehicleMatcher()
        self.db_session = db_session
        
        # Tracking data structures
        self.vehicle_tracks = defaultdict(lambda: {
            'appearances': deque(maxlen=100),
            'plates': set(),
            'attributes': {},
            'anomalies': [],
            'last_seen': None,
            'first_seen': None,
            'signature': None
        })
        
        # Alert management
        self.alert_history = defaultdict(lambda: deque(maxlen=10))
        self.alert_cooldowns = defaultdict(dict)
        
        # Configuration
        self.config = {
            'no_plate_severity': 'high',
            'plate_switch_severity': 'critical',
            'loitering_threshold_minutes': 30,
            'loitering_min_appearances': 5,
            'rapid_reappearance_seconds': 30,
            'alert_cooldown_minutes': 10,
            'vehicle_match_threshold': 0.7
        }
        
        # Statistics
        self.stats = {
            'total_vehicles_tracked': 0,
            'no_plate_vehicles': 0,
            'plate_switches': 0,
            'loitering_incidents': 0
        }
    
    def process_frame_detections(self, detections: List[Dict], 
                               frame_metadata: Dict = None) -> List[Dict]:
        """
        Process all detections from a frame and return anomalies.
        
        Args:
            detections: List of detection dictionaries from frame processing
            frame_metadata: Optional metadata about the frame (timestamp, location, etc.)
            
        Returns:
            List of anomaly dictionaries
        """
        anomalies = []
        current_time = datetime.now()
        frame_vehicles = []
        
        # First pass: Identify all vehicles in frame
        for detection in detections:
            vehicle_info = self._extract_vehicle_info(detection)
            vehicle_info['timestamp'] = current_time
            vehicle_info['frame_metadata'] = frame_metadata
            frame_vehicles.append(vehicle_info)
        
        # Second pass: Match to tracked vehicles and detect anomalies
        for vehicle in frame_vehicles:
            # Find or create vehicle track
            track_id = self._match_or_create_track(vehicle)
            
            # Update track with new information
            self._update_track(track_id, vehicle)
            
            # Check for anomalies
            vehicle_anomalies = self._detect_anomalies(track_id, vehicle)
            anomalies.extend(vehicle_anomalies)
        
        # Update statistics
        self._update_statistics()
        
        return anomalies
    
    def _extract_vehicle_info(self, detection: Dict) -> Dict:
        """Extract comprehensive vehicle information from detection."""
        vehicle_info = {
            'plate_text': detection.get('plate_text'),
            'plate_confidence': detection.get('confidence', 0.0),
            'box': detection.get('box', []),
            'plate_image': detection.get('plate_image_base64'),
            'state': detection.get('state'),
            'state_confidence': detection.get('state_confidence', 0.0),
            'is_no_plate': detection.get('is_vehicle_without_plate', False),
            'attributes': {
                'type': detection.get('vehicle_type'),
                'make': detection.get('vehicle_make'),
                'model': detection.get('vehicle_model'),
                'color': detection.get('vehicle_color'),
                'year': detection.get('vehicle_year'),
                'type_confidence': detection.get('vehicle_type_confidence', 0.0),
                'make_confidence': detection.get('vehicle_make_confidence', 0.0),
                'model_confidence': detection.get('vehicle_model_confidence', 0.0),
                'color_confidence': detection.get('vehicle_color_confidence', 0.0),
                'year_confidence': detection.get('vehicle_year_confidence', 0.0)
            }
        }
        
        # Special handling for no-plate vehicles
        if vehicle_info['is_no_plate'] or vehicle_info['plate_text'] == 'NO_PLATE_DETECTED':
            vehicle_info['is_no_plate'] = True
            vehicle_info['plate_text'] = None
            self.stats['no_plate_vehicles'] += 1
        
        return vehicle_info
    
    def _match_or_create_track(self, vehicle: Dict) -> str:
        """Match vehicle to existing track or create new one."""
        best_match_id = None
        best_match_score = 0.0
        
        # Try to match with existing tracks
        for track_id, track_data in self.vehicle_tracks.items():
            # Skip if track is too old (vehicle likely left)
            if track_data['last_seen']:
                time_since_last = datetime.now() - track_data['last_seen']
                if time_since_last > timedelta(hours=1):
                    continue
            
            # Calculate match score
            score = self._calculate_track_match_score(vehicle, track_data)
            
            if score > best_match_score and score >= self.config['vehicle_match_threshold']:
                best_match_score = score
                best_match_id = track_id
        
        # Create new track if no good match
        if not best_match_id:
            # For vehicles with plates, also check if attributes match existing tracks
            # This helps detect plate switches
            if vehicle['plate_text']:
                for track_id, track_data in self.vehicle_tracks.items():
                    # Skip old tracks
                    if track_data['last_seen']:
                        time_since_last = datetime.now() - track_data['last_seen']
                        if time_since_last > timedelta(minutes=30):
                            continue
                    
                    # Check if attributes match even though plate is different
                    if track_data['attributes'].get('make') and track_data['attributes'].get('model'):
                        similarity = self._calculate_attribute_similarity(
                            vehicle['attributes'], 
                            track_data['attributes']
                        )
                        
                        # High similarity suggests same vehicle with different plate
                        if similarity > 0.8:
                            return track_id
            
            if vehicle['plate_text']:
                track_id = f"plate_{vehicle['plate_text']}"
            else:
                signature = self.vehicle_matcher.create_vehicle_signature(vehicle['attributes'])
                track_id = f"vehicle_{signature}_{datetime.now().timestamp()}"
            
            self.stats['total_vehicles_tracked'] += 1
            return track_id
        
        return best_match_id
        

    def _calculate_track_match_score(self, vehicle: Dict, track_data: Dict) -> float:
        """Calculate how well a vehicle matches a track."""
        # Exact plate match - highest priority
        if vehicle['plate_text'] and vehicle['plate_text'] in track_data['plates']:
            return 1.0
        
        # Check if vehicle attributes match (for plate switch detection)
        # This is key - we need to check attributes even when plates don't match
        if vehicle['attributes'].get('make') and track_data['attributes'].get('make'):
            # Calculate attribute similarity
            similarity = self._calculate_attribute_similarity(vehicle['attributes'], track_data['attributes'])
            
            if similarity > 0.8:
                if vehicle['plate_text'] and track_data['plates'] and vehicle['plate_text'] not in track_data['plates']:
                    return similarity * 0.95
                elif not vehicle['plate_text'] and not track_data['plates']:
                    return similarity
        return 0.0

    def _match_or_create_track(self, vehicle: Dict) -> str:
        """Match vehicle to existing track or create new one."""
        best_match_id = None
        best_match_score = 0.0
        
        # Debug logging
        logger.debug(f"Matching vehicle: plate={vehicle.get('plate_text')}, "
                    f"make={vehicle['attributes'].get('make')}, "
                    f"model={vehicle['attributes'].get('model')}, "
                    f"color={vehicle['attributes'].get('color')}")
        
        # Try to match with ALL existing tracks
        for track_id, track_data in self.vehicle_tracks.items():
            # Skip if track is too old (vehicle likely left)
            if track_data['last_seen']:
                time_since_last = datetime.now() - track_data['last_seen']
                if time_since_last > timedelta(hours=1):
                    continue
            
            # Calculate match score
            score = self._calculate_track_match_score(vehicle, track_data)
            
            logger.debug(f"  Track {track_id} score: {score}")
            
            if score > best_match_score:
                best_match_score = score
                best_match_id = track_id
        
        # Use existing track if score is high enough
        if best_match_score >= self.config['vehicle_match_threshold']:
            logger.debug(f"Matched to existing track: {best_match_id} (score: {best_match_score})")
            return best_match_id
        
        # Create new track if no good match
        if vehicle['plate_text']:
            track_id = f"plate_{vehicle['plate_text']}"
        else:
            signature = self.vehicle_matcher.create_vehicle_signature(vehicle['attributes'])
            track_id = f"vehicle_{signature}_{datetime.now().timestamp()}"
        
        logger.debug(f"Creating new track: {track_id}")
        self.stats['total_vehicles_tracked'] += 1
        return track_id
    
    def _calculate_attribute_similarity(self, attrs1: Dict, attrs2: Dict) -> float:
        """Calculate similarity between two sets of vehicle attributes."""
        if not attrs1 or not attrs2:
            return 0.0
        
        scores = []
        weights = {'type': 0.3, 'make': 0.3, 'model': 0.2, 'color': 0.2}
        
        for attr, weight in weights.items():
            val1 = attrs1.get(attr)
            val2 = attrs2.get(attr)
            
            if val1 and val2:
                if val1.lower() == val2.lower():
                    scores.append(weight)
                elif attr == 'color':
                    # Fuzzy color matching
                    color_sim = self.vehicle_matcher._color_similarity(val1, val2)
                    scores.append(weight * color_sim)
        
        return sum(scores) / sum(weights.values()) if scores else 0.0

    def _update_track(self, track_id: str, vehicle: Dict):
        """Update track with new vehicle information."""
        track = self.vehicle_tracks[track_id]
        current_time = vehicle['timestamp']
        
        # Update timestamps
        if not track['first_seen']:
            track['first_seen'] = current_time
        track['last_seen'] = current_time
        
        # Add appearance
        track['appearances'].append({
            'timestamp': current_time,
            'vehicle': vehicle
        })
        
        # Update plates - IMPORTANT: Check if this is a new plate before adding
        if vehicle['plate_text']:
            # Store the previous plates before adding new one
            previous_plates = list(track['plates'])
            
            track['plates'].add(vehicle['plate_text'])
            
            if len(track['plates']) > len(previous_plates):
                logger.debug(f"New plate detected for track {track_id}: {vehicle['plate_text']}")
                logger.debug(f"Previous plates: {previous_plates}")
                logger.debug(f"All plates now: {list(track['plates'])}")
        
        # Update attributes with highest confidence values
        for attr in ['type', 'make', 'model', 'color', 'year']:
            new_conf = vehicle['attributes'].get(f'{attr}_confidence', 0)
            old_conf = track['attributes'].get(f'{attr}_confidence', 0)
            
            if new_conf > old_conf:
                track['attributes'][attr] = vehicle['attributes'].get(attr)
                track['attributes'][f'{attr}_confidence'] = new_conf
        
        # Update attributes with highest confidence values
        for attr in ['type', 'make', 'model', 'color', 'year']:
            new_conf = vehicle['attributes'].get(f'{attr}_confidence', 0)
            old_conf = track['attributes'].get(f'{attr}_confidence', 0)
            
            if new_conf > old_conf:
                track['attributes'][attr] = vehicle['attributes'].get(attr)
                track['attributes'][f'{attr}_confidence'] = new_conf
    
    def _detect_anomalies(self, track_id: str, vehicle: Dict) -> List[Dict]:
        """Detect various types of anomalies for a vehicle."""
        anomalies = []
        track = self.vehicle_tracks[track_id]
        current_time = vehicle['timestamp']
        
        # 1. No License Plate Detection
        if vehicle['is_no_plate']:
            if self._should_alert('no_plate', track_id):
                anomaly = {
                    'type': 'NO_PLATE_VEHICLE',
                    'severity': self.config['no_plate_severity'],
                    'timestamp': current_time.isoformat(),
                    'track_id': track_id,
                    'message': 'Vehicle detected without visible license plate',
                    'details': {
                        'vehicle_attributes': vehicle['attributes'],
                        'confidence_scores': {
                            k: v for k, v in vehicle['attributes'].items() 
                            if k.endswith('_confidence')
                        },
                        'image': vehicle.get('plate_image')
                    }
                }
                anomalies.append(anomaly)
                self._record_alert('no_plate', track_id)

       # Replace the entire plate switching section (section 2) in _detect_anomalies with this simpler version:

        if vehicle['plate_text'] and track['plates'] and len(track['plates']) > 1:
            all_plates = list(track['plates'])
            
            if self._should_alert('plate_switch', track_id):
                anomaly = {
                    'type': 'PLATE_SWITCH',
                    'severity': self.config['plate_switch_severity'],
                    'timestamp': current_time.isoformat(),
                    'track_id': track_id,
                    'plate': vehicle['plate_text'],
                    'message': f'Vehicle has multiple plates: {", ".join(sorted(all_plates))}',
                    'details': {
                        'all_plates': all_plates,
                        'current_plate': vehicle['plate_text'],
                        'plate_count': len(all_plates),
                        'vehicle_attributes': track['attributes']
                    }
                }
                anomalies.append(anomaly)
                self._record_alert('plate_switch', track_id)
                self.stats['plate_switches'] += 1
        
        # 3. Loitering Detection
        recent_appearances = [
            app for app in track['appearances']
            if (current_time - app['timestamp']) < timedelta(minutes=self.config['loitering_threshold_minutes'])
        ]
        
        # Check if loitering based on recent appearances    
        if len(recent_appearances) >= self.config['loitering_min_appearances']:
            if self._should_alert('loitering', track_id):
                time_span = recent_appearances[-1]['timestamp'] - recent_appearances[0]['timestamp']
                anomaly = {
                    'type': 'LOITERING',
                    'severity': 'medium',
                    'timestamp': current_time.isoformat(),
                    'track_id': track_id,
                    'plate': vehicle['plate_text'] or 'NO_PLATE',
                    'message': f'Vehicle loitering: {len(recent_appearances)} appearances in {time_span.total_seconds() / 60:.1f} minutes',
                    'details': {
                        'appearance_count': len(recent_appearances),
                        'time_span_minutes': time_span.total_seconds() / 60,
                        'vehicle_attributes': track['attributes']
                    }
                }
                anomalies.append(anomaly)
                self._record_alert('loitering', track_id)
                self.stats['loitering_incidents'] += 1
        
        # 4. Rapid Reappearance
        if len(track['appearances']) >= 2:
            last_appearance = track['appearances'][-2]['timestamp']
            time_since_last = current_time - last_appearance
            
            if time_since_last < timedelta(seconds=self.config['rapid_reappearance_seconds']):
                if self._should_alert('rapid_reappearance', track_id):
                    anomaly = {
                        'type': 'RAPID_REAPPEARANCE',
                        'severity': 'low',
                        'timestamp': current_time.isoformat(),
                        'track_id': track_id,
                        'plate': vehicle['plate_text'] or 'NO_PLATE',
                        'message': f'Vehicle reappeared after {time_since_last.total_seconds():.1f} seconds',
                        'details': {
                            'seconds_between': time_since_last.total_seconds(),
                            'vehicle_attributes': track['attributes']
                        }
                    }
                    anomalies.append(anomaly)
                    self._record_alert('rapid_reappearance', track_id)
        
        # 5. Suspicious No-Plate Behavior
        if vehicle['is_no_plate'] and len(track['appearances']) > 3:
            # No-plate vehicle appearing multiple times is extra suspicious
            if self._should_alert('suspicious_no_plate', track_id):
                anomaly = {
                    'type': 'SUSPICIOUS_NO_PLATE',
                    'severity': 'critical',
                    'timestamp': current_time.isoformat(),
                    'track_id': track_id,
                    'message': f'No-plate vehicle with {len(track["appearances"])} appearances',
                    'details': {
                        'total_appearances': len(track['appearances']),
                        'time_span': (current_time - track['first_seen']).total_seconds() / 60,
                        'vehicle_attributes': track['attributes']
                    }
                }
                anomalies.append(anomaly)
                self._record_alert('suspicious_no_plate', track_id)
        
        # Store anomalies in track
        track['anomalies'].extend(anomalies)
        
        # Save to database if session provided
        if self.db_session and anomalies:
            self._save_anomalies_to_db(anomalies)
        
        return anomalies
    
    def _should_alert(self, alert_type: str, track_id: str) -> bool:
        """Check if we should generate an alert based on cooldowns."""
        cooldowns = self.alert_cooldowns[track_id]
        last_alert = cooldowns.get(alert_type)
        
        if not last_alert:
            return True
        
        time_since_last = datetime.now() - last_alert
        return time_since_last > timedelta(minutes=self.config['alert_cooldown_minutes'])
    
    def _record_alert(self, alert_type: str, track_id: str):
        """Record that an alert was generated."""
        self.alert_cooldowns[track_id][alert_type] = datetime.now()
        self.alert_history[track_id].append({
            'type': alert_type,
            'timestamp': datetime.now()
        })
    
    def _save_anomalies_to_db(self, anomalies: List[Dict]):
        """Save anomalies to database."""
        try:
            for anomaly in anomalies:
                alert = SessionAlert(
                    session_id=anomaly.get('session_id', 0),
                    alert_type=anomaly['type'],
                    severity=anomaly['severity'],
                    plate_text=anomaly.get('plate'),
                    message=anomaly['message'],
                    details=json.dumps(anomaly.get('details', {})),
                    alert_time=datetime.fromisoformat(anomaly['timestamp'])
                )
                self.db_session.add(alert)
            self.db_session.commit()
        except Exception as e:
            logger.error(f"Error saving anomalies to database: {e}")
            self.db_session.rollback()
    
    def _update_statistics(self):
        """Update tracking statistics."""
        active_tracks = sum(
            1 for track in self.vehicle_tracks.values()
            if track['last_seen'] and (datetime.now() - track['last_seen']) < timedelta(minutes=5)
        )
        self.stats['active_vehicles'] = active_tracks
    
    def get_tracking_stats(self) -> Dict:
        """Get current tracking statistics."""
        stats = self.stats.copy()
        
        # Add dynamic stats
        stats['total_tracks'] = len(self.vehicle_tracks)
        stats['tracks_with_anomalies'] = sum(
            1 for track in self.vehicle_tracks.values()
            if track['anomalies']
        )
        
        # Get most suspicious vehicles
        suspicious_vehicles = []
        for track_id, track in self.vehicle_tracks.items():
            if track['anomalies']:
                suspicion_score = len(track['anomalies'])
                if any(a['type'] == 'NO_PLATE_VEHICLE' for a in track['anomalies']):
                    suspicion_score *= 2
                if any(a['type'] == 'PLATE_SWITCH' for a in track['anomalies']):
                    suspicion_score *= 3
                
                suspicious_vehicles.append({
                    'track_id': track_id,
                    'plates': list(track['plates']),
                    'anomaly_count': len(track['anomalies']),
                    'suspicion_score': suspicion_score,
                    'last_seen': track['last_seen'].isoformat() if track['last_seen'] else None
                })
        
        # Sort by suspicion score
        suspicious_vehicles.sort(key=lambda x: x['suspicion_score'], reverse=True)
        stats['most_suspicious'] = suspicious_vehicles[:10]
        
        return stats
    
    def get_vehicle_track(self, track_id: str) -> Optional[Dict]:
        """Get detailed information about a specific vehicle track."""
        if track_id not in self.vehicle_tracks:
            return None
        
        track = self.vehicle_tracks[track_id]
        return {
            'track_id': track_id,
            'plates': list(track['plates']),
            'attributes': track['attributes'],
            'first_seen': track['first_seen'].isoformat() if track['first_seen'] else None,
            'last_seen': track['last_seen'].isoformat() if track['last_seen'] else None,
            'total_appearances': len(track['appearances']),
            'anomalies': track['anomalies'],
            'recent_appearances': [
                {
                    'timestamp': app['timestamp'].isoformat(),
                    'plate': app['vehicle'].get('plate_text'),
                    'confidence': app['vehicle'].get('plate_confidence')
                }
                for app in list(track['appearances'])[-10:]  # Last 10 appearances
            ]
        }
    
    def cleanup_old_tracks(self, hours_threshold: int = 24):
        """Remove tracks that haven't been seen in the specified time."""
        current_time = datetime.now()
        tracks_to_remove = []
        
        for track_id, track in self.vehicle_tracks.items():
            if track['last_seen']:
                time_since_last = current_time - track['last_seen']
                if time_since_last > timedelta(hours=hours_threshold):
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.vehicle_tracks[track_id]
            if track_id in self.alert_cooldowns:
                del self.alert_cooldowns[track_id]
        
        logger.info(f"Cleaned up {len(tracks_to_remove)} old tracks")
    
    def export_suspicious_vehicles_report(self) -> Dict:
        """Generate a comprehensive report of suspicious vehicles."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': self.get_tracking_stats(),
            'no_plate_vehicles': [],
            'plate_switchers': [],
            'loiterers': [],
            'all_anomalies': []
        }
        
        for track_id, track in self.vehicle_tracks.items():
            if not track['anomalies']:
                continue
            
            track_info = {
                'track_id': track_id,
                'plates': list(track['plates']),
                'vehicle_description': self._get_vehicle_description(track['attributes']),
                'first_seen': track['first_seen'].isoformat() if track['first_seen'] else None,
                'last_seen': track['last_seen'].isoformat() if track['last_seen'] else None,
                'total_appearances': len(track['appearances'])
            }
            
            for anomaly in track['anomalies']:
                anomaly_info = {**track_info, **anomaly}
                report['all_anomalies'].append(anomaly_info)
                
                if anomaly['type'] in ['NO_PLATE_VEHICLE', 'SUSPICIOUS_NO_PLATE']:
                    report['no_plate_vehicles'].append(anomaly_info)
                elif anomaly['type'] == 'PLATE_SWITCH':
                    report['plate_switchers'].append(anomaly_info)
                elif anomaly['type'] == 'LOITERING':
                    report['loiterers'].append(anomaly_info)
        
        return report
    
    def _get_vehicle_description(self, attributes: Dict) -> str:
        """Create a human-readable vehicle description."""
        parts = []
        
        if attributes.get('color'):
            parts.append(attributes['color'])
        if attributes.get('year'):
            parts.append(attributes['year'])
        if attributes.get('make'):
            parts.append(attributes['make'])
        if attributes.get('model'):
            parts.append(attributes['model'])
        if attributes.get('type'):
            parts.append(f"({attributes['type']})")
        
        return ' '.join(parts) if parts else 'Unknown Vehicle'
    
    def save_tracks_to_db(self, db_session):
        """Save current tracks to database for persistence."""
        try:
            for track_id, track in self.vehicle_tracks.items():
                # Check if track exists in DB
                existing_track = db_session.query(VehicleTrack).filter(
                    VehicleTrack.track_id == track_id
                ).first()
                
                if existing_track:
                    # Update existing track
                    existing_track.last_seen = track['last_seen']
                    existing_track.total_appearances = len(track['appearances'])
                    existing_track.is_suspicious = len(track['anomalies']) > 0
                    existing_track.anomaly_count = len(track['anomalies'])
                    
                    # Update attributes if confidence improved
                    for attr in ['type', 'make', 'model', 'color', 'year']:
                        new_conf = track['attributes'].get(f'{attr}_confidence', 0)
                        old_conf = getattr(existing_track, f'{attr}_confidence', 0)
                        
                        if new_conf > old_conf:
                            setattr(existing_track, f'vehicle_{attr}', track['attributes'].get(attr))
                            setattr(existing_track, f'{attr}_confidence', new_conf)
                else:
                    # Create new track
                    new_track = VehicleTrack(
                        track_id=track_id,
                        first_seen=track['first_seen'],
                        last_seen=track['last_seen'],
                        vehicle_type=track['attributes'].get('type'),
                        vehicle_make=track['attributes'].get('make'),
                        vehicle_model=track['attributes'].get('model'),
                        vehicle_color=track['attributes'].get('color'),
                        vehicle_year=track['attributes'].get('year'),
                        type_confidence=track['attributes'].get('type_confidence', 0),
                        make_confidence=track['attributes'].get('make_confidence', 0),
                        model_confidence=track['attributes'].get('model_confidence', 0),
                        color_confidence=track['attributes'].get('color_confidence', 0),
                        year_confidence=track['attributes'].get('year_confidence', 0),
                        total_appearances=len(track['appearances']),
                        is_suspicious=len(track['anomalies']) > 0,
                        has_no_plate=not bool(track['plates']),
                        anomaly_count=len(track['anomalies'])
                    )
                    db_session.add(new_track)
                
                # Save plate associations
                for plate_text in track['plates']:
                    existing_assoc = db_session.query(TrackPlateAssociation).filter(
                        and_(
                            TrackPlateAssociation.track_id == track_id,
                            TrackPlateAssociation.plate_text == plate_text
                        )
                    ).first()
                    
                    if existing_assoc:
                        existing_assoc.last_seen = track['last_seen']
                        existing_assoc.appearance_count += 1
                    else:
                        new_assoc = TrackPlateAssociation(
                            track_id=track_id,
                            plate_text=plate_text,
                            first_seen=track['first_seen'],
                            last_seen=track['last_seen'],
                            appearance_count=1
                        )
                        db_session.add(new_assoc)
                
                # Save anomalies
                for anomaly in track['anomalies']:
                    # Check if anomaly already exists
                    existing_anomaly = db_session.query(VehicleAnomaly).filter(
                        and_(
                            VehicleAnomaly.track_id == track_id,
                            VehicleAnomaly.anomaly_type == anomaly['type'],
                            VehicleAnomaly.detected_time == datetime.fromisoformat(anomaly['timestamp'])
                        )
                    ).first()
                    
                    if not existing_anomaly:
                        new_anomaly = VehicleAnomaly(
                            track_id=track_id,
                            anomaly_type=anomaly['type'],
                            severity=anomaly['severity'],
                            detected_time=datetime.fromisoformat(anomaly['timestamp']),
                            plate_text=anomaly.get('plate'),
                            message=anomaly['message'],
                            details=json.dumps(anomaly.get('details', {})),
                            image_data=anomaly.get('details', {}).get('image')
                        )
                        db_session.add(new_anomaly)
            
            db_session.commit()
            logger.info(f"Saved {len(self.vehicle_tracks)} tracks to database")
        except Exception as e:
            logger.error(f"Error saving tracks to database: {e}")
            db_session.rollback()


# Convenience function for integration
def create_anomaly_detector(db_session=None) -> EnhancedAnomalyDetector:
    """Create and configure an anomaly detector instance."""
    return EnhancedAnomalyDetector(db_session)


# Test the detector
if __name__ == "__main__":
    # Create test detector
    detector = create_anomaly_detector()
    
    # Test detection 1: Vehicle with plate
    detection1 = {
        'plate_text': 'ABC123',
        'confidence': 0.95,
        'vehicle_type': 'Sedan',
        'vehicle_make': 'Toyota',
        'vehicle_model': 'Camry',
        'vehicle_color': 'Silver',
        'vehicle_type_confidence': 0.9,
        'vehicle_make_confidence': 0.85,
        'vehicle_model_confidence': 0.8,
        'vehicle_color_confidence': 0.95
    }
    
    # Test detection 2: Same vehicle, different plate (suspicious!)
    detection2 = {
        'plate_text': 'XYZ789',
        'confidence': 0.9,
        'vehicle_type': 'Sedan',
        'vehicle_make': 'Toyota',
        'vehicle_model': 'Camry',
        'vehicle_color': 'Silver',
        'vehicle_type_confidence': 0.9,
        'vehicle_make_confidence': 0.85,
        'vehicle_model_confidence': 0.8,
        'vehicle_color_confidence': 0.95
    }
    
    # Test detection 3: Vehicle without plate
    detection3 = {
        'plate_text': 'NO_PLATE_DETECTED',
        'is_vehicle_without_plate': True,
        'confidence': 0.0,
        'vehicle_type': 'SUV',
        'vehicle_make': 'Unknown',
        'vehicle_color': 'Black',
        'vehicle_type_confidence': 0.95,
        'vehicle_color_confidence': 0.9
    }
    
    # Test detection 4: Same no-plate vehicle appears again
    detection4 = {
        'plate_text': 'NO_PLATE_DETECTED',
        'is_vehicle_without_plate': True,
        'confidence': 0.0,
        'vehicle_type': 'SUV',
        'vehicle_make': 'Unknown',
        'vehicle_color': 'Black',
        'vehicle_type_confidence': 0.95,
        'vehicle_color_confidence': 0.9
    }
    
    print("Testing Enhanced Anomaly Detector")
    print("-" * 50)
    
    # Process detections
    anomalies1 = detector.process_frame_detections([detection1])
    print(f"Detection 1 anomalies: {len(anomalies1)}")
    
    # Simulate time passing
    import time
    time.sleep(1)
    
    anomalies2 = detector.process_frame_detections([detection2])
    print(f"Detection 2 anomalies: {len(anomalies2)}")
    for anomaly in anomalies2:
        print(f"  - {anomaly['type']}: {anomaly['message']}")
    
    anomalies3 = detector.process_frame_detections([detection3])
    print(f"Detection 3 anomalies: {len(anomalies3)}")
    for anomaly in anomalies3:
        print(f"  - {anomaly['type']}: {anomaly['message']}")
    
    # Process same no-plate vehicle again
    time.sleep(1)
    anomalies4 = detector.process_frame_detections([detection4])
    print(f"Detection 4 anomalies: {len(anomalies4)}")
    for anomaly in anomalies4:
        print(f"  - {anomaly['type']}: {anomaly['message']}")
    
    # Print statistics
    print("\nStatistics:")
    stats = detector.get_tracking_stats()
    for key, value in stats.items():
        if key != 'most_suspicious':
            print(f"  {key}: {value}")
    
    print("\nMost Suspicious Vehicles:")
    for vehicle in stats.get('most_suspicious', [])[:5]:
        print(f"  Track {vehicle['track_id']}: Score {vehicle['suspicion_score']}, Plates: {vehicle['plates']}")
    
    # Test report generation
    print("\nGenerating Suspicious Vehicles Report...")
    report = detector.export_suspicious_vehicles_report()
    print(f"Total no-plate vehicles: {len(report['no_plate_vehicles'])}")
    print(f"Total plate switchers: {len(report['plate_switchers'])}")
    print(f"Total loiterers: {len(report['loiterers'])}")
    print(f"Total anomalies: {len(report['all_anomalies'])}")