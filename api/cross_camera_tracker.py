"""
Cross-camera vehicle tracking system that maintains vehicle identities
across multiple camera feeds and locations.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json
from sqlalchemy import create_engine, and_, or_
from sqlalchemy.orm import sessionmaker
from models import PlateDetection, VehicleTrack, TrackPlateAssociation
from vehicle_matcher import VehicleMatcher
from dataclasses import dataclass
import math
logger = logging.getLogger(__name__)

import math

class Distance:
    """Simple distance calculator to replace geopy"""
    def __init__(self, km):
        self.kilometers = km

def calculate_distance(coord1, coord2):
    """Calculate distance between two coordinates in kilometers"""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    # Haversine formula
    R = 6371  # Earth's radius in kilometers
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    
    a = (math.sin(dlat/2)**2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
    c = 2 * math.asin(math.sqrt(a))
    
    return Distance(R * c)

@dataclass
class CameraInfo:
    """Information about a camera location."""
    camera_id: str
    location_name: str
    latitude: float
    longitude: float
    direction: Optional[str] = None  # N, S, E, W, etc.
    coverage_radius_meters: float = 50.0

@dataclass
class VehicleSighting:
    """A vehicle sighting at a specific camera."""
    camera_id: str
    timestamp: datetime
    plate_text: Optional[str]
    confidence: float
    vehicle_attributes: Dict
    detection_id: Optional[int] = None
    location: Optional[CameraInfo] = None

class CrossCameraTracker:
    """
    Tracks vehicles across multiple cameras using plate numbers,
    vehicle attributes, and spatiotemporal reasoning.
    """
    
    def __init__(self, db_session=None):
        self.db_session = db_session
        self.vehicle_matcher = VehicleMatcher()
        
        # Camera registry
        self.cameras: Dict[str, CameraInfo] = {}
        
        # Global vehicle registry
        self.global_vehicles: Dict[str, Dict] = {}  # global_id -> vehicle info
        self.plate_to_global_id: Dict[str, str] = {}  # plate -> global_id
        
        # Cross-camera tracking data
        self.vehicle_journeys: Dict[str, List[VehicleSighting]] = defaultdict(list)
        self.active_vehicles: Dict[str, datetime] = {}  # global_id -> last_seen
        
        # Configuration
        self.config = {
            'max_speed_kmh': 200,  # Maximum reasonable vehicle speed
            'inactive_timeout_hours': 24,  # Consider vehicle inactive after this
            'merge_threshold': 0.85,  # Similarity threshold for merging vehicles
            'journey_gap_minutes': 30,  # Gap to split journeys
        }
        
        # Statistics
        self.stats = defaultdict(int)
    
    def register_camera(self, camera_info: CameraInfo):
        """Register a camera in the system."""
        self.cameras[camera_info.camera_id] = camera_info
        logger.info(f"Registered camera: {camera_info.camera_id} at {camera_info.location_name}")
    
    def process_detection(self, detection: Dict, camera_id: str, 
                         timestamp: Optional[datetime] = None) -> Dict:
        """
        Process a vehicle detection from a specific camera.
        
        Returns:
            Dict containing tracking results and any cross-camera matches
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Create sighting record
        sighting = VehicleSighting(
            camera_id=camera_id,
            timestamp=timestamp,
            plate_text=detection.get('plate_text'),
            confidence=detection.get('confidence', 0.0),
            vehicle_attributes={
                'type': detection.get('vehicle_type'),
                'make': detection.get('vehicle_make'),
                'model': detection.get('vehicle_model'),
                'color': detection.get('vehicle_color'),
                'year': detection.get('vehicle_year'),
                'type_confidence': detection.get('vehicle_type_confidence', 0.0),
                'make_confidence': detection.get('vehicle_make_confidence', 0.0),
                'model_confidence': detection.get('vehicle_model_confidence', 0.0),
                'color_confidence': detection.get('vehicle_color_confidence', 0.0),
                'year_confidence': detection.get('vehicle_year_confidence', 0.0),
            },
            location=self.cameras.get(camera_id)
        )
        
        # Find or create global vehicle ID
        global_id = self._identify_vehicle(sighting)
        
        # Update vehicle journey
        self.vehicle_journeys[global_id].append(sighting)
        self.active_vehicles[global_id] = timestamp
        
        # Analyze cross-camera patterns
        journey_info = self._analyze_journey(global_id)
        
        # Check for suspicious patterns
        anomalies = self._detect_cross_camera_anomalies(global_id, sighting)
        
        # Update statistics
        self.stats['total_detections'] += 1
        self.stats[f'camera_{camera_id}_detections'] += 1
        
        return {
            'global_vehicle_id': global_id,
            'camera_id': camera_id,
            'journey_info': journey_info,
            'anomalies': anomalies,
            'vehicle_info': self.global_vehicles.get(global_id, {})
        }
    
    def _identify_vehicle(self, sighting: VehicleSighting) -> str:
        """
        Identify the global vehicle ID for a sighting.
        Uses plate matching first, then attribute matching.
        """
        # First, try exact plate match
        if sighting.plate_text and sighting.plate_text in self.plate_to_global_id:
            return self.plate_to_global_id[sighting.plate_text]
        
        # Try to match with existing vehicles using attributes
        best_match_id = None
        best_match_score = 0.0
        
        detection_dict = {
            'plate_text': sighting.plate_text,
            'confidence': sighting.confidence,
            **{f'vehicle_{k}': v for k, v in sighting.vehicle_attributes.items()}
        }
        
        for global_id, vehicle_info in self.global_vehicles.items():
            # Skip if vehicle has been inactive too long
            if global_id in self.active_vehicles:
                last_seen = self.active_vehicles[global_id]
                if (sighting.timestamp - last_seen).total_seconds() > self.config['inactive_timeout_hours'] * 3600:
                    continue
            
            # Check spatiotemporal feasibility
            if not self._is_spatiotemporally_feasible(global_id, sighting):
                continue
            
            # Calculate similarity
            similarity = self.vehicle_matcher.calculate_similarity(
                detection_dict,
                vehicle_info
            )
            
            if similarity > best_match_score and similarity >= self.config['merge_threshold']:
                best_match_score = similarity
                best_match_id = global_id
        
        # Create new vehicle if no good match
        if best_match_id:
            global_id = best_match_id
            # Update vehicle info with new sighting
            self._update_vehicle_info(global_id, sighting)
        else:
            # Generate new global ID
            global_id = f"vehicle_{datetime.now().timestamp()}_{sighting.camera_id}"
            self.global_vehicles[global_id] = {
                'global_id': global_id,
                'first_seen': sighting.timestamp,
                'plates': set([sighting.plate_text]) if sighting.plate_text else set(),
                **detection_dict
            }
            self.stats['new_vehicles'] += 1
        
        # Update plate mapping
        if sighting.plate_text:
            self.plate_to_global_id[sighting.plate_text] = global_id
            if 'plates' in self.global_vehicles[global_id]:
                self.global_vehicles[global_id]['plates'].add(sighting.plate_text)
        
        return global_id
    
    def _is_spatiotemporally_feasible(self, global_id: str, 
                                     new_sighting: VehicleSighting) -> bool:
        """
        Check if a vehicle could reasonably travel from its last known
        location to the new sighting location in the elapsed time.
        """
        if global_id not in self.vehicle_journeys:
            return True
        
        journey = self.vehicle_journeys[global_id]
        if not journey:
            return True
        
        last_sighting = journey[-1]
        
        # Skip if no location info
        if not last_sighting.location or not new_sighting.location:
            return True
        
        # Calculate time difference
        time_diff = (new_sighting.timestamp - last_sighting.timestamp).total_seconds()
        if time_diff <= 0:
            return False  # Can't go back in time
        
        # Calculate distance
        last_coords = (last_sighting.location.latitude, last_sighting.location.longitude)
        new_coords = (new_sighting.location.latitude, new_sighting.location.longitude)
        distance_km = calculate_distance(last_coords, new_coords).kilometers
        
        # Calculate required speed
        required_speed_kmh = (distance_km / time_diff) * 3600
        
        # Check if speed is reasonable
        return required_speed_kmh <= self.config['max_speed_kmh']
    
    def _update_vehicle_info(self, global_id: str, sighting: VehicleSighting):
        """Update global vehicle info with new sighting data."""
        vehicle = self.global_vehicles[global_id]
        
        # Update last seen
        vehicle['last_seen'] = sighting.timestamp
        
        # Update attributes with higher confidence values
        for attr_key, attr_value in sighting.vehicle_attributes.items():
            if attr_value and f'{attr_key}_confidence' in sighting.vehicle_attributes:
                conf_key = f'{attr_key}_confidence'
                new_conf = sighting.vehicle_attributes[conf_key]
                old_conf = vehicle.get(f'vehicle_{conf_key}', 0)
                
                if new_conf > old_conf:
                    vehicle[f'vehicle_{attr_key}'] = attr_value
                    vehicle[f'vehicle_{conf_key}'] = new_conf
    
    def _analyze_journey(self, global_id: str) -> Dict:
        """Analyze a vehicle's journey across cameras."""
        if global_id not in self.vehicle_journeys:
            return {}
        
        journey = self.vehicle_journeys[global_id]
        if len(journey) < 2:
            return {
                'total_sightings': len(journey),
                'cameras_visited': 1,
                'journey_segments': []
            }
        
        # Split journey into segments based on time gaps
        segments = []
        current_segment = [journey[0]]
        
        for i in range(1, len(journey)):
            time_gap = (journey[i].timestamp - journey[i-1].timestamp).total_seconds() / 60
            
            if time_gap > self.config['journey_gap_minutes']:
                # Start new segment
                segments.append(current_segment)
                current_segment = [journey[i]]
            else:
                current_segment.append(journey[i])
        
        if current_segment:
            segments.append(current_segment)
        
        # Analyze each segment
        journey_segments = []
        cameras_visited = set()
        total_distance = 0.0
        
        for segment in segments:
            cameras_in_segment = list(set(s.camera_id for s in segment))
            cameras_visited.update(cameras_in_segment)
            
            segment_info = {
                'start_time': segment[0].timestamp.isoformat(),
                'end_time': segment[-1].timestamp.isoformat(),
                'duration_minutes': (segment[-1].timestamp - segment[0].timestamp).total_seconds() / 60,
                'cameras': cameras_in_segment,
                'sighting_count': len(segment)
            }
            
            # Calculate segment distance if location info available
            if len(segment) > 1 and segment[0].location and segment[-1].location:
                start_coords = (segment[0].location.latitude, segment[0].location.longitude)
                end_coords = (segment[-1].location.latitude, segment[-1].location.longitude)
                segment_distance = calculate_distance(start_coords, end_coords).kilometers
                segment_info['distance_km'] = segment_distance
                total_distance += segment_distance
            
            journey_segments.append(segment_info)
        
        return {
            'total_sightings': len(journey),
            'cameras_visited': len(cameras_visited),
            'unique_cameras': list(cameras_visited),
            'journey_segments': journey_segments,
            'total_distance_km': total_distance,
            'first_seen': journey[0].timestamp.isoformat(),
            'last_seen': journey[-1].timestamp.isoformat()
        }
    
    def _detect_cross_camera_anomalies(self, global_id: str, 
                                      new_sighting: VehicleSighting) -> List[Dict]:
        """Detect anomalies specific to cross-camera tracking."""
        anomalies = []
        
        if global_id not in self.vehicle_journeys:
            return anomalies
        
        journey = self.vehicle_journeys[global_id]
        vehicle_info = self.global_vehicles.get(global_id, {})
        
        # 1. Impossible travel speed
        if len(journey) >= 2:
            last_sighting = journey[-2]  # -1 is the new sighting
            
            if last_sighting.location and new_sighting.location:
                time_diff = (new_sighting.timestamp - last_sighting.timestamp).total_seconds()
                if time_diff > 0:
                    last_coords = (last_sighting.location.latitude, last_sighting.location.longitude)
                    new_coords = (new_sighting.location.latitude, new_sighting.location.longitude)
                    distance_km = calculate_distance(last_coords, new_coords).kilometers
                    speed_kmh = (distance_km / time_diff) * 3600
                    
                    if speed_kmh > self.config['max_speed_kmh']:
                        anomalies.append({
                            'type': 'IMPOSSIBLE_SPEED',
                            'severity': 'critical',
                            'timestamp': new_sighting.timestamp.isoformat(),
                            'message': f'Vehicle traveled {distance_km:.1f}km in {time_diff/60:.1f} minutes (speed: {speed_kmh:.0f} km/h)',
                            'details': {
                                'from_camera': last_sighting.camera_id,
                                'to_camera': new_sighting.camera_id,
                                'distance_km': distance_km,
                                'time_minutes': time_diff / 60,
                                'calculated_speed_kmh': speed_kmh
                            }
                        })
        
        # 2. Circular route detection
        if len(journey) >= 10:
            recent_cameras = [s.camera_id for s in journey[-10:]]
            camera_counts = defaultdict(int)
            for cam in recent_cameras:
                camera_counts[cam] += 1
            
            # Check for repeated visits
            for cam, count in camera_counts.items():
                if count >= 3:
                    anomalies.append({
                        'type': 'CIRCULAR_ROUTE',
                        'severity': 'medium',
                        'timestamp': new_sighting.timestamp.isoformat(),
                        'message': f'Vehicle repeatedly visiting camera {cam} ({count} times in last 10 sightings)',
                        'details': {
                            'camera_id': cam,
                            'visit_count': count,
                            'recent_pattern': recent_cameras
                        }
                    })
        
        # 3. Multiple plates on same vehicle
        if 'plates' in vehicle_info and len(vehicle_info['plates']) > 1:
            anomalies.append({
                'type': 'MULTIPLE_PLATES',
                'severity': 'critical',
                'timestamp': new_sighting.timestamp.isoformat(),
                'message': f'Vehicle seen with multiple plates: {", ".join(vehicle_info["plates"])}',
                'details': {
                    'all_plates': list(vehicle_info['plates']),
                    'current_plate': new_sighting.plate_text
                }
            })
        
        return anomalies
    
    def get_vehicle_journey(self, global_id: str) -> Optional[Dict]:
        """Get complete journey information for a vehicle."""
        if global_id not in self.vehicle_journeys:
            return None
        
        journey_info = self._analyze_journey(global_id)
        vehicle_info = self.global_vehicles.get(global_id, {})
        
        # Format sightings for response
        sightings = []
        for sighting in self.vehicle_journeys[global_id]:
            sighting_dict = {
                'camera_id': sighting.camera_id,
                'timestamp': sighting.timestamp.isoformat(),
                'plate_text': sighting.plate_text,
                'confidence': sighting.confidence,
                'location': {
                    'name': sighting.location.location_name,
                    'lat': sighting.location.latitude,
                    'lng': sighting.location.longitude
                } if sighting.location else None
            }
            sightings.append(sighting_dict)
        
        return {
            'global_vehicle_id': global_id,
            'vehicle_info': {
                'plates': list(vehicle_info.get('plates', [])),
                'type': vehicle_info.get('vehicle_type'),
                'make': vehicle_info.get('vehicle_make'),
                'model': vehicle_info.get('vehicle_model'),
                'color': vehicle_info.get('vehicle_color'),
                'year': vehicle_info.get('vehicle_year')
            },
            'journey_info': journey_info,
            'sightings': sightings
        }
    
    def find_vehicles_between_cameras(self, camera_a: str, camera_b: str,
                                    time_window_hours: int = 24) -> List[Dict]:
        """Find vehicles that traveled between two cameras."""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        matches = []
        
        for global_id, journey in self.vehicle_journeys.items():
            # Filter recent sightings
            recent_sightings = [s for s in journey if s.timestamp > cutoff_time]
            
            # Check if vehicle visited both cameras
            cameras_visited = set(s.camera_id for s in recent_sightings)
            if camera_a in cameras_visited and camera_b in cameras_visited:
                # Find transitions
                for i in range(len(recent_sightings) - 1):
                    if (recent_sightings[i].camera_id == camera_a and 
                        recent_sightings[i+1].camera_id == camera_b):
                        
                        transition_time = (recent_sightings[i+1].timestamp - 
                                         recent_sightings[i].timestamp).total_seconds() / 60
                        
                        matches.append({
                            'global_vehicle_id': global_id,
                            'vehicle_info': self.global_vehicles.get(global_id, {}),
                            'from_time': recent_sightings[i].timestamp.isoformat(),
                            'to_time': recent_sightings[i+1].timestamp.isoformat(),
                            'transit_time_minutes': transition_time
                        })
        
        return matches
    
    def get_tracking_statistics(self) -> Dict:
        """Get overall cross-camera tracking statistics."""
        active_count = sum(1 for gid, last_seen in self.active_vehicles.items()
                          if (datetime.now() - last_seen).total_seconds() < 3600)
        
        journey_lengths = [len(j) for j in self.vehicle_journeys.values()]
        avg_journey_length = sum(journey_lengths) / len(journey_lengths) if journey_lengths else 0
        
        return {
            'total_vehicles_tracked': len(self.global_vehicles),
            'active_vehicles_last_hour': active_count,
            'total_detections_processed': self.stats['total_detections'],
            'new_vehicles_identified': self.stats['new_vehicles'],
            'average_journey_length': avg_journey_length,
            'cameras_registered': len(self.cameras),
            'camera_statistics': {
                cam_id: {
                    'location': cam.location_name,
                    'detections': self.stats.get(f'camera_{cam_id}_detections', 0)
                }
                for cam_id, cam in self.cameras.items()
            }
        }
    
    def cleanup_inactive_vehicles(self, hours: int = 48):
        """Remove inactive vehicles from memory."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        vehicles_to_remove = []
        
        for global_id, last_seen in self.active_vehicles.items():
            if last_seen < cutoff_time:
                vehicles_to_remove.append(global_id)
        
        for global_id in vehicles_to_remove:
            del self.active_vehicles[global_id]
            if global_id in self.vehicle_journeys:
                del self.vehicle_journeys[global_id]
            if global_id in self.global_vehicles:
                # Remove plate mappings
                vehicle = self.global_vehicles[global_id]
                if 'plates' in vehicle:
                    for plate in vehicle['plates']:
                        if plate in self.plate_to_global_id:
                            del self.plate_to_global_id[plate]
                del self.global_vehicles[global_id]
        
        logger.info(f"Cleaned up {len(vehicles_to_remove)} inactive vehicles")
        return len(vehicles_to_remove)


# Global instance
cross_camera_tracker = CrossCameraTracker()

# Initialize with some example cameras
default_cameras = [
    CameraInfo("cam_01", "Main Entrance", 40.7128, -74.0060, "N", 50),
    CameraInfo("cam_02", "Parking Lot A", 40.7130, -74.0055, "E", 75),
    CameraInfo("cam_03", "East Gate", 40.7135, -74.0050, "W", 50),
    CameraInfo("cam_04", "West Exit", 40.7125, -74.0065, "S", 50),
]

for camera in default_cameras:
    cross_camera_tracker.register_camera(camera)