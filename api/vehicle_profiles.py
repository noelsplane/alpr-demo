"""
Vehicle profile aggregation system that combines multiple detections
into comprehensive vehicle profiles over time.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json
from sqlalchemy import create_engine, func, and_, or_
from sqlalchemy.orm import sessionmaker
from models import PlateDetection
from vehicle_matcher import VehicleMatcher

logger = logging.getLogger(__name__)

class VehicleProfile:
    """Represents a comprehensive vehicle profile built from multiple detections."""
    
    def __init__(self, profile_id: str):
        self.profile_id = profile_id
        self.plate_numbers: Set[str] = set()
        self.appearances: List[Dict] = []
        self.vehicle_attributes = {
            'make': None,
            'model': None,
            'color': None,
            'type': None,
            'year': None,
            'make_confidence': 0.0,
            'model_confidence': 0.0,
            'color_confidence': 0.0,
            'type_confidence': 0.0,
            'year_confidence': 0.0
        }
        self.first_seen = None
        self.last_seen = None
        self.total_sightings = 0
        self.states_seen: Set[str] = set()
        self.anomalies: List[Dict] = []
        self.confidence_score = 0.0
    
    def add_detection(self, detection: PlateDetection):
        """Add a detection to this profile."""
        # Add plate number
        if detection.plate_text:
            self.plate_numbers.add(detection.plate_text)
        
        # Update vehicle attributes with highest confidence values
        if detection.vehicle_make and detection.vehicle_make_confidence > self.vehicle_attributes['make_confidence']:
            self.vehicle_attributes['make'] = detection.vehicle_make
            self.vehicle_attributes['make_confidence'] = detection.vehicle_make_confidence
        
        if detection.vehicle_model and detection.vehicle_model_confidence > self.vehicle_attributes['model_confidence']:
            self.vehicle_attributes['model'] = detection.vehicle_model
            self.vehicle_attributes['model_confidence'] = detection.vehicle_model_confidence
        
        if detection.vehicle_color and detection.vehicle_color_confidence > self.vehicle_attributes['color_confidence']:
            self.vehicle_attributes['color'] = detection.vehicle_color
            self.vehicle_attributes['color_confidence'] = detection.vehicle_color_confidence
        
        if detection.vehicle_type and detection.vehicle_type_confidence > self.vehicle_attributes['type_confidence']:
            self.vehicle_attributes['type'] = detection.vehicle_type
            self.vehicle_attributes['type_confidence'] = detection.vehicle_type_confidence
        
        if detection.vehicle_year and detection.vehicle_year_confidence > self.vehicle_attributes['year_confidence']:
            self.vehicle_attributes['year'] = detection.vehicle_year
            self.vehicle_attributes['year_confidence'] = detection.vehicle_year_confidence
        
        # Add appearance record with plate image
        appearance = {
            'detection_id': detection.id,
            'timestamp': detection.timestamp.isoformat(),
            'plate_text': detection.plate_text,
            'state': detection.state,
            'state_confidence': detection.state_confidence,
            'image_name': detection.image_name,
            'confidence': detection.confidence,
            'plate_image_base64': detection.plate_image_base64  # Include the plate image
        }
        self.appearances.append(appearance)
        
        # Update timestamps
        if not self.first_seen or detection.timestamp < datetime.fromisoformat(self.first_seen):
            self.first_seen = detection.timestamp.isoformat()
        
        if not self.last_seen or detection.timestamp > datetime.fromisoformat(self.last_seen):
            self.last_seen = detection.timestamp.isoformat()
        
        # Update states seen
        if detection.state:
            self.states_seen.add(detection.state)
        
        self.total_sightings += 1
        
        # Check for anomalies
        self._check_anomalies(detection)
    
    def _check_anomalies(self, detection: PlateDetection):
        """Check for anomalies in the detection."""
        # Check for plate switching
        if len(self.plate_numbers) > 1 and detection.plate_text not in self.plate_numbers:
            self.anomalies.append({
                'type': 'PLATE_SWITCH',
                'timestamp': detection.timestamp.isoformat(),
                'details': f'New plate {detection.plate_text} detected, previously seen: {", ".join(self.plate_numbers)}'
            })
        
        # Check for state changes
        if len(self.states_seen) > 1 and detection.state and detection.state not in self.states_seen:
            self.anomalies.append({
                'type': 'STATE_CHANGE',
                'timestamp': detection.timestamp.isoformat(),
                'details': f'New state {detection.state} detected, previously seen: {", ".join(self.states_seen)}'
            })
    
    def calculate_confidence(self):
        """Calculate overall confidence score for this profile."""
        # Factors that increase confidence:
        # - More sightings
        # - Consistent plate numbers
        # - High confidence vehicle attributes
        # - Recent sightings
        
        confidence = 0.0
        
        # Sightings factor (up to 0.3)
        sightings_score = min(self.total_sightings / 10, 1.0) * 0.3
        confidence += sightings_score
        
        # Plate consistency factor (up to 0.3)
        if self.total_sightings > 0:
            plate_consistency = 1.0 if len(self.plate_numbers) == 1 else 0.5
            confidence += plate_consistency * 0.3
        
        # Vehicle attributes factor (up to 0.3)
        attr_scores = []
        for attr in ['make', 'model', 'color', 'type']:
            if self.vehicle_attributes[attr]:
                attr_scores.append(self.vehicle_attributes[f'{attr}_confidence'])
        
        if attr_scores:
            avg_attr_confidence = sum(attr_scores) / len(attr_scores)
            confidence += avg_attr_confidence * 0.3
        
        # Recency factor (up to 0.1)
        if self.last_seen:
            last_seen_date = datetime.fromisoformat(self.last_seen)
            days_since = (datetime.now() - last_seen_date).days
            recency_score = max(0, 1 - (days_since / 30))  # Decay over 30 days
            confidence += recency_score * 0.1
        
        self.confidence_score = min(confidence, 1.0)
        return self.confidence_score
    
    def to_dict(self) -> Dict:
        """Convert profile to dictionary for serialization."""
        # Get sample images (up to 6 most recent)
        sample_images = []
        for appearance in self.appearances[-6:]:
            if appearance.get('plate_image_base64'):
                sample_images.append({
                    'plate_text': appearance['plate_text'],
                    'timestamp': appearance['timestamp'],
                    'image': appearance['plate_image_base64'],
                    'confidence': appearance['confidence']
                })
        
        return {
            'profile_id': self.profile_id,
            'plate_numbers': list(self.plate_numbers),
            'vehicle_attributes': self.vehicle_attributes,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'total_sightings': self.total_sightings,
            'states_seen': list(self.states_seen),
            'confidence_score': self.calculate_confidence(),
            'anomalies': self.anomalies,
            'recent_appearances': self.appearances[-5:],  # Last 5 appearances
            'sample_images': sample_images  # Include sample images
        }


class VehicleProfileAggregator:
    """Manages vehicle profiles and aggregates detections."""
    
    def __init__(self, db_path: str = 'sqlite:///detections.db'):
        self.engine = create_engine(db_path)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.vehicle_matcher = VehicleMatcher()
        self.profiles: Dict[str, VehicleProfile] = {}
        self._profile_index: Dict[str, str] = {}  # Maps plate_text to profile_id
    
    def build_profiles(self, time_window_hours: Optional[int] = None):
        """Build vehicle profiles from detection history."""
        db = self.SessionLocal()
        
        try:
            # Query detections
            query = db.query(PlateDetection).order_by(PlateDetection.timestamp)
            
            if time_window_hours:
                cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
                query = query.filter(PlateDetection.timestamp >= cutoff_time)
            
            detections = query.all()
            logger.info(f"Building profiles from {len(detections)} detections")
            
            # Clear existing profiles
            self.profiles.clear()
            self._profile_index.clear()
            
            # Process each detection
            for detection in detections:
                self._add_detection_to_profile(detection)
            
            # Calculate confidence scores
            for profile in self.profiles.values():
                profile.calculate_confidence()
            
            logger.info(f"Built {len(self.profiles)} vehicle profiles")
            
        finally:
            db.close()
    
    def _add_detection_to_profile(self, detection: PlateDetection):
        """Add a detection to the appropriate profile."""
        # First check if plate already assigned to a profile
        if detection.plate_text in self._profile_index:
            profile_id = self._profile_index[detection.plate_text]
            self.profiles[profile_id].add_detection(detection)
            return
        
        # Try to match with existing profiles
        matched_profile_id = self._find_matching_profile(detection)
        
        if matched_profile_id:
            # Add to existing profile
            self.profiles[matched_profile_id].add_detection(detection)
            if detection.plate_text:
                self._profile_index[detection.plate_text] = matched_profile_id
        else:
            # Create new profile
            profile_id = self._generate_profile_id(detection)
            new_profile = VehicleProfile(profile_id)
            new_profile.add_detection(detection)
            
            self.profiles[profile_id] = new_profile
            if detection.plate_text:
                self._profile_index[detection.plate_text] = profile_id
    
    def _find_matching_profile(self, detection: PlateDetection) -> Optional[str]:
        """Find a profile that matches this detection."""
        # Convert detection to dict format for matcher
        detection_dict = {
            'plate_text': detection.plate_text,
            'confidence': detection.confidence,
            'vehicle_make': detection.vehicle_make,
            'vehicle_make_confidence': detection.vehicle_make_confidence or 0,
            'vehicle_model': detection.vehicle_model,
            'vehicle_model_confidence': detection.vehicle_model_confidence or 0,
            'vehicle_color': detection.vehicle_color,
            'vehicle_color_confidence': detection.vehicle_color_confidence or 0,
            'vehicle_type': detection.vehicle_type,
            'vehicle_type_confidence': detection.vehicle_type_confidence or 0
        }
        
        best_match_id = None
        best_similarity = 0.0
        
        for profile_id, profile in self.profiles.items():
            # Create a representative dict from profile
            profile_dict = {
                'plate_text': list(profile.plate_numbers)[0] if profile.plate_numbers else None,
                'confidence': 0.9,  # High confidence for established profiles
                'vehicle_make': profile.vehicle_attributes['make'],
                'vehicle_make_confidence': profile.vehicle_attributes['make_confidence'],
                'vehicle_model': profile.vehicle_attributes['model'],
                'vehicle_model_confidence': profile.vehicle_attributes['model_confidence'],
                'vehicle_color': profile.vehicle_attributes['color'],
                'vehicle_color_confidence': profile.vehicle_attributes['color_confidence'],
                'vehicle_type': profile.vehicle_attributes['type'],
                'vehicle_type_confidence': profile.vehicle_attributes['type_confidence']
            }
            
            similarity = self.vehicle_matcher.calculate_similarity(detection_dict, profile_dict)
            
            # Higher threshold for matching to existing profile
            if similarity > 0.75 and similarity > best_similarity:
                best_similarity = similarity
                best_match_id = profile_id
        
        return best_match_id
    
    def _generate_profile_id(self, detection: PlateDetection) -> str:
        """Generate a unique profile ID."""
        # Use combination of attributes for ID
        parts = []
        
        if detection.vehicle_type:
            parts.append(detection.vehicle_type.lower().replace(' ', '-'))
        if detection.vehicle_color:
            parts.append(detection.vehicle_color.lower())
        if detection.vehicle_make:
            parts.append(detection.vehicle_make.lower())
        
        base_id = '_'.join(parts) if parts else 'vehicle'
        
        # Add timestamp to ensure uniqueness
        timestamp = int(detection.timestamp.timestamp())
        return f"{base_id}_{timestamp}"
    
    def get_profile_by_plate(self, plate_text: str) -> Optional[VehicleProfile]:
        """Get vehicle profile by plate number."""
        profile_id = self._profile_index.get(plate_text)
        if profile_id:
            return self.profiles.get(profile_id)
        
        # Search through all profiles
        for profile in self.profiles.values():
            if plate_text in profile.plate_numbers:
                return profile
        
        return None
    
    def get_profiles_by_vehicle_type(self, vehicle_type: str) -> List[VehicleProfile]:
        """Get all profiles matching a vehicle type."""
        matching_profiles = []
        
        for profile in self.profiles.values():
            if profile.vehicle_attributes['type'] and \
               profile.vehicle_attributes['type'].lower() == vehicle_type.lower():
                matching_profiles.append(profile)
        
        return matching_profiles
    
    def get_suspicious_profiles(self) -> List[VehicleProfile]:
        """Get profiles with anomalies or suspicious behavior."""
        suspicious = []
        
        for profile in self.profiles.values():
            # Check for multiple plates
            if len(profile.plate_numbers) > 1:
                suspicious.append(profile)
                continue
            
            # Check for anomalies
            if profile.anomalies:
                suspicious.append(profile)
                continue
            
            # Check for frequent appearances (possible loitering)
            if profile.total_sightings > 10:
                # Calculate appearance frequency
                if profile.first_seen and profile.last_seen:
                    first = datetime.fromisoformat(profile.first_seen)
                    last = datetime.fromisoformat(profile.last_seen)
                    time_span_hours = (last - first).total_seconds() / 3600
                    
                    if time_span_hours > 0:
                        frequency = profile.total_sightings / time_span_hours
                        if frequency > 2:  # More than 2 sightings per hour
                            suspicious.append(profile)
        
        return suspicious
    
    def get_all_profiles(self, min_confidence: float = 0.0) -> List[Dict]:
        """Get all profiles above minimum confidence threshold."""
        profiles = []
        
        for profile in self.profiles.values():
            if profile.calculate_confidence() >= min_confidence:
                profiles.append(profile.to_dict())
        
        # Sort by confidence score
        profiles.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return profiles
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics about vehicle profiles."""
        total_profiles = len(self.profiles)
        total_detections = sum(p.total_sightings for p in self.profiles.values())
        
        profiles_with_anomalies = sum(1 for p in self.profiles.values() if p.anomalies)
        profiles_with_multiple_plates = sum(1 for p in self.profiles.values() if len(p.plate_numbers) > 1)
        
        vehicle_type_distribution = defaultdict(int)
        for profile in self.profiles.values():
            if profile.vehicle_attributes['type']:
                vehicle_type_distribution[profile.vehicle_attributes['type']] += 1
        
        return {
            'total_profiles': total_profiles,
            'total_detections': total_detections,
            'avg_sightings_per_vehicle': total_detections / total_profiles if total_profiles > 0 else 0,
            'profiles_with_anomalies': profiles_with_anomalies,
            'profiles_with_multiple_plates': profiles_with_multiple_plates,
            'vehicle_type_distribution': dict(vehicle_type_distribution),
            'high_confidence_profiles': sum(1 for p in self.profiles.values() if p.calculate_confidence() > 0.7)
        }


# Convenience functions
def build_vehicle_profiles(db_path: str = 'sqlite:///detections.db', 
                         time_window_hours: Optional[int] = None) -> VehicleProfileAggregator:
    """Build vehicle profiles from detection history."""
    aggregator = VehicleProfileAggregator(db_path)
    aggregator.build_profiles(time_window_hours)
    return aggregator


def get_vehicle_profile(plate_text: str, 
                       db_path: str = 'sqlite:///detections.db') -> Optional[Dict]:
    """Get vehicle profile for a specific plate."""
    aggregator = VehicleProfileAggregator(db_path)
    aggregator.build_profiles()
    
    profile = aggregator.get_profile_by_plate(plate_text)
    if profile:
        return profile.to_dict()
    return None