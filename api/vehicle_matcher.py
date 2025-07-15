"""
Vehicle matching system using multiple attributes for identification.
"""

import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class VehicleMatcher:
    """Match vehicles using plate, make, model, color, and type."""
    
    def __init__(self):
        # Color groups for fuzzy matching
        self.color_groups = {
            'silver': ['silver', 'gray', 'grey', 'chrome'],
            'white': ['white', 'cream', 'beige', 'pearl'],
            'black': ['black', 'charcoal', 'dark'],
            'blue': ['blue', 'navy', 'azure', 'cobalt'],
            'red': ['red', 'maroon', 'burgundy', 'crimson'],
            'green': ['green', 'olive', 'forest', 'emerald'],
            'brown': ['brown', 'tan', 'bronze', 'copper'],
            'yellow': ['yellow', 'gold', 'amber'],
            'orange': ['orange', 'rust'],
            'purple': ['purple', 'violet', 'lavender']
        }
        
        # Attribute weights for matching
        self.weights = {
            'plate': 0.4,
            'make': 0.2,
            'model': 0.15,
            'color': 0.15,
            'type': 0.1
        }
    
    def calculate_similarity(self, vehicle1: Dict, vehicle2: Dict) -> float:
        """
        Calculate similarity score between two vehicles.
        Returns score between 0.0 and 1.0.
        """
        total_score = 0.0
        total_weight = 0.0
        
        # Plate matching (exact match only)
        if vehicle1.get('plate_text') and vehicle2.get('plate_text'):
            if vehicle1['plate_text'] == vehicle2['plate_text']:
                plate_confidence = min(
                    vehicle1.get('confidence', 0.5),
                    vehicle2.get('confidence', 0.5)
                )
                total_score += self.weights['plate'] * plate_confidence
            total_weight += self.weights['plate']
        
        # Make matching (exact match)
        if vehicle1.get('vehicle_make') and vehicle2.get('vehicle_make'):
            if vehicle1['vehicle_make'].lower() == vehicle2['vehicle_make'].lower():
                make_confidence = min(
                    vehicle1.get('vehicle_make_confidence', 0.5),
                    vehicle2.get('vehicle_make_confidence', 0.5)
                )
                total_score += self.weights['make'] * make_confidence
            total_weight += self.weights['make']
        
        # Model matching (fuzzy match for variations)
        if vehicle1.get('vehicle_model') and vehicle2.get('vehicle_model'):
            model_similarity = self._string_similarity(
                vehicle1['vehicle_model'],
                vehicle2['vehicle_model']
            )
            if model_similarity > 0.8:  # 80% similarity threshold
                model_confidence = min(
                    vehicle1.get('vehicle_model_confidence', 0.5),
                    vehicle2.get('vehicle_model_confidence', 0.5)
                )
                total_score += self.weights['model'] * model_similarity * model_confidence
            total_weight += self.weights['model']
        
        # Color matching (fuzzy match for similar colors)
        if vehicle1.get('vehicle_color') and vehicle2.get('vehicle_color'):
            color_similarity = self._color_similarity(
                vehicle1['vehicle_color'],
                vehicle2['vehicle_color']
            )
            if color_similarity > 0:
                color_confidence = min(
                    vehicle1.get('vehicle_color_confidence', 0.5),
                    vehicle2.get('vehicle_color_confidence', 0.5)
                )
                total_score += self.weights['color'] * color_similarity * color_confidence
            total_weight += self.weights['color']
        
        # Type matching (exact match)
        if vehicle1.get('vehicle_type') and vehicle2.get('vehicle_type'):
            if vehicle1['vehicle_type'].lower() == vehicle2['vehicle_type'].lower():
                type_confidence = min(
                    vehicle1.get('vehicle_type_confidence', 0.5),
                    vehicle2.get('vehicle_type_confidence', 0.5)
                )
                total_score += self.weights['type'] * type_confidence
            total_weight += self.weights['type']
        
        # Normalize score
        if total_weight > 0:
            return total_score / total_weight
        return 0.0
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using SequenceMatcher."""
        if not str1 or not str2:
            return 0.0
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def _color_similarity(self, color1: str, color2: str) -> float:
        """Calculate color similarity considering color groups."""
        if not color1 or not color2:
            return 0.0
        
        color1 = color1.lower()
        color2 = color2.lower()
        
        # Exact match
        if color1 == color2:
            return 1.0
        
        # Check if colors are in the same group
        for group_colors in self.color_groups.values():
            if color1 in group_colors and color2 in group_colors:
                return 0.8  # High similarity for same color group
        
        # Check string similarity for variations not in groups
        return self._string_similarity(color1, color2) * 0.5
    
    def find_matching_vehicles(self, target_vehicle: Dict, 
                             vehicle_database: List[Dict], 
                             threshold: float = 0.7) -> List[Tuple[Dict, float]]:
        """
        Find vehicles in database that match the target vehicle.
        Returns list of (vehicle, similarity_score) tuples.
        """
        matches = []
        
        for vehicle in vehicle_database:
            similarity = self.calculate_similarity(target_vehicle, vehicle)
            if similarity >= threshold:
                matches.append((vehicle, similarity))
        
        # Sort by similarity score (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def is_likely_same_vehicle(self, vehicle1: Dict, vehicle2: Dict, 
                              strict: bool = False) -> bool:
        """
        Determine if two detections are likely the same vehicle.
        
        Args:
            strict: If True, requires higher similarity threshold
        """
        threshold = 0.85 if strict else 0.7
        similarity = self.calculate_similarity(vehicle1, vehicle2)
        return similarity >= threshold
    
    def create_vehicle_signature(self, vehicle: Dict) -> str:
        """
        Create a unique signature for a vehicle based on its attributes.
        Used for tracking when plate is not available.
        """
        parts = []
        
        if vehicle.get('vehicle_type'):
            parts.append(vehicle['vehicle_type'])
        if vehicle.get('vehicle_color'):
            parts.append(vehicle['vehicle_color'])
        if vehicle.get('vehicle_make'):
            parts.append(vehicle['vehicle_make'])
        if vehicle.get('vehicle_model'):
            parts.append(vehicle['vehicle_model'])
        if vehicle.get('vehicle_year'):
            parts.append(vehicle['vehicle_year'])
        
        return '_'.join(parts).lower().replace(' ', '-') if parts else 'unknown'


# Test the matcher
if __name__ == "__main__":
    matcher = VehicleMatcher()
    
    # Test cases
    vehicle_a = {
        'plate_text': 'ABC123',
        'confidence': 0.9,
        'vehicle_make': 'Toyota',
        'vehicle_make_confidence': 0.8,
        'vehicle_model': 'Camry',
        'vehicle_model_confidence': 0.7,
        'vehicle_color': 'Silver',
        'vehicle_color_confidence': 0.9,
        'vehicle_type': 'Sedan',
        'vehicle_type_confidence': 0.95
    }
    
    vehicle_b = {
        'plate_text': 'XYZ789',  # Different plate
        'confidence': 0.85,
        'vehicle_make': 'Toyota',
        'vehicle_make_confidence': 0.85,
        'vehicle_model': 'Camry',
        'vehicle_model_confidence': 0.8,
        'vehicle_color': 'Gray',  # Similar color
        'vehicle_color_confidence': 0.85,
        'vehicle_type': 'Sedan',
        'vehicle_type_confidence': 0.9
    }
    
    similarity = matcher.calculate_similarity(vehicle_a, vehicle_b)
    print(f"Similarity: {similarity:.2%}")
    print(f"Likely same vehicle: {matcher.is_likely_same_vehicle(vehicle_a, vehicle_b)}")