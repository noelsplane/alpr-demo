"""
State classification model wrapper.
Provides pattern-based state classification with future ONNX model support.
"""

import os
import logging
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# US states list
STATE_CLASSES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
]

# Import pattern matcher
try:
    from state_patterns import StatePatternMatcher
except ImportError:
    logger.warning("state_patterns module not found, using basic patterns")
    
    class StatePatternMatcher:
        def __init__(self):
            self.basic_patterns = {
                'CA': r'^[1-9][A-Z]{3}\d{3}$',  # California: 7ABC123
                'TX': r'^[A-Z]{3}\d{4}$',        # Texas: ABC1234
                'NY': r'^[A-Z]{3}\d{4}$',        # New York: ABC1234
                'FL': r'^[A-Z]{4}\d{2}$',        # Florida: ABCD12
                'NJ': r'^[A-Z]\d{2}[A-Z]{3}$',   # New Jersey: A12BCD
            }
        
        def extract_state_from_text(self, plate_text: str) -> Tuple[Optional[str], float]:
            import re
            if not plate_text:
                return None, 0.0
            
            cleaned = plate_text.upper().strip().replace(' ', '').replace('-', '')
            
            for state, pattern in self.basic_patterns.items():
                if re.match(pattern, cleaned):
                    return state, 0.8
            
            return None, 0.0


class StateClassifier:
    """
    State classification for license plates.
    Currently uses pattern matching, with infrastructure for future ML models.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the state classifier.
        
        Args:
            model_path: Path to ONNX model file (for future use)
        """
        self.model_path = model_path
        self.pattern_matcher = StatePatternMatcher()
        
        # Log initialization
        if model_path and os.path.exists(model_path):
            logger.info(f"State classifier initialized with model: {model_path}")
        else:
            logger.info("State classifier using pattern-based classification")
    
    def classify_from_text(self, plate_text: str) -> Tuple[Optional[str], float]:
        """
        Classify state from OCR text using pattern matching.
        
        Args:
            plate_text: OCR-extracted license plate text
            
        Returns:
            Tuple of (state_code, confidence)
        """
        if not plate_text:
            return None, 0.0
            
        return self.pattern_matcher.extract_state_from_text(plate_text)
    
    def get_supported_states(self) -> list:
        """Get list of supported state codes."""
        # Check if pattern matcher has custom list
        if hasattr(self.pattern_matcher, 'patterns'):
            return list(self.pattern_matcher.patterns.keys())
        return STATE_CLASSES


# Global instance
_classifier = None

def get_state_classifier(model_path: Optional[str] = None) -> StateClassifier:
    """
    Get or create global state classifier instance.
    
    Args:
        model_path: Optional path to ONNX model
        
    Returns:
        StateClassifier instance
    """
    global _classifier
    if _classifier is None:
        _classifier = StateClassifier(model_path=model_path)
    return _classifier


# Convenience function for direct use
def classify_state(plate_text: str) -> Tuple[Optional[str], float]:
    """
    Convenience function to classify state from plate text.
    
    Args:
        plate_text: License plate text
        
    Returns:
        Tuple of (state_code, confidence)
    """
    classifier = get_state_classifier()
    return classifier.classify_from_text(plate_text)