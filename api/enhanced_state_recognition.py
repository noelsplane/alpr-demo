"""
Enhanced state recognition using multiple methods.
"""

import re
from typing import Tuple, Optional, Dict, List
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Comprehensive state patterns with variations
STATE_PATTERNS_ENHANCED = {
    'CA': {
        'patterns': [
            r'^[1-9][A-Z]{3}\d{3}$',  # 1ABC234
            r'^\d[A-Z]{3}\d{3}$',     # 7ABC234
        ],
        'prefixes': ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'format': 'NLLLNNN',  # N=Number, L=Letter
    },
    'NY': {
        'patterns': [
            r'^[A-Z]{3}\d{4}$',       # ABC1234
            r'^[A-Z]{3}\d{3}[A-Z]$',  # ABC123A
        ],
        'format': 'LLLNNNN or LLLNNNL',
    },
    'TX': {
        'patterns': [
            r'^[A-Z]{3}\d{4}$',       # ABC1234
            r'^[A-Z]{2}\d[A-Z]\d{3}$', # AB1C234
        ],
        'format': 'LLLNNNN',
    },
    'FL': {
        'patterns': [
            r'^[A-Z]{4}\d{2}$',       # ABCD12
            r'^[A-Z]\d{2}[A-Z]{3}$',  # A12BCD
            r'^[A-Z]{3}[A-Z]\d{2}$',  # ABCD12
        ],
        'format': 'LLLLNN or LNNLLL',
    },
    'IL': {
        'patterns': [
            r'^[A-Z]\d{5,6}$',        # A12345
            r'^[A-Z]{2}\d{5}$',       # AB12345
        ],
        'format': 'LNNNNN or LLNNNNN',
    },
    'PA': {
        'patterns': [
            r'^[A-Z]{3}\d{4}$',       # ABC1234
        ],
        'format': 'LLLNNNN',
    },
    'OH': {
        'patterns': [
            r'^[A-Z]{3}\d{4}$',       # ABC1234
            r'^[A-Z]{3}\d{3}[A-Z]$',  # ABC123A
        ],
        'format': 'LLLNNNN',
    },
    'NJ': {
        'patterns': [
            r'^[A-Z]\d{2}[A-Z]{3}$',  # A12BCD
            r'^[A-Z]{3}\d{2}[A-Z]$',  # ABC12D
        ],
        'format': 'LNNLLL',
    },
    'MI': {
        'patterns': [
            r'^[A-Z]{3}\d{4}$',       # ABC1234
            r'^[A-Z]{3}\d{3}[A-Z]$',  # ABC123A
            r'^\d[A-Z]{2}\d{3}$',     # 1AB234
        ],
        'format': 'LLLNNNN or NLLLNNN',
    },
    'AZ': {
        'patterns': [
            r'^[A-Z]{3}\d{4}$',       # ABC1234
            r'^[A-Z]{2}\d{5}$',       # AB12345
        ],
        'format': 'LLLNNNN or LLNNNNN',
    },
}

# State name keywords that might appear in the image
STATE_KEYWORDS = {
    'CALIFORNIA': 'CA', 'TEXAS': 'TX', 'NEW YORK': 'NY', 'FLORIDA': 'FL',
    'ILLINOIS': 'IL', 'PENNSYLVANIA': 'PA', 'OHIO': 'OH', 'GEORGIA': 'GA',
    'MICHIGAN': 'MI', 'ARIZONA': 'AZ', 'JERSEY': 'NJ', 'GARDEN STATE': 'NJ',
    'GOLDEN STATE': 'CA', 'SUNSHINE': 'FL', 'EMPIRE': 'NY', 'LONE STAR': 'TX',
}

class EnhancedStateRecognizer:
    def __init__(self):
        self.patterns = STATE_PATTERNS_ENHANCED
        self.keywords = STATE_KEYWORDS
        
    def recognize_state(self, plate_text: str, full_image_text: str = "") -> Tuple[Optional[str], float]:
        """
        Recognize state using multiple methods.
        
        Args:
            plate_text: The detected license plate text
            full_image_text: Any other text detected in the image (for context)
            
        Returns:
            Tuple of (state_code, confidence)
        """
        plate_clean = self._clean_text(plate_text)
        
        # Method 1: Pattern matching (60% weight)
        pattern_state, pattern_conf = self._match_patterns(plate_clean)
        
        # Method 2: Keyword detection (20% weight)
        keyword_state, keyword_conf = self._detect_keywords(full_image_text)
        
        # Method 3: Format analysis (20% weight)
        format_state, format_conf = self._analyze_format(plate_clean)
        
        # Combine results
        results = {}
        
        if pattern_state:
            results[pattern_state] = results.get(pattern_state, 0) + pattern_conf * 0.6
            
        if keyword_state and keyword_state == pattern_state:
            # Boost confidence if keyword matches pattern
            results[keyword_state] = results.get(keyword_state, 0) + keyword_conf * 0.3
        elif keyword_state:
            results[keyword_state] = results.get(keyword_state, 0) + keyword_conf * 0.2
            
        if format_state:
            results[format_state] = results.get(format_state, 0) + format_conf * 0.2
        
        # Get best result
        if results:
            best_state = max(results.items(), key=lambda x: x[1])
            return best_state[0], min(best_state[1], 1.0)
            
        return None, 0.0
    
    def _clean_text(self, text: str) -> str:
        """Clean text for pattern matching."""
        return text.upper().replace(" ", "").replace("-", "")
    
    def _match_patterns(self, text: str) -> Tuple[Optional[str], float]:
        """Match against known state patterns."""
        matches = []
        
        for state, info in self.patterns.items():
            for pattern in info['patterns']:
                if re.match(pattern, text):
                    matches.append((state, 0.9))  # High confidence for exact match
                    
        if matches:
            return matches[0]
            
        # Try fuzzy matching
        for state, info in self.patterns.items():
            for pattern in info['patterns']:
                # Convert pattern to a simpler format for fuzzy matching
                pattern_simple = pattern.replace('^', '').replace('$', '').replace('\\d', 'N').replace('[A-Z]', 'L')
                text_format = self._get_format(text)
                
                similarity = SequenceMatcher(None, pattern_simple, text_format).ratio()
                if similarity > 0.8:
                    matches.append((state, similarity * 0.7))
        
        if matches:
            return max(matches, key=lambda x: x[1])
            
        return None, 0.0
    
    def _detect_keywords(self, text: str) -> Tuple[Optional[str], float]:
        """Detect state keywords in the image."""
        text_upper = text.upper()
        
        for keyword, state in self.keywords.items():
            if keyword in text_upper:
                return state, 0.8
                
        return None, 0.0
    
    def _analyze_format(self, text: str) -> Tuple[Optional[str], float]:
        """Analyze the format of the plate."""
        format_str = self._get_format(text)
        
        # Find states that match this format
        matches = []
        
        for state, info in self.patterns.items():
            if 'format' in info:
                if format_str in info['format']:
                    matches.append((state, 0.6))
                    
        if matches:
            # If multiple states match, return with lower confidence
            confidence = 0.6 / len(matches)
            return matches[0][0], confidence
            
        return None, 0.0
    
    def _get_format(self, text: str) -> str:
        """Convert text to format string (L for letter, N for number)."""
        format_str = ""
        for char in text:
            if char.isalpha():
                format_str += "L"
            elif char.isdigit():
                format_str += "N"
        return format_str

# Create global instance
enhanced_recognizer = EnhancedStateRecognizer()

def recognize_state_enhanced(plate_text: str, context_text: str = "") -> Tuple[Optional[str], float]:
    """Enhanced state recognition function."""
    return enhanced_recognizer.recognize_state(plate_text, context_text)
