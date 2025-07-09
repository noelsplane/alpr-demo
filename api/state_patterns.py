"""
State pattern recognition for license plates - All 50 US States + DC.
This module provides regex-based state detection patterns.
"""

import re
from typing import Optional, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)

# Complete state patterns for all US states
STATE_PATTERNS = {
    'AL': [
        r'^[0-9]{1,7}$',              # 1234567
        r'^[0-9][A-Z]{2}[0-9]{4}$',   # 1AB1234
        r'^[A-Z]{1,3}[0-9]{3,4}$',    # ABC123, ABC1234
        r'^[0-9]{2}[A-Z]{3}[0-9]{2}$', # 12ABC34
        r'^[A-Z]\d{6}$',              # A123456
    ],
    'AK': [
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[0-9]{3}[A-Z]{3}$',        # 123ABC
        r'^[A-Z]{2}[0-9]{4}$',        # AB1234
    ],
    'AZ': [
        r'^[A-Z]{3}[0-9]{4}$',        # ABC1234
        r'^[A-Z][0-9]{6}$',           # A123456
        r'^[0-9]{3}[A-Z]{3}$',        # 123ABC
        r'^[A-Z]{2}[0-9]{5}$',        # AB12345
    ],
    'AR': [
        r'^[0-9]{3}[A-Z]{3}$',        # 123ABC
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[0-9]{1,7}$',              # 1234567
        r'^[A-Z]{3}\s[0-9]{3}$',      # ABC 123
    ],
    'CA': [
        r'^[1-9][A-Z]{3}[0-9]{3}$',   # 1ABC234
        r'^[0-9][A-Z]{3}[0-9]{3}$',   # 7ABC234
        r'^[A-Z]{1,2}[0-9]{4,5}$',    # AB1234, AB12345 (commercial)
        r'^[0-9]{3}[A-Z]{3}$',        # 123ABC
    ],
    'CO': [
        r'^[A-Z]{3}-[0-9]{3}$',       # ABC-123
        r'^[0-9]{3}-[A-Z]{3}$',       # 123-ABC
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[0-9]{3}[A-Z]{3}$',        # 123ABC
        r'^[A-Z]{2}[0-9]{4}$',        # AB1234
        r'^[0-9]{2}[A-Z]{4}$',        # 12ABCD
    ],
    'CT': [
        r'^[A-Z]{2}[0-9]{5}$',        # AB12345
        r'^[A-Z]{2}\s[0-9]{5}$',      # AB 12345
        r'^[0-9][A-Z][0-9]{5}$',      # 1A23456
        r'^[0-9][A-Z]\s[0-9]{5}$',    # 1A 23456
        r'^[A-Z]{3}[0-9]{4}$',        # ABC1234
    ],
    'DE': [
        r'^[0-9]{6}$',                # 123456
        r'^[0-9]{7}$',                # 1234567
        r'^PC[0-9]{4,5}$',            # PC1234 or PC12345
        r'^[A-Z]{1,2}[0-9]{3,4}$',    # AB123, AB1234
    ],
    'FL': [
        r'^[A-Z]{4}[0-9]{2}$',        # ABCD12
        r'^[A-Z][0-9]{2}[A-Z]{3}$',   # A12BCD
        r'^[A-Z]{3}[A-Z][0-9]{2}$',   # ABCD12
        r'^[A-Z][0-9]{2}\s[A-Z]{3}$', # A12 BCD
        r'^[A-Z]{3}\s[A-Z][0-9]{2}$', # ABC D12
    ],
    'GA': [
        r'^[A-Z]{3}[0-9]{4}$',        # ABC1234
        r'^[A-Z]{3}[0-9]{3}[A-Z]$',   # ABC123A
        r'^[0-9]{3}[A-Z]{4}$',        # 123ABCD
        r'^[0-9]{1,7}$',              # 1234567
    ],
    'HI': [
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[A-Z]{2}[0-9]{4}$',        # AB1234
        r'^[A-Z][0-9]{5}$',           # A12345
        r'^[0-9]{1,6}$',              # 123456
    ],
    'ID': [
        r'^[A-Z][0-9]{6}$',           # A123456
        r'^[0-9][A-Z][0-9]{5}$',      # 1A23456
        r'^[A-Z]{2}[0-9]{5}$',        # AB12345
        r'^[A-Z]\s[0-9]{6}$',         # A 123456
        r'^[A-Z]{1,2}[0-9]{5,6}$',    # A12345, AB12345
    ],
    'IL': [
        r'^[A-Z][0-9]{5,6}$',         # A12345 or A123456
        r'^[A-Z]{2}[0-9]{5}$',        # AB12345
        r'^[A-Z]\s[0-9]{5,6}$',       # A 12345
        r'^[A-Z]{2}\s[0-9]{5}$',      # AB 12345
        r'^[A-Z]{1,3}[0-9]{4}$',      # ABC1234
    ],
    'IN': [
        r'^[0-9]{3}[A-Z]{3}$',        # 123ABC
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[0-9]{2}[A-Z][0-9]{3}$',   # 12A345
        r'^[0-9]{4}[A-Z]{2}$',        # 1234AB
        r'^[0-9]{1,7}$',              # 1234567
    ],
    'IA': [
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[A-Z]{3}[0-9]{4}$',        # ABC1234
        r'^[0-9]{3}[A-Z]{3}$',        # 123ABC
        r'^[A-Z]{2}[0-9]{4}$',        # AB1234
    ],
    'KS': [
        r'^[0-9]{3}[A-Z]{3}$',        # 123ABC
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[0-9]{1,7}$',              # 1234567
        r'^[A-Z]{2}[0-9]{4}$',        # AB1234
    ],
    'KY': [
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[0-9]{3}[A-Z]{3}$',        # 123ABC
        r'^[A-Z]{3}[0-9]{4}$',        # ABC1234
        r'^[0-9]{1,7}$',              # 1234567
    ],
    'LA': [
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[A-Z]{3}[0-9]{4}$',        # ABC1234
        r'^[0-9]{3}[A-Z]{3}$',        # 123ABC
        r'^[0-9]{1,7}$',              # 1234567
    ],
    'ME': [
        r'^[0-9]{4}[A-Z]{2}$',        # 1234AB
        r'^[A-Z]{2}[0-9]{4}$',        # AB1234
        r'^[0-9]{3}[A-Z]{3}$',        # 123ABC
        r'^[0-9]{1,7}$',              # 1234567
    ],
    'MD': [
        r'^[0-9][A-Z]{2}[0-9]{4}$',   # 1AB2345
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[A-Z]{2}[0-9]{4}[A-Z]$',   # AB1234C
        r'^[A-Z]{3}[0-9]{4}$',        # ABC1234
    ],
    'MA': [
        r'^[0-9][A-Z]{2}[0-9]{3}$',   # 1AB234
        r'^[A-Z]{2}[0-9]{4}$',        # AB1234
        r'^[0-9]{3}[A-Z]{3}$',        # 123ABC
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[0-9]{1,7}$',              # 1234567
    ],
    'MI': [
        r'^[A-Z]{3}[0-9]{4}$',        # ABC1234
        r'^[A-Z]{3}[0-9]{3}[A-Z]$',   # ABC123A
        r'^[A-Z]{3}\s[0-9]{4}$',      # ABC 1234
        r'^[0-9][A-Z]{2}[0-9]{3}$',   # 1AB234
    ],
    'MN': [
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[0-9]{3}[A-Z]{3}$',        # 123ABC
        r'^[A-Z]{3}\s[0-9]{3}$',      # ABC 123
        r'^[0-9]{1,7}$',              # 1234567
    ],
    'MS': [
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[A-Z]{3}[0-9]{4}$',        # ABC1234
        r'^[A-Z]{2}[0-9]{5}$',        # AB12345
        r'^[0-9]{1,7}$',              # 1234567
    ],
    'MO': [
        r'^[A-Z]{2}[0-9][A-Z][0-9]{2}$',    # AB1C23
        r'^[A-Z][0-9]{2}[A-Z]{3}$',          # A12BCD
        r'^[A-Z]{2}[0-9]{4}$',               # AB1234
        r'^[A-Z]{2}[0-9][A-Z]{2}[0-9]$',     # AB1CD2
    ],
    'MT': [
        r'^[0-9]{2}-[0-9]{4}[A-Z]$',  # 12-3456A
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[0-9][A-Z][0-9]{4}$',      # 1A2345
        r'^[0-9]{1,7}[A-Z]$',         # 1234567A
    ],
    'NE': [
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[0-9]{2}[A-Z][0-9]{3}$',   # 12A345 (county prefix)
        r'^[A-Z][0-9]{5}$',           # A12345 (commercial)
        r'^[0-9]{1,7}$',              # 1234567
    ],
    'NV': [
        r'^[0-9]{3}[A-Z]{3}$',        # 123ABC
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[A-Z]{2}[0-9]{4}$',        # AB1234
        r'^[0-9]{2}[A-Z][0-9]{3}$',   # 12A345
    ],
    'NH': [
        r'^[0-9]{3}\s[0-9]{4}$',      # 123 4567
        r'^[0-9]{7}$',                # 1234567
        r'^[A-Z]{3}[0-9]{4}$',        # ABC1234
    ],
    'NJ': [
        r'^[A-Z][0-9]{2}[A-Z]{3}$',   # A12BCD
        r'^[A-Z][0-9]{2}\s[A-Z]{3}$', # A12 BCD
        r'^[A-Z]{3}[0-9]{2}[A-Z]$',   # ABC12D
        r'^[A-Z]{3}\s[0-9]{2}[A-Z]$', # ABC 12D
        r'^[A-Z]{1}[0-9]{2}[A-Z]{3}$', # X12YZA
    ],
    'NM': [
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[0-9]{3}[A-Z]{3}$',        # 123ABC
        r'^[A-Z]{2}[0-9]{4}$',        # AB1234
        r'^[0-9]{3}\s[A-Z]{3}$',      # 123 ABC
        r'^[A-Z]{3}[0-9]{3,4}$',      # ABC123, ABC1234
    ],
    'NY': [
        r'^[A-Z]{3}[0-9]{4}$',        # ABC1234
        r'^[A-Z]{3}\s[0-9]{4}$',      # ABC 1234
        r'^[A-Z]{3}[0-9]{3}[A-Z]$',   # ABC123A
        r'^[A-Z]{3}\s[0-9]{3}[A-Z]$', # ABC 123A
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[0-9]{1,7}$',              # 1234567
    ],
    'NC': [
        r'^[A-Z]{3}[0-9]{4}$',        # ABC1234
        r'^[A-Z]{3}\s[0-9]{4}$',      # ABC 1234
        r'^[A-Z]{2}[0-9]{5}$',        # AB12345
    ],
    'ND': [
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[0-9]{3}[A-Z]{3}$',        # 123ABC
        r'^[A-Z]{2}[0-9]{4}$',        # AB1234
    ],
    'OH': [
        r'^[A-Z]{3}[0-9]{4}$',        # ABC1234
        r'^[A-Z]{3}\s[0-9]{4}$',      # ABC 1234
        r'^[A-Z]{3}[0-9]{3}[A-Z]$',   # ABC123A
        r'^[A-Z]{2}[0-9]{5}$',        # AB12345
    ],
    'OK': [
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[0-9]{3}[A-Z]{3}$',        # 123ABC
        r'^[A-Z]{3}[0-9]{4}$',        # ABC1234
        r'^[0-9]{1,7}$',              # 1234567
    ],
    'OR': [
        r'^[0-9]{3}[A-Z]{3}$',        # 123ABC
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[0-9]{1,6}$',              # 123456
        r'^[A-Z]{2}[0-9]{4}$',        # AB1234
        r'^[A-Z]{3}\s[0-9]{3}$',      # ABC 123
    ],
    'PA': [
        r'^[A-Z]{3}[0-9]{4}$',        # ABC1234
        r'^[A-Z]{3}\s[0-9]{4}$',      # ABC 1234
        r'^[A-Z]{3}[A-Z][0-9]{3}$',   # ABCD123
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[A-Z]{2}[0-9]{5}$',        # AB12345
    ],
    'RI': [
        r'^[0-9]{6}$',                # 123456
        r'^[A-Z]{2}[0-9]{3}$',        # AB123
        r'^[A-Z]{2}\s[0-9]{3}$',      # AB 123
        r'^[0-9]{3}\s[0-9]{3}$',      # 123 456
    ],
    'SC': [
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[A-Z]{3}\s[0-9]{3}$',      # ABC 123
        r'^[A-Z]{3}[0-9]{4}$',        # ABC1234
    ],
    'SD': [
        r'^[0-9][A-Z]{2}[0-9]{3}$',   # 1AB234
        r'^[0-9]{2}[A-Z][0-9]{3}$',   # 12A345
        r'^[A-Z]{2}[0-9]{4}$',        # AB1234
        r'^[0-9]{1,7}[A-Z]$',         # 1234567A
    ],
    'TN': [
        r'^[A-Z][0-9]{2}-[0-9]{2}[A-Z]$',   # A12-34B
        r'^[A-Z]{3}[0-9]{3}$',              # ABC123
        r'^[A-Z]{3}[0-9]{4}$',              # ABC1234
        r'^[A-Z][0-9]{5}$',                 # A12345
    ],
    'TX': [
        r'^[A-Z]{3}[0-9]{4}$',        # ABC1234
        r'^[A-Z]{3}\s[0-9]{4}$',      # ABC 1234
        r'^[A-Z]{2}[0-9][A-Z][0-9]{3}$',    # AB1C234
        r'^[A-Z]{2}[0-9]\s[A-Z][0-9]{3}$',  # AB1 C234
        r'^[A-Z]{2}[0-9]{5}$',        # AB12345
    ],
    'UT': [
        r'^[A-Z][0-9]{2}[A-Z]{2}$',   # A12BC
        r'^[A-Z][0-9]{2}\s[A-Z]{2}$', # A12 BC
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[0-9]{3}[A-Z]{3}$',        # 123ABC
        r'^[A-Z]{1,2}[0-9]{3}[A-Z]{2}$',    # AB123CD
    ],
    'VT': [
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[0-9]{3}[A-Z][0-9]{2}$',   # 123A45
        r'^[A-Z]{2}[0-9]{4}$',        # AB1234
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
    ],
    'VA': [
        r'^[A-Z]{3}[0-9]{4}$',        # ABC1234
        r'^[A-Z]{3}\s[0-9]{4}$',      # ABC 1234
        r'^[A-Z]{2}[0-9]{5}$',        # AB12345
        r'^[A-Z]{3}[0-9]{3}[A-Z]$',   # ABC123A
    ],
    'WA': [
        r'^[A-Z]{3}[0-9]{4}$',        # ABC1234
        r'^[A-Z]{3}\s[0-9]{4}$',      # ABC 1234
        r'^[A-Z]{3}[0-9]{3}[A-Z]$',   # ABC123A
        r'^[0-9]{3}[A-Z]{3}$',        # 123ABC
    ],
    'WV': [
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
        r'^[A-Z][0-9]{5}$',           # A12345
        r'^[0-9][A-Z][0-9]{4}$',      # 1A2345
        r'^[A-Z]{2}[0-9]{4}$',        # AB1234
    ],
    'WI': [
        r'^[A-Z]{3}[0-9]{4}$',        # ABC1234
        r'^[A-Z]{3}\s[0-9]{4}$',      # ABC 1234
        r'^[0-9]{3}[A-Z]{3}$',        # 123ABC
        r'^[A-Z]{2}[0-9]{5}$',        # AB12345
    ],
    'WY': [
        r'^[0-9]{2}-[0-9]{4}$',       # 12-3456
        r'^[0-9]{2}\s[0-9]{4}$',      # 12 3456
        r'^[0-9][A-Z]{3}$',           # 1ABC
        r'^[0-9]\s[A-Z]{3}$',         # 1 ABC
        r'^[0-9]{2}[0-9]{4}$',        # 123456
    ],
    'DC': [
        r'^[A-Z]{2}[0-9]{4}$',        # AB1234
        r'^[0-9]{1,7}$',              # 1234567
        r'^[A-Z]{3}[0-9]{3}$',        # ABC123
    ],
}


class StatePatternMatcher:
    """Pattern-based state recognition for license plates."""
    
    def __init__(self):
        """Initialize the pattern matcher."""
        self.patterns = STATE_PATTERNS
        
    def extract_state_from_text(self, plate_text: str) -> Tuple[Optional[str], float]:
        """
        Extract state from plate text using pattern matching.
        
        Args:
            plate_text: OCR-extracted license plate text
            
        Returns:
            Tuple of (state_code, confidence)
        """
        if not plate_text:
            return None, 0.0
            
        # Clean and normalize the text
        cleaned = self._clean_plate_text(plate_text)
        
        # Also try with spaces preserved (just uppercase)
        cleaned_with_spaces = plate_text.upper().strip()
        
        # Try both versions
        for text_variant in [cleaned, cleaned_with_spaces]:
            # Try specific state patterns
            for state, patterns in self.patterns.items():
                for pattern in patterns:
                    if re.match(pattern, text_variant):
                        # Higher confidence for exact match
                        confidence = 0.9 if text_variant == cleaned else 0.85
                        logger.info(f"Matched {text_variant} to {state} with pattern {pattern}")
                        return state, confidence
        
        # If no exact match, try fuzzy matching for common formats
        return self._fuzzy_match(cleaned)
    
    def _clean_plate_text(self, text: str) -> str:
        """Clean text for pattern matching."""
        # Remove spaces and special characters, convert to uppercase
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        return cleaned
    
    def _fuzzy_match(self, text: str) -> Tuple[Optional[str], float]:
        """Try to match common plate formats even if state-specific pattern not found."""
        # Common formats that might indicate certain states
        common_formats = [
            (r'^[A-Z]\d{2}[A-Z]{3}$', ['NJ', 'FL'], 0.6),  # X12ABC format
            (r'^[A-Z]{3}\d{4}$', ['NY', 'PA', 'TX', 'CA', 'OH', 'GA', 'NC', 'MI', 'VA', 'WA'], 0.5),  # ABC1234 format
            (r'^[A-Z]{2}\d{5}$', ['CT', 'PA', 'IL', 'NC', 'OH'], 0.5),  # AB12345 format
            (r'^\d[A-Z]{3}\d{3}$', ['CA'], 0.7),  # 1ABC234 format (CA specific)
            (r'^\d{3}[A-Z]{3}$', ['OR', 'NV', 'IN', 'MN'], 0.5),  # 123ABC format
            (r'^[A-Z]{3}\d{3}$', ['CO', 'NJ', 'FL', 'KY', 'LA'], 0.5),  # ABC123 format
        ]
        
        for pattern, possible_states, confidence in common_formats:
            if re.match(pattern, text):
                # Return the first possible state with lower confidence
                logger.info(f"Fuzzy matched {text} to possible states: {possible_states}")
                return possible_states[0], confidence
        
        return None, 0.0
    
    def get_supported_states(self) -> List[str]:
        """Get list of supported state codes."""
        return list(self.patterns.keys())
    
    def add_pattern(self, state: str, pattern: str):
        """Add a new pattern for a state."""
        if state not in self.patterns:
            self.patterns[state] = []
        if pattern not in self.patterns[state]:
            self.patterns[state].append(pattern)
            logger.info(f"Added pattern {pattern} for state {state}")


# Convenience function for direct use
def detect_state(plate_text: str) -> Tuple[Optional[str], float]:
    """
    Detect state from plate text.
    
    Args:
        plate_text: OCR-extracted license plate text
        
    Returns:
        Tuple of (state_code, confidence)
    """
    matcher = StatePatternMatcher()
    return matcher.extract_state_from_text(plate_text)

# Function to verify all states are included
def verify_all_states():
    """Verify that all US states are included."""
    all_states = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
    ]
    
    missing = []
    for state in all_states:
        if state not in STATE_PATTERNS:
            missing.append(state)
    
    if missing:
        print(f"Missing states: {missing}")
    else:
        print(f"✓ All {len(all_states)} states/territories are included!")
    
    return len(missing) == 0

# Function to test patterns
def test_patterns():
    """Test the pattern matching with sample plates."""
    test_plates = [
        ("A12BCD", "NJ"),      # New Jersey format
        ("ABC1234", "NY"),     # New York format
        ("7ABC234", "CA"),     # California format
        ("AB12345", "CT"),     # Connecticut format
        ("ABC-123", "CO"),     # Colorado format
        ("XYZ789", "PA"),      # Pennsylvania format
        ("123ABC", "OR"),      # Oregon format
        ("1AB234", "MA"),      # Massachusetts format
    ]
    
    matcher = StatePatternMatcher()
    print("\nTesting state patterns:")
    print("-" * 40)
    
    for plate, expected_state in test_plates:
        detected_state, confidence = matcher.extract_state_from_text(plate)
        status = "✓" if detected_state == expected_state else "✗"
        print(f"{status} {plate} -> Expected: {expected_state}, Got: {detected_state} (conf: {confidence:.2f})")

if __name__ == "__main__":
    # Verify all states are included
    verify_all_states()
    
    # Test some patterns
    test_patterns()
    
    # Count total patterns
    total_patterns = sum(len(patterns) for patterns in STATE_PATTERNS.values())
    print(f"\nTotal patterns: {total_patterns} across {len(STATE_PATTERNS)} states/territories")