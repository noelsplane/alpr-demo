"""
State pattern recognition for license plates - All 50 US States.
This module provides regex-based state detection patterns.
"""

import re
from typing import Optional, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)

# Comprehensive US state patterns
STATE_PATTERNS = {
    # Alabama
    'AL': [
        r'^[A-Z]{2}\d{5}$',           # AB12345
        r'^\d{2}[A-Z]{3}\d{2}$',      # 12ABC34
        r'^[A-Z]\d{6}$',              # A123456
    ],
    
    # Alaska
    'AK': [
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^\d{3}[A-Z]{3}$',           # 123ABC
    ],
    
    # Arizona
    'AZ': [
        r'^[A-Z]{3}\d{4}$',           # ABC1234
        r'^[A-Z]{2}\d{5}$',           # AB12345
        r'^\d{3}[A-Z]{3}$',           # 123ABC
    ],
    
    # Arkansas
    'AR': [
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^\d{3}[A-Z]{3}$',           # 123ABC
        r'^[A-Z]{3}\s\d{3}$',         # ABC 123
    ],
    
    # California
    'CA': [
        r'^[1-9][A-Z]{3}\d{3}$',      # 1ABC234
        r'^\d[A-Z]{3}\d{3}$',         # 7ABC234
        r'^[A-Z]{2}\d{5}$',           # AB12345 (commercial)
    ],
    
    # Colorado
    'CO': [
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^\d{3}[A-Z]{3}$',           # 123ABC
        r'^[A-Z]{2}\d{4}$',           # AB1234
        r'^\d{2}[A-Z]{4}$',           # 12ABCD
    ],
    
    # Connecticut
    'CT': [
        r'^[A-Z]{2}\d{5}$',           # AB12345
        r'^[A-Z]{2}\s\d{5}$',         # AB 12345
        r'^\d[A-Z]\d{5}$',            # 1A23456
        r'^\d[A-Z]\s\d{5}$',          # 1A 23456
    ],
    
    # Delaware
    'DE': [
        r'^\d{6}$',                   # 123456
        r'^\d{7}$',                   # 1234567
        r'^PC\d{4,5}$',               # PC1234 or PC12345
    ],
    
    # Florida
    'FL': [
        r'^[A-Z]{4}\d{2}$',           # ABCD12
        r'^[A-Z]\d{2}[A-Z]{3}$',      # A12BCD
        r'^[A-Z]{3}[A-Z]\d{2}$',      # ABCD12
        r'^[A-Z]\d{2}\s[A-Z]{3}$',    # A12 BCD
        r'^[A-Z]{3}\s[A-Z]\d{2}$',    # ABC D12
    ],
    
    # Georgia
    'GA': [
        r'^[A-Z]{3}\d{4}$',           # ABC1234
        r'^[A-Z]{3}\d{3}[A-Z]$',      # ABC123A
        r'^\d{3}[A-Z]{4}$',           # 123ABCD
    ],
    
    # Hawaii
    'HI': [
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^[A-Z]{2}\d{4}$',           # AB1234
        r'^[A-Z]\d{5}$',              # A12345
    ],
    
    # Idaho
    'ID': [
        r'^[A-Z]\d{6}$',              # A123456
        r'^\d[A-Z]\d{5}$',            # 1A23456
        r'^[A-Z]{2}\d{5}$',           # AB12345
        r'^[A-Z]\s\d{6}$',            # A 123456
    ],
    
    # Illinois
    'IL': [
        r'^[A-Z]\d{5,6}$',            # A12345 or A123456
        r'^[A-Z]{2}\d{5}$',           # AB12345
        r'^[A-Z]\s\d{5,6}$',          # A 12345
        r'^[A-Z]{2}\s\d{5}$',         # AB 12345
    ],
    
    # Indiana
    'IN': [
        r'^\d{3}[A-Z]{3}$',           # 123ABC
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^\d{2}[A-Z]\d{3}$',         # 12A345
        r'^\d{4}[A-Z]{2}$',           # 1234AB
    ],
    
    # Iowa
    'IA': [
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^[A-Z]{3}\d{4}$',           # ABC1234
        r'^\d{3}[A-Z]{3}$',           # 123ABC
        r'^[A-Z]{2}\d{4}$',           # AB1234
    ],
    
    # Kansas
    'KS': [
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^\d{3}[A-Z]{3}$',           # 123ABC
        r'^[A-Z]{2}\d{4}$',           # AB1234
    ],
    
    # Kentucky
    'KY': [
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^\d{3}[A-Z]{3}$',           # 123ABC
        r'^[A-Z]{3}\d{4}$',           # ABC1234
    ],
    
    # Louisiana
    'LA': [
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^[A-Z]{3}\d{4}$',           # ABC1234
        r'^\d{3}[A-Z]{3}$',           # 123ABC
    ],
    
    # Maine
    'ME': [
        r'^\d{4}[A-Z]{2}$',           # 1234AB
        r'^[A-Z]{2}\d{4}$',           # AB1234
        r'^\d{3}[A-Z]{3}$',           # 123ABC
    ],
    
    # Maryland
    'MD': [
        r'^\d[A-Z]{2}\d{4}$',         # 1AB2345
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^[A-Z]{2}\d{4}[A-Z]$',      # AB1234C
        r'^[A-Z]{3}\d{4}$',           # ABC1234
    ],
    
    # Massachusetts
    'MA': [
        r'^\d[A-Z]{2}\d{3}$',         # 1AB234
        r'^[A-Z]{2}\d{4}$',           # AB1234
        r'^\d{3}[A-Z]{3}$',           # 123ABC
        r'^[A-Z]{3}\d{3}$',           # ABC123
    ],
    
    # Michigan
    'MI': [
        r'^[A-Z]{3}\d{4}$',           # ABC1234
        r'^[A-Z]{3}\d{3}[A-Z]$',      # ABC123A
        r'^[A-Z]{3}\s\d{4}$',         # ABC 1234
        r'^\d[A-Z]{2}\d{3}$',         # 1AB234
    ],
    
    # Minnesota
    'MN': [
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^\d{3}[A-Z]{3}$',           # 123ABC
        r'^[A-Z]{3}\s\d{3}$',         # ABC 123
    ],
    
    # Mississippi
    'MS': [
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^[A-Z]{3}\d{4}$',           # ABC1234
        r'^[A-Z]{2}\d{5}$',           # AB12345
    ],
    
    # Missouri
    'MO': [
        r'^[A-Z]{2}\d[A-Z]\d{2}$',    # AB1C23
        r'^[A-Z]\d{2}[A-Z]{3}$',      # A12BCD
        r'^[A-Z]{2}\d{4}$',           # AB1234
        r'^[A-Z]{2}\d[A-Z]{2}\d$',    # AB1CD2
    ],
    
    # Montana
    'MT': [
        r'^\d{2}\d{4}[A-Z]$',         # 12-3456A (county-number-letter)
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^\d[A-Z]\d{4}$',            # 1A2345
    ],
    
    # Nebraska
    'NE': [
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^\d{2}[A-Z]\d{3}$',         # 12A345 (county prefix)
        r'^[A-Z]\d{5}$',              # A12345 (commercial)
    ],
    
    # Nevada
    'NV': [
        r'^\d{3}[A-Z]{3}$',           # 123ABC
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^[A-Z]{2}\d{4}$',           # AB1234
        r'^\d{2}[A-Z]\d{3}$',         # 12A345
    ],
    
    # New Hampshire
    'NH': [
        r'^\d{3}\s\d{4}$',            # 123 4567
        r'^\d{7}$',                   # 1234567
        r'^[A-Z]{3}\d{4}$',           # ABC1234
    ],
    
    # New Jersey
    'NJ': [
        r'^[A-Z]\d{2}[A-Z]{3}$',      # A12BCD
        r'^[A-Z]\d{2}\s[A-Z]{3}$',    # A12 BCD
        r'^[A-Z]{3}\d{2}[A-Z]$',      # ABC12D
        r'^[A-Z]{3}\s\d{2}[A-Z]$',    # ABC 12D
    ],
    
    # New Mexico
    'NM': [
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^\d{3}[A-Z]{3}$',           # 123ABC
        r'^[A-Z]{2}\d{4}$',           # AB1234
        r'^\d{3}\s[A-Z]{3}$',         # 123 ABC
    ],
    
    # New York
    'NY': [
        r'^[A-Z]{3}\d{4}$',           # ABC1234
        r'^[A-Z]{3}\s\d{4}$',         # ABC 1234
        r'^[A-Z]{3}\d{3}[A-Z]$',      # ABC123A
        r'^[A-Z]{3}\s\d{3}[A-Z]$',    # ABC 123A
    ],
    
    # North Carolina
    'NC': [
        r'^[A-Z]{3}\d{4}$',           # ABC1234
        r'^[A-Z]{3}\s\d{4}$',         # ABC 1234
        r'^[A-Z]{2}\d{5}$',           # AB12345
    ],
    
    # North Dakota
    'ND': [
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^\d{3}[A-Z]{3}$',           # 123ABC
        r'^[A-Z]{2}\d{4}$',           # AB1234
    ],
    
    # Ohio
    'OH': [
        r'^[A-Z]{3}\d{4}$',           # ABC1234
        r'^[A-Z]{3}\s\d{4}$',         # ABC 1234
        r'^[A-Z]{3}\d{3}[A-Z]$',      # ABC123A
        r'^[A-Z]{2}\d{5}$',           # AB12345
    ],
    
    # Oklahoma
    'OK': [
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^\d{3}[A-Z]{3}$',           # 123ABC
        r'^[A-Z]{3}\d{4}$',           # ABC1234
    ],
    
    # Oregon
    'OR': [
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^\d{3}[A-Z]{3}$',           # 123ABC
        r'^[A-Z]{2}\d{4}$',           # AB1234
        r'^[A-Z]{3}\s\d{3}$',         # ABC 123
    ],
    
    # Pennsylvania
    'PA': [
        r'^[A-Z]{3}\d{4}$',           # ABC1234
        r'^[A-Z]{3}\s\d{4}$',         # ABC 1234
        r'^[A-Z]{3}[A-Z]\d{3}$',      # ABCD123
    ],
    
    # Rhode Island
    'RI': [
        r'^\d{6}$',                   # 123456
        r'^[A-Z]{2}\d{3}$',           # AB123
        r'^[A-Z]{2}\s\d{3}$',         # AB 123
        r'^\d{3}\s\d{3}$',            # 123 456
    ],
    
    # South Carolina
    'SC': [
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^[A-Z]{3}\s\d{3}$',         # ABC 123
        r'^[A-Z]{3}\d{4}$',           # ABC1234
    ],
    
    # South Dakota
    'SD': [
        r'^\d[A-Z]{2}\d{3}$',         # 1AB234
        r'^\d{2}[A-Z]\d{3}$',         # 12A345
        r'^[A-Z]{2}\d{4}$',           # AB1234
    ],
    
    # Tennessee
    'TN': [
        r'^[A-Z]\d{2}\d{2}[A-Z]$',    # A12-34B (county format)
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^[A-Z]{3}\d{4}$',           # ABC1234
        r'^[A-Z]\d{5}$',              # A12345
    ],
    
    # Texas
    'TX': [
        r'^[A-Z]{3}\d{4}$',           # ABC1234
        r'^[A-Z]{3}\s\d{4}$',         # ABC 1234
        r'^[A-Z]{2}\d[A-Z]\d{3}$',    # AB1C234 (truck)
        r'^[A-Z]{2}\d\s[A-Z]\d{3}$',  # AB1 C234
    ],
    
    # Utah
    'UT': [
        r'^[A-Z]\d{2}[A-Z]{2}$',      # A12BC
        r'^[A-Z]\d{2}\s[A-Z]{2}$',    # A12 BC
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^\d{3}[A-Z]{3}$',           # 123ABC
    ],
    
    # Vermont
    'VT': [
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^\d{3}[A-Z]\d{2}$',         # 123A45
        r'^[A-Z]{2}\d{4}$',           # AB1234
    ],
    
    # Virginia
    'VA': [
        r'^[A-Z]{3}\d{4}$',           # ABC1234
        r'^[A-Z]{3}\s\d{4}$',         # ABC 1234
        r'^[A-Z]{2}\d{5}$',           # AB12345
        r'^[A-Z]{3}\d{3}[A-Z]$',      # ABC123A
    ],
    
    # Washington
    'WA': [
        r'^[A-Z]{3}\d{4}$',           # ABC1234
        r'^[A-Z]{3}\s\d{4}$',         # ABC 1234
        r'^[A-Z]{3}\d{3}[A-Z]$',      # ABC123A
        r'^\d{3}[A-Z]{3}$',           # 123ABC
    ],
    
    # West Virginia
    'WV': [
        r'^[A-Z]{3}\d{3}$',           # ABC123
        r'^[A-Z]\d{5}$',              # A12345
        r'^\d[A-Z]\d{4}$',            # 1A2345
        r'^[A-Z]{2}\d{4}$',           # AB1234
    ],
    
    # Wisconsin
    'WI': [
        r'^[A-Z]{3}\d{4}$',           # ABC1234
        r'^[A-Z]{3}\s\d{4}$',         # ABC 1234
        r'^\d{3}[A-Z]{3}$',           # 123ABC
        r'^[A-Z]{2}\d{5}$',           # AB12345
    ],
    
    # Wyoming
    'WY': [
        r'^\d{2}\d{4}$',              # 12-3456 (county-number)
        r'^\d{2}\s\d{4}$',            # 12 3456
        r'^\d[A-Z]{3}$',              # 1ABC
        r'^\d\s[A-Z]{3}$',            # 1 ABC
    ],
}

# Generic patterns that could match multiple states
GENERIC_PATTERNS = [
    (r'^[A-Z]{3}\d{4}$', ['TX', 'NY', 'PA', 'OH', 'NC', 'MI', 'GA', 'VA', 'WA']),  # ABC1234
    (r'^[A-Z]{3}\d{3}$', ['CA', 'FL', 'IL', 'MN', 'AR', 'KS', 'OR']),  # ABC123
    (r'^[A-Z]{2}\d{4,5}$', ['Generic']),  # AB1234 or AB12345
    (r'^\d{3}[A-Z]{3}$', ['Generic']),  # 123ABC
]


class StatePatternMatcher:
    """Pattern-based state recognition for license plates."""
    
    def __init__(self):
        """Initialize the pattern matcher."""
        self.patterns = STATE_PATTERNS
        self.generic_patterns = GENERIC_PATTERNS
        
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
        
        # Check for explicit state mentions (e.g., "CALIFORNIA", "TEXAS")
        state, conf = self._check_state_names(plate_text)
        if state:
            return state, conf
        
        # Try specific state patterns first
        matches = []
        for state, patterns in self.patterns.items():
            for pattern in patterns:
                if re.match(pattern, cleaned):
                    matches.append((state, 0.8))  # High confidence for specific patterns
                    
        if matches:
            # Return the first match (could be improved with scoring)
            return matches[0]
        
        # Try generic patterns
        for pattern, possible_states in self.generic_patterns:
            if re.match(pattern, cleaned):
                # Lower confidence for generic patterns
                return possible_states[0] if len(possible_states) == 1 else None, 0.3
                
        return None, 0.0
    
    def _clean_plate_text(self, text: str) -> str:
        """Clean and normalize plate text."""
        # Remove common OCR errors and normalize
        cleaned = text.upper().strip()
        
        # Remove non-alphanumeric except spaces and hyphens
        cleaned = re.sub(r'[^A-Z0-9\s-]', '', cleaned)
        
        # Normalize spacing
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove spaces for pattern matching
        cleaned_no_space = cleaned.replace(' ', '').replace('-', '')
        
        return cleaned_no_space
    
    def _check_state_names(self, text: str) -> Tuple[Optional[str], float]:
        """Check for explicit state names in the text."""
        state_names = {
            # Full state names
            'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
            'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE',
            'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 'IDAHO': 'ID',
            'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
            'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
            'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS',
            'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV',
            'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY',
            'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK',
            'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
            'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT',
            'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV',
            'WISCONSIN': 'WI', 'WYOMING': 'WY',
            
            # Common nicknames
            'GARDEN STATE': 'NJ', 'EMPIRE STATE': 'NY', 'GOLDEN STATE': 'CA',
            'SUNSHINE STATE': 'FL', 'LONE STAR': 'TX', 'CONSTITUTION STATE': 'CT',
            'FIRST STATE': 'DE', 'ALOHA STATE': 'HI', 'LAND OF LINCOLN': 'IL',
            'HOOSIER STATE': 'IN', 'HAWKEYE STATE': 'IA', 'PELICAN STATE': 'LA',
            'OLD DOMINION': 'VA', 'VOLUNTEER STATE': 'TN', 'GRANITE STATE': 'NH',
            'SILVER STATE': 'NV', 'BUCKEYE STATE': 'OH', 'KEYSTONE STATE': 'PA',
            'OCEAN STATE': 'RI', 'PALMETTO STATE': 'SC', 'MOUNT RUSHMORE': 'SD',
            'GREEN MOUNTAIN': 'VT', 'EVERGREEN STATE': 'WA', 'MOUNTAIN STATE': 'WV',
        }
        
        text_upper = text.upper()
        for name, code in state_names.items():
            if name in text_upper:
                return code, 0.9  # High confidence when state name is found
                
        return None, 0.0
    
    def get_supported_states(self) -> List[str]:
        """Get list of supported state codes."""
        return list(self.patterns.keys())


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