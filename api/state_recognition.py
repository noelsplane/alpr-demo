# enhanced_state_recognition.py
"""
Enhanced state recognition module with better pattern matching
and state text detection from OCR results.
"""

import re
from typing import Optional, Dict, List, Tuple
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class EnhancedStateRecognizer:
    """
    Enhanced state recognizer that looks for both state codes and names in OCR text.
    """
    
    def __init__(self):
        # Initialize state patterns (same as before but with additions)
        self.state_patterns = self._initialize_state_patterns()
        
        # State names and abbreviations
        self.state_names = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
            'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
            'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
            'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
            'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
            'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
            'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
            'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
            'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
            'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
            'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'Washington D.C.'
        }
        
        # Common state mottos/phrases that appear on plates
        self.state_phrases = {
            'SUNSHINE': 'FL',  # Florida - Sunshine State
            'EMPIRE': 'NY',    # New York - Empire State
            'GOLDEN': 'CA',    # California - Golden State
            'LONE STAR': 'TX', # Texas - Lone Star State
            'VOLUNTEER': 'TN', # Tennessee - Volunteer State
            'PEACH': 'GA',     # Georgia - Peach State
            'GARDEN': 'NJ',    # New Jersey - Garden State
            'ALOHA': 'HI',     # Hawaii - Aloha State
            'LAND OF LINCOLN': 'IL',  # Illinois
            'FIRST IN FLIGHT': 'NC',  # North Carolina
            'LIVE FREE': 'NH',        # New Hampshire
            'OCEAN STATE': 'RI',      # Rhode Island
            'CONSTITUTION': 'CT',      # Connecticut
            'KEYSTONE': 'PA',         # Pennsylvania
            'SHOW ME': 'MO',          # Missouri
            'EVERGREEN': 'WA',        # Washington
            'GRAND CANYON': 'AZ',     # Arizona
            'BUCKEYE': 'OH',          # Ohio
            'HOOSIER': 'IN',          # Indiana
            'SPORTSMAN': 'LA',        # Louisiana
            'NATURAL STATE': 'AR',    # Arkansas
            'BLUEGRASS': 'KY',        # Kentucky
            'PALMETTO': 'SC',         # South Carolina
            'MOUNTAIN': 'WV',         # West Virginia
            'CENTENNIAL': 'CO',       # Colorado
            'SOONER': 'OK',           # Oklahoma
            'BEEHIVE': 'UT',          # Utah
            'BEAVER': 'OR',           # Oregon
            'SILVER': 'NV',           # Nevada
            'TREASURE': 'MT',         # Montana
            'EQUALITY': 'WY',         # Wyoming
            'HAWKEYE': 'IA',          # Iowa
            'SUNFLOWER': 'KS',        # Kansas
            'CORNHUSKER': 'NE',       # Nebraska
            'PEACE GARDEN': 'ND',     # North Dakota
            'MOUNT RUSHMORE': 'SD',   # South Dakota
            'DAIRY': 'WI',            # Wisconsin
            'NORTH STAR': 'MN',       # Minnesota
            'MAGNOLIA': 'MS',         # Mississippi
            'PELICAN': 'LA',          # Louisiana
            'VACATIONLAND': 'ME',     # Maine
            'GREEN MOUNTAIN': 'VT',   # Vermont
            'GRANITE': 'NH',          # New Hampshire
            'FIRST STATE': 'DE',      # Delaware
            'FREE STATE': 'MD',       # Maryland
            'FAMOUS POTATOES': 'ID',  # Idaho
            'LAST FRONTIER': 'AK',    # Alaska
            'LAND OF ENCHANTMENT': 'NM', # New Mexico
        }
        
        # Create reverse mapping of state names
        self.state_name_to_code = {name.upper(): code for code, name in self.state_names.items()}
        
    def _initialize_state_patterns(self) -> Dict[str, List[str]]:
        """Initialize comprehensive state patterns."""
        # Include all the patterns from the original state_recognition.py
        # This is a subset for brevity - you should include all patterns
        return {
            'CA': [r'^[0-9][A-Z]{3}[0-9]{3}$', r'^[A-Z]{3}[0-9]{4}$', r'^[0-9]{3}[A-Z]{3}$'],
            'TX': [r'^[A-Z]{3}[0-9]{4}$', r'^[A-Z]{2}[0-9]{5}$', r'^[0-9]{3}[A-Z]{3}$'],
            'NY': [r'^[A-Z]{3}[0-9]{4}$', r'^[A-Z]{3}-[0-9]{4}$', r'^[0-9]{3}-[A-Z]{3}$'],
            'FL': [r'^[A-Z]{3}[0-9]{3}$', r'^[A-Z]{4}[0-9]{2}$', r'^[0-9]{3}[A-Z]{3}$'],
            # Add all other states...
        }
    
    def recognize_state_from_multiple_texts(self, ocr_texts: List[str], plate_number: str) -> Optional[Dict[str, str]]:
        """
        Recognize state from multiple OCR text attempts.
        
        Args:
            ocr_texts: List of all OCR text attempts
            plate_number: The cleaned plate number
            
        Returns:
            State information or None
        """
        # First try pattern matching on the plate number
        state_from_pattern = self._match_plate_pattern(plate_number)
        if state_from_pattern:
            return state_from_pattern
        
        # Look for state information in all OCR texts
        all_text = ' '.join(ocr_texts).upper()
        
        # Check for state abbreviations
        state_from_abbr = self._find_state_abbreviation(all_text)
        if state_from_abbr:
            return state_from_abbr
        
        # Check for state names
        state_from_name = self._find_state_name(all_text)
        if state_from_name:
            return state_from_name
        
        # Check for state phrases/mottos
        state_from_phrase = self._find_state_phrase(all_text)
        if state_from_phrase:
            return state_from_phrase
        
        # Try fuzzy matching as last resort
        state_from_fuzzy = self._fuzzy_match_state(all_text)
        if state_from_fuzzy:
            return state_from_fuzzy
        
        return None
    
    def _match_plate_pattern(self, plate_text: str) -> Optional[Dict[str, str]]:
        """Match plate text against known state patterns."""
        cleaned = self._clean_for_pattern_matching(plate_text)
        
        # Try multiple variations
        variations = [
            cleaned,
            cleaned.replace(' ', ''),
            cleaned.replace('-', ''),
            cleaned.replace(' ', '-'),
        ]
        
        for state_code, patterns in self.state_patterns.items():
            for pattern in patterns:
                for variant in variations:
                    if re.match(pattern, variant):
                        return {
                            "code": state_code,
                            "name": self.state_names[state_code],
                            "confidence": "high",
                            "method": "pattern"
                        }
        
        return None
    
    def _find_state_abbreviation(self, text: str) -> Optional[Dict[str, str]]:
        """Find state abbreviation in text."""
        # Look for standalone 2-letter state codes
        words = text.split()
        
        for word in words:
            # Clean the word
            cleaned_word = re.sub(r'[^A-Z]', '', word)
            
            # Check if it's exactly 2 letters and matches a state
            if len(cleaned_word) == 2 and cleaned_word in self.state_names:
                return {
                    "code": cleaned_word,
                    "name": self.state_names[cleaned_word],
                    "confidence": "high",
                    "method": "abbreviation"
                }
        
        # Also check with regex for state codes at boundaries
        for state_code in self.state_names.keys():
            if re.search(r'\b' + state_code + r'\b', text):
                return {
                    "code": state_code,
                    "name": self.state_names[state_code],
                    "confidence": "high",
                    "method": "abbreviation"
                }
        
        return None
    
    def _find_state_name(self, text: str) -> Optional[Dict[str, str]]:
        """Find full state name in text."""
        # Remove spaces for better matching
        text_no_spaces = text.replace(' ', '')
        
        for state_name, state_code in self.state_name_to_code.items():
            # Try exact match first
            if state_name in text:
                return {
                    "code": state_code,
                    "name": self.state_names[state_code],
                    "confidence": "high",
                    "method": "name"
                }
            
            # Try without spaces
            if state_name.replace(' ', '') in text_no_spaces:
                return {
                    "code": state_code,
                    "name": self.state_names[state_code],
                    "confidence": "high",
                    "method": "name"
                }
        
        return None
    
    def _find_state_phrase(self, text: str) -> Optional[Dict[str, str]]:
        """Find state by motto or phrase."""
        for phrase, state_code in self.state_phrases.items():
            if phrase in text:
                return {
                    "code": state_code,
                    "name": self.state_names[state_code],
                    "confidence": "medium",
                    "method": "phrase"
                }
        
        return None
    
    def _fuzzy_match_state(self, text: str) -> Optional[Dict[str, str]]:
        """Use fuzzy matching to find state names."""
        # Extract potential state name candidates (2-20 character words)
        words = re.findall(r'\b[A-Z]{2,20}\b', text)
        
        best_match = None
        best_score = 0
        
        for word in words:
            # Check against state names
            for state_name, state_code in self.state_name_to_code.items():
                # Calculate similarity
                score = SequenceMatcher(None, word, state_name.replace(' ', '')).ratio()
                
                if score > 0.8 and score > best_score:  # 80% similarity threshold
                    best_score = score
                    best_match = {
                        "code": state_code,
                        "name": self.state_names[state_code],
                        "confidence": "low" if score < 0.9 else "medium",
                        "method": "fuzzy"
                    }
        
        return best_match
    
    def _clean_for_pattern_matching(self, text: str) -> str:
        """Clean text specifically for pattern matching."""
        # Convert to uppercase
        cleaned = text.upper().strip()
        
        # Smart character replacement for common OCR errors
        # This is more conservative than the general cleaning
        replacements = {
            ('O', '0'): lambda i, chars: i > 0 and chars[i-1].isdigit(),
            ('0', 'O'): lambda i, chars: i > 0 and chars[i-1].isalpha(),
            ('I', '1'): lambda i, chars: i > 0 and chars[i-1].isdigit(),
            ('1', 'I'): lambda i, chars: i > 0 and chars[i-1].isalpha(),
        }
        
        chars = list(cleaned)
        for i, char in enumerate(chars):
            for (from_char, to_char), condition in replacements.items():
                if char == from_char and condition(i, chars):
                    chars[i] = to_char
                    break
        
        cleaned = ''.join(chars)
        
        # Remove special characters except hyphens and spaces
        cleaned = re.sub(r'[^A-Z0-9\-\s]', '', cleaned)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned
    
    def get_state_statistics(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Get statistics about state detection.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary with state counts
        """
        state_counts = {}
        
        for detection in detections:
            state = detection.get('state')
            if state and state.get('code'):
                code = state['code']
                state_counts[code] = state_counts.get(code, 0) + 1
        
        return state_counts


# Helper function for integration
def enhance_detection_with_state(detection: Dict, state_recognizer: EnhancedStateRecognizer) -> Dict:
    """
    Enhance a detection dictionary with state information.
    
    Args:
        detection: Detection dictionary with 'text' and optionally 'all_ocr_texts'
        state_recognizer: State recognizer instance
        
    Returns:
        Enhanced detection dictionary
    """
    plate_text = detection.get('text', '')
    all_texts = detection.get('all_ocr_texts', [plate_text])
    
    # Recognize state
    state_info = state_recognizer.recognize_state_from_multiple_texts(all_texts, plate_text)
    
    # Add to detection
    if state_info:
        detection['state'] = state_info
        detection['state_code'] = state_info['code']
    else:
        detection['state'] = None
        detection['state_code'] = None
    
    return detection


if __name__ == "__main__":
    # Test the enhanced recognizer
    recognizer = EnhancedStateRecognizer()
    
    # Test cases
    test_cases = [
        (["ABC123", "CALIFORNIA", "ABC 123"], "ABC123"),
        (["XYZ789", "SUNSHINE STATE", "XYZ 789"], "XYZ789"),
        (["123ABC", "TX", "LONE STAR"], "123ABC"),
        (["DEF456", "NEW YORK", "EMPIRE"], "DEF456"),
    ]
    
    for texts, plate in test_cases:
        result = recognizer.recognize_state_from_multiple_texts(texts, plate)
        print(f"\nTexts: {texts}")
        print(f"Plate: {plate}")
        print(f"Result: {result}")
