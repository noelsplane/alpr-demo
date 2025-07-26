"""
State Name Filtering for ALPR System
Filters out US state names from plate text detection to prevent false positives.
"""

import re
from typing import Set, List, Optional
import logging

logger = logging.getLogger(__name__)

# Comprehensive list of US state names and abbreviations
US_STATES = {
    # Full state names
    'ALABAMA', 'ALASKA', 'ARIZONA', 'ARKANSAS', 'CALIFORNIA', 'COLORADO',
    'CONNECTICUT', 'DELAWARE', 'FLORIDA', 'GEORGIA', 'HAWAII', 'IDAHO',
    'ILLINOIS', 'INDIANA', 'IOWA', 'KANSAS', 'KENTUCKY', 'LOUISIANA',
    'MAINE', 'MARYLAND', 'MASSACHUSETTS', 'MICHIGAN', 'MINNESOTA',
    'MISSISSIPPI', 'MISSOURI', 'MONTANA', 'NEBRASKA', 'NEVADA',
    'NEW HAMPSHIRE', 'NEW JERSEY', 'NEW MEXICO', 'NEW YORK',
    'NORTH CAROLINA', 'NORTH DAKOTA', 'OHIO', 'OKLAHOMA', 'OREGON',
    'PENNSYLVANIA', 'RHODE ISLAND', 'SOUTH CAROLINA', 'SOUTH DAKOTA',
    'TENNESSEE', 'TEXAS', 'UTAH', 'VERMONT', 'VIRGINIA', 'WASHINGTON',
    'WEST VIRGINIA', 'WISCONSIN', 'WYOMING',
    
    # State abbreviations
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID',
    'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS',
    'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
    'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV',
    'WI', 'WY',
    
    # Common variations and partial matches
    'NEWHAMPSHIRE', 'NEWJERSEY', 'NEWMEXICO', 'NEWYORK', 'NORTHCAROLINA',
    'NORTHDAKOTA', 'RHODEISLAND', 'SOUTHCAROLINA', 'SOUTHDAKOTA',
    'WESTVIRGINIA',
    
    # District of Columbia and territories
    'DISTRICT OF COLUMBIA', 'WASHINGTON DC', 'WASHINGTON D.C.', 'DC',
    'PUERTO RICO', 'PR', 'GUAM', 'GU', 'VIRGIN ISLANDS', 'VI',
    'AMERICAN SAMOA', 'AS', 'NORTHERN MARIANA', 'MP',
    
    # Common license plate text that appears with state names
    'STATE', 'PLATE', 'LICENSE', 'LICENCE', 'DMV', 'MOTOR', 'VEHICLE',
    'REGISTRATION', 'REG', 'DEPT', 'DEPARTMENT', 'TRANSPORTATION',
    'TRANSPORT', 'GOVT', 'GOVERNMENT', 'OFFICIAL', 'EXPIRES', 'EXP',
    'VALID', 'ISSUED', 'COUNTY', 'CITY', 'MUNICIPAL', 'POLICE',
    
    # Canadian provinces (in case of cross-border detection)
    'ALBERTA', 'BRITISH COLUMBIA', 'MANITOBA', 'NEW BRUNSWICK',
    'NEWFOUNDLAND', 'NORTHWEST TERRITORIES', 'NOVA SCOTIA', 'NUNAVUT',
    'ONTARIO', 'PRINCE EDWARD ISLAND', 'QUEBEC', 'SASKATCHEWAN', 'YUKON',
    'AB', 'BC', 'MB', 'NB', 'NL', 'NT', 'NS', 'NU', 'ON', 'PE', 'QC', 'SK', 'YT',
    
    # Common month abbreviations (often on plates)
    'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC',
    'JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'JUNE', 'JULY', 'AUGUST',
    'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER'
}

# Common words that appear on license plates but aren't plate numbers
COMMON_PLATE_WORDS = {
    'THE', 'AND', 'OF', 'IN', 'TO', 'FOR', 'WITH', 'ON', 'AT', 'BY', 'FROM',
    'YEAR', 'MONTH', 'DAY', 'WEEK', 'TIME', 'DATE', 'GOOD', 'UNTIL', 'THRU',
    'THROUGH', 'BEFORE', 'AFTER', 'SINCE', 'DURING', 'WITHIN', 'BETWEEN',
    'AMONG', 'ACROSS', 'UNDER', 'OVER', 'ABOVE', 'BELOW', 'NEAR', 'FAR',
    'HERE', 'THERE', 'WHERE', 'WHEN', 'HOW', 'WHY', 'WHAT', 'WHO', 'WHICH',
    'THIS', 'THAT', 'THESE', 'THOSE', 'SOME', 'ANY', 'ALL', 'EACH', 'EVERY',
    'BOTH', 'EITHER', 'NEITHER', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE',
    'FIRST', 'SECOND', 'THIRD', 'LAST', 'NEXT', 'PREVIOUS', 'PRIOR',
    'STICKER', 'TAB', 'REGISTRATION', 'RENEWAL', 'FEE', 'PAID', 'DUE',
    'COMMERCIAL', 'PASSENGER', 'TRUCK', 'MOTORCYCLE', 'TRAILER', 'RV',
    'TEMPORARY', 'TEMP', 'PERMANENT', 'PERM', 'DEALER', 'SPECIALTY',
    'CLASSIC', 'ANTIQUE', 'VINTAGE', 'CUSTOM', 'PERSONALIZED', 'VANITY'
}

class StateNameFilter:
    """Filter to remove state names and common words from plate text detections."""
    
    def __init__(self):
        # Combine all filter sets and normalize to uppercase
        self.filtered_words = set()
        for word_set in [US_STATES, COMMON_PLATE_WORDS]:
            self.filtered_words.update(word.upper().strip() for word in word_set)
        
        # Compile regex patterns for efficient matching
        self.exact_match_pattern = re.compile(
            r'^(' + '|'.join(re.escape(word) for word in self.filtered_words) + r')$',
            re.IGNORECASE
        )
        
        # Pattern for detecting if text contains mostly state-related words
        self.state_heavy_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(word) for word in US_STATES) + r')\b',
            re.IGNORECASE
        )
        
        logger.info(f"StateNameFilter initialized with {len(self.filtered_words)} filtered terms")
    
    def is_state_name(self, text: str) -> bool:
        """Check if the text is exactly a state name or common plate word."""
        if not text or not isinstance(text, str):
            return False
        
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            return False
        
        # Check exact match
        return bool(self.exact_match_pattern.match(cleaned_text))
    
    def contains_state_info(self, text: str) -> bool:
        """Check if text contains significant state-related information."""
        if not text or not isinstance(text, str):
            return False
        
        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            return False
        
        # Count state-related words
        state_matches = self.state_heavy_pattern.findall(cleaned_text)
        total_words = len(cleaned_text.split())
        
        # If more than 50% of words are state-related, filter it out
        if total_words > 0 and len(state_matches) / total_words > 0.5:
            return True
        
        return False
    
    def is_valid_plate_format(self, text: str) -> bool:
        """Check if text matches typical license plate formats."""
        if not text or not isinstance(text, str):
            return False
        
        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            return False
        
        # Remove spaces for format checking
        no_spaces = re.sub(r'\s+', '', cleaned_text)
        
        # Check length (typical plates are 3-8 characters)
        if len(no_spaces) < 3 or len(no_spaces) > 8:
            return False
        
        # Check for typical plate patterns
        plate_patterns = [
            r'^[A-Z0-9]{3,8}$',           # Basic alphanumeric
            r'^[A-Z]{2,3}[0-9]{3,4}$',    # Letters followed by numbers
            r'^[0-9]{3}[A-Z]{2,3}$',      # Numbers followed by letters
            r'^[A-Z][0-9]{2,3}[A-Z]{2,3}$', # Mixed pattern
            r'^[0-9]{3}[A-Z]{3}$',        # 3 numbers, 3 letters
            r'^[A-Z]{3}[0-9]{3}$',        # 3 letters, 3 numbers
        ]
        
        for pattern in plate_patterns:
            if re.match(pattern, no_spaces):
                return True
        
        return False
    
    def should_filter_detection(self, text: str, confidence: float = 0.0) -> bool:
        """
        Determine if a detection should be filtered out.
        
        Args:
            text: The detected text
            confidence: OCR confidence score (0.0 to 1.0)
            
        Returns:
            True if the detection should be filtered out
        """
        if not text or not isinstance(text, str):
            return True
        
        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            return True
        
        # Filter if it's exactly a state name or common word
        if self.is_state_name(cleaned_text):
            logger.debug(f"Filtered state name/common word: '{text}'")
            return True
        
        # Filter if it contains mostly state information
        if self.contains_state_info(cleaned_text):
            logger.debug(f"Filtered state-heavy text: '{text}'")
            return True
        
        # Filter if it doesn't match typical plate formats
        if not self.is_valid_plate_format(cleaned_text):
            logger.debug(f"Filtered invalid plate format: '{text}'")
            return True
        
        # Filter very low confidence detections of questionable text
        if confidence < 0.3 and len(cleaned_text) < 4:
            logger.debug(f"Filtered low confidence short text: '{text}' (confidence: {confidence})")
            return True
        
        return False
    
    def filter_detections(self, detections: List[dict]) -> List[dict]:
        """
        Filter a list of detections, removing state names and invalid plates.
        
        Args:
            detections: List of detection dictionaries with 'text' and optionally 'confidence'
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        filtered_detections = []
        original_count = len(detections)
        
        for detection in detections:
            text = detection.get('text', '') or detection.get('plate_text', '')
            confidence = detection.get('confidence', 0.0)
            
            if not self.should_filter_detection(text, confidence):
                filtered_detections.append(detection)
            else:
                logger.debug(f"Filtered detection: {detection}")
        
        filtered_count = len(filtered_detections)
        if filtered_count != original_count:
            logger.info(f"Filtered {original_count - filtered_count} detections, kept {filtered_count}")
        
        return filtered_detections
    
    def clean_plate_text(self, text: str) -> Optional[str]:
        """
        Clean and validate plate text, returning None if it should be filtered.
        
        Args:
            text: Raw plate text
            
        Returns:
            Cleaned plate text or None if invalid
        """
        if not text or not isinstance(text, str):
            return None
        
        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            return None
        
        if self.should_filter_detection(cleaned_text):
            return None
        
        return cleaned_text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for processing."""
        if not text:
            return ""
        
        # Convert to uppercase and strip whitespace
        cleaned = text.upper().strip()
        
        # Remove special characters except spaces, letters, and numbers
        cleaned = re.sub(r'[^A-Z0-9\s]', '', cleaned)
        
        # Normalize multiple spaces to single space
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def add_custom_filter(self, word: str):
        """Add a custom word to the filter list."""
        if word and isinstance(word, str):
            normalized_word = word.upper().strip()
            self.filtered_words.add(normalized_word)
            logger.info(f"Added custom filter word: '{normalized_word}'")
    
    def remove_custom_filter(self, word: str):
        """Remove a word from the filter list (only custom words)."""
        if word and isinstance(word, str):
            normalized_word = word.upper().strip()
            if normalized_word in self.filtered_words and normalized_word not in US_STATES:
                self.filtered_words.remove(normalized_word)
                logger.info(f"Removed custom filter word: '{normalized_word}'")
    
    def get_filter_stats(self) -> dict:
        """Get statistics about the filter."""
        return {
            "total_filtered_words": len(self.filtered_words),
            "us_states_count": len(US_STATES),
            "common_words_count": len(COMMON_PLATE_WORDS)
        }

# Global filter instance
state_filter = StateNameFilter()

def filter_state_names(detections: List[dict]) -> List[dict]:
    """Convenience function to filter state names from detections."""
    return state_filter.filter_detections(detections)

def is_valid_plate_text(text: str, confidence: float = 0.0) -> bool:
    """Convenience function to check if text is valid plate text."""
    return not state_filter.should_filter_detection(text, confidence)

def clean_plate_text(text: str) -> Optional[str]:
    """Convenience function to clean and validate plate text."""
    return state_filter.clean_plate_text(text)