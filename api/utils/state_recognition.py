import re
from typing import Optional, Dict, List

class StateRecognizer:

    def __init__(self):

        self.state_patterns = {
            'AL': [
                r'^[0-9]{1,7}$',  # 1234567
                r'^[0-9][A-Z]{2}[0-9]{4}$',  # 1AB1234
                r'^[A-Z]{1,3}[0-9]{3,4}$',  # ABC123, ABC1234
            ],
            'AK': [
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
                r'^[0-9]{3}[A-Z]{3}$',  # 123ABC
                r'^[A-Z]{2}[0-9]{4}$',  # AB1234
            ],
            'AZ': [
                r'^[A-Z]{3}[0-9]{4}$',  # ABC1234
                r'^[A-Z][0-9]{6}$',  # A123456
                r'^[0-9]{7}$',  # 1234567
            ],
            'AR': [
                r'^[0-9]{3}[A-Z]{3}$',  # 123ABC
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
            ],
            'CA': [
                r'^[0-9][A-Z]{3}[0-9]{3}$',  # 1ABC123
                r'^[A-Z]{1,2}[0-9]{4}$',  # AB1234
                r'^[0-9]{3}[A-Z]{3}$',  # 123ABC
            ],
            'CO': [
                r'^[A-Z]{3}-[0-9]{3}$',  # ABC-123
                r'^[0-9]{3}-[A-Z]{3}$',  # 123-ABC
                r'^[A-Z]{2}[0-9]{4}$',  # AB1234
            ],
            'CT': [
                r'^[0-9]{3}-[A-Z]{3}$',  # 123-ABC
                r'^[A-Z]{2}[0-9]{4}$',  # AB1234
                r'^[0-9]{1,7}$',  # 1234567
            ],
            'DE': [
                r'^[0-9]{1,6}$',  # 123456
                r'^[A-Z]{1,2}[0-9]{3,4}$',  # AB123, AB1234
                r'^PC[0-9]{4}$',  # PC1234
            ],
            'FL': [
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{1,2}[0-9]{4}$',  # AB1234
            ],
            'GA': [
                r'^[A-Z]{3}[0-9]{4}$',  # ABC1234
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{1,2}[0-9]{4}$',  # AB1234
            ],
            'HI': [
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
                r'^[0-9]{1,6}$',  # 123456
                r'^[A-Z]{2}[0-9]{4}$',  # AB1234
            ],
            'ID': [
                r'^[A-Z]{1,2}[0-9]{5,6}$',  # A12345, AB12345
                r'^[0-9][A-Z][0-9]{5}$',  # 1A12345
                r'^[0-9]{1,7}$',  # 1234567
            ],
            'IL': [
                r'^[A-Z]{1,3}[0-9]{4}$',  # ABC1234
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{2}[0-9]{3}[A-Z]$',  # AB123C
            ],
            'IN': [
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
                r'^[0-9]{3}[A-Z]{3}$',  # 123ABC
                r'^[A-Z]{1,2}[0-9]{4}$',  # AB1234
            ],
            'IA': [
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
                r'^[0-9]{3}[A-Z]{3}$',  # 123ABC
                r'^[0-9]{1,7}$',  # 1234567
            ],
            'KS': [
                r'^[0-9]{3}[A-Z]{3}$',  # 123ABC
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
                r'^[0-9]{1,7}$',  # 1234567
            ],
            'KY': [
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{1,2}[0-9]{4}$',  # AB1234
            ],
            'LA': [
                r'^[A-Z]{3}[0-9]{4}$',  # ABC1234
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{1,2}[0-9]{3}[A-Z]$',  # AB123C
            ],
            'ME': [
                r'^[0-9]{4}[A-Z]{2}$',  # 1234AB
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
            ],
            'MD': [
                r'^[A-Z]{3}[0-9]{4}$',  # ABC1234
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{1,2}[0-9]{5}$',  # AB12345
            ],
            'MA': [
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
                r'^[0-9]{3}[A-Z]{3}$',  # 123ABC
            ],
            'MI': [
                r'^[A-Z]{3}[0-9]{4}$',  # ABC1234
                r'^[A-Z]{1}[0-9]{6}$',  # A123456
                r'^[0-9]{1,7}$',  # 1234567
            ],
            'MN': [
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
                r'^[0-9]{3}[A-Z]{3}$',  # 123ABC
                r'^[0-9]{1,7}$',  # 1234567
            ],
            'MS': [
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{1,2}[0-9]{4}$',  # AB1234
            ],
            'MO': [
                r'^[A-Z]{2}[0-9]{1}[A-Z]{1}[0-9]{3}$',  # AB1C234
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
            ],
            'MT': [
                r'^[0-9]{1,7}[A-Z]$',  # 1234567A
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
                r'^[0-9]{5}[A-Z]{2}$',  # 12345AB
            ],
            'NE': [
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
                r'^[0-9]{1,7}$',  # 1234567
                r'^[0-9]{3}[A-Z]{3}$',  # 123ABC
            ],
            'NV': [
                r'^[0-9]{3}[A-Z]{3}$',  # 123ABC
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
                r'^[0-9]{1,7}$',  # 1234567
            ],
            'NH': [
                r'^[0-9]{3}[A-Z]{3}$',  # 123ABC
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
                r'^[0-9]{1,7}$',  # 1234567
            ],
            'NJ': [
                r'^[A-Z]{1}[0-9]{2}[A-Z]{3}$',  # A12BCD
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{3}[0-9]{4}$',  # ABC1234
            ],
            'NM': [
                r'^[A-Z]{3}[0-9]{3,4}$',  # ABC123, ABC1234
                r'^[0-9]{1,7}$',  # 1234567
                r'^[0-9]{3}[A-Z]{3}$',  # 123ABC
            ],
            'NY': [
                r'^[A-Z]{3}[0-9]{4}$',  # ABC1234
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{1,2}[0-9]{3}[A-Z]{2}$',  # AB123CD
            ],
            'NC': [
                r'^[A-Z]{3}[0-9]{4}$',  # ABC1234
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{1,2}[0-9]{4}$',  # AB1234
            ],
            'ND': [
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{2}[0-9]{4}$',  # AB1234
            ],
            'OH': [
                r'^[A-Z]{3}[0-9]{4}$',  # ABC1234
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{2}[0-9]{4}$',  # AB1234
            ],
            'OK': [
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{2}[0-9]{4}$',  # AB1234
            ],
            'OR': [
                r'^[0-9]{3}[A-Z]{3}$',  # 123ABC
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
                r'^[0-9]{1,6}$',  # 123456
            ],
            'PA': [
                r'^[A-Z]{3}[0-9]{4}$',  # ABC1234
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{1,2}[0-9]{3}[A-Z]{2}$',  # AB123CD
            ],
            'RI': [
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{2}[0-9]{3}$',  # AB123
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
            ],
            'SC': [
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{2}[0-9]{4}$',  # AB1234
            ],
            'SD': [
                r'^[0-9]{1,7}[A-Z]$',  # 1234567A
                r'^[A-Z]{1,3}[0-9]{3}$',  # ABC123
                r'^[0-9]{3}[A-Z]{3}$',  # 123ABC
            ],
            'TN': [
                r'^[A-Z]{3}[0-9]{4}$',  # ABC1234
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{2}[0-9]{3}[A-Z]$',  # AB123C
            ],
            'TX': [
                r'^[A-Z]{3}[0-9]{4}$',  # ABC1234
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{1,2}[0-9]{3}[A-Z]{2}$',  # AB123CD
            ],
            'UT': [
                r'^[A-Z]{1,2}[0-9]{3}[A-Z]{2}$',  # AB123CD
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
            ],
            'VT': [
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{2}[0-9]{3}[A-Z]$',  # AB123C
            ],
            'VA': [
                r'^[A-Z]{3}[0-9]{4}$',  # ABC1234
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{1,2}[0-9]{5}$',  # AB12345
            ],
            'WA': [
                r'^[A-Z]{3}[0-9]{4}$',  # ABC1234
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{2}[0-9]{5}$',  # AB12345
            ],
            'WV': [
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{1,2}[0-9]{3,4}$',  # AB123, AB1234
                r'^[0-9]{3}[A-Z]{3}$',  # 123ABC
            ],
            'WI': [
                r'^[A-Z]{3}[0-9]{4}$',  # ABC1234
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{2}[0-9]{3}[A-Z]$',  # AB123C
            ],
            'WY': [
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{1}[0-9]{5}$',  # A12345
                r'^[0-9]{3}[A-Z]{3}$',  # 123ABC
            ],
            'DC': [
                r'^[A-Z]{2}[0-9]{4}$',  # AB1234
                r'^[0-9]{1,7}$',  # 1234567
                r'^[A-Z]{3}[0-9]{3}$',  # ABC123
            ],
        }

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
            'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
        }

    def clean_plate_text(self, plate_text: str) -> str:
        if not plate_text:
            return ""
        
        cleaned = plate_text.upper().strip()
        cleaned = cleaned.replace('O', '0')
        cleaned = cleaned.replace('I', '1')
        cleaned = cleaned.replace('S', '5')

        cleaned = re.sub(r'[^A-Z0-9\-\s]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)

        return cleaned

    def recognize_state(self, plate_text: str) -> Optional[Dict[str, str]]:
        """
        Recognizes the state from the plate text using predefined patterns.
        
        Args:
            plate_text (str): The text of the license plate.
        
        Returns:
            Optional[Dict[str, str]]: A dictionary with the state code and matched pattern if recognized, otherwise None.
        """
        if not plate_text:
            return None
        
        cleaned_text = self.clean_plate_text(plate_text)
        if not cleaned_text:
            return None
        
        # Create variations of the text to try matching
        text_variations = [
            cleaned_text,
            cleaned_text.upper(),
            cleaned_text.replace(' ', ''),
            cleaned_text.upper().replace(' ', '')
        ]
        
        matched_states = []
        
        # Try matching each variation against all patterns
        for state_code, patterns in self.state_patterns.items():
            for pattern in patterns:
                for text_var in text_variations:
                    if re.match(pattern, text_var):
                        matched_states.append({
                            "code": state_code,
                            "name": self.state_names[state_code],
                            "confidence": "high",
                            "pattern_matched": pattern,
                            "text_used": text_var
                        })
        
        if matched_states:
            return {
                "code": matched_states[0]["code"],
                "name": matched_states[0]["name"],
                "confidence": matched_states[0]["confidence"],
            }

        return self._fallback_recognition(cleaned_text)
    
    def _fallback_recognition(self, cleaned_text: str) -> Optional[Dict[str, str]]:
        """
        Fallback recognition when no exact matches are found.
        
        Args:
            cleaned_text (str): The cleaned license plate text.
            
        Returns:
            Optional[Dict[str, str]]: A dictionary with state information if recognized, otherwise None.
        """
        common_patterns = [
            (r'^[A-Z]{3}[0-9]{3,4}$', 'medium'),
            (r'^[0-9]{3}[A-Z]{3}$', 'medium'),
            (r'^[A-Z]{2}[0-9]{4,5}$', 'medium')
        ]

        for pattern, confidence in common_patterns:
            if re.match(pattern, cleaned_text):
            
                return {
                    "code": "UNKNOWN",
                    "name": "Unkown State",
                    "confidence": confidence
                }
        return None

    def get_supported_states(self) -> List[Dict[str, str]]:
        """Return List of all supported states."""
        return [
            {"code": code, "name": name}
            for code, name in self.state_names.items()
        ]
