"""
Utilities for filtering and extracting actual license plate numbers from OCR results.
"""

import re
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def is_likely_plate_text(text: str) -> bool:
    """
    Check if text looks like a license plate number.
    
    License plates typically:
    - Are 2-8 characters long
    - Contain mix of letters and numbers (or all letters/numbers for some states)
    - Don't contain common words
    - Are not just years (4 digits starting with 19 or 20)
    """
    text = text.strip().upper()
    
    # Remove spaces and special chars for analysis
    clean = re.sub(r'[^A-Z0-9]', '', text)
    
    # Skip if it's just a year (like 2018, 2019, etc.)
    if re.match(r'^(19|20)\d{2}$', clean):
        return False
    
    # Skip month abbreviations
    if clean in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']:
        return False
    
    # Too short or too long
    if len(clean) < 2 or len(clean) > 8:
        return False
    
    # Skip if it's a common word or phrase
    skip_words = [
        'OPEN', 'ROAD', 'COM', 'JERSEY', 'STATE', 'GARDEN', 'NISSAN', 
        'MORRISTOWN', 'NEW', 'FAN', 'ATIUW', 'ARLIUW', 'ROADCOM',
        'OPENROAD', 'WEBSITE', 'DEALER', 'SALES', 'SERVICE', 'AUTO',
        'CALIFORNIA', 'TEXAS', 'FLORIDA', 'YORK', 'MONTH', 'YEAR',
        'GOV', 'DMV', 'DEPT', 'MOTOR', 'VEHICLES', 'USA', 'AMERICA',
        'REGISTRATION', 'EXPIRES', 'VALID', 'THRU', 'COUNTY'
    ]
    
    for word in skip_words:
        if word in clean:
            return False
    
    # California specific pattern: #XXX### (digit + 3 letters + 3 digits)
    if re.match(r'^\d[A-Z]{3}\d{3}$', clean):
        return True
    
    # Check if it looks like a plate format
    # Must be 4-8 characters
    if 4 <= len(clean) <= 8:
        # All numbers (some states)
        if clean.isdigit() and len(clean) >= 5:
            return True
        # All letters (rare but possible)
        if clean.isalpha() and len(clean) >= 4:
            return True
        # Mix of letters and numbers
        has_letter = any(c.isalpha() for c in clean)
        has_number = any(c.isdigit() for c in clean)
        if has_letter and has_number:
            return True
    
    return False

def extract_plate_number(ocr_results: List) -> Tuple[str, float]:
    """
    Extract the most likely license plate number from OCR results.
    """
    plate_candidates = []
    all_texts = []
    
    # First pass: collect all texts and look for plate patterns
    for result in ocr_results:
        if len(result) >= 2:
            text = result[1].strip().upper()
            conf = result[2] if len(result) > 2 else 1.0
            
            # Store all texts for debugging
            all_texts.append((text, conf))
            
            # Check if it looks like a plate
            if is_likely_plate_text(text):
                # Clean the text
                cleaned = re.sub(r'[^A-Z0-9\s\-]', '', text)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                
                # California pattern: #XXX### 
                ca_pattern = r'^\d[A-Z]{3}\d{3}$'
                if re.match(ca_pattern, cleaned.replace(' ', '')):
                    plate_candidates.append((cleaned, conf * 1.5))  # Boost CA format
                    
                # New Jersey pattern: X##XXX
                nj_pattern = r'^[A-Z]\d{2}[A-Z]{3}$'
                if re.match(nj_pattern, cleaned.replace(' ', '')):
                    plate_candidates.append((cleaned, conf * 1.5))  # Boost NJ format
                    
                # Generic patterns
                else:
                    plate_candidates.append((cleaned, conf))
    
    # Second pass: try to find plate numbers that might be split
    # Look for California pattern in parts (like "6XSU" + "832")
    for i in range(len(all_texts)):
        for j in range(i + 1, min(i + 3, len(all_texts))):  # Look ahead up to 2 texts
            combined = all_texts[i][0].replace(' ', '') + all_texts[j][0].replace(' ', '')
            combined_conf = (all_texts[i][1] + all_texts[j][1]) / 2
            
            if is_likely_plate_text(combined):
                # Format with space in typical position
                if re.match(r'^\d[A-Z]{3}\d{3}$', combined):  # California
                    formatted = f"{combined[0]}{combined[1:4]} {combined[4:]}"
                    plate_candidates.append((formatted, combined_conf * 1.2))
                elif re.match(r'^[A-Z]\d{2}[A-Z]{3}$', combined):  # New Jersey
                    formatted = f"{combined[:3]} {combined[3:]}"
                    plate_candidates.append((formatted, combined_conf * 1.2))
                else:
                    plate_candidates.append((combined, combined_conf))
    
    # Third pass: Look for the longest alphanumeric sequence that's not a skip word
    if not plate_candidates:
        for text, conf in all_texts:
            cleaned = re.sub(r'[^A-Z0-9]', '', text)
            if 5 <= len(cleaned) <= 8 and not any(skip in cleaned for skip in ['GOV', 'DMV', 'CALIFORNIA']):
                # Check if it has the right character mix
                has_letter = any(c.isalpha() for c in cleaned)
                has_number = any(c.isdigit() for c in cleaned)
                if (has_letter and has_number) or len(cleaned) >= 6:
                    plate_candidates.append((cleaned, conf * 0.8))
    
    # Log all candidates for debugging
    if plate_candidates:
        logger.info(f"Plate candidates found: {plate_candidates}")
    
    # Return best candidate
    if plate_candidates:
        # Sort by confidence
        plate_candidates.sort(key=lambda x: x[1], reverse=True)
        return plate_candidates[0]
    
    return "", 0.0

def detect_state_from_context(ocr_results: List) -> Tuple[Optional[str], float]:
    """
    Detect state from contextual text in the image.
    """
    context_text = ""
    for result in ocr_results:
        if len(result) >= 2:
            context_text += " " + result[1].upper()
    
    # Log context for debugging
    logger.info(f"Context text for state detection: {context_text}")
    
    # State mappings - comprehensive list
    state_indicators = {
        # Direct state names
        'CALIFORNIA': 'CA', 'CALIFORNIA,': 'CA', 'CALIFORNIA.': 'CA',
        'NEW JERSEY': 'NJ', 'NEWJERSEY': 'NJ', 'JERSEY': 'NJ',
        'CONNECTICUT': 'CT', 'NEW YORK': 'NY', 'NEWYORK': 'NY',
        'TEXAS': 'TX', 'FLORIDA': 'FL', 'ILLINOIS': 'IL',
        'PENNSYLVANIA': 'PA', 'OHIO': 'OH', 'GEORGIA': 'GA',
        'MICHIGAN': 'MI', 'VIRGINIA': 'VA', 'MASSACHUSETTS': 'MA',
        'INDIANA': 'IN', 'ARIZONA': 'AZ', 'TENNESSEE': 'TN',
        'MISSOURI': 'MO', 'MARYLAND': 'MD', 'WISCONSIN': 'WI',
        'MINNESOTA': 'MN', 'COLORADO': 'CO', 'ALABAMA': 'AL',
        'SOUTH CAROLINA': 'SC', 'LOUISIANA': 'LA', 'KENTUCKY': 'KY',
        'OREGON': 'OR', 'OKLAHOMA': 'OK', 'NEVADA': 'NV',
        'UTAH': 'UT', 'IOWA': 'IA', 'ARKANSAS': 'AR',
        'MISSISSIPPI': 'MS', 'KANSAS': 'KS', 'NEW MEXICO': 'NM',
        'NEBRASKA': 'NE', 'WEST VIRGINIA': 'WV', 'IDAHO': 'ID',
        'HAWAII': 'HI', 'NEW HAMPSHIRE': 'NH', 'MAINE': 'ME',
        'MONTANA': 'MT', 'RHODE ISLAND': 'RI', 'DELAWARE': 'DE',
        'SOUTH DAKOTA': 'SD', 'NORTH DAKOTA': 'ND', 'ALASKA': 'AK',
        'VERMONT': 'VT', 'WYOMING': 'WY', 'WASHINGTON': 'WA',
        'NORTH CAROLINA': 'NC',
        
        # State codes (sometimes visible on plates)
        ' CA ': 'CA', ' NJ ': 'NJ', ' NY ': 'NY', ' TX ': 'TX',
        ' FL ': 'FL', ' IL ': 'IL', ' PA ': 'PA', ' OH ': 'OH',
        
        # State nicknames/slogans
        'GARDEN STATE': 'NJ', 'GARDENSTATE': 'NJ',
        'EMPIRE STATE': 'NY', 'EMPIRESTATE': 'NY',
        'GOLDEN STATE': 'CA', 'GOLDENSTATE': 'CA',
        'SUNSHINE STATE': 'FL', 'SUNSHINESTATE': 'FL',
        'LONE STAR': 'TX', 'LONESTAR': 'TX',
        'CONSTITUTION STATE': 'CT',
        'FIRST STATE': 'DE', 'ALOHA STATE': 'HI',
        'LAND OF LINCOLN': 'IL', 'HOOSIER STATE': 'IN',
        'HAWKEYE STATE': 'IA', 'PELICAN STATE': 'LA',
        'OLD DOMINION': 'VA', 'VOLUNTEER STATE': 'TN',
        'GRANITE STATE': 'NH', 'SILVER STATE': 'NV',
        'BUCKEYE STATE': 'OH', 'KEYSTONE STATE': 'PA',
        'OCEAN STATE': 'RI', 'PALMETTO STATE': 'SC',
        'MOUNT RUSHMORE': 'SD', 'GREEN MOUNTAIN': 'VT',
        'EVERGREEN STATE': 'WA', 'MOUNTAIN STATE': 'WV',
        
        # Domain patterns (like dmv.ca.gov)
        'CA.GOV': 'CA', 'DMV.CA': 'CA',
        'NJ.GOV': 'NJ', 'DMV.NJ': 'NJ',
        'NY.GOV': 'NY', 'DMV.NY': 'NY',
    }
    
    # Check each indicator
    for indicator, state_code in state_indicators.items():
        if indicator in context_text:
            logger.info(f"Found state indicator: '{indicator}' -> {state_code}")
            return state_code, 0.95
    
    # Check for domain patterns like "dmv.XX.gov"
    dmv_pattern = r'DMV\.([A-Z]{2})\.GOV'
    dmv_match = re.search(dmv_pattern, context_text)
    if dmv_match:
        state_code = dmv_match.group(1)
        logger.info(f"Found DMV domain pattern: {state_code}")
        return state_code, 0.9
    
    # Check for standalone state codes
    state_code_pattern = r'\b([A-Z]{2})\b'
    matches = re.findall(state_code_pattern, context_text)
    valid_state_codes = {
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
    }
    
    for match in matches:
        if match in valid_state_codes and match not in ['IN', 'OR', 'OK']:  # Avoid common words
            logger.info(f"Found standalone state code: {match}")
            return match, 0.7
    
    return None, 0.0