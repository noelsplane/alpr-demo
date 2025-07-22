"""
unified_state_recognition.py
Consolidated state recognition system that combines multiple detection methods.
"""

import re
import logging
from typing import Optional, Dict, List, Tuple
from difflib import SequenceMatcher
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class UnifiedStateRecognizer:
    """
    Unified state recognition system that combines:
    1. Pattern matching on plate text
    2. Context detection from surrounding text
    3. Visual features from the plate region
    4. Confidence-based voting
    """
    
    def __init__(self):
        # Initialize all state data
        self._init_state_patterns()
        self._init_state_keywords()
        self._init_state_visual_features()
        
    def _init_state_patterns(self):
        """Initialize comprehensive state plate patterns for all 50 states + DC."""
        self.state_patterns = {
            'AL': {
                'patterns': [
                    r'^\d{7}$',               # 1234567
                    r'^\d[A-Z]{2}\d{4}$',     # 1AB2345
                    r'^[A-Z]{1,3}\d{3,4}$',   # ABC123, ABC1234
                    r'^\d{2}[A-Z]{2}\d{3}$',   # 12AB345
                ],
                'weight': 0.85
            },
            'AK': {
                'patterns': [
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                    r'^[A-Z]{2}\d{4}$',       # AB1234
                ],
                'weight': 0.85
            },
            'AZ': {
                'patterns': [
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                    r'^[A-Z]\d{6}$',          # A123456
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                    r'^[A-Z]{2}\d{5}$',       # AB12345
                ],
                'weight': 0.85
            },
            'AR': {
                'patterns': [
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^\d{3}\s?[A-Z]{3}$',    # 123 ABC
                ],
                'weight': 0.85
            },
            'CA': {
                'patterns': [
                    r'^[1-9][A-Z]{3}\d{3}$',  # 1ABC234 - Most common CA format
                    r'^\d[A-Z]{3}\d{3}$',     # 7ABC234
                    r'^[A-Z]{2}\d{5}$',       # AB12345 - Commercial
                ],
                'weight': 1.0,
                'notes': 'CA plates always start with a digit'
            },
            'CO': {
                'patterns': [
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                    r'^[A-Z]{2}\d{4}$',       # AB1234
                    r'^\d{2}[A-Z]{4}$',       # 12ABCD
                    r'^[A-Z]{3}-\d{3}$',      # ABC-123
                ],
                'weight': 0.85
            },
            'CT': {
                'patterns': [
                    r'^[A-Z]{2}\d{5}$',       # AB12345
                    r'^[A-Z]{2}\s?\d{5}$',    # AB 12345
                    r'^\d[A-Z]\d{5}$',        # 1A23456
                    r'^\d[A-Z]\s?\d{5}$',     # 1A 23456
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                ],
                'weight': 0.85
            },
            'DE': {
                'patterns': [
                    r'^\d{6}$',               # 123456
                    r'^\d{7}$',               # 1234567
                    r'^PC\d{4,5}$',           # PC1234 or PC12345
                ],
                'weight': 0.85
            },
            'FL': {
                'patterns': [
                    r'^[A-Z]{4}\d{2}$',       # ABCD12
                    r'^[A-Z]\d{2}[A-Z]{3}$',  # A12BCD
                    r'^[A-Z]{3}[A-Z]\d{2}$',  # ABCD12
                    r'^[A-Z]\d{2}\s?[A-Z]{3}$', # A12 BCD
                    r'^[A-Z]{3}\s?[A-Z]\d{2}$', # ABC D12
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                ],
                'weight': 0.95,
                'notes': 'FL has unique letter-heavy patterns'
            },
            'GA': {
                'patterns': [
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                    r'^[A-Z]{3}\d{3}[A-Z]$',  # ABC123A
                    r'^\d{3}[A-Z]{4}$',       # 123ABCD
                ],
                'weight': 0.85
            },
            'HI': {
                'patterns': [
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^[A-Z]{2}\d{4}$',       # AB1234
                    r'^[A-Z]\d{5}$',          # A12345
                ],
                'weight': 0.85
            },
            'ID': {
                'patterns': [
                    r'^[A-Z]\d{6}$',          # A123456
                    r'^\d[A-Z]\d{5}$',        # 1A23456
                    r'^[A-Z]{2}\d{5}$',       # AB12345
                    r'^[A-Z]\s?\d{6}$',       # A 123456
                    r'^[A-Z]{2}\d{3}[A-Z]$',  # AB123C
                ],
                'weight': 0.85
            },
            'IL': {
                'patterns': [
                    r'^[A-Z]\d{5}$',          # A12345
                    r'^[A-Z]\d{6}$',          # A123456
                    r'^[A-Z]{2}\d{5}$',       # AB12345
                    r'^[A-Z]\s?\d{5,6}$',     # A 12345
                    r'^[A-Z]{2}\s?\d{5}$',    # AB 12345
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                ],
                'weight': 0.85
            },
            'IN': {
                'patterns': [
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^\d{2}[A-Z]\d{3}$',     # 12A345
                    r'^\d{4}[A-Z]{2}$',       # 1234AB
                ],
                'weight': 0.85
            },
            'IA': {
                'patterns': [
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                    r'^[A-Z]{2}\d{4}$',       # AB1234
                ],
                'weight': 0.85
            },
            'KS': {
                'patterns': [
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                    r'^[A-Z]{2}\d{4}$',       # AB1234
                ],
                'weight': 0.85
            },
            'KY': {
                'patterns': [
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                ],
                'weight': 0.85
            },
            'LA': {
                'patterns': [
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                ],
                'weight': 0.85
            },
            'ME': {
                'patterns': [
                    r'^\d{4}[A-Z]{2}$',       # 1234AB
                    r'^[A-Z]{2}\d{4}$',       # AB1234
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                    r'^\d{4}\s?[A-Z]{2}$',    # 1234 AB
                ],
                'weight': 0.85
            },
            'MD': {
                'patterns': [
                    r'^\d[A-Z]{2}\d{4}$',     # 1AB2345
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^[A-Z]{2}\d{4}[A-Z]$',  # AB1234C
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                ],
                'weight': 0.85
            },
            'MA': {
                'patterns': [
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^\d{3}[A-Z]{2}\d$',     # 284FH8
                    r'^\d{3}[A-Z]{2}$',       # 284FH (without last digit)
                    r'^\d[A-Z]{2}\d{3}$',     # 1AB234
                    r'^\d{2}[A-Z]{2}\d{2}$',  # 12AB34
                    r'^[A-Z]{2}\d{4}$',       # AB1234
                    r'^\d{4}[A-Z]{2}$',       # 1234AB
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                ],
                'weight': 0.9,
                'notes': 'MA uses various patterns including ###LL#'
            },
            'MI': {
                'patterns': [
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                    r'^[A-Z]{3}\d{3}[A-Z]$',  # ABC123A
                    r'^[A-Z]{3}\s?\d{4}$',    # ABC 1234
                    r'^\d[A-Z]{2}\d{3}$',     # 1AB234
                    r'^[A-Z]{2}\d{4}$',       # AB1234
                ],
                'weight': 0.85
            },
            'MN': {
                'patterns': [
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                    r'^[A-Z]{3}\s?\d{3}$',    # ABC 123
                ],
                'weight': 0.85
            },
            'MS': {
                'patterns': [
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                    r'^[A-Z]{2}\d{5}$',       # AB12345
                ],
                'weight': 0.85
            },
            'MO': {
                'patterns': [
                    r'^[A-Z]{2}\d[A-Z]\d{2}$',    # AB1C23
                    r'^[A-Z]\d{2}[A-Z]{3}$',      # A12BCD
                    r'^[A-Z]{2}\d{4}$',           # AB1234
                    r'^[A-Z]{2}\d[A-Z]{2}\d$',    # AB1CD2
                    r'^[A-Z]{2}\d\s?[A-Z]\d{2}$', # AB1 C23
                ],
                'weight': 0.85
            },
            'MT': {
                'patterns': [
                    r'^\d{2}-\d{4}[A-Z]$',    # 12-3456A (county-number-letter)
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^\d[A-Z]\d{4}$',        # 1A2345
                    r'^\d{6}[A-Z]$',          # 123456A
                ],
                'weight': 0.85
            },
            'NE': {
                'patterns': [
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^\d{2}[A-Z]\d{3}$',     # 12A345 (county prefix)
                    r'^[A-Z]\d{5}$',          # A12345 (commercial)
                    r'^\d{1,2}-[A-Z]\d{4}$',  # 1-A2345 or 12-A3456
                ],
                'weight': 0.85
            },
            'NV': {
                'patterns': [
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^[A-Z]{2}\d{4}$',       # AB1234
                    r'^\d{2}[A-Z]\d{3}$',     # 12A345
                ],
                'weight': 0.85
            },
            'NH': {
                'patterns': [
                    r'^\d{3}\s?\d{4}$',       # 123 4567
                    r'^\d{7}$',               # 1234567
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                ],
                'weight': 0.85
            },
            'NJ': {
                'patterns': [
                    r'^[A-Z]\d{2}[A-Z]{3}$',  # A12BCD - Very distinctive
                    r'^[A-Z]\d{2}\s?[A-Z]{3}$', # A12 BCD
                    r'^[A-Z]{3}\d{2}[A-Z]$',  # ABC12D
                    r'^[A-Z]{3}\s?\d{2}[A-Z]$', # ABC 12D
                ],
                'weight': 0.95,
                'notes': 'NJ has very distinctive A##XXX pattern'
            },
            'NM': {
                'patterns': [
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                    r'^[A-Z]{2}\d{4}$',       # AB1234
                    r'^\d{3}\s?[A-Z]{3}$',    # 123 ABC
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                ],
                'weight': 0.85
            },
            'NY': {
                'patterns': [
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                    r'^[A-Z]{3}\s?\d{4}$',    # ABC 1234
                    r'^[A-Z]{3}\d{3}[A-Z]$',  # ABC123A
                    r'^[A-Z]{3}\s?\d{3}[A-Z]$', # ABC 123A
                ],
                'weight': 0.9
            },
            'NC': {
                'patterns': [
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                    r'^[A-Z]{3}\s?\d{4}$',    # ABC 1234
                    r'^[A-Z]{2}\d{5}$',       # AB12345
                ],
                'weight': 0.85
            },
            'ND': {
                'patterns': [
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                    r'^[A-Z]{2}\d{4}$',       # AB1234
                ],
                'weight': 0.85
            },
            'OH': {
                'patterns': [
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                    r'^[A-Z]{3}\s?\d{4}$',    # ABC 1234
                    r'^[A-Z]{3}\d{3}[A-Z]$',  # ABC123A
                    r'^[A-Z]{2}\d{5}$',       # AB12345
                ],
                'weight': 0.85
            },
            'OK': {
                'patterns': [
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                ],
                'weight': 0.85
            },
            'OR': {
                'patterns': [
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                    r'^[A-Z]{2}\d{4}$',       # AB1234
                    r'^[A-Z]{3}\s?\d{3}$',    # ABC 123
                ],
                'weight': 0.85
            },
            'PA': {
                'patterns': [
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                    r'^[A-Z]{3}\s?\d{4}$',    # ABC 1234
                    r'^[A-Z]{3}[A-Z]\d{3}$',  # ABCD123
                ],
                'weight': 0.85
            },
            'RI': {
                'patterns': [
                    r'^\d{6}$',               # 123456
                    r'^[A-Z]{2}\d{3}$',       # AB123
                    r'^[A-Z]{2}\s?\d{3}$',    # AB 123
                    r'^\d{3}\s?\d{3}$',       # 123 456
                    r'^[A-Z]{2}\d{4}$',       # AB1234
                ],
                'weight': 0.85
            },
            'SC': {
                'patterns': [
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^[A-Z]{3}\s?\d{3}$',    # ABC 123
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                ],
                'weight': 0.85
            },
            'SD': {
                'patterns': [
                    r'^\d[A-Z]{2}\d{3}$',     # 1AB234
                    r'^\d{2}[A-Z]\d{3}$',     # 12A345
                    r'^[A-Z]{2}\d{4}$',       # AB1234
                    r'^\d{6}[A-Z]$',          # 123456A
                ],
                'weight': 0.85
            },
            'TN': {
                'patterns': [
                    r'^[A-Z]\d{2}\d{2}[A-Z]$',    # A12-34B (county format)
                    r'^[A-Z]{3}\d{3}$',           # ABC123
                    r'^[A-Z]{3}\d{4}$',           # ABC1234
                    r'^[A-Z]\d{5}$',              # A12345
                    r'^[A-Z]\d{2}-\d{2}[A-Z]$',   # A12-34B
                ],
                'weight': 0.85
            },
            'TX': {
                'patterns': [
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                    r'^[A-Z]{3}\s?\d{4}$',    # ABC 1234
                    r'^[A-Z]{2}\d[A-Z]\d{3}$', # AB1C234 (truck)
                    r'^[A-Z]{2}\d\s?[A-Z]\d{3}$', # AB1 C234
                    r'^[A-Z]{2}\d{5}$',       # AB12345
                ],
                'weight': 0.9
            },
            'UT': {
                'patterns': [
                    r'^[A-Z]\d{2}[A-Z]{2}$',  # A12BC
                    r'^[A-Z]\d{2}\s?[A-Z]{2}$', # A12 BC
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                    r'^[A-Z]{2}\d{4}$',       # AB1234
                ],
                'weight': 0.85
            },
            'VT': {
                'patterns': [
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^\d{3}[A-Z]\d{2}$',     # 123A45
                    r'^[A-Z]{2}\d{4}$',       # AB1234
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                ],
                'weight': 0.85
            },
            'VA': {
                'patterns': [
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                    r'^[A-Z]{3}\s?\d{4}$',    # ABC 1234
                    r'^[A-Z]{2}\d{5}$',       # AB12345
                    r'^[A-Z]{3}\d{3}[A-Z]$',  # ABC123A
                ],
                'weight': 0.85
            },
            'WA': {
                'patterns': [
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                    r'^[A-Z]{3}\s?\d{4}$',    # ABC 1234
                    r'^[A-Z]{3}\d{3}[A-Z]$',  # ABC123A
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                ],
                'weight': 0.85
            },
            'WV': {
                'patterns': [
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                    r'^[A-Z]\d{5}$',          # A12345
                    r'^\d[A-Z]\d{4}$',        # 1A2345
                    r'^[A-Z]{2}\d{4}$',       # AB1234
                    r'^\d{7}$',               # 1234567
                ],
                'weight': 0.85
            },
            'WI': {
                'patterns': [
                    r'^[A-Z]{3}\d{4}$',       # ABC1234
                    r'^[A-Z]{3}\s?\d{4}$',    # ABC 1234
                    r'^\d{3}[A-Z]{3}$',       # 123ABC
                    r'^[A-Z]{2}\d{5}$',       # AB12345
                ],
                'weight': 0.85
            },
            'WY': {
                'patterns': [
                    r'^\d{2}-\d{4}$',         # 12-3456 (county-number)
                    r'^\d{2}\s?\d{4}$',       # 12 3456
                    r'^\d[A-Z]{3}$',          # 1ABC
                    r'^\d\s?[A-Z]{3}$',       # 1 ABC
                    r'^\d{2}-\d{5}$',         # 12-12345
                ],
                'weight': 0.85
            },
            'DC': {
                'patterns': [
                    r'^[A-Z]{2}\d{4}$',       # AB1234
                    r'^[A-Z]{2}\s?\d{4}$',    # AB 1234
                    r'^[A-Z]{3}\d{3}$',       # ABC123
                ],
                'weight': 0.85
            }
        }
        
    def _init_state_keywords(self):
        """Initialize state keywords and context clues."""
        self.state_keywords = {
            # State names
            'CALIFORNIA': 'CA', 'TEXAS': 'TX', 'NEW YORK': 'NY', 'FLORIDA': 'FL',
            'NEW JERSEY': 'NJ', 'ILLINOIS': 'IL', 'PENNSYLVANIA': 'PA', 'OHIO': 'OH',
            'GEORGIA': 'GA', 'MICHIGAN': 'MI', 'VIRGINIA': 'VA', 'MASSACHUSETTS': 'MA',
            'ARIZONA': 'AZ', 'WASHINGTON': 'WA', 'TENNESSEE': 'TN', 'INDIANA': 'IN',
            'MISSOURI': 'MO', 'MARYLAND': 'MD', 'WISCONSIN': 'WI', 'COLORADO': 'CO',
            'MINNESOTA': 'MN', 'SOUTH CAROLINA': 'SC', 'ALABAMA': 'AL', 'LOUISIANA': 'LA',
            'KENTUCKY': 'KY', 'OREGON': 'OR', 'OKLAHOMA': 'OK', 'CONNECTICUT': 'CT',
            'UTAH': 'UT', 'IOWA': 'IA', 'NEVADA': 'NV', 'ARKANSAS': 'AR', 'MISSISSIPPI': 'MS',
            'KANSAS': 'KS', 'NEW MEXICO': 'NM', 'NEBRASKA': 'NE', 'WEST VIRGINIA': 'WV',
            'IDAHO': 'ID', 'HAWAII': 'HI', 'NEW HAMPSHIRE': 'NH', 'MAINE': 'ME',
            'MONTANA': 'MT', 'RHODE ISLAND': 'RI', 'DELAWARE': 'DE', 'SOUTH DAKOTA': 'SD',
            'NORTH DAKOTA': 'ND', 'ALASKA': 'AK', 'VERMONT': 'VT', 'WYOMING': 'WY',
            'DISTRICT OF COLUMBIA': 'DC', 'WASHINGTON DC': 'DC', 'WASHINGTON D.C.': 'DC',
            
            # State mottos and nicknames
            'GARDEN STATE': 'NJ', 'EMPIRE STATE': 'NY', 'GOLDEN STATE': 'CA',
            'SUNSHINE STATE': 'FL', 'LONE STAR': 'TX', 'LAND OF LINCOLN': 'IL',
            'KEYSTONE STATE': 'PA', 'BUCKEYE STATE': 'OH', 'PEACH STATE': 'GA',
            'WOLVERINE STATE': 'MI', 'OLD DOMINION': 'VA', 'BAY STATE': 'MA',
            'GRAND CANYON STATE': 'AZ', 'EVERGREEN STATE': 'WA', 'VOLUNTEER STATE': 'TN',
            'HOOSIER STATE': 'IN', 'SHOW ME STATE': 'MO', 'FREE STATE': 'MD',
            'BADGER STATE': 'WI', 'CENTENNIAL STATE': 'CO', 'NORTH STAR STATE': 'MN',
            'PALMETTO STATE': 'SC', 'YELLOWHAMMER STATE': 'AL', 'PELICAN STATE': 'LA',
            'BLUEGRASS STATE': 'KY', 'BEAVER STATE': 'OR', 'SOONER STATE': 'OK',
            'CONSTITUTION STATE': 'CT', 'BEEHIVE STATE': 'UT', 'HAWKEYE STATE': 'IA',
            'SILVER STATE': 'NV', 'NATURAL STATE': 'AR', 'MAGNOLIA STATE': 'MS',
            'SUNFLOWER STATE': 'KS', 'LAND OF ENCHANTMENT': 'NM', 'CORNHUSKER STATE': 'NE',
            'MOUNTAIN STATE': 'WV', 'GEM STATE': 'ID', 'ALOHA STATE': 'HI',
            'GRANITE STATE': 'NH', 'PINE TREE STATE': 'ME', 'TREASURE STATE': 'MT',
            'OCEAN STATE': 'RI', 'FIRST STATE': 'DE', 'MOUNT RUSHMORE': 'SD',
            'PEACE GARDEN STATE': 'ND', 'LAST FRONTIER': 'AK', 'GREEN MOUNTAIN': 'VT',
            'EQUALITY STATE': 'WY', 'NATIONS CAPITAL': 'DC',
            
            # DMV URLs
            'DMV.CA.GOV': 'CA', 'DMV.NY.GOV': 'NY', 'FLHSMV.GOV': 'FL',
            'NJMVC.GOV': 'NJ', 'TXD.TEXAS.GOV': 'TX', 'DMV.PA.GOV': 'PA',
            'BMV.OHIO.GOV': 'OH', 'DMV.VIRGINIA.GOV': 'VA', 'MASS.GOV/RMV': 'MA',
            'AZDOT.GOV': 'AZ', 'DOL.WA.GOV': 'WA', 'TN.GOV/SAFETY': 'TN',
            'IN.GOV/BMV': 'IN', 'DOR.MO.GOV': 'MO', 'MVA.MARYLAND.GOV': 'MD',
            
            # State-specific terms
            'SACRAMENTO': 'CA', 'ALBANY': 'NY', 'TALLAHASSEE': 'FL',
            'AUSTIN': 'TX', 'TRENTON': 'NJ', 'SPRINGFIELD': 'IL',
            'HARRISBURG': 'PA', 'COLUMBUS': 'OH', 'ATLANTA': 'GA',
            'LANSING': 'MI', 'RICHMOND': 'VA', 'BOSTON': 'MA',
            'PHOENIX': 'AZ', 'OLYMPIA': 'WA', 'NASHVILLE': 'TN',
            'INDIANAPOLIS': 'IN', 'JEFFERSON CITY': 'MO', 'ANNAPOLIS': 'MD',
        }
        
    def _init_state_visual_features(self):
        """Initialize visual features for states (colors, logos, etc)."""
        self.state_visual_features = {
            'CA': {
                'colors': ['red', 'blue', 'white'],
                'has_bear': True,  # California bear logo
                'script_style': 'cursive',
            },
            'TX': {
                'colors': ['black', 'white'],
                'has_star': True,  # Lone star
                'script_style': 'bold',
            },
            'NY': {
                'colors': ['blue', 'white', 'yellow'],
                'has_statue': True,  # Statue of Liberty
                'script_style': 'serif',
            },
            'MA': {
                'colors': ['red', 'white'],
                'script_style': 'serif',
                'notes': 'Red text on white background'
            },
            # Add more visual features...
        }
    
    def recognize_state(self, 
                       plate_text: str, 
                       ocr_results: List = None,
                       plate_image: np.ndarray = None,
                       context_image: np.ndarray = None) -> Dict:
        """
        Main entry point for state recognition using all available data.
        
        Args:
            plate_text: The cleaned license plate text
            ocr_results: All OCR results from the plate region
            plate_image: The cropped plate image
            context_image: Larger context around the plate
            
        Returns:
            Dictionary with state info and confidence breakdown
        """
        results = {
            'state_code': None,
            'state_name': None,
            'confidence': 0.0,
            'method': 'none',
            'confidence_breakdown': {}
        }
        
        # Method 1: Pattern matching (40% weight)
        pattern_result = self._match_patterns(plate_text)
        if pattern_result:
            results['confidence_breakdown']['pattern'] = pattern_result
        
        # Method 2: Context detection (30% weight)
        if ocr_results:
            context_result = self._detect_from_context(ocr_results)
            if context_result:
                results['confidence_breakdown']['context'] = context_result
        
        # Method 3: Visual features (20% weight)
        if plate_image is not None:
            visual_result = self._detect_visual_features(plate_image)
            if visual_result:
                results['confidence_breakdown']['visual'] = visual_result
        
        # Method 4: Extended context (10% weight)
        if context_image is not None and ocr_results:
            extended_result = self._detect_extended_context(context_image, ocr_results)
            if extended_result:
                results['confidence_breakdown']['extended'] = extended_result
        
        # Combine results using weighted voting
        final_state, final_confidence = self._combine_results(results['confidence_breakdown'])
        
        if final_state:
            results['state_code'] = final_state
            results['state_name'] = self._get_state_name(final_state)
            results['confidence'] = final_confidence
            results['method'] = self._determine_method(results['confidence_breakdown'])
        
        return results
    
    def _match_patterns(self, plate_text: str) -> Optional[Dict]:
        """Match plate text against known patterns."""
        if not plate_text:
            return None
            
        cleaned = self._clean_plate_text(plate_text)
        best_matches = []
        
        for state_code, state_info in self.state_patterns.items():
            for pattern in state_info['patterns']:
                if re.match(pattern, cleaned):
                    confidence = state_info.get('weight', 0.8)
                    best_matches.append({
                        'state': state_code,
                        'confidence': confidence,
                        'pattern': pattern
                    })
        
        if best_matches:
            # Return the best match
            best = max(best_matches, key=lambda x: x['confidence'])
            return best
            
        return None
    
    def _detect_from_context(self, ocr_results: List) -> Optional[Dict]:
        """Detect state from OCR context."""
        all_text = ' '.join([str(r[1]) if len(r) > 1 else str(r) for r in ocr_results]).upper()
        
        # Check for state keywords
        for keyword, state_code in self.state_keywords.items():
            if keyword in all_text:
                # Higher confidence for full state names
                confidence = 0.9 if len(keyword) > 5 else 0.7
                return {
                    'state': state_code,
                    'confidence': confidence,
                    'keyword': keyword
                }
        
        # Check for state codes
        state_code_pattern = r'\b([A-Z]{2})\b'
        matches = re.findall(state_code_pattern, all_text)
        valid_states = set(self.state_patterns.keys())
        
        for match in matches:
            if match in valid_states:
                return {
                    'state': match,
                    'confidence': 0.6,
                    'keyword': f'Code: {match}'
                }
        
        return None
    
    def _detect_visual_features(self, plate_image: np.ndarray) -> Optional[Dict]:
        """Detect state from visual features (placeholder for now)."""
        # This is where you could add:
        # - Color detection
        # - Logo/symbol detection
        # - Font style analysis
        # For now, return None
        return None
    
    def _detect_extended_context(self, context_image: np.ndarray, ocr_results: List) -> Optional[Dict]:
        """Look for state clues in extended context around the plate."""
        # This could look for:
        # - Dealer frames with state names
        # - Registration stickers
        # - State-specific decorations
        return None
    
    def _combine_results(self, confidence_breakdown: Dict) -> Tuple[Optional[str], float]:
        """Combine results from multiple methods using weighted voting."""
        if not confidence_breakdown:
            return None, 0.0
        
        # Weight for each method
        weights = {
            'pattern': 0.4,
            'context': 0.3,
            'visual': 0.2,
            'extended': 0.1
        }
        
        # Aggregate votes
        state_scores = {}
        
        for method, result in confidence_breakdown.items():
            if result and 'state' in result:
                state = result['state']
                confidence = result['confidence']
                weight = weights.get(method, 0.1)
                
                if state not in state_scores:
                    state_scores[state] = 0.0
                
                state_scores[state] += confidence * weight
        
        if not state_scores:
            return None, 0.0
        
        # Get the best state
        best_state = max(state_scores.items(), key=lambda x: x[1])
        
        # Normalize confidence to 0-1 range
        total_weight = sum(weights[m] for m in confidence_breakdown.keys())
        normalized_confidence = best_state[1] / total_weight if total_weight > 0 else 0
        
        return best_state[0], min(normalized_confidence, 1.0)
    
    def _determine_method(self, confidence_breakdown: Dict) -> str:
        """Determine which method contributed most to the result."""
        if not confidence_breakdown:
            return 'none'
        
        methods = []
        for method, result in confidence_breakdown.items():
            if result:
                methods.append(method)
        
        if len(methods) > 1:
            return 'combined'
        elif methods:
            return methods[0]
        else:
            return 'none'
    
    def _clean_plate_text(self, text: str) -> str:
        """Clean plate text for pattern matching."""
        if not text:
            return ""
        
        # Basic cleaning
        cleaned = text.upper().strip()
        cleaned = re.sub(r'[^A-Z0-9\s\-]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove spaces for pattern matching
        cleaned = cleaned.replace(' ', '').replace('-', '')
        
        return cleaned
    
    def _get_state_name(self, state_code: str) -> str:
        """Get full state name from code."""
        state_names = {
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
        return state_names.get(state_code, 'Unknown')
    
    def get_debug_info(self, result: Dict) -> str:
        """Get detailed debug information about the recognition."""
        lines = []
        lines.append(f"State: {result.get('state_code', 'None')} ({result.get('state_name', 'Unknown')})")
        lines.append(f"Confidence: {result.get('confidence', 0):.2%}")
        lines.append(f"Method: {result.get('method', 'none')}")
        
        if 'confidence_breakdown' in result:
            lines.append("\nConfidence Breakdown:")
            for method, details in result['confidence_breakdown'].items():
                if details:
                    lines.append(f"  {method}: {details.get('confidence', 0):.2f}")
                    if 'pattern' in details:
                        lines.append(f"    Pattern: {details['pattern']}")
                    if 'keyword' in details:
                        lines.append(f"    Keyword: {details['keyword']}")
        
        return '\n'.join(lines)


# Convenience function for easy integration
def recognize_state(plate_text: str, 
                   ocr_results: List = None,
                   plate_image: np.ndarray = None,
                   context_image: np.ndarray = None,
                   debug: bool = False) -> Tuple[Optional[str], float, Optional[Dict]]:
    """
    Convenience function for state recognition.
    
    Returns:
        Tuple of (state_code, confidence, full_result_if_debug)
    """
    recognizer = UnifiedStateRecognizer()
    result = recognizer.recognize_state(plate_text, ocr_results, plate_image, context_image)
    
    state_code = result.get('state_code')
    confidence = result.get('confidence', 0.0)
    
    if debug:
        return state_code, confidence, result
    else:
        return state_code, confidence, None