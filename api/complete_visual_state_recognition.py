# complete_visual_state_recognition.py
"""
Production-ready visual state recognition system with actual implementation
of color analysis, template matching, and visual feature detection.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import re
from collections import Counter
import logging
from pathlib import Path
import json
from dataclasses import dataclass
from enum import Enum
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import scikit-learn only if available
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Color clustering will use fallback method.")

class StateColorScheme:
    """Comprehensive state license plate color schemes."""
    
    # Color ranges in HSV format for better color detection
    # Format: (hue_low, sat_low, val_low), (hue_high, sat_high, val_high)
    STATE_COLORS_HSV = {
        'CA': {
            'background': [(0, 0, 200), (180, 30, 255)],  # White background
            'text': [(100, 50, 50), (130, 255, 150)],     # Dark blue text
            'description': 'White background with dark blue text'
        },
        'NY': {
            'background': [(15, 50, 150), (35, 255, 255)],  # Gold/Yellow
            'text': [(100, 50, 50), (130, 255, 150)],       # Dark blue
            'accent': [(0, 0, 0), (180, 255, 50)],          # Black accents
            'description': 'Gold/yellow background with blue text'
        },
        'TX': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(0, 0, 0), (180, 255, 50)],            # Black
            'accent': [(100, 50, 50), (130, 255, 150)],     # Blue accent
            'description': 'White with black text and blue state name'
        },
        'FL': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(40, 50, 50), (80, 255, 255)],         # Green
            'accent': [(10, 100, 100), (25, 255, 255)],     # Orange
            'description': 'White with green text and orange elements'
        },
        'NJ': {
            'background': [(25, 50, 150), (45, 255, 255)],  # Yellow/cream
            'text': [(0, 0, 0), (180, 255, 50)],            # Black
            'description': 'Yellow/cream background with black text'
        },
        'PA': {
            'background': [(100, 30, 150), (130, 80, 255)], # Light blue
            'text': [(15, 50, 50), (35, 255, 255)],         # Yellow/gold
            'description': 'Blue background with yellow text'
        },
        'IL': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'accent': [(0, 100, 100), (10, 255, 255)],      # Red accents
            'description': 'White with blue text and red accents'
        },
        'OH': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'accent': [(0, 100, 100), (10, 255, 255)],      # Red
            'description': 'White with blue and red elements'
        },
        'GA': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(0, 100, 100), (10, 255, 255)],        # Red
            'accent': [(100, 50, 50), (130, 255, 150)],     # Blue
            'description': 'White with red text and blue accents'
        },
        'NC': {
            'background': [(100, 30, 150), (130, 80, 255)], # Light blue
            'text': [(0, 100, 50), (10, 255, 255)],         # Red
            'description': 'Blue background with red text'
        },
        'MI': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'description': 'White with blue text'
        },
        'VA': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'description': 'White with blue text'
        },
        'MA': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(0, 100, 100), (10, 255, 255)],        # Red
            'description': 'White with red text'
        },
        'AZ': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(15, 100, 100), (25, 255, 255)],       # Orange/copper
            'description': 'White with copper/orange text'
        },
        'WA': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'description': 'White with blue text'
        },
        'TN': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(40, 50, 50), (80, 255, 255)],         # Green
            'description': 'White with green elements'
        },
        'IN': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'description': 'White with blue text'
        },
        'MO': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'accent': [(0, 100, 100), (10, 255, 255)],      # Red
            'description': 'White with blue text and red accents'
        },
        'MD': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(0, 0, 0), (180, 255, 50)],            # Black
            'accent': [(0, 100, 100), (10, 255, 255)],      # Red flag elements
            'description': 'White with black text and flag elements'
        },
        'WI': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(0, 100, 100), (10, 255, 255)],        # Red
            'description': 'White with red text'
        },
        'MN': {
            'background': [(100, 30, 150), (130, 80, 255)], # Light blue
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'description': 'Light blue background with blue text'
        },
        'CO': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(40, 50, 50), (80, 255, 255)],         # Green (mountains)
            'description': 'White with green mountain graphics'
        },
        'AL': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(0, 100, 100), (10, 255, 255)],        # Red
            'accent': [(15, 50, 150), (35, 255, 255)],      # Yellow stars
            'description': 'White with red text and yellow elements'
        },
        'SC': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'description': 'White with blue text and palmetto tree'
        },
        'LA': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'accent': [(0, 100, 100), (10, 255, 255)],      # Red
            'description': 'White with blue text and pelican graphic'
        },
        'KY': {
            'background': [(100, 30, 150), (130, 80, 255)], # Light blue
            'text': [(0, 0, 200), (180, 30, 255)],          # White
            'description': 'Blue background with white text'
        },
        'OR': {
            'background': [(40, 30, 150), (80, 80, 255)],   # Light green
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'description': 'Green/teal background with blue text'
        },
        'OK': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'description': 'White with blue text'
        },
        'CT': {
            'background': [(100, 30, 150), (130, 80, 255)], # Light blue
            'text': [(100, 50, 50), (130, 255, 150)],       # Dark blue
            'description': 'Blue gradient background'
        },
        'IA': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'description': 'White with blue text'
        },
        'MS': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'description': 'White with blue text and magnolia'
        },
        'AR': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(0, 100, 100), (10, 255, 255)],        # Red
            'description': 'White with red text and diamond graphic'
        },
        'UT': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(15, 100, 100), (25, 255, 255)],       # Orange
            'accent': [(100, 50, 50), (130, 255, 150)],     # Blue
            'description': 'White with orange arch and blue sky'
        },
        'NV': {
            'background': [(100, 30, 150), (130, 80, 255)], # Light blue
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'description': 'Blue background with mountain graphics'
        },
        'WV': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'description': 'White with blue text'
        },
        'NE': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'description': 'White with blue text'
        },
        'ID': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(0, 100, 100), (10, 255, 255)],        # Red
            'description': 'White with red text and scenic graphics'
        },
        'HI': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(0, 0, 0), (180, 255, 50)],            # Black
            'accent': [(30, 100, 100), (40, 255, 255)],     # Rainbow colors
            'description': 'White with rainbow graphics'
        },
        'ME': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(0, 100, 100), (10, 255, 255)],        # Red
            'accent': [(100, 50, 50), (130, 255, 150)],     # Blue
            'description': 'White with red text and lobster graphic'
        },
        'NH': {
            'background': [(40, 30, 150), (80, 80, 255)],   # Green
            'text': [(0, 0, 200), (180, 30, 255)],          # White
            'description': 'Green background with white text'
        },
        'RI': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'description': 'White with blue text and wave graphics'
        },
        'MT': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'description': 'White with blue text and mountain graphics'
        },
        'DE': {
            'background': [(15, 50, 150), (35, 255, 255)],  # Gold
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'description': 'Gold background with blue text'
        },
        'SD': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(0, 0, 0), (180, 255, 50)],            # Black
            'description': 'White with black text and Mount Rushmore'
        },
        'ND': {
            'background': [(100, 30, 150), (130, 80, 255)], # Light blue
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'description': 'Blue background with buffalo graphic'
        },
        'AK': {
            'background': [(15, 50, 150), (35, 255, 255)],  # Yellow
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'description': 'Yellow background with blue text'
        },
        'VT': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(40, 50, 50), (80, 255, 255)],         # Green
            'description': 'White with green text'
        },
        'WY': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'description': 'White with blue text and cowboy graphic'
        },
        'NM': {
            'background': [(15, 100, 150), (35, 255, 255)], # Yellow/turquoise
            'text': [(0, 100, 100), (10, 255, 255)],        # Red
            'description': 'Yellow/turquoise with red text'
        },
        'KS': {
            'background': [(0, 0, 200), (180, 30, 255)],    # White
            'text': [(100, 50, 50), (130, 255, 150)],       # Blue
            'description': 'White with blue text'
        }
    }


@dataclass
class ColorAnalysisResult:
    """Result of color analysis for a license plate."""
    dominant_colors: List[Tuple[int, int, int]]  # BGR colors
    color_percentages: List[float]
    background_color: Optional[Tuple[int, int, int]]
    text_color: Optional[Tuple[int, int, int]]
    matched_states: Dict[str, float]  # State -> confidence


class VisualFeatureDetector:
    """Detects visual features in license plates."""
    
    def __init__(self, template_dir: Optional[str] = None):
        self.template_dir = Path(template_dir) if template_dir else Path("plate_templates")
        self.color_scheme = StateColorScheme()
        self.templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load state logo/symbol templates if available."""
        if not self.template_dir.exists():
            logger.info(f"Template directory {self.template_dir} not found. Template matching disabled.")
            return
        
        # Load any available templates
        for template_file in self.template_dir.glob("*.png"):
            state_code = template_file.stem.upper()
            template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
            if template is not None:
                self.templates[state_code] = template
                logger.info(f"Loaded template for {state_code}")
    
    def analyze_colors(self, plate_img: np.ndarray) -> ColorAnalysisResult:
        """
        Analyze colors in the license plate image.
        
        Args:
            plate_img: BGR image of the license plate
            
        Returns:
            ColorAnalysisResult with color analysis
        """
        if plate_img is None or plate_img.size == 0:
            return ColorAnalysisResult([], [], None, None, {})
        
        # Convert to HSV for better color analysis
        hsv_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
        
        # Get dominant colors
        dominant_colors, percentages = self._extract_dominant_colors(plate_img)
        
        # Identify background and text colors
        background_color, text_color = self._identify_background_text_colors(
            plate_img, dominant_colors, percentages
        )
        
        # Match against state color schemes
        matched_states = self._match_state_colors(hsv_img, dominant_colors)
        
        return ColorAnalysisResult(
            dominant_colors=dominant_colors,
            color_percentages=percentages,
            background_color=background_color,
            text_color=text_color,
            matched_states=matched_states
        )
    
    def _extract_dominant_colors(self, img: np.ndarray, n_colors: int = 4) -> Tuple[List, List]:
        """Extract dominant colors from image."""
        # Reshape image to pixels
        pixels = img.reshape(-1, 3)
        
        if SKLEARN_AVAILABLE and len(pixels) > n_colors:
            # Use KMeans clustering
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get colors and their percentages
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            
            # Calculate percentages
            percentages = []
            for i in range(n_colors):
                percentage = (labels == i).sum() / len(labels)
                percentages.append(percentage)
            
            # Sort by percentage
            sorted_indices = np.argsort(percentages)[::-1]
            colors = colors[sorted_indices]
            percentages = [percentages[i] for i in sorted_indices]
            
        else:
            # Fallback: Use histogram-based method
            colors, percentages = self._histogram_dominant_colors(img, n_colors)
        
        return colors.tolist(), percentages
    
    def _histogram_dominant_colors(self, img: np.ndarray, n_colors: int) -> Tuple[np.ndarray, List]:
        """Fallback method using color histograms."""
        # Quantize colors to reduce complexity
        quantized = (img // 32) * 32
        
        # Get unique colors and counts
        unique_colors, counts = np.unique(
            quantized.reshape(-1, 3), axis=0, return_counts=True
        )
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1][:n_colors]
        
        colors = unique_colors[sorted_indices]
        total_pixels = img.shape[0] * img.shape[1]
        percentages = [counts[i] / total_pixels for i in sorted_indices]
        
        return colors, percentages
    
    def _identify_background_text_colors(self, img: np.ndarray, 
                                       dominant_colors: List, 
                                       percentages: List) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """Identify which colors are likely background vs text."""
        if not dominant_colors:
            return None, None
        
        # Background is usually the most dominant color
        background_color = tuple(dominant_colors[0])
        
        # Text color is usually high contrast with background
        # Convert to grayscale for contrast calculation
        bg_gray = np.dot(background_color, [0.114, 0.587, 0.299])  # BGR weights
        
        max_contrast = 0
        text_color = None
        
        for i in range(1, min(len(dominant_colors), 3)):
            color = dominant_colors[i]
            color_gray = np.dot(color, [0.114, 0.587, 0.299])
            contrast = abs(color_gray - bg_gray)
            
            if contrast > max_contrast and percentages[i] > 0.05:  # At least 5% of image
                max_contrast = contrast
                text_color = tuple(color)
        
        return background_color, text_color
    
    def _match_state_colors(self, hsv_img: np.ndarray, dominant_colors: List) -> Dict[str, float]:
        """Match image colors against known state color schemes."""
        matched_states = {}
        h, w = hsv_img.shape[:2]
        total_pixels = h * w
        
        for state, color_ranges in self.color_scheme.STATE_COLORS_HSV.items():
            total_match_score = 0.0
            
            # Check each color range (background, text, accent)
            for color_type, (lower, upper) in color_ranges.items():
                if color_type == 'description':
                    continue
                
                # Create mask for this color range
                lower_bound = np.array(lower)
                upper_bound = np.array(upper)
                mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
                
                # Calculate percentage of pixels matching this color
                matching_pixels = cv2.countNonZero(mask)
                match_percentage = matching_pixels / total_pixels
                
                # Weight different color types
                if color_type == 'background':
                    weight = 0.5  # Background is most important
                elif color_type == 'text':
                    weight = 0.3
                else:  # accent colors
                    weight = 0.2
                
                total_match_score += match_percentage * weight
            
            # Only include states with significant color match
            if total_match_score > 0.1:  # At least 10% match
                matched_states[state] = min(total_match_score, 1.0)
        
        return matched_states
    
    def detect_state_logos(self, plate_img: np.ndarray) -> Dict[str, float]:
        """
        Detect state-specific logos or symbols using template matching.
        
        Args:
            plate_img: License plate image
            
        Returns:
            Dictionary of state -> confidence scores
        """
        if not self.templates:
            return {}
        
        # Convert to grayscale for template matching
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img
        
        detected_states = {}
        
        for state, template in self.templates.items():
            # Multi-scale template matching
            scales = [0.8, 1.0, 1.2]
            best_match = 0
            
            for scale in scales:
                # Resize template
                scaled_template = cv2.resize(
                    template, 
                    (int(template.shape[1] * scale), int(template.shape[0] * scale))
                )
                
                # Skip if template is larger than image
                if (scaled_template.shape[0] > gray.shape[0] or 
                    scaled_template.shape[1] > gray.shape[1]):
                    continue
                
                # Apply template matching
                result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                best_match = max(best_match, max_val)
            
            # Only include significant matches
            if best_match > 0.6:  # 60% similarity threshold
                detected_states[state] = best_match
        
        return detected_states
    
    def extract_text_regions(self, plate_img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Extract specific regions where state information typically appears.
        
        Returns:
            List of (region_name, region_image) tuples
        """
        height, width = plate_img.shape[:2]
        regions = []
        
        # Top region (state name often appears here)
        top_region = plate_img[0:int(height * 0.25), :]
        regions.append(('top', top_region))
        
        # Bottom region (state name, DMV URL)
        bottom_region = plate_img[int(height * 0.75):, :]
        regions.append(('bottom', bottom_region))
        
        # Left side (some states have vertical text)
        left_region = plate_img[:, 0:int(width * 0.15)]
        regions.append(('left', left_region))
        
        # Right side
        right_region = plate_img[:, int(width * 0.85):]
        regions.append(('right', right_region))
        
        # Center region (main plate area)
        center_region = plate_img[int(height * 0.25):int(height * 0.75), 
                                 int(width * 0.15):int(width * 0.85)]
        regions.append(('center', center_region))
        
        return regions


class EnhancedStateRecognizer:
    """Complete state recognition system using multiple methods."""
    
    def __init__(self, template_dir: Optional[str] = None):
        self.visual_detector = VisualFeatureDetector(template_dir)
        self.state_keywords = self._load_state_keywords()
        self.ocr_reader = None  # Initialize when needed
        
    def _load_state_keywords(self) -> Dict[str, List[str]]:
        """Load comprehensive state keywords and phrases."""
        return {
            'AL': ['ALABAMA', 'HEART OF DIXIE', 'SWEET HOME', 'STARS FELL'],
            'AK': ['ALASKA', 'LAST FRONTIER', 'NORTH TO FUTURE'],
            'AZ': ['ARIZONA', 'GRAND CANYON', 'COPPER STATE'],
            'AR': ['ARKANSAS', 'NATURAL STATE', 'DIAMOND STATE'],
            'CA': ['CALIFORNIA', 'GOLDEN STATE', 'EUREKA', 'DMV.CA.GOV'],
            'CO': ['COLORADO', 'CENTENNIAL', 'COLORFUL', 'PIONEER'],
            'CT': ['CONNECTICUT', 'CONSTITUTION', 'NUTMEG'],
            'DE': ['DELAWARE', 'FIRST STATE', 'DIAMOND STATE'],
            'FL': ['FLORIDA', 'SUNSHINE', 'SUNSHINE STATE', 'MYFLORIDA'],
            'GA': ['GEORGIA', 'PEACH STATE', 'ON MY MIND'],
            'HI': ['HAWAII', 'ALOHA STATE', 'ALOHA', 'RAINBOW'],
            'ID': ['IDAHO', 'FAMOUS POTATOES', 'SCENIC', 'GEM STATE'],
            'IL': ['ILLINOIS', 'LAND OF LINCOLN', 'LINCOLN', 'PRAIRIE'],
            'IN': ['INDIANA', 'HOOSIER', 'CROSSROADS', 'AMBER WAVES'],
            'IA': ['IOWA', 'HAWKEYE', 'FIELDS OF OPPORTUNITY'],
            'KS': ['KANSAS', 'SUNFLOWER', 'MIDWAY USA', 'AD ASTRA'],
            'KY': ['KENTUCKY', 'BLUEGRASS', 'UNBRIDLED SPIRIT'],
            'LA': ['LOUISIANA', 'SPORTSMAN', 'PELICAN', 'BAYOU'],
            'ME': ['MAINE', 'VACATIONLAND', 'LOBSTER', 'PINE TREE'],
            'MD': ['MARYLAND', 'OLD LINE', 'CHESAPEAKE', 'FREE STATE'],
            'MA': ['MASSACHUSETTS', 'SPIRIT OF AMERICA', 'BAY STATE'],
            'MI': ['MICHIGAN', 'GREAT LAKES', 'PURE MICHIGAN', 'WATER WONDERLAND'],
            'MN': ['MINNESOTA', 'NORTH STAR', '10,000 LAKES', 'LAND OF LAKES'],
            'MS': ['MISSISSIPPI', 'MAGNOLIA', 'HOSPITALITY STATE'],
            'MO': ['MISSOURI', 'SHOW ME STATE', 'GATEWAY'],
            'MT': ['MONTANA', 'TREASURE STATE', 'BIG SKY', 'MOUNTAIN'],
            'NE': ['NEBRASKA', 'CORNHUSKER', 'GOOD LIFE'],
            'NV': ['NEVADA', 'SILVER STATE', 'BATTLE BORN', 'SAGEBRUSH'],
            'NH': ['NEW HAMPSHIRE', 'LIVE FREE', 'GRANITE STATE', 'SCENIC'],
            'NJ': ['NEW JERSEY', 'GARDEN STATE', 'CROSSROADS', 'SHORE'],
            'NM': ['NEW MEXICO', 'LAND OF ENCHANTMENT', 'CHILE CAPITAL'],
            'NY': ['NEW YORK', 'EMPIRE STATE', 'EXCELSIOR', 'LIBERTY'],
            'NC': ['NORTH CAROLINA', 'FIRST IN FLIGHT', 'TAR HEEL'],
            'ND': ['NORTH DAKOTA', 'PEACE GARDEN', 'DISCOVER', 'LEGENDARY'],
            'OH': ['OHIO', 'BUCKEYE', 'BIRTHPLACE OF AVIATION', 'HEART OF IT ALL'],
            'OK': ['OKLAHOMA', 'SOONER', 'NATIVE AMERICA', 'OK'],
            'OR': ['OREGON', 'PACIFIC WONDERLAND', 'BEAVER STATE', 'EXPLORE'],
            'PA': ['PENNSYLVANIA', 'KEYSTONE', 'PURSUE YOUR HAPPINESS'],
            'RI': ['RHODE ISLAND', 'OCEAN STATE', 'DISCOVER BEAUTIFUL'],
            'SC': ['SOUTH CAROLINA', 'PALMETTO', 'SMILING FACES'],
            'SD': ['SOUTH DAKOTA', 'MOUNT RUSHMORE', 'GREAT FACES'],
            'TN': ['TENNESSEE', 'VOLUNTEER', 'MUSIC CITY', 'SOUNDS GOOD'],
            'TX': ['TEXAS', 'LONE STAR', 'DRIVE FRIENDLY'],
            'UT': ['UTAH', 'BEEHIVE', 'GREATEST SNOW', 'LIFE ELEVATED'],
            'VT': ['VERMONT', 'GREEN MOUNTAIN', 'FREEDOM AND UNITY'],
            'VA': ['VIRGINIA', 'OLD DOMINION', 'MOTHER OF PRESIDENTS', 'VIRGINIA IS FOR LOVERS'],
            'WA': ['WASHINGTON', 'EVERGREEN', 'SALMON', 'ENDLESS POSSIBILITIES'],
            'WV': ['WEST VIRGINIA', 'MOUNTAIN STATE', 'WILD AND WONDERFUL'],
            'WI': ['WISCONSIN', 'BADGER', 'DAIRY', 'AMERICA\'S DAIRYLAND'],
            'WY': ['WYOMING', 'EQUALITY STATE', 'COWBOY STATE', 'FOREVER WEST'],
            'DC': ['WASHINGTON DC', 'DISTRICT OF COLUMBIA', 'TAXATION WITHOUT']
        }
    
    def recognize_state(self, 
                       plate_img: np.ndarray,
                       plate_text: str,
                       ocr_texts: List[str],
                       use_ocr_on_regions: bool = True) -> Dict[str, Any]:
        """
        Comprehensive state recognition using all available methods.
        
        Args:
            plate_img: The license plate image (BGR)
            plate_text: The main plate number
            ocr_texts: All OCR text attempts from the plate
            use_ocr_on_regions: Whether to run OCR on specific regions
            
        Returns:
            Complete state recognition results
        """
        results = {
            'state_code': None,
            'state_name': None,
            'confidence': 0.0,
            'method': 'unknown',
            'details': {
                'pattern_match': {'state': None, 'confidence': 0.0},
                'color_match': {'states': {}, 'dominant_colors': []},
                'keyword_match': {'state': None, 'confidence': 0.0, 'keyword': None},
                'logo_match': {'states': {}},
                'region_text': []
            }
        }
        
        # 1. Pattern-based recognition
        pattern_state, pattern_conf = self._pattern_recognition(plate_text)
        results['details']['pattern_match'] = {
            'state': pattern_state,
            'confidence': pattern_conf
        }
        
        # 2. Color analysis
        color_result = self.visual_detector.analyze_colors(plate_img)
        results['details']['color_match'] = {
            'states': color_result.matched_states,
            'dominant_colors': color_result.dominant_colors,
            'background': color_result.background_color,
            'text_color': color_result.text_color
        }
        
        # 3. Keyword detection from OCR texts
        keyword_state, keyword_conf, keyword = self._keyword_detection(ocr_texts)
        results['details']['keyword_match'] = {
            'state': keyword_state,
            'confidence': keyword_conf,
            'keyword': keyword
        }
        
        # 4. Logo/symbol detection
        logo_matches = self.visual_detector.detect_state_logos(plate_img)
        results['details']['logo_match']['states'] = logo_matches
        
        # 5. Region-specific OCR (if requested)
        if use_ocr_on_regions:
            region_texts = self._ocr_regions(plate_img)
            results['details']['region_text'] = region_texts
            
            # Check regions for state keywords
            region_state, region_conf, region_keyword = self._keyword_detection(
                [text for _, text in region_texts]
            )
            if region_conf > keyword_conf:
                results['details']['keyword_match'] = {
                    'state': region_state,
                    'confidence': region_conf,
                    'keyword': region_keyword
                }
                keyword_state = region_state
                keyword_conf = region_conf
        
        # 6. Combine all evidence
        final_state, final_conf, method = self._combine_evidence(
            pattern_state, pattern_conf,
            color_result.matched_states,
            keyword_state, keyword_conf,
            logo_matches
        )
        
        results['state_code'] = final_state
        results['confidence'] = final_conf
        results['method'] = method
        
        # Get state name
        if final_state:
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
            results['state_name'] = state_names.get(final_state, 'Unknown')
        
        return results
    
    def _pattern_recognition(self, plate_text: str) -> Tuple[Optional[str], float]:
        """Pattern-based state recognition."""
        try:
            from state_patterns import StatePatternMatcher
            matcher = StatePatternMatcher()
            return matcher.extract_state_from_text(plate_text)
        except ImportError:
            # Fallback to using the existing state_model
            try:
                from state_model import get_state_classifier
                classifier = get_state_classifier()
                return classifier.classify_from_text(plate_text)
            except ImportError:
                # If all else fails, return no match
                logger.warning("No state pattern recognition available")
                return None, 0.0
    
    def _keyword_detection(self, texts: List[str]) -> Tuple[Optional[str], float, Optional[str]]:
        """Detect state from keywords in text."""
        all_text = ' '.join(texts).upper()
        
        best_match = None
        best_conf = 0.0
        best_keyword = None
        
        for state, keywords in self.state_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    # Higher confidence for longer, more specific keywords
                    confidence = min(0.9 + (len(keyword) / 100), 1.0)
                    
                    if confidence > best_conf:
                        best_match = state
                        best_conf = confidence
                        best_keyword = keyword
        
        return best_match, best_conf, best_keyword
    
    def _ocr_regions(self, plate_img: np.ndarray) -> List[Tuple[str, str]]:
        """Run OCR on specific regions of the plate."""
        if self.ocr_reader is None:
            try:
                import easyocr
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            except ImportError:
                logger.warning("EasyOCR not available for region analysis")
                return []
        
        regions = self.visual_detector.extract_text_regions(plate_img)
        region_texts = []
        
        for region_name, region_img in regions:
            if region_img.size == 0:
                continue
            
            try:
                # Run OCR on region
                results = self.ocr_reader.readtext(region_img)
                
                # Combine all text from region
                region_text = ' '.join([text for _, text, _ in results])
                if region_text.strip():
                    region_texts.append((region_name, region_text))
                    
            except Exception as e:
                logger.error(f"OCR failed on region {region_name}: {e}")
        
        return region_texts
    
    def _combine_evidence(self, 
                         pattern_state: Optional[str], pattern_conf: float,
                         color_matches: Dict[str, float],
                         keyword_state: Optional[str], keyword_conf: float,
                         logo_matches: Dict[str, float]) -> Tuple[Optional[str], float, str]:
        """
        Combine evidence from all methods to determine final state.
        
        Returns:
            (state_code, confidence, primary_method)
        """
        # Score each state based on all evidence
        state_scores = {}
        
        # Add pattern evidence (weight: 40%)
        if pattern_state:
            state_scores[pattern_state] = state_scores.get(pattern_state, 0) + pattern_conf * 0.4
        
        # Add color evidence (weight: 25%)
        for state, conf in color_matches.items():
            state_scores[state] = state_scores.get(state, 0) + conf * 0.25
        
        # Add keyword evidence (weight: 25%)
        if keyword_state:
            state_scores[keyword_state] = state_scores.get(keyword_state, 0) + keyword_conf * 0.25
        
        # Add logo evidence (weight: 10%)
        for state, conf in logo_matches.items():
            state_scores[state] = state_scores.get(state, 0) + conf * 0.1
        
        if not state_scores:
            return None, 0.0, 'none'
        
        # Get best state
        best_state = max(state_scores.items(), key=lambda x: x[1])
        
        # Determine primary method
        method = 'ensemble'
        if pattern_state == best_state[0] and pattern_conf > 0.7:
            method = 'pattern'
        elif keyword_state == best_state[0] and keyword_conf > 0.8:
            method = 'keyword'
        elif best_state[0] in color_matches and color_matches[best_state[0]] > 0.6:
            method = 'color'
        elif best_state[0] in logo_matches and logo_matches[best_state[0]] > 0.7:
            method = 'logo'
        
        return best_state[0], min(best_state[1], 1.0), method


# Helper function for easy integration
def recognize_state_comprehensive(plate_img: np.ndarray, 
                                plate_text: str,
                                ocr_texts: List[str],
                                template_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Easy-to-use function for comprehensive state recognition.
    
    Args:
        plate_img: License plate image (BGR)
        plate_text: Main plate number text
        ocr_texts: All OCR attempts
        template_dir: Directory containing state logo templates
        
    Returns:
        Complete recognition results
    """
    recognizer = EnhancedStateRecognizer(template_dir)
    return recognizer.recognize_state(plate_img, plate_text, ocr_texts)


if __name__ == "__main__":
    # Test the system
    print("Enhanced State Recognition System Ready")
    print(f"scikit-learn available: {SKLEARN_AVAILABLE}")
    
    # Example usage
    test_img = np.zeros((100, 300, 3), dtype=np.uint8)  # Dummy image
    test_plate = "ABC1234"
    test_texts = ["CALIFORNIA", "ABC 1234", "CA"]
    
    results = recognize_state_comprehensive(test_img, test_plate, test_texts)
    print(f"\nTest Results: {results}")