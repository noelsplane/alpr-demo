# generate_state_templates.py
"""
Generate or prepare template images for state logo detection.
This creates placeholder templates that can be replaced with actual state logos.
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, Tuple

class StateTemplateGenerator:
    """Generate template images for state recognition."""
    
    def __init__(self, output_dir: str = "plate_templates"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define state-specific visual elements
        self.state_visuals = {
            'CA': {'text': 'California', 'symbol': 'bear'},
            'TX': {'text': 'TEXAS', 'symbol': 'star'},
            'NY': {'text': 'NEW YORK', 'symbol': 'statue'},
            'FL': {'text': 'FLORIDA', 'symbol': 'orange'},
            'IL': {'text': 'Land of Lincoln', 'symbol': 'lincoln'},
            'PA': {'text': 'PENNSYLVANIA', 'symbol': 'keystone'},
            'OH': {'text': 'OHIO', 'symbol': 'circle'},
            'GA': {'text': 'GEORGIA', 'symbol': 'peach'},
            'NC': {'text': 'North Carolina', 'symbol': 'plane'},
            'MI': {'text': 'MICHIGAN', 'symbol': 'lakes'},
            'NJ': {'text': 'Garden State', 'symbol': 'garden'},
            'VA': {'text': 'Virginia', 'symbol': 'lovers'},
            'WA': {'text': 'Washington', 'symbol': 'evergreen'},
            'AZ': {'text': 'ARIZONA', 'symbol': 'cactus'},
            'MA': {'text': 'Massachusetts', 'symbol': 'cod'},
            'TN': {'text': 'Tennessee', 'symbol': 'music'},
            'IN': {'text': 'INDIANA', 'symbol': 'race'},
            'MO': {'text': 'Show-Me State', 'symbol': 'arch'},
            'MD': {'text': 'Maryland', 'symbol': 'flag'},
            'WI': {'text': 'Wisconsin', 'symbol': 'cheese'},
            'CO': {'text': 'COLORADO', 'symbol': 'mountain'},
            'MN': {'text': 'Minnesota', 'symbol': 'lakes'},
            'SC': {'text': 'South Carolina', 'symbol': 'palmetto'},
            'AL': {'text': 'Alabama', 'symbol': 'stars'},
            'LA': {'text': 'Louisiana', 'symbol': 'pelican'},
            'KY': {'text': 'Kentucky', 'symbol': 'horse'},
            'OR': {'text': 'Oregon', 'symbol': 'tree'},
            'OK': {'text': 'Oklahoma', 'symbol': 'native'},
            'CT': {'text': 'Connecticut', 'symbol': 'charter'},
            'UT': {'text': 'Utah', 'symbol': 'arch'},
            'IA': {'text': 'IOWA', 'symbol': 'corn'},
            'NV': {'text': 'Nevada', 'symbol': 'silver'},
            'AR': {'text': 'Arkansas', 'symbol': 'diamond'},
            'MS': {'text': 'Mississippi', 'symbol': 'magnolia'},
            'KS': {'text': 'Kansas', 'symbol': 'sunflower'},
            'NM': {'text': 'New Mexico', 'symbol': 'zia'},
            'NE': {'text': 'Nebraska', 'symbol': 'corn'},
            'WV': {'text': 'West Virginia', 'symbol': 'mountain'},
            'ID': {'text': 'Idaho', 'symbol': 'potato'},
            'HI': {'text': 'Hawaii', 'symbol': 'rainbow'},
            'NH': {'text': 'New Hampshire', 'symbol': 'granite'},
            'ME': {'text': 'Maine', 'symbol': 'lobster'},
            'RI': {'text': 'Rhode Island', 'symbol': 'anchor'},
            'MT': {'text': 'Montana', 'symbol': 'mountain'},
            'DE': {'text': 'Delaware', 'symbol': 'first'},
            'SD': {'text': 'South Dakota', 'symbol': 'rushmore'},
            'ND': {'text': 'North Dakota', 'symbol': 'buffalo'},
            'AK': {'text': 'Alaska', 'symbol': 'bear'},
            'VT': {'text': 'Vermont', 'symbol': 'maple'},
            'WY': {'text': 'Wyoming', 'symbol': 'cowboy'},
        }
    
    def generate_text_template(self, state_code: str, text: str, 
                             size: Tuple[int, int] = (200, 50)) -> np.ndarray:
        """Generate a simple text template for a state."""
        # Create blank image
        template = np.ones((size[1], size[0]), dtype=np.uint8) * 255
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Get text size
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Center text
        x = (size[0] - text_size[0]) // 2
        y = (size[1] + text_size[1]) // 2
        
        # Draw text
        cv2.putText(template, text, (x, y), font, font_scale, 0, thickness)
        
        return template
    
    def generate_symbol_template(self, state_code: str, symbol_type: str,
                               size: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """Generate a simple symbol template for a state."""
        template = np.ones((size[1], size[0]), dtype=np.uint8) * 255
        
        # Draw different symbols based on type
        center = (size[0] // 2, size[1] // 2)
        
        if symbol_type == 'star':
            # Draw a star
            pts = self._get_star_points(center, 30)
            cv2.fillPoly(template, [pts], 0)
            
        elif symbol_type == 'circle':
            # Draw a circle
            cv2.circle(template, center, 30, 0, -1)
            
        elif symbol_type == 'mountain':
            # Draw mountain shape
            pts = np.array([[20, 80], [50, 20], [80, 80]], np.int32)
            cv2.fillPoly(template, [pts], 0)
            
        elif symbol_type == 'tree':
            # Draw simple tree
            cv2.rectangle(template, (45, 60), (55, 80), 0, -1)  # trunk
            cv2.circle(template, (50, 40), 20, 0, -1)  # foliage
            
        elif symbol_type == 'palmetto':
            # Draw palm tree shape
            cv2.rectangle(template, (48, 50), (52, 80), 0, -1)  # trunk
            # Palm fronds
            for angle in range(0, 360, 45):
                x = int(50 + 25 * np.cos(np.radians(angle)))
                y = int(40 + 15 * np.sin(np.radians(angle)))
                cv2.line(template, (50, 40), (x, y), 0, 2)
                
        else:
            # Default: draw state code
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(template, state_code, (20, 60), font, 1, 0, 2)
        
        return template
    
    def _get_star_points(self, center: Tuple[int, int], size: int) -> np.ndarray:
        """Get points for a 5-pointed star."""
        pts = []
        for i in range(10):
            angle = i * np.pi / 5
            if i % 2 == 0:
                r = size
            else:
                r = size * 0.5
            x = int(center[0] + r * np.cos(angle - np.pi / 2))
            y = int(center[1] + r * np.sin(angle - np.pi / 2))
            pts.append([x, y])
        return np.array(pts, np.int32)
    
    def generate_all_templates(self):
        """Generate templates for all states."""
        print("Generating state templates...")
        
        metadata = {}
        
        for state_code, visual_info in self.state_visuals.items():
            # Generate text template
            text_template = self.generate_text_template(
                state_code, visual_info['text']
            )
            text_path = self.output_dir / f"{state_code}_text.png"
            cv2.imwrite(str(text_path), text_template)
            
            # Generate symbol template
            symbol_template = self.generate_symbol_template(
                state_code, visual_info['symbol']
            )
            symbol_path = self.output_dir / f"{state_code}_symbol.png"
            cv2.imwrite(str(symbol_path), symbol_template)
            
            # Combine for main template
            combined = np.ones((100, 300), dtype=np.uint8) * 255
            combined[25:75, 10:110] = cv2.resize(symbol_template, (100, 50))
            combined[25:75, 120:290] = cv2.resize(text_template, (170, 50))
            
            main_path = self.output_dir / f"{state_code}.png"
            cv2.imwrite(str(main_path), combined)
            
            metadata[state_code] = {
                'text_template': str(text_path.name),
                'symbol_template': str(symbol_path.name),
                'main_template': str(main_path.name),
                'text': visual_info['text'],
                'symbol_type': visual_info['symbol']
            }
            
            print(f"  Generated templates for {state_code}")
        
        # Save metadata
        metadata_path = self.output_dir / "template_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nGenerated {len(metadata)} state templates in {self.output_dir}")
        print(f"Metadata saved to {metadata_path}")
        
        return metadata
    
    def create_from_real_plates(self, plate_images_dir: str):
        """
        Extract templates from real license plate images.
        This is more effective than synthetic templates.
        """
        plate_dir = Path(plate_images_dir)
        if not plate_dir.exists():
            print(f"Plate images directory {plate_dir} not found")
            return
        
        print("\nExtracting templates from real plates...")
        
        # Group images by state
        state_images = {}
        for img_path in plate_dir.glob("*.jpg"):
            # Assume filename format: STATE_platetext_confidence_index.jpg
            parts = img_path.stem.split('_')
            if len(parts) >= 2:
                state = parts[0]
                if state not in state_images:
                    state_images[state] = []
                state_images[state].append(img_path)
        
        # Extract templates for each state
        for state, images in state_images.items():
            if state == 'UNKNOWN':
                continue
                
            print(f"\nProcessing {len(images)} images for {state}")
            
            # Use first few images to extract common elements
            templates = []
            for img_path in images[:5]:  # Use up to 5 images
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Extract top and bottom regions (where state text usually is)
                h, w = img.shape
                top_region = img[0:int(h*0.3), :]
                bottom_region = img[int(h*0.7):, :]
                
                # Save regions as potential templates
                top_path = self.output_dir / f"{state}_real_top_{len(templates)}.png"
                bottom_path = self.output_dir / f"{state}_real_bottom_{len(templates)}.png"
                
                cv2.imwrite(str(top_path), top_region)
                cv2.imwrite(str(bottom_path), bottom_region)
                
                templates.append({
                    'source': img_path.name,
                    'top': top_path.name,
                    'bottom': bottom_path.name
                })
            
            print(f"  Extracted {len(templates)} template sets for {state}")


def download_real_templates():
    """
    Instructions for getting real state logo templates.
    """
    instructions = """
    To get real state logo templates for better accuracy:
    
    1. Search for "US license plate state logos" or visit:
       - State DMV websites
       - Wikipedia pages for each state's license plates
       - License plate collector forums
    
    2. For each state, save small images of:
       - State name text as it appears on plates
       - State symbols/logos (flags, landmarks, etc.)
       - Distinctive graphics or patterns
    
    3. Save images as: STATE_CODE.png (e.g., CA.png, NY.png)
       - Keep images small (100-200px wide)
       - Convert to grayscale
       - High contrast is better
    
    4. Place in the plate_templates directory
    
    Note: Ensure you have rights to use any images you download.
    """
    print(instructions)


if __name__ == "__main__":
    # Generate synthetic templates
    generator = StateTemplateGenerator()
    generator.generate_all_templates()
    
    # If you have real plate images, extract templates from them
    # generator.create_from_real_plates("plate_dataset")
    
    # Show instructions for getting real templates
    print("\n" + "="*60)
    download_real_templates()