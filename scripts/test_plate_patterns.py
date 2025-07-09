"""
test_plate_patterns.py
Test if plate patterns match correctly
"""

import re

def test_plate_pattern(plate_text):
    """Test what state patterns match a given plate."""
    
    # Clean the plate text
    plate_clean = plate_text.upper().strip().replace(' ', '').replace('-', '')
    
    print(f"\nTesting plate: '{plate_text}' -> '{plate_clean}'")
    print(f"Length: {len(plate_clean)}")
    print(f"Format: ", end="")
    for c in plate_clean:
        if c.isalpha():
            print("L", end="")
        elif c.isdigit():
            print("N", end="")
        else:
            print("?", end="")
    print("\n")
    
    # Define state patterns (subset for testing)
    state_patterns = {
        'CA': [
            r'^[1-9][A-Z]{3}\d{3}$',  # 1ABC234
            r'^\d[A-Z]{3}\d{3}$',     # 7ABC234
        ],
        'MA': [
            r'^\d{3}[A-Z]{3}$',       # 123ABC
            r'^[A-Z]{3}\d{3}$',       # ABC123
            r'^\d{3}[A-Z]{2}\d$',     # 284FH8 - Massachusetts pattern
            r'^\d[A-Z]{2}\d{3}$',     # 1AB234
        ],
        'TX': [
            r'^[A-Z]{3}\d{4}$',       # ABC1234
            r'^[A-Z]{2}\d{5}$',       # AB12345
        ],
        'NY': [
            r'^[A-Z]{3}\d{4}$',       # ABC1234
            r'^[A-Z]{3}\d{3}[A-Z]$',  # ABC123A
        ],
        'FL': [
            r'^[A-Z]{4}\d{2}$',       # ABCD12
            r'^[A-Z]\d{2}[A-Z]{3}$',  # A12BCD
        ],
        'NJ': [
            r'^[A-Z]\d{2}[A-Z]{3}$',  # A12BCD
        ],
        'IL': [
            r'^[A-Z]\d{5}$',          # A12345
            r'^[A-Z]{2}\d{5}$',       # AB12345
        ],
        'PA': [
            r'^[A-Z]{3}\d{4}$',       # ABC1234
        ],
        'OH': [
            r'^[A-Z]{3}\d{4}$',       # ABC1234
            r'^[A-Z]{3}\d{3}[A-Z]$',  # ABC123A
        ],
        'MI': [
            r'^[A-Z]{3}\d{4}$',       # ABC1234
            r'^[A-Z]{3}\d{3}[A-Z]$',  # ABC123A
            r'^\d[A-Z]{2}\d{3}$',     # 1AB234
        ],
    }
    
    matches = []
    
    for state, patterns in state_patterns.items():
        for pattern in patterns:
            if re.match(pattern, plate_clean):
                matches.append((state, pattern))
                print(f"✓ Matches {state}: {pattern}")
    
    if not matches:
        print("✗ No state patterns matched")
        
        # Suggest possible patterns
        print("\nThis plate might need pattern:")
        format_str = ""
        for c in plate_clean:
            if c.isalpha():
                format_str += "[A-Z]"
            elif c.isdigit():
                format_str += r"\d"
        print(f"  r'^{format_str}$'")
    
    return matches


if __name__ == "__main__":
    # Test the Massachusetts plate from your image
    test_plates = [
        "284FH8",      # Your MA plate
        "1ABC234",     # CA format
        "ABC1234",     # Common format (TX, NY, PA, etc.)
        "A12BCD",      # NJ format
        "123ABC",      # Some states
        "ABCD12",      # FL format
    ]
    
    for plate in test_plates:
        matches = test_plate_pattern(plate)