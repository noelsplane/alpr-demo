#!/usr/bin/env python3
"""
Test script for state name filtering functionality.
"""

import sys
import os
sys.path.append('api')

from state_filter import StateNameFilter, clean_plate_text, is_valid_plate_text

def test_state_filtering():
    """Test the state filtering functionality."""
    print("🧪 Testing State Name Filtering System")
    print("=" * 50)
    
    # Test cases: (text, should_be_filtered, description)
    test_cases = [
        # State names that should be filtered
        ("CALIFORNIA", True, "Full state name"),
        ("CA", True, "State abbreviation"),
        ("TEXAS", True, "Another state name"),
        ("NEW YORK", True, "Multi-word state"),
        ("FLORIDA", True, "State name"),
        
        # Common plate words that should be filtered
        ("STATE", True, "Common plate word"),
        ("LICENSE", True, "License word"),
        ("EXPIRES", True, "Expiration text"),
        ("REGISTRATION", True, "Registration text"),
        
        # Valid license plates that should NOT be filtered
        ("ABC123", False, "Valid alphanumeric plate"),
        ("1ABC234", False, "Valid plate with numbers first"),
        ("XYZ987", False, "Valid 6-character plate"),
        ("AB12CD", False, "Valid mixed format"),
        ("123ABC", False, "Numbers then letters"),
        
        # Invalid/questionable text that should be filtered
        ("A", True, "Too short"),
        ("AB", True, "Too short"),
        ("ABCDEFGHI", True, "Too long"),
        ("CALIFORNIA LICENSE", True, "Contains state name"),
        ("TX PLATE", True, "Contains state abbreviation"),
        
        # Edge cases
        ("", True, "Empty string"),
        ("   ", True, "Whitespace only"),
        ("ABC-123", False, "Valid plate with hyphen (cleaned)"),
        ("ABC 123", False, "Valid plate with space (cleaned)"),
    ]
    
    filter_system = StateNameFilter()
    passed = 0
    failed = 0
    
    print(f"Testing {len(test_cases)} cases...")
    print()
    
    for i, (text, should_filter, description) in enumerate(test_cases, 1):
        # Test the filtering
        is_filtered = filter_system.should_filter_detection(text)
        cleaned = clean_plate_text(text)
        is_valid = is_valid_plate_text(text)
        
        # Determine if test passed
        test_passed = (is_filtered == should_filter)
        
        if test_passed:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        
        print(f"{i:2d}. {status} | '{text}' -> '{cleaned}' | {description}")
        print(f"    Expected filtered: {should_filter}, Got: {is_filtered}, Valid: {is_valid}")
        
        if not test_passed:
            print(f"    ⚠️  MISMATCH: Expected {'FILTERED' if should_filter else 'ALLOWED'}, "
                  f"but got {'FILTERED' if is_filtered else 'ALLOWED'}")
        print()
    
    # Summary
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    print(f"Success rate: {(passed / len(test_cases)) * 100:.1f}%")
    
    # Filter statistics
    stats = filter_system.get_filter_stats()
    print(f"\n📈 Filter Statistics:")
    print(f"   Total filtered words: {stats['total_filtered_words']}")
    print(f"   US states: {stats['us_states_count']}")
    print(f"   Common words: {stats['common_words_count']}")
    
    return failed == 0

def test_detection_list_filtering():
    """Test filtering a list of detections."""
    print("\n🔬 Testing Detection List Filtering")
    print("=" * 50)
    
    # Sample detections (simulating OCR results)
    sample_detections = [
        {"text": "ABC123", "confidence": 0.9},
        {"text": "CALIFORNIA", "confidence": 0.8},
        {"text": "XYZ789", "confidence": 0.85},
        {"text": "STATE", "confidence": 0.7},
        {"text": "TX", "confidence": 0.6},
        {"text": "123ABC", "confidence": 0.95},
        {"text": "LICENSE", "confidence": 0.75},
        {"text": "DEF456", "confidence": 0.88},
        {"text": "FLORIDA", "confidence": 0.82},
        {"text": "GHI789", "confidence": 0.91},
    ]
    
    filter_system = StateNameFilter()
    
    print(f"Original detections: {len(sample_detections)}")
    for det in sample_detections:
        print(f"  - '{det['text']}' (conf: {det['confidence']})")
    
    # Filter the detections
    filtered_detections = filter_system.filter_detections(sample_detections)
    
    print(f"\nFiltered detections: {len(filtered_detections)}")
    for det in filtered_detections:
        print(f"  - '{det['text']}' (conf: {det['confidence']})")
    
    filtered_count = len(sample_detections) - len(filtered_detections)
    print(f"\n📊 Filtered out: {filtered_count} detections")
    print(f"Retention rate: {(len(filtered_detections) / len(sample_detections)) * 100:.1f}%")

if __name__ == "__main__":
    print("🚀 ALPR State Name Filter Test Suite")
    print("Testing automatic filtering of state names from plate detections")
    print()
    
    try:
        # Run basic filtering tests
        success = test_state_filtering()
        
        # Run detection list filtering tests
        test_detection_list_filtering()
        
        print("\n" + "=" * 50)
        if success:
            print("🎉 All tests passed! State filtering is working correctly.")
            sys.exit(0)
        else:
            print("⚠️  Some tests failed. Check the filtering logic.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)