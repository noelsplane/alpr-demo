# baseline_metrics.py
"""
Establish baseline metrics for state recognition patterns.
This will help track improvements as you enhance the system.
"""

import json
import sqlite3
from datetime import datetime
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple
import os

class StateRecognitionMetrics:
    def __init__(self, db_path: str = "detections.db"):
        self.db_path = db_path
        self.metrics = {
            'pattern_based': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'confidence_distribution': [],
            'state_distribution': defaultdict(int),
            'method_effectiveness': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'common_errors': defaultdict(int),
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_baseline_metrics(self):
        """Calculate baseline metrics from existing data."""
        if not os.path.exists(self.db_path):
            print(f"Database {self.db_path} not found!")
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all detections
        query = """
        SELECT plate_text, confidence, state, state_confidence, image_name
        FROM detections
        WHERE plate_text IS NOT NULL
        """
        
        cursor.execute(query)
        detections = cursor.fetchall()
        
        print(f"Analyzing {len(detections)} detections...")
        
        for plate_text, conf, state, state_conf, image_name in detections:
            # Track state distribution
            if state and state != 'null':
                self.metrics['state_distribution'][state] += 1
            
            # Track confidence distribution
            if state_conf:
                self.metrics['confidence_distribution'].append(state_conf)
            
            # Analyze pattern matching effectiveness
            self._analyze_pattern_effectiveness(plate_text, state)
        
        conn.close()
        
        # Calculate summary statistics
        self._calculate_summary_stats()
        
        # Save metrics
        self._save_metrics()
        
        # Print report
        self._print_report()
    
    def _analyze_pattern_effectiveness(self, plate_text: str, detected_state: str):
        """Analyze how well patterns match states."""
        try:
            from state_patterns import StatePatternMatcher
            
            matcher = StatePatternMatcher()
            predicted_state, confidence = matcher.extract_state_from_text(plate_text)
            
            if predicted_state:
                self.metrics['pattern_based'][predicted_state]['total'] += 1
                if predicted_state == detected_state:
                    self.metrics['pattern_based'][predicted_state]['correct'] += 1
        except Exception as e:
            print(f"Pattern matching error: {e}")
    
    def _calculate_summary_stats(self):
        """Calculate summary statistics."""
        # Overall accuracy
        total_with_state = sum(self.metrics['state_distribution'].values())
        
        # Confidence statistics
        if self.metrics['confidence_distribution']:
            conf_array = np.array(self.metrics['confidence_distribution'])
            self.metrics['confidence_stats'] = {
                'mean': float(np.mean(conf_array)),
                'median': float(np.median(conf_array)),
                'std': float(np.std(conf_array)),
                'min': float(np.min(conf_array)),
                'max': float(np.max(conf_array))
            }
        
        # Pattern accuracy by state
        pattern_accuracy = {}
        for state, data in self.metrics['pattern_based'].items():
            if data['total'] > 0:
                accuracy = data['correct'] / data['total']
                pattern_accuracy[state] = {
                    'accuracy': accuracy,
                    'total': data['total'],
                    'correct': data['correct']
                }
        
        self.metrics['pattern_accuracy'] = pattern_accuracy
        
        # Overall metrics
        total_pattern_correct = sum(d['correct'] for d in self.metrics['pattern_based'].values())
        total_pattern_attempts = sum(d['total'] for d in self.metrics['pattern_based'].values())
        
        self.metrics['overall'] = {
            'total_detections': total_with_state,
            'pattern_accuracy': total_pattern_correct / total_pattern_attempts if total_pattern_attempts > 0 else 0,
            'states_covered': len(self.metrics['state_distribution']),
            'avg_confidence': self.metrics['confidence_stats']['mean'] if 'confidence_stats' in self.metrics else 0
        }
    
    def _save_metrics(self):
        """Save metrics to JSON file."""
        filename = f"baseline_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert defaultdicts to regular dicts for JSON serialization
        save_data = {
            'timestamp': self.metrics['timestamp'],
            'overall': self.metrics['overall'],
            'state_distribution': dict(self.metrics['state_distribution']),
            'confidence_stats': self.metrics.get('confidence_stats', {}),
            'pattern_accuracy': self.metrics.get('pattern_accuracy', {})
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\nMetrics saved to: {filename}")
    
    def _print_report(self):
        """Print a formatted metrics report."""
        print("\n" + "="*60)
        print("STATE RECOGNITION BASELINE METRICS REPORT")
        print("="*60)
        
        print(f"\nReport Generated: {self.metrics['timestamp']}")
        
        # Overall metrics
        print("\n### OVERALL METRICS ###")
        overall = self.metrics['overall']
        print(f"Total Detections with States: {overall['total_detections']}")
        print(f"Pattern Recognition Accuracy: {overall['pattern_accuracy']:.2%}")
        print(f"Number of States Detected: {overall['states_covered']}")
        print(f"Average Confidence Score: {overall['avg_confidence']:.2%}")
        
        # State distribution
        print("\n### STATE DISTRIBUTION ###")
        print(f"{'State':<10} {'Count':<10} {'Percentage':<10}")
        print("-" * 30)
        total = sum(self.metrics['state_distribution'].values())
        for state, count in sorted(self.metrics['state_distribution'].items(), 
                                  key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{state:<10} {count:<10} {percentage:<10.1f}%")
        
        # Confidence statistics
        if 'confidence_stats' in self.metrics:
            print("\n### CONFIDENCE STATISTICS ###")
            stats = self.metrics['confidence_stats']
            print(f"Mean Confidence: {stats['mean']:.2%}")
            print(f"Median Confidence: {stats['median']:.2%}")
            print(f"Std Deviation: {stats['std']:.2%}")
            print(f"Min Confidence: {stats['min']:.2%}")
            print(f"Max Confidence: {stats['max']:.2%}")
        
        # Pattern accuracy by state
        print("\n### PATTERN RECOGNITION ACCURACY BY STATE ###")
        print(f"{'State':<10} {'Accuracy':<12} {'Correct':<10} {'Total':<10}")
        print("-" * 45)
        
        if 'pattern_accuracy' in self.metrics:
            for state, data in sorted(self.metrics['pattern_accuracy'].items(), 
                                    key=lambda x: x[1]['accuracy'], reverse=True)[:10]:
                print(f"{state:<10} {data['accuracy']:<12.2%} {data['correct']:<10} {data['total']:<10}")
        
        print("\n" + "="*60)
        
        # Recommendations
        print("\n### RECOMMENDATIONS ###")
        if overall['pattern_accuracy'] < 0.7:
            print("⚠️  Pattern recognition accuracy is below 70%")
            print("   - Consider collecting more training data")
            print("   - Review and update state patterns")
            print("   - Implement machine learning model")
        
        if overall['avg_confidence'] < 0.6:
            print("⚠️  Average confidence is below 60%")
            print("   - Enhance OCR preprocessing")
            print("   - Add visual feature detection")
            print("   - Implement ensemble methods")
        
        low_coverage_states = [s for s, c in self.metrics['state_distribution'].items() if c < 5]
        if low_coverage_states:
            print(f"⚠️  Low data for states: {', '.join(low_coverage_states[:5])}")
            print("   - Collect more samples from these states")

def compare_metrics(baseline_file: str, current_file: str):
    """Compare baseline metrics with current metrics."""
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    
    with open(current_file, 'r') as f:
        current = json.load(f)
    
    print("\n" + "="*60)
    print("METRICS COMPARISON")
    print("="*60)
    
    # Compare overall metrics
    print("\n### OVERALL METRICS COMPARISON ###")
    print(f"{'Metric':<30} {'Baseline':<15} {'Current':<15} {'Change':<15}")
    print("-" * 75)
    
    metrics_to_compare = [
        ('Pattern Accuracy', 'pattern_accuracy', lambda x: f"{x:.2%}"),
        ('Average Confidence', 'avg_confidence', lambda x: f"{x:.2%}"),
        ('States Covered', 'states_covered', lambda x: str(x)),
        ('Total Detections', 'total_detections', lambda x: str(x))
    ]
    
    for name, key, formatter in metrics_to_compare:
        baseline_val = baseline['overall'].get(key, 0)
        current_val = current['overall'].get(key, 0)
        
        if isinstance(baseline_val, (int, float)) and isinstance(current_val, (int, float)):
            change = current_val - baseline_val
            change_str = f"+{change:.2%}" if key.endswith('accuracy') or key.endswith('confidence') else f"+{change}"
            if change < 0:
                change_str = change_str.replace('+', '')
        else:
            change_str = "N/A"
        
        print(f"{name:<30} {formatter(baseline_val):<15} {formatter(current_val):<15} {change_str:<15}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Calculate baseline metrics
    print("Calculating baseline metrics for state recognition...")
    metrics = StateRecognitionMetrics()
    metrics.calculate_baseline_metrics()
    
    # If you have a previous baseline, you can compare them
    # Example of how to use the comparison function:
    # compare_metrics("baseline_metrics_20240101_120000.json", "baseline_metrics_20240115_120000.json")
    
    print("\nTo compare with future metrics, run:")
    print("python baseline_metrics.py")
    print("Then use: compare_metrics('old_metrics.json', 'new_metrics.json')")