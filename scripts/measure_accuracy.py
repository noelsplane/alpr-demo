"""
measure_accuracy.py
Tool to measure state recognition accuracy and identify problem areas
"""

import os
import json
import cv2
import pandas as pd
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from unified_state_recognition import UnifiedStateRecognizer
from plate_filter_utils import extract_plate_number
import easyocr

class AccuracyMeasurement:
    def __init__(self, test_data_dir="test_data"):
        self.test_data_dir = test_data_dir
        self.recognizer = UnifiedStateRecognizer()
        self.ocr = easyocr.Reader(['en'], gpu=False)
        self.results = defaultdict(list)
        
    def create_test_structure(self):
        """Create directory structure for test data."""
        states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                  'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                  'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                  'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                  'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC']
        
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        for state in states:
            state_dir = os.path.join(self.test_data_dir, state)
            os.makedirs(state_dir, exist_ok=True)
            
        print(f"Created test directory structure in {self.test_data_dir}/")
        print("Place your test images in the appropriate state folders")
        
    def test_single_image(self, image_path, true_state):
        """Test a single image and return results."""
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run OCR
        ocr_results = self.ocr.readtext(img_rgb)
        
        # Extract plate text
        plate_text, ocr_conf = extract_plate_number(ocr_results)
        
        if not plate_text:
            return {
                'true_state': true_state,
                'predicted_state': None,
                'confidence': 0.0,
                'plate_text': None,
                'correct': False,
                'error_type': 'no_plate_detected'
            }
        
        # Get state prediction
        state_result = self.recognizer.recognize_state(
            plate_text=plate_text,
            ocr_results=ocr_results,
            plate_image=img
        )
        
        predicted_state = state_result.get('state_code')
        confidence = state_result.get('confidence', 0.0)
        
        return {
            'true_state': true_state,
            'predicted_state': predicted_state,
            'confidence': confidence,
            'plate_text': plate_text,
            'correct': predicted_state == true_state,
            'error_type': None if predicted_state == true_state else 'misclassification',
            'method': state_result.get('method', 'none')
        }
    
    def test_all_images(self):
        """Test all images in the test directory."""
        print(f"\nTesting all images in {self.test_data_dir}")
        
        total_tested = 0
        total_correct = 0
        
        # Results by state
        state_results = defaultdict(lambda: {'total': 0, 'correct': 0})
        
        # Confusion matrix data
        confusion_data = defaultdict(lambda: defaultdict(int))
        
        # Process each state directory
        for state_dir in os.listdir(self.test_data_dir):
            state_path = os.path.join(self.test_data_dir, state_dir)
            if not os.path.isdir(state_path):
                continue
                
            true_state = state_dir
            print(f"\nTesting {true_state} images...")
            
            # Process each image in the state directory
            for img_file in os.listdir(state_path):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                    
                img_path = os.path.join(state_path, img_file)
                result = self.test_single_image(img_path, true_state)
                
                if result:
                    total_tested += 1
                    state_results[true_state]['total'] += 1
                    
                    if result['correct']:
                        total_correct += 1
                        state_results[true_state]['correct'] += 1
                    
                    # Update confusion matrix
                    predicted = result['predicted_state'] or 'NONE'
                    confusion_data[true_state][predicted] += 1
                    
                    # Store detailed result
                    self.results[true_state].append(result)
                    
                    # Print result
                    status = "✓" if result['correct'] else "✗"
                    print(f"  {status} {img_file}: {result['plate_text']} -> "
                          f"{result['predicted_state']} (conf: {result['confidence']:.2%})")
        
        # Calculate overall accuracy
        overall_accuracy = total_correct / total_tested if total_tested > 0 else 0
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_tested': total_tested,
            'total_correct': total_correct,
            'state_results': dict(state_results),
            'confusion_data': dict(confusion_data),
            'detailed_results': dict(self.results)
        }
    
    def generate_report(self, results):
        """Generate a comprehensive accuracy report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = f"accuracy_report_{timestamp}"
        os.makedirs(report_dir, exist_ok=True)
        
        # 1. Overall Summary
        summary = {
            'timestamp': timestamp,
            'overall_accuracy': results['overall_accuracy'],
            'total_images_tested': results['total_tested'],
            'total_correct': results['total_correct'],
            'states_tested': len(results['state_results'])
        }
        
        # 2. Per-State Accuracy
        state_accuracy = []
        for state, data in results['state_results'].items():
            if data['total'] > 0:
                accuracy = data['correct'] / data['total']
                state_accuracy.append({
                    'state': state,
                    'accuracy': accuracy,
                    'total_tested': data['total'],
                    'correct': data['correct']
                })
        
        state_accuracy.sort(key=lambda x: x['accuracy'])
        
        # 3. Save detailed results
        with open(os.path.join(report_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        with open(os.path.join(report_dir, 'state_accuracy.json'), 'w') as f:
            json.dump(state_accuracy, f, indent=2)
        
        # 4. Generate accuracy plot
        if state_accuracy:
            plt.figure(figsize=(12, 8))
            states = [s['state'] for s in state_accuracy]
            accuracies = [s['accuracy'] * 100 for s in state_accuracy]
            
            bars = plt.bar(states, accuracies)
            
            # Color bars based on accuracy
            for bar, acc in zip(bars, accuracies):
                if acc >= 95:
                    bar.set_color('green')
                elif acc >= 80:
                    bar.set_color('yellow')
                else:
                    bar.set_color('red')
            
            plt.axhline(y=95, color='r', linestyle='--', label='95% Target')
            plt.xlabel('State')
            plt.ylabel('Accuracy (%)')
            plt.title('State Recognition Accuracy by State')
            plt.xticks(rotation=90)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, 'accuracy_by_state.png'))
            plt.close()
        
        # 5. Generate text report
        report_text = f"""
State Recognition Accuracy Report
Generated: {timestamp}
=====================================

Overall Performance:
- Total Images Tested: {results['total_tested']}
- Correctly Identified: {results['total_correct']}
- Overall Accuracy: {results['overall_accuracy']:.2%}
- States Tested: {len(results['state_results'])}

Per-State Performance:
"""
        
        for state_data in state_accuracy:
            report_text += f"\n{state_data['state']}: "
            report_text += f"{state_data['accuracy']:.1%} "
            report_text += f"({state_data['correct']}/{state_data['total_tested']})"
            
            # Add warning for low accuracy
            if state_data['accuracy'] < 0.95:
                report_text += " ⚠️ Below 95% target"
        
        # Identify problem patterns
        report_text += "\n\nProblem Areas:\n"
        problem_patterns = self._identify_problem_patterns(results['detailed_results'])
        for pattern in problem_patterns:
            report_text += f"- {pattern}\n"
        
        with open(os.path.join(report_dir, 'report.txt'), 'w') as f:
            f.write(report_text)
        
        print(f"\nReport generated in {report_dir}/")
        print(report_text)
        
        return report_dir
    
    def _identify_problem_patterns(self, detailed_results):
        """Identify common failure patterns."""
        problems = []
        
        # Analyze failures
        failures = []
        for state, results in detailed_results.items():
            for result in results:
                if not result['correct']:
                    failures.append(result)
        
        if not failures:
            return ["No failures detected!"]
        
        # Group by error type
        error_types = defaultdict(int)
        misclassified_patterns = defaultdict(int)
        
        for failure in failures:
            error_types[failure['error_type']] += 1
            
            if failure['plate_text']:
                # Analyze pattern
                pattern = self._get_pattern(failure['plate_text'])
                misclassified_patterns[pattern] += 1
        
        # Report findings
        if error_types['no_plate_detected'] > 0:
            problems.append(f"OCR failed to detect plate text in {error_types['no_plate_detected']} images")
        
        # Most problematic patterns
        if misclassified_patterns:
            top_patterns = sorted(misclassified_patterns.items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            for pattern, count in top_patterns:
                problems.append(f"Pattern {pattern} failed {count} times")
        
        return problems
    
    def _get_pattern(self, text):
        """Convert text to pattern format."""
        pattern = ""
        for char in text:
            if char.isalpha():
                pattern += "L"
            elif char.isdigit():
                pattern += "N"
            else:
                pattern += "?"
        return pattern


def main():
    """Run accuracy measurement."""
    import sys
    
    tester = AccuracyMeasurement()
    
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        tester.create_test_structure()
        print("\nNow add test images to the state folders and run again without 'setup'")
        return
    
    # Check if test directory exists
    if not os.path.exists(tester.test_data_dir):
        print(f"Test directory '{tester.test_data_dir}' not found!")
        print("Run 'python measure_accuracy.py setup' to create the directory structure")
        return
    
    # Run tests
    results = tester.test_all_images()
    
    if results['total_tested'] == 0:
        print("No test images found! Add images to the test_data/STATE folders")
        return
    
    # Generate report
    tester.generate_report(results)


if __name__ == "__main__":
    main()