"""
Finger Counting Benchmark Tool
Validates finger detection accuracy by comparing CV vs MediaPipe on same frames.
Reuses calibration code from main.py for consistency.
Automatically runs detailed analysis after benchmark.
"""

import cv2
import numpy as np
import sys
import os
import json
import subprocess
import time
from datetime import datetime
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detectors.cv.cv_detector import CVDetector
from src.detectors.mediapipe_detector import MediaPipeDetector
from src.detectors.cv.finger_detection import draw_finger_visualization
from src.core.config import CALIBRATION_DURATION


class FingerCountingBenchmark:
    """Benchmarks CV vs MediaPipe finger counting on same frames"""
    
    def __init__(self, cv_detector=None, mp_detector=None):
        # Accept pre-initialized detectors (e.g., after calibration)
        self.cv_detector = cv_detector if cv_detector is not None else CVDetector()
        self.mp_detector = mp_detector if mp_detector is not None else MediaPipeDetector()
        
        # Separate results for each detector
        self.cv_results = defaultdict(lambda: {
            'samples': 0,
            'correct': 0,
            'detected_counts': [],
            'errors': []
        })
        self.mp_results = defaultdict(lambda: {
            'samples': 0,
            'correct': 0,
            'detected_counts': [],
            'errors': []
        })
        
    def run_test_sequence(self, duration_per_pose=5, cap=None):
        """
        Run finger counting test with user holding specific finger counts.
        Both detectors run simultaneously on same frames for fair comparison.
        
        Args:
            duration_per_pose: Seconds to collect samples per finger count (default: 5s)
            cap: OpenCV VideoCapture object (reuses camera from calibration)
        """
        own_cap = False
        if cap is None:
            cap = cv2.VideoCapture(0)
            own_cap = True
            
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        # Test sequence: 0, 1, 2, 3, 4, 5 fingers
        test_sequence = [0, 1, 2, 3, 4, 5]
        
        print("\n" + "="*80)
        print("FINGER COUNTING BENCHMARK - CV vs MediaPipe")
        print("="*80)
        print(f"Detectors: CV vs MediaPipe (parallel)")
        print(f"Duration per pose: {duration_per_pose}s")
        print("\nInstructions:")
        print("  - Position your hand clearly in front of the camera")
        print("  - Hold each finger count steady for the duration")
        print("  - Both detectors process the SAME frames simultaneously")
        print("  - Press 'q' to skip a pose or 'ESC' to quit early")
        print("="*80)
        
        for expected_fingers in test_sequence:
            print(f"\nGET READY: Show {expected_fingers} finger(s)")
            print("   Press any key when ready...")
            
            # Wait for user to get ready
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Show preview
                display = frame.copy()
                text = f"Show {expected_fingers} finger(s) - Press any key to start"
                cv2.putText(display, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow('Finger Counting Benchmark', display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key != 255:  # Any key pressed
                    break
            
            # Start collecting samples
            print(f"RECORDING {expected_fingers} finger(s)...")
            start_time = cv2.getTickCount()
            cv_samples = 0
            mp_samples = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Process frame with both detectors simultaneously
                cv_result = self.cv_detector.process_frame(frame.copy())
                mp_result = self.mp_detector.process_frame(frame.copy())
                
                # Record CV results
                if cv_result and cv_result['detected']:
                    detected_fingers = cv_result.get('finger_count', -1)
                    self.cv_results[expected_fingers]['samples'] += 1
                    self.cv_results[expected_fingers]['detected_counts'].append(detected_fingers)
                    
                    if detected_fingers == expected_fingers:
                        self.cv_results[expected_fingers]['correct'] += 1
                    else:
                        self.cv_results[expected_fingers]['errors'].append(detected_fingers)
                    
                    cv_samples += 1
                
                # Record MediaPipe results
                if mp_result and mp_result['detected']:
                    detected_fingers = mp_result.get('finger_count', -1)
                    self.mp_results[expected_fingers]['samples'] += 1
                    self.mp_results[expected_fingers]['detected_counts'].append(detected_fingers)
                    
                    if detected_fingers == expected_fingers:
                        self.mp_results[expected_fingers]['correct'] += 1
                    else:
                        self.mp_results[expected_fingers]['errors'].append(detected_fingers)
                    
                    mp_samples += 1
                
                # Display feedback with visualizations
                display = frame.copy()
                elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                remaining = duration_per_pose - elapsed
                
                # Draw CV visualizations (hand contour, convex hull, fingertips)
                if cv_result and cv_result.get('contour') is not None:
                    contour = cv_result['contour']
                    # Draw hand contour in green
                    cv2.drawContours(display, [contour], 0, (0, 255, 0), 2)
                    
                    # Draw convex hull in yellow
                    hull = cv2.convexHull(contour)
                    cv2.drawContours(display, [hull], 0, (0, 255, 255), 2)
                    
                    # Draw debug info if available
                    if 'debug_info' in cv_result:
                        debug = cv_result['debug_info']
                        
                        # Draw fingertip candidates (green circles)
                        if 'fingertips' in debug:
                            for pt in debug['fingertips']:
                                cv2.circle(display, tuple(pt), 8, (0, 255, 0), -1)
                        
                        # Draw valleys (blue circles)
                        if 'valleys' in debug:
                            for pt in debug['valleys']:
                                cv2.circle(display, tuple(pt), 6, (255, 0, 0), -1)
                        
                        # Draw hand center (magenta)
                        if 'hand_center' in debug:
                            cv2.circle(display, tuple(debug['hand_center']), 8, (255, 0, 255), -1)
                
                y_pos = 30
                
                # Draw CV detection result with debug metrics
                if cv_result and cv_result['detected']:
                    cv_detected = cv_result.get('finger_count', '?')
                    cv_color = (0, 255, 0) if cv_detected == expected_fingers else (0, 0, 255)
                    cv_status = "✓" if cv_detected == expected_fingers else "✗"
                    
                    # Main result
                    cv2.putText(display, f"CV: {cv_detected} {cv_status}", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cv_color, 2)
                    
                    # Debug metrics
                    if 'debug_info' in cv_result:
                        debug = cv_result['debug_info']
                        y_debug = display.shape[0] - 120
                        cv2.putText(display, "CV Debug:", 
                                   (10, y_debug), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        y_debug += 20
                        
                        if 'solidity' in debug:
                            cv2.putText(display, f"Solidity: {debug['solidity']:.2f}", 
                                       (10, y_debug), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                            y_debug += 20
                        
                        if 'method_counts' in debug:
                            counts = debug['method_counts']
                            cv2.putText(display, f"Methods: {counts}", 
                                       (10, y_debug), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                            y_debug += 20
                        
                        if 'area' in debug:
                            cv2.putText(display, f"Area: {debug['area']:.0f}", 
                                       (10, y_debug), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                else:
                    cv2.putText(display, "CV: No hand", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                
                y_pos += 30
                
                # Draw MediaPipe detection result
                if mp_result and mp_result['detected']:
                    mp_detected = mp_result.get('finger_count', '?')
                    mp_color = (0, 255, 0) if mp_detected == expected_fingers else (0, 0, 255)
                    mp_status = "✓" if mp_detected == expected_fingers else "✗"
                    cv2.putText(display, f"MP: {mp_detected} {mp_status}", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mp_color, 2)
                else:
                    cv2.putText(display, "MP: No hand", 
                               (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                
                y_pos += 40
                
                cv2.putText(display, f"Expected: {expected_fingers} fingers", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display, f"Time: {remaining:.1f}s | CV:{cv_samples} MP:{mp_samples}", 
                           (10, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display, "Press 'q' to skip | ESC to quit", 
                           (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow('Finger Counting Benchmark', display)
                
                # Check for user input
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC - quit
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key == ord('q'):  # Skip this pose
                    print(f"  Skipped after CV:{cv_samples} MP:{mp_samples} samples")
                    break
                
                # Check if duration elapsed
                if elapsed >= duration_per_pose:
                    print(f"   ✓ Collected CV:{cv_samples} MP:{mp_samples} samples")
                    break
        
        if own_cap:
            cap.release()
        cv2.destroyAllWindows()
        print("\nTest sequence complete!")
    
    def generate_report(self):
        """Generate detailed accuracy report for both detectors"""
        reports = {}
        
        print("\n" + "="*80)
        print("FINGER COUNTING ACCURACY REPORT - CV DETECTOR")
        print("="*80)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        cv_report = self._generate_single_report(self.cv_results, "CV")
        reports['cv'] = cv_report
        
        print("\n" + "="*80)
        print("FINGER COUNTING ACCURACY REPORT - MEDIAPIPE DETECTOR")
        print("="*80)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        mp_report = self._generate_single_report(self.mp_results, "MediaPipe")
        reports['mediapipe'] = mp_report
        
        # Print comparison
        self._print_comparison(reports['cv'], reports['mediapipe'])
        
        return reports
    
    def _generate_single_report(self, results, detector_name):
        """Generate report for a single detector"""
        total_samples = 0
        total_correct = 0
        
        print("\n--- PER-FINGER ACCURACY ---")
        for expected in sorted(results.keys()):
            data = results[expected]
            samples = data['samples']
            correct = data['correct']
            accuracy = (correct / samples * 100) if samples > 0 else 0
            
            total_samples += samples
            total_correct += correct
            
            print(f"\n{expected} finger(s):")
            print(f"  Samples:  {samples}")
            print(f"  Correct:  {correct}")
            print(f"  Accuracy: {accuracy:.1f}%")
            
            if data['detected_counts']:
                counts = np.array(data['detected_counts'])
                print(f"  Mean:     {np.mean(counts):.2f}")
                print(f"  Std Dev:  {np.std(counts):.2f}")
                print(f"  Median:   {np.median(counts):.0f}")
                print(f"  Mode:     {np.bincount(counts).argmax()}")
            
            if data['errors']:
                error_counts = defaultdict(int)
                for err in data['errors']:
                    error_counts[err] += 1
                print(f"  Errors:   {dict(error_counts)}")
        
        # Overall statistics
        overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
        
        print("\n" + "-"*80)
        print("--- OVERALL STATISTICS ---")
        print(f"Total Samples:    {total_samples}")
        print(f"Total Correct:    {total_correct}")
        print(f"Overall Accuracy: {overall_accuracy:.1f}%")
        
        # Confusion matrix
        print("\n--- CONFUSION MATRIX ---")
        print("Expected → Detected")
        for expected in sorted(results.keys()):
            data = results[expected]
            if data['detected_counts']:
                counts = np.array(data['detected_counts'])
                distribution = {i: np.sum(counts == i) for i in range(6)}
                print(f"{expected}: {distribution}")
        
        print("="*80)
        
        return {
            'detector': detector_name.lower(),
            'timestamp': datetime.now().isoformat(),
            'overall_accuracy': overall_accuracy,
            'total_samples': total_samples,
            'total_correct': total_correct,
            'per_finger_results': dict(results)
        }
    
    def _print_comparison(self, cv_report, mp_report):
        """Print side-by-side comparison of both detectors"""
        print("\n" + "="*80)
        print("DIRECT COMPARISON (Same Frames)")
        print("="*80)
        
        print(f"\n{'Metric':<20} {'CV':<15} {'MediaPipe':<15} {'Winner'}")
        print("-" * 80)
        
        cv_acc = cv_report['overall_accuracy']
        mp_acc = mp_report['overall_accuracy']
        winner = "CV" if cv_acc > mp_acc else ("MediaPipe" if mp_acc > cv_acc else "Tie")
        print(f"{'Overall Accuracy':<20} {cv_acc:>6.1f}%{'':<8} {mp_acc:>6.1f}%{'':<8} {winner}")
        
        print(f"\n{'Fingers':<10} {'CV Acc':<12} {'MP Acc':<12} {'Difference'}")
        print("-" * 80)
        
        for finger in sorted(set(cv_report['per_finger_results'].keys()) | set(mp_report['per_finger_results'].keys())):
            cv_data = cv_report['per_finger_results'].get(finger, {'samples': 0, 'correct': 0})
            mp_data = mp_report['per_finger_results'].get(finger, {'samples': 0, 'correct': 0})
            
            cv_finger_acc = (cv_data['correct'] / cv_data['samples'] * 100) if cv_data['samples'] > 0 else 0
            mp_finger_acc = (mp_data['correct'] / mp_data['samples'] * 100) if mp_data['samples'] > 0 else 0
            diff = cv_finger_acc - mp_finger_acc
            
            print(f"{finger:<10} {cv_finger_acc:>6.1f}%{'':<5} {mp_finger_acc:>6.1f}%{'':<5} {diff:>+6.1f}%")
        
        print("="*80)
    
    def save_results(self):
        """Save results to JSON files"""
        os.makedirs('benchmarks', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        reports = self.generate_report()
        saved_files = []
        
        for detector_name, report in reports.items():
            filename = f"finger_accuracy_{detector_name}_{timestamp}.json"
            filepath = os.path.join('benchmarks', filename)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"{detector_name.upper()} results saved to: {filepath}")
            saved_files.append(filepath)
        
        return saved_files


def run_mediapipe_calibration(cap):
    """
    Reuse calibration logic from main.py
    Returns (cv_detector, mp_detector) with calibrated CV detector
    """
    print("\n" + "=" * 70)
    print("MEDIAPIPE-BASED CALIBRATION (10 seconds)")
    print("=" * 70)
    print("MediaPipe will detect your hand automatically")
    print("Move your hand around slowly for best calibration")
    print("Press SPACE to start (ESC to skip)")
    print("=" * 70)
    
    mp_detector = MediaPipeDetector()
    
    # Preview mode
    print("\nPreview - Press SPACE to start calibration")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        mp_result = mp_detector.process_frame(frame.copy(), use_palm_center=True)
        
        cv2.putText(mp_result['annotated_frame'], "Move hand around slowly", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(mp_result['annotated_frame'], "Press SPACE to start (ESC to skip)", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Calibration', mp_result['annotated_frame'])
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            break
        elif key == 27:  # ESC
            print("Calibration skipped")
            return None, None
    
    # Calibration phase
    ycrcb_samples, hsv_samples = [], []
    duration = 10
    start_time = time.time()
    hand_regions_collected = 0
    
    print("\nCalibrating with MediaPipe...")
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        
        remaining = duration - (time.time() - start_time)
        mp_result = mp_detector.process_frame(frame.copy(), use_palm_center=True)
        
        # Progress bar
        progress = int(((duration - remaining) / duration) * 600)
        cv2.rectangle(mp_result['annotated_frame'], (20, 420), (20 + progress, 440), (0, 255, 0), -1)
        cv2.rectangle(mp_result['annotated_frame'], (20, 420), (620, 440), (255, 255, 255), 2)
        
        cv2.putText(mp_result['annotated_frame'], f"Calibrating: {remaining:.1f}s", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(mp_result['annotated_frame'], f"Regions: {hand_regions_collected}", (20, 450), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if mp_result['detected']:
            h, w = frame.shape[:2]
            center_x = mp_result['hand_x']
            center_y = mp_result['hand_y']
            
            # Skip face region
            if center_y < 0.3:
                cv2.imshow('Calibration', mp_result['annotated_frame'])
                cv2.waitKey(1)
                continue
            
            # Sample 100x100 region around hand center
            cx_pixel = int(center_x * w)
            cy_pixel = int(center_y * h)
            
            y1 = max(0, cy_pixel - 50)
            y2 = min(h, cy_pixel + 50)
            x1 = max(0, cx_pixel - 50)
            x2 = min(w, cx_pixel + 50)
            
            if y2 - y1 > 20 and x2 - x1 > 20:
                hand_region = frame[y1:y2, x1:x2]
                
                ycrcb_region = cv2.cvtColor(hand_region, cv2.COLOR_BGR2YCrCb)
                hsv_region = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
                
                ycrcb_samples.append(ycrcb_region.reshape(-1, 3))
                hsv_samples.append(hsv_region.reshape(-1, 3))
                hand_regions_collected += 1
                
                # Visual feedback
                cv2.rectangle(mp_result['annotated_frame'], (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imshow('Calibration', mp_result['annotated_frame'])
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cv2.destroyAllWindows()
    
    if len(ycrcb_samples) < 5:
        print("Not enough hand regions collected")
        return None, None
    
    # Calculate bounds using shared function (percentile-based with wrap-around support)
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from tools.debug_utils import calculate_skin_bounds
    
    ycrcb_lower, ycrcb_upper, hsv_lower, hsv_upper = calculate_skin_bounds(
        ycrcb_samples, hsv_samples, margin_factor=0.15
    )
    
    print(f"Calibration complete! Collected {hand_regions_collected} regions")
    print(f"YCrCb: {ycrcb_lower.tolist()} - {ycrcb_upper.tolist()}")
    print(f"HSV:   {hsv_lower.tolist()} - {hsv_upper.tolist()}")
    
    # Create CV detector with calibrated bounds
    cv_detector = CVDetector()
    cv_detector.update_calibration(ycrcb_lower, ycrcb_upper, hsv_lower, hsv_upper)
    
    return cv_detector, mp_detector


def main():
    print("\nFINGER COUNTING BENCHMARK - CV vs MediaPipe")
    print("Both detectors process the SAME frames for fair comparison.")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    # Run calibration first
    print("\nStep 1: Calibration")
    cv_detector, mp_detector = run_mediapipe_calibration(cap)
    
    if cv_detector is None or mp_detector is None:
        print("Calibration failed or skipped")
        cap.release()
        return
    
    print("\nStep 2: Benchmark Test")
    print("\nSelect test duration per pose:")
    print("  1. Quick test (3 seconds per pose = ~18s total)")
    print("  2. Standard test (5 seconds per pose = ~30s total)")
    print("  3. Thorough test (10 seconds per pose = ~60s total)")
    
    duration_choice = input("\nEnter choice (1-3): ").strip()
    duration_map = {'1': 3, '2': 5, '3': 10}
    duration = duration_map.get(duration_choice, 5)
    
    # Run benchmark with calibrated detectors (reuse same camera)
    print("\nRunning parallel comparison...")
    benchmark = FingerCountingBenchmark(cv_detector, mp_detector)
    benchmark.run_test_sequence(duration_per_pose=duration, cap=cap)
    saved_files = benchmark.save_results()
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Run detailed analysis automatically
    print("\n" + "="*80)
    print("RUNNING DETAILED ANALYSIS...")
    print("="*80)
    subprocess.run([sys.executable, 'tools/analyze_finger_detection.py'])


if __name__ == '__main__':
    main()
