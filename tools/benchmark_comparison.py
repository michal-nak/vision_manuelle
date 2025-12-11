"""
Benchmark Comparison Tool - CV vs MediaPipe
Compares performance metrics between CV and MediaPipe detection methods
"""
import cv2
import numpy as np
import time
import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detectors import CVDetector, MediaPipeDetector
from src.core.utils import find_camera, setup_camera
from src.core.config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS


def run_mediapipe_calibration():
    """Run MediaPipe-based calibration for CV detector (reused from main.py)"""
    print("\n" + "=" * 70)
    print("MEDIAPIPE-BASED CALIBRATION (10 seconds)")
    print("=" * 70)
    print("MediaPipe will detect your hand automatically")
    print("Move your hand around slowly for best calibration")
    print("Press SPACE to start (ESC to skip)")
    print("=" * 70)
    
    cap = find_camera()
    if not cap:
        print("Could not open camera, skipping calibration")
        return False
    
    setup_camera(cap, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
    
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
        
        cv2.imshow('MediaPipe Calibration', mp_result['annotated_frame'])
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            break
        elif key == 27:  # ESC
            print("Calibration skipped")
            cap.release()
            cv2.destroyAllWindows()
            mp_detector.cleanup()
            return False
    
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
        cv2.putText(mp_result['annotated_frame'], "Move hand slowly", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(mp_result['annotated_frame'], f"Regions: {hand_regions_collected}", (20, 450), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if mp_result['detected']:
            h, w = frame.shape[:2]
            center_x = mp_result['hand_x']
            center_y = mp_result['hand_y']
            
            if center_y < 0.3:  # Skip face region
                continue
            
            center_x = max(0.1, min(0.9, center_x))
            center_y = max(0.1, min(0.9, center_y))
            
            bbox_w = int(w * 0.22)
            bbox_h = int(h * 0.28)
            x1 = max(0, int(center_x * w - bbox_w // 2))
            y1 = max(0, int(center_y * h - bbox_h // 2))
            x2 = min(w, x1 + bbox_w)
            y2 = min(h, y1 + bbox_h)
            
            cv2.rectangle(mp_result['annotated_frame'], (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            hand_region = frame[y1:y2, x1:x2]
            if hand_region.size > 0 and hand_region.shape[0] > 10 and hand_region.shape[1] > 10:
                roi_ycrcb = cv2.cvtColor(hand_region, cv2.COLOR_BGR2YCrCb)
                roi_hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
                
                sample_size = min(150, hand_region.shape[0] * hand_region.shape[1])
                if sample_size > 0:
                    indices = np.random.choice(hand_region.shape[0] * hand_region.shape[1], 
                                             size=sample_size, replace=False)
                    
                    for idx in indices:
                        ycrcb_samples.append(roi_ycrcb.reshape(-1, 3)[idx])
                        hsv_samples.append(roi_hsv.reshape(-1, 3)[idx])
                    
                    hand_regions_collected += 1
        
        cv2.imshow('MediaPipe Calibration', mp_result['annotated_frame'])
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cv2.destroyAllWindows()
    mp_detector.cleanup()
    
    # Calculate and save thresholds
    if len(ycrcb_samples) < 100 or len(hsv_samples) < 100:
        print(f"\nInsufficient samples ({len(ycrcb_samples)} collected)")
        cap.release()
        return False
    
    ycrcb_array = np.array(ycrcb_samples)
    hsv_array = np.array(hsv_samples)
    
    # Calculate bounds with margin
    margin_factor = 1.5
    ycrcb_mean = np.mean(ycrcb_array, axis=0)
    ycrcb_std = np.std(ycrcb_array, axis=0)
    hsv_mean = np.mean(hsv_array, axis=0)
    hsv_std = np.std(hsv_array, axis=0)
    
    ycrcb_lower = np.clip(ycrcb_mean - margin_factor * ycrcb_std, 0, 255).astype(int)
    ycrcb_upper = np.clip(ycrcb_mean + margin_factor * ycrcb_std, 0, 255).astype(int)
    hsv_lower = np.clip(hsv_mean - margin_factor * hsv_std, 0, 255).astype(int)
    hsv_upper = np.clip(hsv_mean + margin_factor * hsv_std, 0, 255).astype(int)
    
    # Save to config
    config_path = Path(__file__).parent.parent / 'skin_detection_config.json'
    config = {
        'ycrcb_lower': ycrcb_lower.tolist(),
        'ycrcb_upper': ycrcb_upper.tolist(),
        'hsv_lower': hsv_lower.tolist(),
        'hsv_upper': hsv_upper.tolist()
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Calibration complete! Config saved to {config_path}")
    print(f"   Collected {len(ycrcb_samples)} samples from {hand_regions_collected} hand regions")
    
    cap.release()
    return True


class BenchmarkResults:
    """Stores and analyzes benchmark results"""
    
    def __init__(self, name):
        self.name = name
        self.frame_times = []
        self.detection_times = []
        self.detections = []
        self.finger_counts = []
        self.gestures = []
        self.false_positives = 0
        self.false_negatives = 0
        self.total_frames = 0
        
    def add_frame(self, frame_time, detection_time, detected, finger_count, gesture):
        """Add frame metrics"""
        self.frame_times.append(frame_time)
        self.detection_times.append(detection_time)
        self.detections.append(detected)
        self.finger_counts.append(finger_count)
        self.gestures.append(gesture)
        self.total_frames += 1
    
    def get_fps_stats(self):
        """Calculate FPS statistics"""
        if not self.frame_times:
            return {"mean": 0, "min": 0, "max": 0, "std": 0}
        
        fps_values = [1.0 / ft if ft > 0 else 0 for ft in self.frame_times]
        return {
            "mean": np.mean(fps_values),
            "min": np.min(fps_values),
            "max": np.max(fps_values),
            "std": np.std(fps_values),
            "median": np.median(fps_values)
        }
    
    def get_detection_rate(self):
        """Calculate detection success rate"""
        if not self.detections:
            return 0.0
        return (sum(self.detections) / len(self.detections)) * 100
    
    def get_latency_stats(self):
        """Calculate latency statistics in ms"""
        if not self.detection_times:
            return {"mean": 0, "min": 0, "max": 0, "std": 0}
        
        latencies_ms = [dt * 1000 for dt in self.detection_times]
        return {
            "mean": np.mean(latencies_ms),
            "min": np.min(latencies_ms),
            "max": np.max(latencies_ms),
            "std": np.std(latencies_ms),
            "median": np.median(latencies_ms)
        }
    
    def get_finger_count_stability(self):
        """Calculate finger count stability (variance)"""
        if len(self.finger_counts) < 2:
            return 0.0
        
        # Calculate variance in sliding windows
        window_size = 10
        variances = []
        for i in range(len(self.finger_counts) - window_size):
            window = self.finger_counts[i:i+window_size]
            variances.append(np.var(window))
        
        return np.mean(variances) if variances else 0.0
    
    def get_summary(self):
        """Get complete summary"""
        fps_stats = self.get_fps_stats()
        latency_stats = self.get_latency_stats()
        
        return {
            "name": self.name,
            "total_frames": self.total_frames,
            "fps": fps_stats,
            "latency_ms": latency_stats,
            "detection_rate": self.get_detection_rate(),
            "finger_stability": self.get_finger_count_stability(),
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives
        }


def compare_detectors(duration_seconds=60):
    """
    Compare CV and MediaPipe detectors
    
    Args:
        duration_seconds: Duration of benchmark in seconds
    """
    print("=" * 80)
    print("BENCHMARK COMPARISON: CV vs MediaPipe")
    print("=" * 80)
    print(f"Duration: {duration_seconds} seconds per detector")
    print("\nInstructions:")
    print("1. Position your hand in front of the camera")
    print("2. Perform various gestures (0-5 fingers)")
    print("3. Try different angles, distances, and speeds")
    print("4. Keep consistent between both tests")
    print("\nPress 'q' to skip to next detector or ESC to abort")
    print("=" * 80)
    
    # Initialize camera
    cap = find_camera()
    if not cap:
        print("ERROR: Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    results = {}
    
    # Test each detector
    for detector_name, detector_class in [("CV", CVDetector), ("MediaPipe", MediaPipeDetector)]:
        print(f"\n{'='*80}")
        print(f"Testing {detector_name} Detector - {duration_seconds}s")
        print(f"{'='*80}")
        
        input(f"Press ENTER to start {detector_name} test...")
        
        detector = detector_class()
        benchmark = BenchmarkResults(detector_name)
        
        start_time = time.time()
        frame_count = 0
        
        # Warm-up frames
        for _ in range(10):
            ret, frame = cap.read()
            if ret:
                detector.process_frame(frame)
        
        print(f"Recording {detector_name} metrics...")
        
        while time.time() - start_time < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            
            # Detect
            detection_start = time.time()
            result = detector.process_frame(frame)
            detection_time = time.time() - detection_start
            
            # Extract metrics - check result structure
            detected = False
            finger_count = 0
            gesture = 'None'
            
            if result is not None:
                detected = result.get('detected', False)
                if detected:
                    finger_count = result.get('finger_count', 0)
                    gesture = result.get('gesture', 'None')
            
            # Display
            display = result.get('annotated_frame', frame) if result else frame
            
            # Add overlay with metrics and debug
            fps = 1.0 / (time.time() - frame_start) if (time.time() - frame_start) > 0 else 0
            elapsed = time.time() - start_time
            remaining = max(0, duration_seconds - elapsed)
            
            # Detection status color
            status_color = (0, 255, 0) if detected else (0, 0, 255)
            status_text = "DETECTED" if detected else "NO HAND"
            
            cv2.putText(display, f"{detector_name} Benchmark", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, f"Status: {status_text}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display, f"FPS: {fps:.1f} | Time: {remaining:.1f}s", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, f"Fingers: {finger_count} | Gesture: {gesture}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Benchmark', display)
            
            # Record metrics
            frame_time = time.time() - frame_start
            benchmark.add_frame(frame_time, detection_time, detected, finger_count, gesture)
            frame_count += 1
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print(f"\nSkipping {detector_name} test...")
                break
        
        # Store results
        results[detector_name] = benchmark
        
        print(f"\n{detector_name} Test Complete:")
        print(f"  Frames processed: {frame_count}")
        print(f"  Average FPS: {benchmark.get_fps_stats()['mean']:.2f}")
        print(f"  Detection rate: {benchmark.get_detection_rate():.1f}%")
    
    cv2.destroyAllWindows()
    cap.release()
    
    # Generate comparison report
    generate_report(results)
    
    return results


def generate_report(results):
    """Generate detailed comparison report"""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS - COMPARISON")
    print("=" * 80)
    
    # Collect summaries
    summaries = {name: res.get_summary() for name, res in results.items()}
    
    # FPS Comparison
    print("\n--- FRAMES PER SECOND (FPS) ---")
    print(f"{'Detector':<15} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10} {'Std Dev':<10}")
    print("-" * 80)
    for name, summary in summaries.items():
        fps = summary['fps']
        print(f"{name:<15} {fps['mean']:<10.2f} {fps['median']:<10.2f} "
              f"{fps['min']:<10.2f} {fps['max']:<10.2f} {fps['std']:<10.2f}")
    
    # Latency Comparison
    print("\n--- LATENCY (milliseconds) ---")
    print(f"{'Detector':<15} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10} {'Std Dev':<10}")
    print("-" * 80)
    for name, summary in summaries.items():
        lat = summary['latency_ms']
        print(f"{name:<15} {lat['mean']:<10.2f} {lat['median']:<10.2f} "
              f"{lat['min']:<10.2f} {lat['max']:<10.2f} {lat['std']:<10.2f}")
    
    # Detection Rate
    print("\n--- DETECTION METRICS ---")
    print(f"{'Detector':<15} {'Frames':<12} {'Detection Rate':<20} {'Finger Stability':<20}")
    print("-" * 80)
    for name, summary in summaries.items():
        print(f"{name:<15} {summary['total_frames']:<12} "
              f"{summary['detection_rate']:<20.2f}% "
              f"{summary['finger_stability']:<20.4f}")
    
    # Relative Comparison
    if len(summaries) == 2:
        cv_sum = summaries.get('CV')
        mp_sum = summaries.get('MediaPipe')
        
        if cv_sum and mp_sum:
            print("\n--- RELATIVE COMPARISON (CV vs MediaPipe) ---")
            print("-" * 80)
            
            fps_diff = ((cv_sum['fps']['mean'] / mp_sum['fps']['mean']) - 1) * 100
            lat_diff = ((cv_sum['latency_ms']['mean'] / mp_sum['latency_ms']['mean']) - 1) * 100
            det_diff = cv_sum['detection_rate'] - mp_sum['detection_rate']
            stab_diff = ((cv_sum['finger_stability'] / mp_sum['finger_stability']) - 1) * 100 if mp_sum['finger_stability'] > 0 else 0
            
            print(f"FPS Difference:             {fps_diff:+.2f}% "
                  f"({'CV faster' if fps_diff > 0 else 'MediaPipe faster'})")
            print(f"Latency Difference:         {lat_diff:+.2f}% "
                  f"({'CV slower' if lat_diff > 0 else 'CV faster'})")
            print(f"Detection Rate Difference:  {det_diff:+.2f}% "
                  f"({'CV better' if det_diff > 0 else 'MediaPipe better'})")
            print(f"Stability Difference:       {stab_diff:+.2f}% "
                  f"({'CV more stable' if stab_diff < 0 else 'MediaPipe more stable'})")
    
    # Save to file in benchmarks folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmarks_dir = Path(__file__).parent.parent / "benchmarks"
    benchmarks_dir.mkdir(exist_ok=True)
    
    output_file = benchmarks_dir / f"benchmark_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(summaries, f, indent=2)
    
    print(f"\n--- RESULTS SAVED ---")
    print(f"Full results saved to: {output_file}")
    print("=" * 80)


def quick_test(duration_seconds=20):
    """Quick test with fixed duration (more fair comparison than frame count)"""
    print("=" * 80)
    print("QUICK BENCHMARK TEST")
    print("=" * 80)
    print(f"Testing {duration_seconds} seconds per detector")
    print("This ensures fair comparison regardless of FPS differences")
    print("=" * 80)
    
    cap = find_camera()
    if not cap:
        print("ERROR: Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    results = {}
    
    for detector_name, detector_class in [("CV", CVDetector), ("MediaPipe", MediaPipeDetector)]:
        print(f"\nTesting {detector_name}...")
        
        detector = detector_class()
        benchmark = BenchmarkResults(detector_name)
        
        # Warm-up
        for _ in range(10):
            ret, frame = cap.read()
            if ret:
                detector.process_frame(frame)
        
        # Test with time-based duration
        start_time = time.time()
        frame_count = 0
        
        print(f"Testing {detector_name} for {duration_seconds} seconds...")
        
        while time.time() - start_time < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            detection_start = time.time()
            result = detector.process_frame(frame)
            detection_time = time.time() - detection_start
            
            # Extract metrics properly
            detected = False
            finger_count = 0
            gesture = 'None'
            
            if result is not None:
                detected = result.get('detected', False)
                if detected:
                    finger_count = result.get('finger_count', 0)
                    gesture = result.get('gesture', 'None')
            
            # Display video feedback
            display = result.get('annotated_frame', frame) if result else frame
            fps = 1.0 / (time.time() - frame_start) if (time.time() - frame_start) > 0 else 0
            
            # Calculate progress
            elapsed = time.time() - start_time
            remaining = max(0, duration_seconds - elapsed)
            progress_pct = (elapsed / duration_seconds) * 100
            
            # Detection status
            status_color = (0, 255, 0) if detected else (0, 0, 255)
            status_text = "DETECTED" if detected else "NO HAND"
            
            # Add comprehensive overlay
            cv2.putText(display, f"{detector_name} Benchmark - {progress_pct:.0f}%", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Status: {status_text}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display, f"Time: {remaining:.1f}s | FPS: {fps:.1f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, f"Frames: {frame_count}", (10, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(display, f"Fingers: {finger_count} | Gesture: {gesture}", (10, 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow(f'Benchmark - {detector_name}', display)
            
            frame_time = time.time() - frame_start
            benchmark.add_frame(frame_time, detection_time, detected, finger_count, gesture)
            frame_count += 1
            
            # Progress updates every 5 seconds
            if int(elapsed) % 5 == 0 and elapsed > 0 and frame_count % 30 == 0:
                print(f"  {elapsed:.0f}s elapsed - {frame_count} frames processed")
            
            # Allow early exit
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to abort
                print(f"\n  Aborted at {elapsed:.1f}s")
                break
        
        print(f"  Completed: {frame_count} frames in {duration_seconds}s")
        
        results[detector_name] = benchmark
        cv2.destroyAllWindows()
    
    cap.release()
    cv2.destroyAllWindows()
    generate_report(results)
    return results


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Benchmark CV vs MediaPipe detectors')
    parser.add_argument('-s', '--skip-calibration', action='store_true',
                       help='Skip MediaPipe calibration phase')
    args = parser.parse_args()
    
    # Run calibration unless skipped
    if not args.skip_calibration:
        print("\n" + "=" * 70)
        print("CALIBRATION PHASE")
        print("=" * 70)
        print("Before benchmarking, we need to calibrate the CV detector")
        print("This uses MediaPipe to automatically detect your hand")
        print("=" * 70)
        
        calibration_success = run_mediapipe_calibration()
        
        if not calibration_success:
            print("\n⚠️  Calibration was not completed")
            print("The benchmark will use existing config (if available)")
            response = input("Continue with benchmark anyway? (y/n): ").strip().lower()
            if response != 'y':
                print("Benchmark aborted")
                return
    else:
        print("\n⚠️  Calibration skipped (using existing config)")
    
    # Run benchmark
    print("\n" + "=" * 70)
    print("BENCHMARK OPTIONS")
    print("=" * 70)
    print("1. Full benchmark (60s per detector)")
    print("2. Quick test (20s per detector)")
    print("3. Custom duration")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        compare_detectors(duration_seconds=60)
    elif choice == "2":
        quick_test(duration_seconds=20)
    elif choice == "3":
        try:
            duration = int(input("Enter duration in seconds: ").strip())
            compare_detectors(duration_seconds=duration)
        except ValueError:
            print("Invalid duration, using default 60s")
            compare_detectors(duration_seconds=60)
    else:
        print("Invalid choice, running quick test")
        quick_test(frames=300)


if __name__ == "__main__":
    main()
