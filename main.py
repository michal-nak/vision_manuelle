"""
Hand Detection System - Main Entry Point
Launch the gesture paint application with optional calibration
"""
import sys
import cv2
from src.ui.gesture_paint import GesturePaintApp
from src.detectors import CVDetector
from src.core.utils import find_camera, setup_camera
from src.core.config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, CALIBRATION_RECT, CALIBRATION_DURATION
import tkinter as tk
import numpy as np
import time

def mediapipe_based_calibration():
    """Use MediaPipe to detect hand and calibrate CV detector from those regions"""
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
    
    from src.detectors import MediaPipeDetector
    mp_detector = MediaPipeDetector()
    
    # Preview mode
    print("\nPreview - Press SPACE to start calibration")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        mp_result = mp_detector.process_frame(frame.copy())
        
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
    
    # Calibration phase - collect hand regions from MediaPipe detections
    ycrcb_samples, hsv_samples = [], []
    duration = 10  # 10 seconds for better sampling
    start_time = time.time()
    hand_regions_collected = 0
    
    print("\nCalibrating with MediaPipe...")
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        
        remaining = duration - (time.time() - start_time)
        
        # Process with MediaPipe
        mp_result = mp_detector.process_frame(frame.copy())
        
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
        
        # If hand is detected, sample from hand region
        if mp_result['detected']:
            h, w = frame.shape[:2]
            
            # Get palm center from MediaPipe landmarks (not thumb tip)
            # Palm center is average of wrist (0) and middle finger base (9)
            if hasattr(mp_detector, 'last_landmarks') and mp_detector.last_landmarks:
                landmarks = mp_detector.last_landmarks
                wrist = landmarks.landmark[0]
                middle_mcp = landmarks.landmark[9]  # Middle finger base
                
                # Calculate palm center (between wrist and middle finger base)
                center_x = (wrist.x + middle_mcp.x) / 2
                center_y = (wrist.y + middle_mcp.y) / 2
            else:
                # Fallback: use thumb tip from result
                center_x = mp_result['hand_x']
                center_y = mp_result['hand_y']
            
            # Filter out detections in top 30% of frame (likely face)
            if center_y < 0.3:  # Skip if in top 30% (face region)
                continue
            
            # Clamp to frame bounds
            center_x = max(0.1, min(0.9, center_x))
            center_y = max(0.1, min(0.9, center_y))
            
            # Bounding box centered on hand (includes palm and fingers)
            bbox_w = int(w * 0.22)
            bbox_h = int(h * 0.28)
            x1 = max(0, int(center_x * w - bbox_w // 2))
            y1 = max(0, int(center_y * h - bbox_h // 2))
            x2 = min(w, x1 + bbox_w)
            y2 = min(h, y1 + bbox_h)
            
            # Draw bbox (green for hand sampling area)
            cv2.rectangle(mp_result['annotated_frame'], (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw estimated hand center
            center_px_x = int(center_x * w)
            center_px_y = int(center_y * h)
            cv2.circle(mp_result['annotated_frame'], (center_px_x, center_px_y), 8, (0, 255, 0), -1)
            cv2.putText(mp_result['annotated_frame'], "Hand Center", (center_px_x + 10, center_px_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Extract from entire hand region (palm + fingers)
            hand_region = frame[y1:y2, x1:x2]
            if hand_region.size > 0 and hand_region.shape[0] > 10 and hand_region.shape[1] > 10:
                roi_ycrcb = cv2.cvtColor(hand_region, cv2.COLOR_BGR2YCrCb)
                roi_hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
                
                # Sample from entire hand region
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
    
    # Calculate thresholds
    if len(ycrcb_samples) < 500:
        print(f"Not enough samples ({len(ycrcb_samples)}), skipping calibration")
        cap.release()
        return False
    
    ycrcb_array = np.array(ycrcb_samples)
    hsv_array = np.array(hsv_samples)
    
    # Use 10th and 90th percentile for tighter, more accurate bounds
    # This avoids outliers from shadows, highlights, or background
    ycrcb_lower = np.percentile(ycrcb_array, 10, axis=0).astype(np.uint8)
    ycrcb_upper = np.percentile(ycrcb_array, 90, axis=0).astype(np.uint8)
    hsv_lower = np.percentile(hsv_array, 10, axis=0).astype(np.uint8)
    hsv_upper = np.percentile(hsv_array, 90, axis=0).astype(np.uint8)
    
    # Create calibration dict
    base_calibration = {
        'ycrcb_lower': ycrcb_lower,
        'ycrcb_upper': ycrcb_upper,
        'hsv_lower': hsv_lower,
        'hsv_upper': hsv_upper
    }
    
    print("\n" + "=" * 70)
    print("COLOR CALIBRATION COMPLETE")
    print("=" * 70)
    print(f"Hand regions collected: {hand_regions_collected}")
    print(f"Total samples: {len(ycrcb_samples)}")
    print(f"YCrCb: {ycrcb_lower} - {ycrcb_upper}")
    print(f"HSV: {hsv_lower} - {hsv_upper}")
    print("=" * 70)
    
    # Run auto-optimization to find best processing parameters
    print("\nRunning auto-optimization to find best processing parameters...")
    print("This will test different parameter combinations for 18 seconds.")
    print("MediaPipe will be used as ground truth for detection quality.")
    print("Keep your hand in frame for accurate results.")
    input("\nPress ENTER to start optimization...")
    
    from tools.calibration.auto_optimize import auto_optimize
    optimized_config = auto_optimize(cap, base_calibration, use_mediapipe_validation=True)
    
    cap.release()
    
    if not optimized_config:
        print("Optimization cancelled, using color calibration only")
        # Apply color calibration to CV detector
        CVDetector.ycrcb_lower = ycrcb_lower
        CVDetector.ycrcb_upper = ycrcb_upper
        CVDetector.hsv_lower = hsv_lower
        CVDetector.hsv_upper = hsv_upper
        return True
    
    # Save optimized configuration
    from tools.calibration.config_io import save_calibration
    save_calibration(optimized_config)
    
    # Apply to CV detector class defaults
    CVDetector.ycrcb_lower = optimized_config['ycrcb_lower']
    CVDetector.ycrcb_upper = optimized_config['ycrcb_upper']
    CVDetector.hsv_lower = optimized_config['hsv_lower']
    CVDetector.hsv_upper = optimized_config['hsv_upper']
    
    return True

def main():
    # Get detection mode from command line or default to mediapipe
    detection_mode = sys.argv[1] if len(sys.argv) > 1 else "mediapipe"
    skip_calibration = "--skip-calibration" in sys.argv or "-s" in sys.argv
    debug_mode = "--debug" in sys.argv or "-d" in sys.argv
    
    if detection_mode not in ["mediapipe", "cv"]:
        print("Usage: python main.py [mediapipe|cv] [--skip-calibration|-s] [--debug|-d]")
        print("  mediapipe: Use MediaPipe hand detection")
        print("  cv: Use CV-based hand detection")
        print("  --skip-calibration, -s: Skip calibration and use saved config (cv mode only)")
        print("  --debug, -d: Show debug overlay with masks and detection info")
        detection_mode = "mediapipe"
    
    # Run MediaPipe-based calibration if using CV mode (unless skipped)
    if detection_mode == "cv":
        if skip_calibration:
            print("\n" + "=" * 70)
            print("Skipping calibration - using saved configuration")
            print("=" * 70)
            from pathlib import Path
            config_path = Path(__file__).parent / 'skin_detection_config.json'
            if config_path.exists():
                print(f"Loading config from: {config_path}")
            else:
                print("⚠️  No saved config found! Consider running calibration.")
            print("=" * 70)
        else:
            calibrated = mediapipe_based_calibration()
            if not calibrated:
                print("\nProceeding with saved calibration values...")
            else:
                print("\nCalibration successful! CV detector updated.")
    
    # Launch main app
    root = tk.Tk()
    app = GesturePaintApp(root, detection_mode=detection_mode, show_debug=debug_mode)
    root.mainloop()

if __name__ == "__main__":
    main()
