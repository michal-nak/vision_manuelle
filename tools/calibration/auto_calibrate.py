"""Auto calibration logic - collects hand samples and computes color ranges"""

import cv2
import numpy as np
from .ui_display import draw_progress_bar, draw_roi_box, print_calibration_results


def auto_calibrate(cap):
    """
    Automatic calibration by sampling hand skin in ROI
    User places hand in center box, samples collected for 5 seconds
    Returns calibration dictionary with color ranges
    """
    print("\n" + "=" * 70)
    print("AUTO-CALIBRATION MODE")
    print("=" * 70)
    print("Place hand in green box")
    print("Press SPACE to start (5s countdown)")
    print("Press ESC to cancel")
    print("=" * 70)
    
    # Preview mode
    while True:
        ret, frame = cap.read()
        if not ret:
            return None
        
        draw_roi_box(frame)
        cv2.imshow('Auto Calibrate', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        elif key == 27:  # ESC
            print("\nCancelled")
            cv2.destroyAllWindows()
            return None
    
    # Collect samples
    ycrcb_samples = []
    hsv_samples = []
    frame_count = 0
    total_frames = 150  # ~5 seconds at 30 FPS
    roi_size = 200
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        x = (w - roi_size) // 2
        y = (h - roi_size) // 2
        
        roi = frame[y:y+roi_size, x:x+roi_size]
        
        ycrcb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        ycrcb_samples.extend(ycrcb_roi.reshape(-1, 3).tolist())
        hsv_samples.extend(hsv_roi.reshape(-1, 3).tolist())
        
        progress = frame_count / total_frames
        draw_progress_bar(frame, progress, "Sampling hand...")
        draw_roi_box(frame)
        
        cv2.imshow('Auto Calibrate', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            print("\nCancelled")
            cv2.destroyAllWindows()
            return None
        frame_count += 1
    
    cv2.destroyAllWindows()
    
    if len(ycrcb_samples) < 100:
        print(f"\nNot enough samples ({len(ycrcb_samples)})")
        return None
    
    # Calculate color ranges from samples
    ycrcb_samples, hsv_samples = np.array(ycrcb_samples), np.array(hsv_samples)
    
    ycrcb_lower = np.maximum(
        np.percentile(ycrcb_samples, 5, axis=0).astype(np.uint8) - [10, 15, 15], 
        [0, 0, 0]
    ).astype(np.uint8)
    ycrcb_upper = np.minimum(
        np.percentile(ycrcb_samples, 95, axis=0).astype(np.uint8) + [10, 15, 15], 
        [255, 255, 255]
    ).astype(np.uint8)
    hsv_lower = np.maximum(
        np.percentile(hsv_samples, 5, axis=0).astype(np.uint8) - [5, 20, 30], 
        [0, 0, 0]
    ).astype(np.uint8)
    hsv_upper = np.minimum(
        np.percentile(hsv_samples, 95, axis=0).astype(np.uint8) + [5, 20, 30], 
        [180, 255, 255]
    ).astype(np.uint8)
    
    print_calibration_results(ycrcb_lower, ycrcb_upper, hsv_lower, hsv_upper, len(ycrcb_samples))
    
    return {
        'ycrcb_lower': ycrcb_lower, 
        'ycrcb_upper': ycrcb_upper, 
        'hsv_lower': hsv_lower, 
        'hsv_upper': hsv_upper
    }
