"""Manual tuning interface with trackbars for real-time calibration"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.detectors.cv.cv_detector import CVDetector
from .ui_display import show_masks_comparison
from .auto_calibrate import auto_calibrate


def nothing(x):
    """Empty callback for trackbars"""
    pass


def manual_tune(cap, initial_calibration=None):
    """
    Manual tuning mode with trackbars for all color parameters
    Displays live detection and mask comparison
    Returns calibration dictionary when user presses 's' to save
    """
    detector = CVDetector()
    
    cv2.namedWindow('Detection')
    cv2.namedWindow('Masks')
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Controls', 400, 700)
    
    # Initialize trackbar values
    if initial_calibration:
        vals = initial_calibration
        y_min, cr_min, cb_min = vals['ycrcb_lower']
        y_max, cr_max, cb_max = vals['ycrcb_upper']
        h_min, s_min, v_min = vals['hsv_lower']
        h_max, s_max, v_max = vals['hsv_upper']
    else:
        # Default values
        y_min, cr_min, cb_min, y_max, cr_max, cb_max = 0, 133, 77, 255, 173, 127
        h_min, s_min, v_min, h_max, s_max, v_max = 0, 30, 60, 20, 150, 255
    
    # Create trackbars
    cv2.createTrackbar('Y_min', 'Controls', int(y_min), 255, nothing)
    cv2.createTrackbar('Y_max', 'Controls', int(y_max), 255, nothing)
    cv2.createTrackbar('Cr_min', 'Controls', int(cr_min), 255, nothing)
    cv2.createTrackbar('Cr_max', 'Controls', int(cr_max), 255, nothing)
    cv2.createTrackbar('Cb_min', 'Controls', int(cb_min), 255, nothing)
    cv2.createTrackbar('Cb_max', 'Controls', int(cb_max), 255, nothing)
    cv2.createTrackbar('H_min', 'Controls', int(h_min), 180, nothing)
    cv2.createTrackbar('H_max', 'Controls', int(h_max), 180, nothing)
    cv2.createTrackbar('S_min', 'Controls', int(s_min), 255, nothing)
    cv2.createTrackbar('S_max', 'Controls', int(s_max), 255, nothing)
    cv2.createTrackbar('V_min', 'Controls', int(v_min), 255, nothing)
    cv2.createTrackbar('V_max', 'Controls', int(v_max), 255, nothing)
    
    print("\n" + "=" * 70)
    print("MANUAL TUNING")
    print("=" * 70)
    print("Controls: 's'=save, 'r'=reset bg, 'c'=recalibrate, 'q'=quit")
    print("=" * 70)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get trackbar values
        vals = {
            'y_min': cv2.getTrackbarPos('Y_min', 'Controls'),
            'y_max': cv2.getTrackbarPos('Y_max', 'Controls'),
            'cr_min': cv2.getTrackbarPos('Cr_min', 'Controls'),
            'cr_max': cv2.getTrackbarPos('Cr_max', 'Controls'),
            'cb_min': cv2.getTrackbarPos('Cb_min', 'Controls'),
            'cb_max': cv2.getTrackbarPos('Cb_max', 'Controls'),
            'h_min': cv2.getTrackbarPos('H_min', 'Controls'),
            'h_max': cv2.getTrackbarPos('H_max', 'Controls'),
            's_min': cv2.getTrackbarPos('S_min', 'Controls'),
            's_max': cv2.getTrackbarPos('S_max', 'Controls'),
            'v_min': cv2.getTrackbarPos('V_min', 'Controls'),
            'v_max': cv2.getTrackbarPos('V_max', 'Controls')
        }
        
        # Update detector with current values
        import numpy as np
        detector.update_calibration(
            np.array([vals['y_min'], vals['cr_min'], vals['cb_min']], dtype=np.uint8),
            np.array([vals['y_max'], vals['cr_max'], vals['cb_max']], dtype=np.uint8),
            np.array([vals['h_min'], vals['s_min'], vals['v_min']], dtype=np.uint8),
            np.array([vals['h_max'], vals['s_max'], vals['v_max']], dtype=np.uint8)
        )
        
        # Process frame
        result = detector.process_frame(frame)
        
        # Create individual masks for comparison
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(
            ycrcb, 
            np.array([vals['y_min'], vals['cr_min'], vals['cb_min']], dtype=np.uint8),
            np.array([vals['y_max'], vals['cr_max'], vals['cb_max']], dtype=np.uint8)
        )
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Handle HSV hue wrap-around (e.g., 170-180 and 0-10 for red/pink skin tones)
        hsv_lower = np.array([vals['h_min'], vals['s_min'], vals['v_min']], dtype=np.uint8)
        hsv_upper = np.array([vals['h_max'], vals['s_max'], vals['v_max']], dtype=np.uint8)
        
        if vals['h_min'] > vals['h_max']:
            # Wrapped range: combine two masks
            mask_hsv1 = cv2.inRange(hsv, hsv_lower, np.array([180, vals['s_max'], vals['v_max']], dtype=np.uint8))
            mask_hsv2 = cv2.inRange(hsv, np.array([0, vals['s_min'], vals['v_min']], dtype=np.uint8), hsv_upper)
            mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
        else:
            # Normal range
            mask_hsv = cv2.inRange(hsv, hsv_lower, hsv_upper)
        
        mask_combined = cv2.bitwise_and(mask_ycrcb, mask_hsv)
        
        # Display
        masks = show_masks_comparison(mask_ycrcb, mask_hsv, mask_combined)
        cv2.imshow('Detection', result['annotated_frame'])
        cv2.imshow('Masks', masks)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset_background()
            print("\nBackground reset")
        elif key == ord('c'):
            # Recalibrate
            cv2.destroyAllWindows()
            new_cal = auto_calibrate(cap)
            if new_cal:
                return manual_tune(cap, new_cal)
            cv2.namedWindow('Detection')
            cv2.namedWindow('Masks')
            cv2.namedWindow('Controls')
        elif key == ord('s'):
            # Save and return
            print("\n" + "=" * 70)
            print(f"YCrCb: [{vals['y_min']}, {vals['cr_min']}, {vals['cb_min']}] to [{vals['y_max']}, {vals['cr_max']}, {vals['cb_max']}]")
            print(f"HSV:   [{vals['h_min']}, {vals['s_min']}, {vals['v_min']}] to [{vals['h_max']}, {vals['s_max']}, {vals['v_max']}]")
            print("=" * 70)
            cv2.destroyAllWindows()
            return {
                'ycrcb_lower': np.array([vals['y_min'], vals['cr_min'], vals['cb_min']], dtype=np.uint8),
                'ycrcb_upper': np.array([vals['y_max'], vals['cr_max'], vals['cb_max']], dtype=np.uint8),
                'hsv_lower': np.array([vals['h_min'], vals['s_min'], vals['v_min']], dtype=np.uint8),
                'hsv_upper': np.array([vals['h_max'], vals['s_max'], vals['v_max']], dtype=np.uint8)
            }
    
    cv2.destroyAllWindows()
    return None
