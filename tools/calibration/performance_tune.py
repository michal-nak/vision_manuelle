"""Performance tuning mode - interactive adjustment of all parameters with FPS monitoring"""

import cv2
import numpy as np
import time
from collections import deque
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.detectors.cv.cv_detector import CVDetector
from .ui_display import draw_performance_info, show_masks_comparison


def nothing(x):
    """Empty callback for trackbars"""
    pass


def performance_tuning(cap):
    """
    Interactive performance tuning mode with all parameters
    Real-time FPS monitoring and bottleneck detection
    Returns calibration dictionary when user saves
    """
    print("\n" + "=" * 70)
    print("PERFORMANCE TUNING MODE")
    print("=" * 70)
    print("Adjust detection parameters in real-time")
    print("=" * 70)
    
    detector = CVDetector()
    
    cv2.namedWindow('Detection')
    cv2.namedWindow('Masks')
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Controls', 400, 700)
    
    # Color range trackbars
    cv2.createTrackbar('Y_min', 'Controls', 99, 255, nothing)
    cv2.createTrackbar('Y_max', 'Controls', 184, 255, nothing)
    cv2.createTrackbar('Cr_min', 'Controls', 127, 255, nothing)
    cv2.createTrackbar('Cr_max', 'Controls', 164, 255, nothing)
    cv2.createTrackbar('Cb_min', 'Controls', 28, 255, nothing)
    cv2.createTrackbar('Cb_max', 'Controls', 133, 255, nothing)
    cv2.createTrackbar('H_min', 'Controls', 2, 180, nothing)
    cv2.createTrackbar('H_max', 'Controls', 35, 180, nothing)
    cv2.createTrackbar('S_min', 'Controls', 34, 255, nothing)
    cv2.createTrackbar('S_max', 'Controls', 255, 255, nothing)
    cv2.createTrackbar('V_min', 'Controls', 107, 255, nothing)
    cv2.createTrackbar('V_max', 'Controls', 255, 255, nothing)
    
    # Processing parameter trackbars
    cv2.createTrackbar('Denoise', 'Controls', 10, 30, nothing)
    cv2.createTrackbar('Morph_small', 'Controls', 5, 15, nothing)
    cv2.createTrackbar('Morph_large', 'Controls', 11, 25, nothing)
    cv2.createTrackbar('Open_iter', 'Controls', 2, 5, nothing)
    cv2.createTrackbar('Close_iter', 'Controls', 3, 7, nothing)
    cv2.createTrackbar('Min_area', 'Controls', 30, 100, nothing)  # x100
    cv2.createTrackbar('BG_threshold', 'Controls', 16, 50, nothing)
    cv2.createTrackbar('Smooth_pos', 'Controls', 5, 10, nothing)
    cv2.createTrackbar('Smooth_fing', 'Controls', 3, 7, nothing)
    
    print("\nControls:")
    print("  Adjust trackbars to tune detection")
    print("  's' = save current settings")
    print("  'r' = reset background subtractor")
    print("  'd' = toggle debug info")
    print("  'q' = quit")
    print("=" * 70)
    
    show_debug = False
    fps_history = deque(maxlen=30)
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get all parameter values
        params = {
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
            'v_max': cv2.getTrackbarPos('V_max', 'Controls'),
            'denoise': max(1, cv2.getTrackbarPos('Denoise', 'Controls')),
            'morph_small': max(3, cv2.getTrackbarPos('Morph_small', 'Controls')),
            'morph_large': max(5, cv2.getTrackbarPos('Morph_large', 'Controls')),
            'open_iter': max(1, cv2.getTrackbarPos('Open_iter', 'Controls')),
            'close_iter': max(1, cv2.getTrackbarPos('Close_iter', 'Controls')),
            'min_area': cv2.getTrackbarPos('Min_area', 'Controls') * 100,
            'bg_threshold': max(1, cv2.getTrackbarPos('BG_threshold', 'Controls')),
            'smooth_pos': max(1, cv2.getTrackbarPos('Smooth_pos', 'Controls')),
            'smooth_fing': max(1, cv2.getTrackbarPos('Smooth_fing', 'Controls'))
        }
        
        # Apply tuned denoising
        t_start = time.perf_counter()
        if params['denoise'] > 0:
            denoised = cv2.fastNlMeansDenoisingColored(frame, None, params['denoise'], params['denoise'], 7, 21)
        else:
            denoised = frame.copy()
        t_denoise = (time.perf_counter() - t_start) * 1000
        
        # Apply color detection
        t_start = time.perf_counter()
        ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, 
                                  np.array([params['y_min'], params['cr_min'], params['cb_min']], dtype=np.uint8),
                                  np.array([params['y_max'], params['cr_max'], params['cb_max']], dtype=np.uint8))
        
        hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
        
        # Handle HSV hue wrap-around (e.g., 170-180 and 0-10 for red/pink skin tones)
        hsv_lower = np.array([params['h_min'], params['s_min'], params['v_min']], dtype=np.uint8)
        hsv_upper = np.array([params['h_max'], params['s_max'], params['v_max']], dtype=np.uint8)
        
        if params['h_min'] > params['h_max']:
            # Wrapped range: combine two masks
            mask_hsv1 = cv2.inRange(hsv, hsv_lower, np.array([180, params['s_max'], params['v_max']], dtype=np.uint8))
            mask_hsv2 = cv2.inRange(hsv, np.array([0, params['s_min'], params['v_min']], dtype=np.uint8), hsv_upper)
            mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
        else:
            # Normal range
            mask_hsv = cv2.inRange(hsv, hsv_lower, hsv_upper)
        
        mask_combined = cv2.bitwise_and(mask_ycrcb, mask_hsv)
        t_color = (time.perf_counter() - t_start) * 1000
        
        # Apply tuned morphology
        t_start = time.perf_counter()
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                  (params['morph_small'], params['morph_small']))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                  (params['morph_large'], params['morph_large']))
        
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel_small, 
                                         iterations=params['open_iter'])
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel_large,
                                         iterations=params['close_iter'])
        mask_combined = cv2.GaussianBlur(mask_combined, (5, 5), 0)
        t_morph = (time.perf_counter() - t_start) * 1000
        
        # Find contours with tuned area threshold
        t_start = time.perf_counter()
        contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        annotated = frame.copy()
        hand_detected = False
        
        if contours:
            # Filter by area
            valid_contours = [c for c in contours if cv2.contourArea(c) > params['min_area']]
            if valid_contours:
                hand_contour = max(valid_contours, key=cv2.contourArea)
                cv2.drawContours(annotated, [hand_contour], -1, (0, 255, 0), 2)
                hull = cv2.convexHull(hand_contour)
                cv2.drawContours(annotated, [hull], -1, (255, 0, 0), 2)
                
                M = cv2.moments(hand_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(annotated, (cx, cy), 10, (0, 255, 0), -1)
                    hand_detected = True
        
        t_contour = (time.perf_counter() - t_start) * 1000
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
        prev_time = curr_time
        fps_history.append(fps)
        avg_fps = np.mean(fps_history)
        
        # Display info
        total_time = t_denoise + t_color + t_morph + t_contour
        
        draw_performance_info(annotated, avg_fps, total_time, hand_detected, show_debug, 
                            t_denoise, t_color, t_morph, t_contour)
        
        cv2.imshow('Detection', annotated)
        
        # Show mask comparison
        masks = show_masks_comparison(mask_ycrcb, mask_hsv, mask_combined)
        cv2.imshow('Masks', masks)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save optimized settings
            calibration = {
                'ycrcb_lower': np.array([params['y_min'], params['cr_min'], params['cb_min']]),
                'ycrcb_upper': np.array([params['y_max'], params['cr_max'], params['cb_max']]),
                'hsv_lower': np.array([params['h_min'], params['s_min'], params['v_min']]),
                'hsv_upper': np.array([params['h_max'], params['s_max'], params['v_max']])
            }
            
            print(f"\nSaving optimized settings:")
            print(f"   FPS: {int(avg_fps)}")
            print(f"   Processing time: {total_time:.1f}ms")
            print(f"   Denoise strength: {params['denoise']}")
            print(f"   Morphology kernels: {params['morph_small']}, {params['morph_large']}")
            print(f"   Iterations: open={params['open_iter']}, close={params['close_iter']}")
            print(f"   Min area: {params['min_area']}")
            print(f"   Smoothing: pos={params['smooth_pos']}, fingers={params['smooth_fing']}")
            
            cv2.destroyAllWindows()
            return calibration
        elif key == ord('r'):
            detector.reset_background()
            print("Background reset")
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"Debug mode: {'ON' if show_debug else 'OFF'}")
    
    cv2.destroyAllWindows()
    return None
