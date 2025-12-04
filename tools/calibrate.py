"""
Unified Calibration Tool for CV Hand Detector
Combines quick calibration, tuning, and verification in one script
Press '1' for auto-calibrate, '2' for manual tune, '3' for verify
"""
import cv2
import numpy as np
import time
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detectors import CVDetector
from src.core.config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    CALIBRATION_DURATION, CALIBRATION_SAMPLE_INTERVAL,
    CALIBRATION_MIN_SAMPLES, CALIBRATION_RECT,
    CALIBRATION_FILE, CALIBRATION_BACKUP_FILE,
    YCRCB_LOWER, YCRCB_UPPER, HSV_LOWER, HSV_UPPER
)
from src.core.utils import find_camera, setup_camera, draw_text_with_background

def nothing(x):
    pass

def auto_calibrate(cap, duration=CALIBRATION_DURATION):
    print("\n" + "=" * 70)
    print("AUTO-CALIBRATION")
    print("=" * 70)
    print("Instructions: Position hand in box, press SPACE to start")
    print("=" * 70)
    
    rect_x, rect_y, rect_w, rect_h = CALIBRATION_RECT
    
    # Preview mode
    print("\nPREVIEW - Press SPACE to start, ESC to cancel")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 255), 3)
        cv2.putText(frame, "PREVIEW - Press SPACE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(frame, "Position hand in YELLOW BOX", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "ESC to cancel", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.imshow('Calibration', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            print("\n▶️  CALIBRATION STARTED!")
            break
        elif key == 27:
            print("\n⚠️  Cancelled")
            cv2.destroyAllWindows()
            return None
    
    ycrcb_samples, hsv_samples = [], []
    start_time, frame_count = time.time(), 0
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        
        remaining = duration - (time.time() - start_time)
        color = (0, 255, 0) if remaining > 1 else (0, 255, 255)
        
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), color, 3 if remaining > 1 else 5)
        
        # Progress bar
        progress = int(((duration - remaining) / duration) * 600)
        cv2.rectangle(frame, (20, 450), (20 + progress, 470), (0, 255, 0), -1)
        cv2.rectangle(frame, (20, 450), (620, 470), (255, 255, 255), 2)
        
        cv2.putText(frame, f"Calibrating: {remaining:.1f}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        cv2.putText(frame, "Move hand slowly in box", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        if frame_count % 3 == 0:
            roi = frame[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w]
            roi_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            sample_size = min(500, roi.shape[0] * roi.shape[1])
            indices = np.random.choice(roi.shape[0] * roi.shape[1], size=sample_size, replace=False)
            
            for idx in indices:
                ycrcb_samples.append(roi_ycrcb.reshape(-1, 3)[idx])
                hsv_samples.append(roi_hsv.reshape(-1, 3)[idx])
            
            cv2.putText(frame, f"Samples: {len(ycrcb_samples)}", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.imshow('Calibration', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            print("\nCancelled")
            cv2.destroyAllWindows()
            return None
        frame_count += 1
    
    cv2.destroyAllWindows()
    
    if len(ycrcb_samples) < 100:
        print(f"\nNot enough samples ({len(ycrcb_samples)})")
        return None
    
    ycrcb_samples, hsv_samples = np.array(ycrcb_samples), np.array(hsv_samples)
    
    ycrcb_lower = np.maximum(np.percentile(ycrcb_samples, 5, axis=0).astype(np.uint8) - [10, 15, 15], [0, 0, 0]).astype(np.uint8)
    ycrcb_upper = np.minimum(np.percentile(ycrcb_samples, 95, axis=0).astype(np.uint8) + [10, 15, 15], [255, 255, 255]).astype(np.uint8)
    hsv_lower = np.maximum(np.percentile(hsv_samples, 5, axis=0).astype(np.uint8) - [5, 20, 30], [0, 0, 0]).astype(np.uint8)
    hsv_upper = np.minimum(np.percentile(hsv_samples, 95, axis=0).astype(np.uint8) + [5, 20, 30], [180, 255, 255]).astype(np.uint8)
    
    print(f"\nCalibration complete! ({len(ycrcb_samples)} samples)")
    print(f"   YCrCb: {ycrcb_lower.tolist()} to {ycrcb_upper.tolist()}")
    print(f"   HSV: {hsv_lower.tolist()} to {hsv_upper.tolist()}")
    
    return {'ycrcb_lower': ycrcb_lower, 'ycrcb_upper': ycrcb_upper, 'hsv_lower': hsv_lower, 'hsv_upper': hsv_upper}

def manual_tune(cap, initial_calibration=None):
    detector = CVDetector()
    
    cv2.namedWindow('Detection')
    cv2.namedWindow('Masks')
    cv2.namedWindow('Controls')
    
    if initial_calibration:
        vals = initial_calibration
        y_min, cr_min, cb_min = vals['ycrcb_lower']
        y_max, cr_max, cb_max = vals['ycrcb_upper']
        h_min, s_min, v_min = vals['hsv_lower']
        h_max, s_max, v_max = vals['hsv_upper']
    else:
        y_min, cr_min, cb_min, y_max, cr_max, cb_max = 0, 133, 77, 255, 173, 127
        h_min, s_min, v_min, h_max, s_max, v_max = 0, 30, 60, 20, 150, 255
    
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
        
        detector.set_ycrcb_range([vals['y_min'], vals['cr_min'], vals['cb_min']], [vals['y_max'], vals['cr_max'], vals['cb_max']])
        detector.set_hsv_range([vals['h_min'], vals['s_min'], vals['v_min']], [vals['h_max'], vals['s_max'], vals['v_max']])
        
        result = detector.process_frame(frame)
        
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, np.array([vals['y_min'], vals['cr_min'], vals['cb_min']], dtype=np.uint8),
                                  np.array([vals['y_max'], vals['cr_max'], vals['cb_max']], dtype=np.uint8))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, np.array([vals['h_min'], vals['s_min'], vals['v_min']], dtype=np.uint8),
                                np.array([vals['h_max'], vals['s_max'], vals['v_max']], dtype=np.uint8))
        mask_combined = cv2.bitwise_and(mask_ycrcb, mask_hsv)
        
        masks = np.hstack([cv2.cvtColor(mask_ycrcb, cv2.COLOR_GRAY2BGR),
                           cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR),
                           cv2.cvtColor(mask_combined, cv2.COLOR_GRAY2BGR)])
        cv2.putText(masks, "YCrCb", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(masks, "HSV", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(masks, "Combined", (430, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Detection', result['annotated_frame'])
        cv2.imshow('Masks', masks)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset_background()
            print("\n✅ Background reset")
        elif key == ord('c'):
            cv2.destroyAllWindows()
            new_cal = auto_calibrate(cap)
            if new_cal:
                return manual_tune(cap, new_cal)
            cv2.namedWindow('Detection')
            cv2.namedWindow('Masks')
            cv2.namedWindow('Controls')
        elif key == ord('s'):
            print("\n" + "=" * 70)
            print(f"YCrCb: [{vals['y_min']}, {vals['cr_min']}, {vals['cb_min']}] to [{vals['y_max']}, {vals['cr_max']}, {vals['cb_max']}]")
            print(f"HSV:   [{vals['h_min']}, {vals['s_min']}, {vals['v_min']}] to [{vals['h_max']}, {vals['s_max']}, {vals['v_max']}]")
            print("=" * 70)
            return {
                'ycrcb_lower': np.array([vals['y_min'], vals['cr_min'], vals['cb_min']], dtype=np.uint8),
                'ycrcb_upper': np.array([vals['y_max'], vals['cr_max'], vals['cb_max']], dtype=np.uint8),
                'hsv_lower': np.array([vals['h_min'], vals['s_min'], vals['v_min']], dtype=np.uint8),
                'hsv_upper': np.array([vals['h_max'], vals['s_max'], vals['v_max']], dtype=np.uint8)
            }
    
    cv2.destroyAllWindows()
    return None

def auto_optimize(cap, base_calibration=None):
    print("\n" + "=" * 70)
    print("AUTO-OPTIMIZATION MODE")
    print("=" * 70)
    print("Testing parameter combinations to maximize FPS and detection quality...")
    print("This will take ~2 minutes. Position your hand in frame.")
    print("=" * 70)
    
    if base_calibration is None:
        print("\nFirst, let's calibrate color ranges...")
        base_calibration = auto_calibrate(cap)
        if not base_calibration:
            print("❌ Calibration cancelled")
            return None
    
    best_config = {
        'calibration': base_calibration,
        'denoise_h': 10,
        'kernel_small': 5,
        'kernel_large': 11,
        'morph_iter': 2,
        'min_area': 3000,
        'fps': 0,
        'quality_score': 0
    }
    
    # Parameter ranges to test
    test_configs = [
        # Fast (prioritize FPS)
        {'denoise_h': 5, 'kernel_small': 3, 'kernel_large': 7, 'morph_iter': 1, 'min_area': 4000, 'name': 'Fast'},
        # Balanced
        {'denoise_h': 7, 'kernel_small': 5, 'kernel_large': 9, 'morph_iter': 2, 'min_area': 3000, 'name': 'Balanced'},
        # Quality (prioritize detection)
        {'denoise_h': 10, 'kernel_small': 5, 'kernel_large': 11, 'morph_iter': 2, 'min_area': 2500, 'name': 'Quality'},
        # Ultra Fast
        {'denoise_h': 3, 'kernel_small': 3, 'kernel_large': 7, 'morph_iter': 1, 'min_area': 5000, 'name': 'Ultra-Fast'},
        # Custom variants
        {'denoise_h': 7, 'kernel_small': 3, 'kernel_large': 9, 'morph_iter': 1, 'min_area': 3500, 'name': 'Fast+Quality'},
        {'denoise_h': 5, 'kernel_small': 5, 'kernel_large': 9, 'morph_iter': 2, 'min_area': 3000, 'name': 'Smooth'},
    ]
    
    results = []
    
    for idx, config in enumerate(test_configs):
        print(f"\n[{idx+1}/{len(test_configs)}] Testing {config['name']} preset...")
        
        detector = CVDetector()
        detector.ycrcb_lower = base_calibration['ycrcb_lower']
        detector.ycrcb_upper = base_calibration['ycrcb_upper']
        detector.hsv_lower = base_calibration['hsv_lower']
        detector.hsv_upper = base_calibration['hsv_upper']
        detector.denoise_h = config['denoise_h']
        detector.kernel_small = np.ones((config['kernel_small'], config['kernel_small']), np.uint8)
        detector.kernel_large = np.ones((config['kernel_large'], config['kernel_large']), np.uint8)
        detector.morph_iterations = config['morph_iter']
        detector.min_contour_area = config['min_area']
        
        # Test for 3 seconds
        fps_samples = []
        detection_count = 0
        total_frames = 0
        start_time = time.time()
        
        while time.time() - start_time < 3.0:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            result = detector.process_frame(frame)
            
            if result['detected']:
                detection_count += 1
            total_frames += 1
            
            frame_time = time.time() - frame_start
            if frame_time > 0:
                fps_samples.append(1.0 / frame_time)
            
            cv2.imshow('Optimizing', result['annotated_frame'])
            cv2.waitKey(1)
        
        avg_fps = np.mean(fps_samples) if fps_samples else 0
        detection_rate = (detection_count / total_frames * 100) if total_frames > 0 else 0
        
        # Quality score: balance FPS and detection rate
        quality_score = (avg_fps * 0.4) + (detection_rate * 0.6)
        
        result_entry = {
            'config': config,
            'fps': avg_fps,
            'detection_rate': detection_rate,
            'quality_score': quality_score
        }
        results.append(result_entry)
        
        print(f"  → FPS: {avg_fps:.1f} | Detection: {detection_rate:.1f}% | Score: {quality_score:.1f}")
        
        if quality_score > best_config['quality_score']:
            best_config.update({
                'denoise_h': config['denoise_h'],
                'kernel_small': config['kernel_small'],
                'kernel_large': config['kernel_large'],
                'morph_iter': config['morph_iter'],
                'min_area': config['min_area'],
                'fps': avg_fps,
                'quality_score': quality_score,
                'detection_rate': detection_rate,
                'name': config['name']
            })
    
    cv2.destroyAllWindows()
    
    # Display results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    results.sort(key=lambda x: x['quality_score'], reverse=True)
    for idx, r in enumerate(results):
        marker = "★ BEST" if r['config']['name'] == best_config['name'] else ""
        print(f"{idx+1}. {r['config']['name']:12} | FPS: {r['fps']:5.1f} | Detection: {r['detection_rate']:5.1f}% | Score: {r['quality_score']:5.1f} {marker}")
    
    print("\n" + "=" * 70)
    print(f"🏆 WINNER: {best_config['name']}")
    print(f"   FPS: {best_config['fps']:.1f} | Detection: {best_config['detection_rate']:.1f}%")
    print("=" * 70)
    print(f"   Denoise: {best_config['denoise_h']}")
    print(f"   Kernel Small: {best_config['kernel_small']}x{best_config['kernel_small']}")
    print(f"   Kernel Large: {best_config['kernel_large']}x{best_config['kernel_large']}")
    print(f"   Morph Iterations: {best_config['morph_iter']}")
    print(f"   Min Area: {best_config['min_area']}")
    print("=" * 70)
    
    return best_config

def save_calibration(calibration):
    backup = {
        'timestamp': datetime.now().isoformat(),
        'ycrcb_lower': calibration['ycrcb_lower'].tolist(),
        'ycrcb_upper': calibration['ycrcb_upper'].tolist(),
        'hsv_lower': calibration['hsv_lower'].tolist(),
        'hsv_upper': calibration['hsv_upper'].tolist()
    }
    backup_path = Path(__file__).parent.parent / 'calibration_backup.json'
    with open(backup_path, 'w') as f:
        json.dump(backup, f, indent=2)
    print(f"\n Backup saved to {backup_path}")
    
    detector_path = Path(__file__).parent.parent / 'src' / 'detectors' / 'cv_detector.py'
    with open(detector_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Check if this is an optimized config with processing parameters
    has_processing_params = 'denoise_h' in calibration
    
    updated_lines = []
    for line in lines:
        if 'self.ycrcb_lower = np.array' in line:
            updated_lines.append(f'        self.ycrcb_lower = np.array([{calibration["ycrcb_lower"][0]}, {calibration["ycrcb_lower"][1]}, {calibration["ycrcb_lower"][2]}], dtype=np.uint8)\n')
        elif 'self.ycrcb_upper = np.array' in line:
            updated_lines.append(f'        self.ycrcb_upper = np.array([{calibration["ycrcb_upper"][0]}, {calibration["ycrcb_upper"][1]}, {calibration["ycrcb_upper"][2]}], dtype=np.uint8)\n')
        elif 'self.hsv_lower = np.array' in line:
            updated_lines.append(f'        self.hsv_lower = np.array([{calibration["hsv_lower"][0]}, {calibration["hsv_lower"][1]}, {calibration["hsv_lower"][2]}], dtype=np.uint8)\n')
        elif 'self.hsv_upper = np.array' in line:
            updated_lines.append(f'        self.hsv_upper = np.array([{calibration["hsv_upper"][0]}, {calibration["hsv_upper"][1]}, {calibration["hsv_upper"][2]}], dtype=np.uint8)\n')
        elif has_processing_params and 'self.denoise_h =' in line:
            updated_lines.append(f'        self.denoise_h = {calibration["denoise_h"]}\n')
        elif has_processing_params and 'self.kernel_small = np.ones' in line:
            updated_lines.append(f'        self.kernel_small = np.ones(({calibration["kernel_small"]}, {calibration["kernel_small"]}), np.uint8)\n')
        elif has_processing_params and 'self.kernel_large = np.ones' in line:
            updated_lines.append(f'        self.kernel_large = np.ones(({calibration["kernel_large"]}, {calibration["kernel_large"]}), np.uint8)\n')
        elif has_processing_params and 'self.morph_iterations =' in line:
            updated_lines.append(f'        self.morph_iterations = {calibration["morph_iter"]}\n')
        elif has_processing_params and 'self.min_contour_area =' in line:
            updated_lines.append(f'        self.min_contour_area = {calibration["min_area"]}\n')
        else:
            updated_lines.append(line)
    
    with open(detector_path, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    print(f"✅ {detector_path.name} updated!")

def verify_calibration():
    print("\n" + "=" * 70)
    print("CALIBRATION STATUS")
    print("=" * 70)
    
    detector_path = Path(__file__).parent.parent / 'src' / 'detectors' / 'cv_detector.py'
    with open(detector_path, 'r') as f:
        content = f.read()
    
    ycrcb_l = re.search(r'self\.ycrcb_lower = np\.array\(\[(\d+), (\d+), (\d+)\]', content)
    ycrcb_u = re.search(r'self\.ycrcb_upper = np\.array\(\[(\d+), (\d+), (\d+)\]', content)
    hsv_l = re.search(r'self\.hsv_lower = np\.array\(\[(\d+), (\d+), (\d+)\]', content)
    hsv_u = re.search(r'self\.hsv_upper = np\.array\(\[(\d+), (\d+), (\d+)\]', content)
    
    if all([ycrcb_l, ycrcb_u, hsv_l, hsv_u]):
        ycrcb_l_vals = [int(x) for x in ycrcb_l.groups()]
        ycrcb_u_vals = [int(x) for x in ycrcb_u.groups()]
        hsv_l_vals = [int(x) for x in hsv_l.groups()]
        hsv_u_vals = [int(x) for x in hsv_u.groups()]
        
        print("Current values in cv_detector.py:")
        print(f"  YCrCb: {ycrcb_l_vals} to {ycrcb_u_vals}")
        print(f"  HSV:   {hsv_l_vals} to {hsv_u_vals}")
        
        is_default = (ycrcb_l_vals == [0, 133, 77] and ycrcb_u_vals == [255, 173, 127] and
                     hsv_l_vals == [0, 30, 60] and hsv_u_vals == [20, 150, 255])
        
        if is_default:
            print("\n Using DEFAULT values (not calibrated)")
        else:
            print("\n Using CALIBRATED values")
    
    backup_path = Path(__file__).parent.parent / 'calibration_backup.json'
    if backup_path.exists():
        with open(backup_path, 'r') as f:
            backup = json.load(f)
        print(f"\nBackup found: {backup.get('timestamp', 'unknown')}")
    else:
        print("\n  No backup file")
    print("=" * 70)

def performance_tuning(cap):
    """Interactive performance tuning mode with all parameters"""
    print("\n" + "=" * 70)
    print("PERFORMANCE TUNING MODE")
    print("=" * 70)
    print("Adjust detection parameters in real-time")
    print("=" * 70)
    
    detector = CVDetector()
    
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Performance', cv2.WINDOW_NORMAL)
    
    # Color range trackbars
    cv2.createTrackbar('Y_min', 'Performance', 99, 255, nothing)
    cv2.createTrackbar('Y_max', 'Performance', 184, 255, nothing)
    cv2.createTrackbar('Cr_min', 'Performance', 127, 255, nothing)
    cv2.createTrackbar('Cr_max', 'Performance', 164, 255, nothing)
    cv2.createTrackbar('Cb_min', 'Performance', 28, 255, nothing)
    cv2.createTrackbar('Cb_max', 'Performance', 133, 255, nothing)
    cv2.createTrackbar('H_min', 'Performance', 2, 180, nothing)
    cv2.createTrackbar('H_max', 'Performance', 35, 180, nothing)
    cv2.createTrackbar('S_min', 'Performance', 34, 255, nothing)
    cv2.createTrackbar('S_max', 'Performance', 255, 255, nothing)
    cv2.createTrackbar('V_min', 'Performance', 107, 255, nothing)
    cv2.createTrackbar('V_max', 'Performance', 255, 255, nothing)
    
    # Processing parameter trackbars
    cv2.createTrackbar('Denoise', 'Performance', 10, 30, nothing)
    cv2.createTrackbar('Morph_small', 'Performance', 5, 15, nothing)
    cv2.createTrackbar('Morph_large', 'Performance', 11, 25, nothing)
    cv2.createTrackbar('Open_iter', 'Performance', 2, 5, nothing)
    cv2.createTrackbar('Close_iter', 'Performance', 3, 7, nothing)
    cv2.createTrackbar('Min_area', 'Performance', 30, 100, nothing)  # x100
    cv2.createTrackbar('BG_threshold', 'Performance', 16, 50, nothing)
    cv2.createTrackbar('Smooth_pos', 'Performance', 5, 10, nothing)
    cv2.createTrackbar('Smooth_fing', 'Performance', 3, 7, nothing)
    
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
            'y_min': cv2.getTrackbarPos('Y_min', 'Performance'),
            'y_max': cv2.getTrackbarPos('Y_max', 'Performance'),
            'cr_min': cv2.getTrackbarPos('Cr_min', 'Performance'),
            'cr_max': cv2.getTrackbarPos('Cr_max', 'Performance'),
            'cb_min': cv2.getTrackbarPos('Cb_min', 'Performance'),
            'cb_max': cv2.getTrackbarPos('Cb_max', 'Performance'),
            'h_min': cv2.getTrackbarPos('H_min', 'Performance'),
            'h_max': cv2.getTrackbarPos('H_max', 'Performance'),
            's_min': cv2.getTrackbarPos('S_min', 'Performance'),
            's_max': cv2.getTrackbarPos('S_max', 'Performance'),
            'v_min': cv2.getTrackbarPos('V_min', 'Performance'),
            'v_max': cv2.getTrackbarPos('V_max', 'Performance'),
            'denoise': max(1, cv2.getTrackbarPos('Denoise', 'Performance')),
            'morph_small': max(3, cv2.getTrackbarPos('Morph_small', 'Performance')),
            'morph_large': max(5, cv2.getTrackbarPos('Morph_large', 'Performance')),
            'open_iter': max(1, cv2.getTrackbarPos('Open_iter', 'Performance')),
            'close_iter': max(1, cv2.getTrackbarPos('Close_iter', 'Performance')),
            'min_area': cv2.getTrackbarPos('Min_area', 'Performance') * 100,
            'bg_threshold': max(1, cv2.getTrackbarPos('BG_threshold', 'Performance')),
            'smooth_pos': max(1, cv2.getTrackbarPos('Smooth_pos', 'Performance')),
            'smooth_fing': max(1, cv2.getTrackbarPos('Smooth_fing', 'Performance'))
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
        mask_hsv = cv2.inRange(hsv,
                               np.array([params['h_min'], params['s_min'], params['v_min']], dtype=np.uint8),
                               np.array([params['h_max'], params['s_max'], params['v_max']], dtype=np.uint8))
        
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
        
        cv2.putText(annotated, f"FPS: {int(avg_fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"Processing: {total_time:.1f}ms", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, "Detected" if hand_detected else "Not detected", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if hand_detected else (0, 0, 255), 2)
        
        if show_debug:
            y_off = 120
            cv2.putText(annotated, f"Denoise: {t_denoise:.1f}ms", (10, y_off), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated, f"Color: {t_color:.1f}ms", (10, y_off + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated, f"Morph: {t_morph:.1f}ms", (10, y_off + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated, f"Contour: {t_contour:.1f}ms", (10, y_off + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show which operation is slowest
            timings = [('Denoise', t_denoise), ('Color', t_color), ('Morph', t_morph), ('Contour', t_contour)]
            slowest = max(timings, key=lambda x: x[1])
            cv2.putText(annotated, f"Bottleneck: {slowest[0]}", (10, y_off + 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        cv2.imshow('Detection', annotated)
        
        # Show mask comparison
        masks = np.hstack([
            cv2.cvtColor(mask_ycrcb, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(mask_combined, cv2.COLOR_GRAY2BGR)
        ])
        cv2.putText(masks, "YCrCb", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(masks, "HSV", (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(masks, "Combined", (430, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Performance', masks)
        
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
            
            print(f"\n✅ Saving optimized settings:")
            print(f"   FPS: {int(avg_fps)}")
            print(f"   Processing time: {total_time:.1f}ms")
            print(f"   Denoise strength: {params['denoise']}")
            print(f"   Morphology kernels: {params['morph_small']}, {params['morph_large']}")
            print(f"   Iterations: open={params['open_iter']}, close={params['close_iter']}")
            print(f"   Min area: {params['min_area']}")
            print(f"   Smoothing: pos={params['smooth_pos']}, fingers={params['smooth_fing']}")
            
            return calibration
        elif key == ord('r'):
            detector.reset_background()
            print("Background reset")
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"Debug mode: {'ON' if show_debug else 'OFF'}")
    
    return None

def main():
    print("=" * 70)
    print("UNIFIED CALIBRATION TOOL")
    print("=" * 70)
    print("\n1. Auto-Calibrate (5 seconds, recommended)")
    print("2. Manual Tuning (trackbars)")
    print("3. Performance Tuning (manual FPS optimization)")
    print("4. AUTO-OPTIMIZE (automatic best settings) ⭐ NEW")
    print("5. Verify Current Calibration")
    print("6. Exit")
    print("=" * 70)
    
    choice = input("\nChoice (1-6): ").strip()
    
    if choice == '5':
        verify_calibration()
        return
    elif choice == '6':
        return
    
    # Open camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    calibration = None
    
    if choice == '1':
        calibration = auto_calibrate(cap)
    elif choice == '2':
        calibration = manual_tune(cap)
    elif choice == '3':
        calibration = performance_tuning(cap)
    elif choice == '4':
        calibration = auto_optimize(cap)
    
    cap.release()
    cv2.destroyAllWindows()
    
    if calibration:
        save_choice = input("\nSave calibration? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_calibration(calibration)
            print("\nCalibration saved! Run gesture_paint.py to test.")

if __name__ == "__main__":
    main()
