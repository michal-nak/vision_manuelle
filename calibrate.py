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
from datetime import datetime
from cv_detector import CVDetector
from config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    CALIBRATION_DURATION, CALIBRATION_SAMPLE_INTERVAL,
    CALIBRATION_MIN_SAMPLES, CALIBRATION_RECT,
    CALIBRATION_FILE, CALIBRATION_BACKUP_FILE,
    YCRCB_LOWER, YCRCB_UPPER, HSV_LOWER, HSV_UPPER
)
from utils import find_camera, setup_camera, draw_text_with_background

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
    print("\n⏸️  PREVIEW - Press SPACE to start, ESC to cancel")
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
            print("\n⚠️  Cancelled")
            cv2.destroyAllWindows()
            return None
        frame_count += 1
    
    cv2.destroyAllWindows()
    
    if len(ycrcb_samples) < 100:
        print(f"\n⚠️  Not enough samples ({len(ycrcb_samples)})")
        return None
    
    ycrcb_samples, hsv_samples = np.array(ycrcb_samples), np.array(hsv_samples)
    
    ycrcb_lower = np.maximum(np.percentile(ycrcb_samples, 5, axis=0).astype(np.uint8) - [10, 15, 15], [0, 0, 0]).astype(np.uint8)
    ycrcb_upper = np.minimum(np.percentile(ycrcb_samples, 95, axis=0).astype(np.uint8) + [10, 15, 15], [255, 255, 255]).astype(np.uint8)
    hsv_lower = np.maximum(np.percentile(hsv_samples, 5, axis=0).astype(np.uint8) - [5, 20, 30], [0, 0, 0]).astype(np.uint8)
    hsv_upper = np.minimum(np.percentile(hsv_samples, 95, axis=0).astype(np.uint8) + [5, 20, 30], [180, 255, 255]).astype(np.uint8)
    
    print(f"\n✅ Calibration complete! ({len(ycrcb_samples)} samples)")
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

def save_calibration(calibration):
    backup = {
        'timestamp': datetime.now().isoformat(),
        'ycrcb_lower': calibration['ycrcb_lower'].tolist(),
        'ycrcb_upper': calibration['ycrcb_upper'].tolist(),
        'hsv_lower': calibration['hsv_lower'].tolist(),
        'hsv_upper': calibration['hsv_upper'].tolist()
    }
    with open('calibration_backup.json', 'w') as f:
        json.dump(backup, f, indent=2)
    print("\n💾 Backup saved to calibration_backup.json")
    
    with open('cv_detector.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
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
        else:
            updated_lines.append(line)
    
    with open('cv_detector.py', 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    print("✅ cv_detector.py updated!")

def verify_calibration():
    print("\n" + "=" * 70)
    print("CALIBRATION STATUS")
    print("=" * 70)
    
    with open('cv_detector.py', 'r') as f:
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
            print("\n⚠️  Using DEFAULT values (not calibrated)")
        else:
            print("\n✅ Using CALIBRATED values")
    
    if os.path.exists('calibration_backup.json'):
        with open('calibration_backup.json', 'r') as f:
            backup = json.load(f)
        print(f"\nBackup found: {backup.get('timestamp', 'unknown')}")
    else:
        print("\n⚠️  No backup file")
    print("=" * 70)

def main():
    print("=" * 70)
    print("UNIFIED CALIBRATION TOOL")
    print("=" * 70)
    print("\n1. Auto-Calibrate (5 seconds, recommended)")
    print("2. Manual Tuning (trackbars)")
    print("3. Verify Current Calibration")
    print("4. Exit")
    print("=" * 70)
    
    choice = input("\nChoice (1-4): ").strip()
    
    if choice == '3':
        verify_calibration()
        return
    elif choice == '4':
        return
    
    # Open camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    calibration = None
    
    if choice == '1':
        calibration = auto_calibrate(cap)
    elif choice == '2':
        calibration = manual_tune(cap)
    
    cap.release()
    cv2.destroyAllWindows()
    
    if calibration:
        save_choice = input("\nSave calibration? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_calibration(calibration)
            print("\n🎉 Calibration saved! Run gesture_paint.py to test.")

if __name__ == "__main__":
    main()
