"""
Unified demo/test script for CV hand detector
Includes: detector comparison, live test, edge detection demo
Press '1' for detector comparison, '2' for live test, '3' for edge detection
"""
import cv2
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detectors import CVDetector, MediaPipeDetector
from src.core.config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, HSV_LOWER, HSV_UPPER
from src.core.utils import find_camera, setup_camera, FPSCounter

def compare_detectors():
    print("\n" + "=" * 70)
    print("DETECTOR COMPARISON")
    print("=" * 70)
    print("Left: Simple HSV | Right: Enhanced CV (YCrCb+HSV+Background Sub)")
    print("Press 'q' to quit")
    print("=" * 70)
    
    cap = find_camera()
    if not cap:
        print("Could not open camera")
        return
    
    setup_camera(cap, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
    
    class OldCVDetector:
        def __init__(self):
            self.hsv_lower = np.array([0, 30, 60], dtype=np.uint8)
            self.hsv_upper = np.array([20, 150, 255], dtype=np.uint8)
        
        def process_frame(self, frame):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            annotated = frame.copy()
            detected = False
            
            if contours:
                hand_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(hand_contour) > 3000:
                    cv2.drawContours(annotated, [hand_contour], -1, (0, 255, 0), 2)
                    M = cv2.moments(hand_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(annotated, (cx, cy), 10, (0, 255, 0), -1)
                        detected = True
            
            return {'detected': detected, 'annotated_frame': annotated}
    
    old_detector = OldCVDetector()
    new_detector = CVDetector()
    
    fps_old = FPSCounter()
    fps_new = FPSCounter()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_old = frame.copy()
        frame_new = frame.copy()
        
        result_old = old_detector.process_frame(frame_old)
        result_new = new_detector.process_frame(frame_new)
        
        fps_old.update()
        fps_new.update()
        
        cv2.putText(result_old['annotated_frame'], f"Old HSV | FPS: {fps_old.get_fps()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(result_new['annotated_frame'], f"New Enhanced | FPS: {fps_new.get_fps()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        combined = np.hstack([result_old['annotated_frame'], result_new['annotated_frame']])
        cv2.imshow('Comparison', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    new_detector.cleanup()

def live_test():
    print("\n" + "=" * 70)
    print("LIVE DETECTOR TEST")
    print("=" * 70)
    print("Press '1' for MediaPipe, '2' for CV, 'r' to reset bg, 'q' to quit")
    print("=" * 70)
    
    cap = find_camera()
    if not cap:
        print("Could not open camera")
        return
    
    setup_camera(cap, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
    
    mp_detector = MediaPipeDetector()
    cv_detector = CVDetector()
    current_mode = 'mediapipe'
    
    fps_counter = FPSCounter()
    
    print(f"\nMode: {current_mode.upper()}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detector = mp_detector if current_mode == 'mediapipe' else cv_detector
        result = detector.process_frame(frame)
        
        fps = fps_counter.update()
        
        mode_label = "MediaPipe" if current_mode == 'mediapipe' else "CV Enhanced"
        cv2.putText(result['annotated_frame'], f"{mode_label} | FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        if result['detected']:
            cv2.putText(result['annotated_frame'], f"Fingers: {result['finger_count']}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(result['annotated_frame'], f"Pos: ({result['hand_x']:.0f}, {result['hand_y']:.0f})", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(result['annotated_frame'], "No hand detected", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        cv2.imshow('Live Test', result['annotated_frame'])
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_mode = 'mediapipe'
            print(f"\n Switched to: MEDIAPIPE")
        elif key == ord('2'):
            current_mode = 'cv'
            print(f"\n Switched to: CV ENHANCED")
        elif key == ord('r') and current_mode == 'cv':
            cv_detector.reset_background()
            print(f"\nBackground reset")
    
    cap.release()
    cv2.destroyAllWindows()
    mp_detector.cleanup()
    cv_detector.cleanup()

def edge_detection_demo():
    print("\n" + "=" * 70)
    print("EDGE DETECTION DEMO (Sobel)")
    print("=" * 70)
    print("Press 'q' to quit")
    print("=" * 70)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = np.uint8(magnitude * 255 / np.max(magnitude))
        
        edges_colored = cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)
        
        combined = np.hstack([frame, edges_colored])
        cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, "Sobel Edges", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('Edge Detection', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("=" * 70)
    print("UNIFIED DEMO/TEST SCRIPT")
    print("=" * 70)
    print("\n1. Detector Comparison (Old vs New)")
    print("2. Live Test (MediaPipe vs CV)")
    print("3. Edge Detection Demo")
    print("4. Exit")
    print("=" * 70)
    
    choice = input("\nChoice (1-4): ").strip()
    
    if choice == '1':
        compare_detectors()
    elif choice == '2':
        live_test()
    elif choice == '3':
        edge_detection_demo()
    elif choice == '4':
        return
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
