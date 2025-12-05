"""
Enhanced Demo Tool for Hand Detection
Features:
1. CV vs MediaPipe comparison (side-by-side)
2. Gesture recognition live demo (with thumb visualization)
3. Live test with detector switching
4. Edge detection demo
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
    print("CV vs MEDIAPIPE COMPARISON")
    print("=" * 70)
    print("Left: Enhanced CV | Right: MediaPipe with Gestures")
    print("Press 'q' to quit, 'r' to reset CV background")
    print("=" * 70)
    
    cap = find_camera()
    if not cap:
        print("Could not open camera")
        return
    
    setup_camera(cap, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
    
    cv_detector = CVDetector()
    mp_detector = MediaPipeDetector()
    
    fps_cv = FPSCounter()
    fps_mp = FPSCounter()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_cv = frame.copy()
        frame_mp = frame.copy()
        
        result_cv = cv_detector.process_frame(frame_cv)
        result_mp = mp_detector.process_frame(frame_mp)
        
        fps_cv.update()
        fps_mp.update()
        
        # Add headers and info
        cv2.putText(result_cv['annotated_frame'], f"CV Enhanced | FPS: {fps_cv.get_fps()}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(result_mp['annotated_frame'], f"MediaPipe | FPS: {fps_mp.get_fps()}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if result_cv['detected']:
            cv2.putText(result_cv['annotated_frame'], f"Fingers: {result_cv['finger_count']}", 
                       (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        combined = np.hstack([result_cv['annotated_frame'], result_mp['annotated_frame']])
        
        # Add divider line
        h = combined.shape[0]
        cv2.line(combined, (640, 0), (640, h), (255, 255, 255), 2)
        
        cv2.imshow('CV vs MediaPipe Comparison', combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            cv_detector.reset_background()
            print("CV background reset")
    
    cap.release()
    cv2.destroyAllWindows()
    cv_detector.cleanup()
    mp_detector.cleanup()

def gesture_demo():
    print("\n" + "=" * 70)
    print("GESTURE RECOGNITION DEMO")
    print("=" * 70)
    print("Shows real-time gesture detection with thumb visualization")
    print("Gestures: Draw (1 finger), Erase (2 fingers), Clear (5 fingers)")
    print("Press 'q' to quit")
    print("=" * 70)
    
    cap = find_camera()
    if not cap:
        print("Could not open camera")
        return
    
    setup_camera(cap, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
    
    mp_detector = MediaPipeDetector()
    fps_counter = FPSCounter()
    
    gesture_colors = {
        'Draw': (0, 255, 0),
        'Erase': (0, 165, 255),
        'Clear': (0, 0, 255),
        'None': (128, 128, 128)
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = mp_detector.process_frame(frame)
        fps = fps_counter.update()
        
        gesture = result.get('gesture', 'None')
        color = gesture_colors.get(gesture, (255, 255, 255))
        
        # Add large gesture indicator
        if result['detected']:
            cv2.rectangle(result['annotated_frame'], (10, 400), (630, 470), (0, 0, 0), -1)
            cv2.rectangle(result['annotated_frame'], (10, 400), (630, 470), color, 3)
            cv2.putText(result['annotated_frame'], f"GESTURE: {gesture}", 
                       (20, 445), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
        
        cv2.putText(result['annotated_frame'], f"FPS: {int(fps)}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Gesture Recognition Demo', result['annotated_frame'])
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    mp_detector.cleanup()

def live_test():
    print("\n" + "=" * 70)
    print("LIVE DETECTOR SWITCHING TEST")
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
        mode_color = (255, 255, 0) if current_mode == 'mediapipe' else (0, 255, 255)
        
        # Header bar
        cv2.rectangle(result['annotated_frame'], (0, 0), (640, 50), (0, 0, 0), -1)
        cv2.putText(result['annotated_frame'], f"{mode_label} | FPS: {int(fps)}", 
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        
        # Info overlay
        if result['detected']:
            info_y = 420
            cv2.rectangle(result['annotated_frame'], (5, info_y-5), (250, 475), (0, 0, 0), -1)
            cv2.putText(result['annotated_frame'], f"Fingers: {result['finger_count']}", 
                       (10, info_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result['annotated_frame'], f"X: {result['hand_x']:.2f}", 
                       (10, info_y+45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(result['annotated_frame'], f"Y: {result['hand_y']:.2f}", 
                       (130, info_y+45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            if current_mode == 'mediapipe':
                gesture = result.get('gesture', 'None')
                cv2.putText(result['annotated_frame'], f"Gesture: {gesture}", 
                           (10, info_y+70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(result['annotated_frame'], "No hand detected", 
                       (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow('Live Detector Test', result['annotated_frame'])
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_mode = 'mediapipe'
            fps_counter = FPSCounter()
            print("\nSwitched to: MEDIAPIPE")
        elif key == ord('2'):
            current_mode = 'cv'
            fps_counter = FPSCounter()
            print("\nSwitched to: CV ENHANCED")
        elif key == ord('r') and current_mode == 'cv':
            cv_detector.reset_background()
            print("Background reset")
    
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
    
    cap = find_camera()
    if not cap:
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
    print("ENHANCED HAND DETECTION DEMO")
    print("=" * 70)
    print("\n1. CV vs MediaPipe Comparison (side-by-side)")
    print("2. Gesture Recognition Demo (with thumb visualization)")
    print("3. Live Detector Switching Test")
    print("4. Edge Detection Demo")
    print("5. Exit")
    print("=" * 70)
    
    choice = input("\nChoice (1-5): ").strip()
    
    if choice == '1':
        compare_detectors()
    elif choice == '2':
        gesture_demo()
    elif choice == '3':
        live_test()
    elif choice == '4':
        edge_detection_demo()
    elif choice == '5':
        return
    else:
        print("Invalid choice")
        return
    
    # Ask to continue
    print("\n" + "=" * 70)
    again = input("Run another demo? (y/n): ").strip().lower()
    if again == 'y':
        main()

if __name__ == "__main__":
    main()
