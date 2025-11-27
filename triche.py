import cv2
import mediapipe as mp
import time
import numpy as np
import ctypes

# Use Windows API for faster mouse movement
try:
    import ctypes
    user32 = ctypes.windll.user32
    
    def move_mouse(dx, dy):
        """Move mouse relative using Windows API (much faster than pyautogui)"""
        user32.mouse_event(1, int(dx), int(dy), 0, 0)  # MOUSEEVENTF_MOVE = 1
except:
    # Fallback to pyautogui if Windows API not available
    import pyautogui
    pyautogui.FAILSAFE = False
    def move_mouse(dx, dy):
        pyautogui.move(dx, dy, duration=0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Try to find available webcam
cap = None
for camera_index in range(5):
    print(f"Trying camera index {camera_index}...")
    test_cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if test_cap.isOpened():
        ret, frame = test_cap.read()
        if ret:
            print(f"Successfully opened camera at index {camera_index}")
            cap = test_cap
            break
        else:
            test_cap.release()
    else:
        test_cap.release()

if cap is None:
    print("Error: Could not open any webcam")
    print("Please check:")
    print("1. Camera is connected and not used by another application")
    print("2. Camera permissions are granted")
    print("3. Camera drivers are installed")
    exit()

print("Webcam opened successfully")
print("Press 'q' to quit")

# Debug mode - set to True to show timing information
DEBUG_MODE = False

# Set camera resolution (high res for display)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Processing resolution (downscaled for MediaPipe)
SQRT_SCALER = 2
PROCESS_WIDTH = CAMERA_WIDTH // SQRT_SCALER
PROCESS_HEIGHT = CAMERA_HEIGHT // SQRT_SCALER

CAMERA_CENTER_X = CAMERA_WIDTH // 2
CAMERA_CENTER_Y = CAMERA_HEIGHT // 2

# Mouse control deadzone
DEADZONE = 50

# FPS calculation variables
prev_time = 0
fps = 0

# Timing breakdown variables
timing_info = {
    "capture": 0,
    "preprocess": 0,
    "mediapipe": 0,
    "drawing": 0,
    "mouse": 0,
    "display": 0
}

# Configure hand detection
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
) as hands:
    
    while cap.isOpened():
        t0 = time.perf_counter()
        
        success, frame = cap.read()
        if not success:
            print("Failed to read from webcam")
            break
        
        t1 = time.perf_counter()
        timing_info["capture"] = (t1 - t0) * 1000
        
        # Flip the frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Downscale for faster processing
        small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        t2 = time.perf_counter()
        timing_info["preprocess"] = (t2 - t1) * 1000
        
        # Process the downscaled frame and detect hands
        results = hands.process(rgb_frame)
        
        t3 = time.perf_counter()
        timing_info["mediapipe"] = (t3 - t2) * 1000
        
        # Calculate scale factors for coordinate conversion
        scale_x = w / PROCESS_WIDTH
        scale_y = h / PROCESS_HEIGHT
        
        # Draw hand landmarks and control mouse
        finger_count = 0
        t4 = time.perf_counter()
        
        if results.multi_hand_landmarks:
            # Only use the first detected hand for mouse control
            hand_landmarks = results.multi_hand_landmarks[0]
            
            for idx, hand_landmarks_iter in enumerate(results.multi_hand_landmarks):
                # Draw landmarks and connections on full resolution frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks_iter,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            
            t5 = time.perf_counter()
            timing_info["drawing"] = (t5 - t4) * 1000
            
            # Use only the first hand for mouse control
            # Calculate palm center from finger bases (landmarks 5, 9, 13, 17 = thumb, index, middle, ring, pinky bases)
            # Also include wrist (0) for better center calculation
            palm_landmarks = [0, 5, 9, 13, 17]
            palm_x = sum(hand_landmarks.landmark[i].x for i in palm_landmarks) / len(palm_landmarks)
            palm_y = sum(hand_landmarks.landmark[i].y for i in palm_landmarks) / len(palm_landmarks)
            
            cx = int(palm_x * w)
            cy = int(palm_y * h)
            
            # Draw circle on palm center (only for controlling hand)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
            
            # Mouse control with deadzone (use same value as displayed)
            if abs(cx - CAMERA_CENTER_X) > DEADZONE or abs(cy - CAMERA_CENTER_Y) > DEADZONE:
                t_mouse_start = time.perf_counter()
                
                move_x = cx - CAMERA_CENTER_X
                move_y = cy - CAMERA_CENTER_Y
                
                max_rate = 20
                move_x = max(-max_rate, min(max_rate, move_x))
                move_y = max(-max_rate, min(max_rate, move_y))
                move_mouse(move_x, move_y)
                
                t_mouse_end = time.perf_counter()
                timing_info["mouse"] = (t_mouse_end - t_mouse_start) * 1000
            else:
                timing_info["mouse"] = 0
            
            # Count extended fingers (only for first hand)
            finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
            thumb_tip = 4
            finger_count = 0
            
            # Check thumb (different logic)
            if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
                finger_count += 1
            
            # Check other fingers
            for tip_id in finger_tips:
                if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                    finger_count += 1
            
            # Display finger count
            cv2.putText(frame, f"Fingers: {finger_count}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Main non detectee", (10, h-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            timing_info["drawing"] = 0
            timing_info["mouse"] = 0
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Draw deadzone rectangle
        cv2.rectangle(frame, 
                     (CAMERA_CENTER_X - DEADZONE, CAMERA_CENTER_Y - DEADZONE),
                     (CAMERA_CENTER_X + DEADZONE, CAMERA_CENTER_Y + DEADZONE),
                     (0, 0, 255), 2)
        cv2.circle(frame, (CAMERA_CENTER_X, CAMERA_CENTER_Y), 5, (0, 0, 255), -1)
        
        # Display FPS and timing info on frame
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display timing breakdown (in milliseconds) only if DEBUG_MODE is on
        if DEBUG_MODE:
            y_offset = 110
            cv2.putText(frame, f'Capture: {timing_info["capture"]:.1f}ms', (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Preproc: {timing_info["preprocess"]:.1f}ms', (10, y_offset+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'MediaPipe: {timing_info["mediapipe"]:.1f}ms', (10, y_offset+40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Drawing: {timing_info["drawing"]:.1f}ms', (10, y_offset+60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Mouse: {timing_info["mouse"]:.1f}ms', (10, y_offset+80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        t6 = time.perf_counter()
        
        # Display the frame
        cv2.imshow('Hand Detection', frame)
        
        if DEBUG_MODE:
            t7 = time.perf_counter()
            timing_info["display"] = (t7 - t6) * 1000
            
            cv2.putText(frame, f'Display: {timing_info["display"]:.1f}ms', (10, y_offset+100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
