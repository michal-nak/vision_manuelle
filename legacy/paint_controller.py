"""
Paint controller using hand gestures
- Index finger up: move cursor
- Pinch (thumb + index close): click and drag to draw
- Open hand (3+ fingers): release/stop drawing
- Press 'r'/'g'/'b' for red/green/blue brush
- Press '1'/'2'/'3' for small/medium/large brush
- ESC to quit
"""
import cv2
import numpy as np
import math
import pyautogui
import time
import subprocess

pyautogui.FAILSAFE = False

def nothing(x):
    pass

def get_largest_contour(contours, min_area=1000):
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    return largest

def angle_between(a, b, c):
    # angle at b between points a-b-c
    ab = (a[0]-b[0], a[1]-b[1])
    cb = (c[0]-b[0], c[1]-b[1])
    dot = ab[0]*cb[0] + ab[1]*cb[1]
    mag1 = math.hypot(ab[0], ab[1])
    mag2 = math.hypot(cb[0], cb[1])
    if mag1*mag2 == 0:
        return 180.0
    cosang = dot/(mag1*mag2)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))

def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

# --- Main ---
cap = cv2.VideoCapture(0)

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_CENTER_X = CAMERA_WIDTH // 2
CAMERA_CENTER_Y = CAMERA_HEIGHT // 2

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not cap.isOpened():
    print("Error: cannot open webcam")
    exit()

# HSV trackbars for tuning skin detection
cv2.namedWindow("Controls")
cv2.createTrackbar("H_min","Controls",0,179,nothing)
cv2.createTrackbar("H_max","Controls",20,179,nothing)
cv2.createTrackbar("S_min","Controls",48,255,nothing)
cv2.createTrackbar("S_max","Controls",255,255,nothing)
cv2.createTrackbar("V_min","Controls",80,255,nothing)
cv2.createTrackbar("V_max","Controls",255,255,nothing)

# Paint control state
is_drawing = False
prev_finger_pos = None
smoothing_buffer = []
smoothing_window = 3

# Screen size for cursor mapping
screen_w, screen_h = pyautogui.size()

# Launch Paint
print("Opening Paint...")
try:
    paint_process = subprocess.Popen(['mspaint.exe'])
    time.sleep(3)  # Give Paint time to open and focus
    print("Paint opened successfully!")
    print("IMPORTANT: Make sure Paint is in focus and Brush tool is selected!")
    time.sleep(2)  # Extra time for user to see message
except Exception as e:
    print(f"Warning: Could not open Paint automatically: {e}")
    print("Please open Paint manually before using gestures.")

print("=== Paint Controller ===")
print("Gestures:")
print("  - 1 finger (index): move cursor")
print("  - Pinch (thumb+index close <40px): click & draw")
print("  - Open hand (3+ fingers): stop drawing")
print("\nKeyboard shortcuts:")
print("  r: red, g: green, b: blue, y: yellow, k: black, w: white")
print("  t: test drawing (draws a small circle)")
print("  ESC: quit")
print("========================\n")

print("Testing drawing capability...")
print("Moving cursor to center and drawing test circle...")
time.sleep(1)
test_x, test_y = screen_w // 2, screen_h // 2
pyautogui.moveTo(test_x, test_y)
time.sleep(0.5)
pyautogui.mouseDown()
for angle in range(0, 360, 10):
    rad = math.radians(angle)
    x = test_x + int(50 * math.cos(rad))
    y = test_y + int(50 * math.sin(rad))
    pyautogui.moveTo(x, y, duration=0.01)
pyautogui.mouseUp()
print("Test circle drawn! If you see a circle in Paint, drawing works!")
print("If not, make sure Paint window is focused and Brush tool is selected.")
print("\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Preprocessing
    blur = cv2.GaussianBlur(frame, (7,7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Get HSV range from trackbars
    hmin = cv2.getTrackbarPos("H_min","Controls")
    hmax = cv2.getTrackbarPos("H_max","Controls")
    smin = cv2.getTrackbarPos("S_min","Controls")
    smax = cv2.getTrackbarPos("S_max","Controls")
    vmin = cv2.getTrackbarPos("V_min","Controls")
    vmax = cv2.getTrackbarPos("V_max","Controls")
    lower = np.array([hmin, smin, vmin], dtype=np.uint8)
    upper = np.array([hmax, smax, vmax], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological ops
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (7,7), 0)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_contour = get_largest_contour(contours, min_area=3000)

    display = frame.copy()
    cv2.imshow("Mask", mask)

    gesture_text = "No hand detected"
    
    if hand_contour is not None:
        cv2.drawContours(display, [hand_contour], -1, (0,255,0), 2)

        hull = cv2.convexHull(hand_contour, returnPoints=True)
        cv2.drawContours(display, [hull], -1, (255,0,0), 2)

        hull_idx = cv2.convexHull(hand_contour, returnPoints=False)
        if len(hull_idx) > 3:
            defects = cv2.convexityDefects(hand_contour, hull_idx)
            if defects is not None:
                finger_tips = []
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    start = tuple(hand_contour[s][0])
                    end   = tuple(hand_contour[e][0])
                    far   = tuple(hand_contour[f][0])

                    ang = angle_between(start, far, end)
                    if d > 10000 and ang < 90.0:
                        cv2.circle(display, start, 6, (0,255,255), -1)
                        cv2.circle(display, end, 6, (0,255,255), -1)
                        cv2.circle(display, far, 6, (0,0,255), -1)
                        finger_tips.append(start)
                        finger_tips.append(end)

                # Reduce duplicates
                filtered_tips = []
                for p in finger_tips:
                    if not any(distance(p, q) < 40 for q in filtered_tips):
                        filtered_tips.append(p)

                for i, pt in enumerate(filtered_tips):
                    cv2.circle(display, pt, 10, (0,255,0), 2)
                    cv2.putText(display, f"{i+1}", (pt[0]-10, pt[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                num_fingers = len(filtered_tips)

                # Gesture recognition
                if num_fingers >= 3:
                    # Open hand: stop drawing
                    if is_drawing:
                        pyautogui.mouseUp()
                        is_drawing = False
                    gesture_text = f"Open hand ({num_fingers} fingers) - Stop drawing"
                    
                elif num_fingers >= 2:
                    # Check for pinch: thumb+index close together
                    # Assume first two tips are thumb and index
                    if len(filtered_tips) >= 2:
                        dist = distance(filtered_tips[0], filtered_tips[1])
                        if dist < 40:
                            # Pinch detected: start drawing if not already
                            if not is_drawing:
                                pyautogui.mouseDown()
                                is_drawing = True
                                print(f"[DEBUG] Started drawing - mouseDown() called")
                            gesture_text = f"Pinch (dist={dist:.0f}) - DRAWING ACTIVE"
                        else:
                            # Two fingers but not pinching: move cursor
                            if is_drawing:
                                pyautogui.mouseUp()
                                is_drawing = False
                                print(f"[DEBUG] Stopped drawing - mouseUp() called")
                            gesture_text = f"Two fingers (dist={dist:.0f}) - Moving"
                    
                    # Move cursor based on index finger (first tip)
                    if len(filtered_tips) > 0:
                        cx, cy = filtered_tips[0]
                        # Smooth cursor movement
                        smoothing_buffer.append((cx, cy))
                        if len(smoothing_buffer) > smoothing_window:
                            smoothing_buffer.pop(0)
                        avg_x = int(np.mean([p[0] for p in smoothing_buffer]))
                        avg_y = int(np.mean([p[1] for p in smoothing_buffer]))
                        
                        # Map camera coords to screen coords
                        mouse_x = np.interp(avg_x, [0, CAMERA_WIDTH], [0, screen_w])
                        mouse_y = np.interp(avg_y, [0, CAMERA_HEIGHT], [0, screen_h])
                        pyautogui.moveTo(mouse_x, mouse_y, duration=0.05)

                elif num_fingers == 1:
                    # One finger: move cursor (no drawing unless pinch)
                    if is_drawing:
                        pyautogui.mouseUp()
                        is_drawing = False
                    gesture_text = "One finger - Moving cursor"
                    
                    cx, cy = filtered_tips[0]
                    smoothing_buffer.append((cx, cy))
                    if len(smoothing_buffer) > smoothing_window:
                        smoothing_buffer.pop(0)
                    avg_x = int(np.mean([p[0] for p in smoothing_buffer]))
                    avg_y = int(np.mean([p[1] for p in smoothing_buffer]))
                    
                    mouse_x = np.interp(avg_x, [0, CAMERA_WIDTH], [0, screen_w])
                    mouse_y = np.interp(avg_y, [0, CAMERA_HEIGHT], [0, screen_h])
                    pyautogui.moveTo(mouse_x, mouse_y, duration=0.05)
                else:
                    if is_drawing:
                        pyautogui.mouseUp()
                        is_drawing = False
                    gesture_text = "Hand detected - No clear gesture"

    else:
        if is_drawing:
            pyautogui.mouseUp()
            is_drawing = False

    # Display gesture info
    cv2.putText(display, gesture_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    
    if is_drawing:
        cv2.putText(display, "DRAWING", (10, h-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)

    cv2.imshow("Paint Controller", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        if is_drawing:
            pyautogui.mouseUp()
        break
    elif key == ord('s'):
        cv2.imwrite("paint_snapshot.png", display)
        print("Snapshot saved: paint_snapshot.png")
    elif key == ord('t'):
        # Test drawing
        print("Testing drawing...")
        curr_x, curr_y = pyautogui.position()
        pyautogui.mouseDown()
        pyautogui.move(50, 0, duration=0.2)
        pyautogui.move(0, 50, duration=0.2)
        pyautogui.move(-50, 0, duration=0.2)
        pyautogui.move(0, -50, duration=0.2)
        pyautogui.mouseUp()
        print("Drew test square - if you see it, drawing works!")
    # Color change via Edit Colors dialog
    elif key == ord('r'):
        # Open Edit Colors and select red
        pyautogui.hotkey('alt', 'h')  # Home ribbon
        time.sleep(0.1)
        pyautogui.press('ec')  # Edit Colors
        time.sleep(0.2)
        pyautogui.hotkey('alt', 'r')  # Red slider
        pyautogui.press('right', presses=5)
        pyautogui.press('enter')
        print("Switched to RED")
    elif key == ord('g'):
        pyautogui.hotkey('alt', 'h')
        time.sleep(0.1)
        pyautogui.press('ec')
        time.sleep(0.2)
        pyautogui.hotkey('alt', 'g')  # Green slider
        pyautogui.press('right', presses=5)
        pyautogui.press('enter')
        print("Switched to GREEN")
    elif key == ord('b'):
        pyautogui.hotkey('alt', 'h')
        time.sleep(0.1)
        pyautogui.press('ec')
        time.sleep(0.2)
        pyautogui.hotkey('alt', 'b')  # Blue slider (use 'l' for blue slider)
        pyautogui.press('right', presses=5)
        pyautogui.press('enter')
        print("Switched to BLUE")
    elif key == ord('k'):
        # Black - first color in palette
        pyautogui.hotkey('alt', 'h')
        time.sleep(0.1)
        pyautogui.press('c')  # Colors dropdown
        time.sleep(0.1)
        pyautogui.press('down')  # Navigate to black
        pyautogui.press('enter')
        print("Switched to BLACK")
    elif key == ord('w'):
        # White
        pyautogui.hotkey('alt', 'h')
        time.sleep(0.1)
        pyautogui.press('c')
        time.sleep(0.1)
        pyautogui.press('down', presses=2)
        pyautogui.press('enter')
        print("Switched to WHITE")
    elif key == ord('1'):
        # Smaller brush
        pyautogui.hotkey('ctrl', 'pageup')
        print("Brush size: SMALLER")
    elif key == ord('2'):
        # Larger brush
        pyautogui.hotkey('ctrl', 'pagedown')
        print("Brush size: LARGER")

cap.release()
cv2.destroyAllWindows()
