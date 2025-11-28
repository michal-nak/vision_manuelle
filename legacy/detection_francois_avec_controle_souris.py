import cv2
import numpy as np
import math

import pyautogui
pyautogui.FAILSAFE = False

def nothing(x):
    pass

# --- Helper functions ---
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
    mag1 = math.hypot(ab[0], ab[1]) # pour calculer la norme
    mag2 = math.hypot(cb[0], cb[1])
    if mag1*mag2 == 0:
        return 180.0
    cosang = dot/(mag1*mag2)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))

# --- Main: capture webcam and process frames ---
cap = cv2.VideoCapture(0)

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

CAMERA_CENTER_X = CAMERA_WIDTH // 2
CAMERA_CENTER_Y = CAMERA_HEIGHT // 2

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    print("Erreur: impossible d'ouvrir la webcam")
    exit()

# Optionnel: create trackbars to fine tune HSV skin range in real time
cv2.namedWindow("Controls")
cv2.createTrackbar("H_min","Controls",0,179,nothing)
cv2.createTrackbar("H_max","Controls",20,179,nothing)
cv2.createTrackbar("S_min","Controls",48,255,nothing)
cv2.createTrackbar("S_max","Controls",255,255,nothing)
cv2.createTrackbar("V_min","Controls",80,255,nothing)
cv2.createTrackbar("V_max","Controls",255,255,nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # miroir
    h, w = frame.shape[:2]

    # --- Preprocessing ---
    blur = cv2.bilateralFilter(frame, d=5, sigmaColor=75, sigmaSpace=75) # pour réduire le bruit et petites variations

    # --- Skin color segmentation in HSV ---
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    hmin = cv2.getTrackbarPos("H_min","Controls")
    hmax = cv2.getTrackbarPos("H_max","Controls")
    smin = cv2.getTrackbarPos("S_min","Controls")
    smax = cv2.getTrackbarPos("S_max","Controls")
    vmin = cv2.getTrackbarPos("V_min","Controls")
    vmax = cv2.getTrackbarPos("V_max","Controls")
    lower = np.array([hmin, smin, vmin], dtype=np.uint8)
    upper = np.array([hmax, smax, vmax], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # --- Morphological ops to clean the mask ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (7,7), 0)

    # --- Edge detection (Canny) on the blurred frame -- optional/visual ---
    edges = cv2.Canny(blur, 50, 150)

    # --- Find contours on mask ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_contour = get_largest_contour(contours, min_area=3000)

    display = frame.copy()
    cv2.imshow("Mask", mask)
    cv2.imshow("Edges", edges)

    if hand_contour is not None:
        # draw contour
        cv2.drawContours(display, [hand_contour], -1, (0,255,0), 2)

        # get center coordinates of the hand using moments
        # M = cv2.moments(hand_contour)
        # if M["m00"] != 0:
        #     cx = int(M["m10"] / M["m00"])
        #     cy = int(M["m01"] / M["m00"])
            # draw center point
            # cv2.circle(display, (cx, cy), 5, (255, 0, 255), -1)

            # # Move the mouse cursor based on hand position with a center deadzone
            # deadzone = 40
            # if abs(cx - CAMERA_CENTER_X) > deadzone or abs(cy - CAMERA_CENTER_Y) > deadzone:
            #     # mouse_x = np.interp(cx, [0, CAMERA_WIDTH], [0, screen_w])
            #     # mouse_y = np.interp(cy, [0, CAMERA_HEIGHT], [0, screen_h])
            #     # pyautogui.moveTo(mouse_x, mouse_y, duration=0.1)
            #     move_x = cx - CAMERA_CENTER_X
            #     move_y = cy - CAMERA_CENTER_Y

            #     max_rate = 20
            #     # Limit movement to ±100 pixels
            #     move_x = max(-max_rate, min(max_rate, move_x))
            #     move_y = max(-max_rate, min(max_rate, move_y))
            #     pyautogui.move(move_x, move_y, duration=0.1)

        # convex hull
        hull = cv2.convexHull(hand_contour, returnPoints=True)
        cv2.drawContours(display, [hull], -1, (255,0,0), 2)

        # convex hull indices (for defects)
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

                    # compute angle between start-far-end
                    ang = angle_between(start, far, end)
                    # filter defects by depth and angle to identify fingers
                    if d > 10000 and ang < 90.0:
                        # mark points
                        cv2.circle(display, start, 6, (0,255,255), -1)
                        cv2.circle(display, end, 6, (0,255,255), -1)
                        cv2.circle(display, far, 6, (0,0,255), -1)
                        finger_tips.append(start)
                        finger_tips.append(end)

                # reduce duplicates among finger_tips and draw them
                # cluster nearby points (simple)
                filtered_tips = []
                for p in finger_tips:
                    if not any(math.hypot(p[0]-q[0], p[1]-q[1]) < 40 for q in filtered_tips):
                        filtered_tips.append(p)
                for i,pt in enumerate(filtered_tips):
                    cv2.circle(display, pt, 10, (0,255,0), 2)
                    cv2.putText(display, f"{i+1}", (pt[0]-10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)



                # Move the mouse cursor based on hand position with a center deadzone
                cx, cy = CAMERA_CENTER_X, CAMERA_CENTER_Y
                if len(filtered_tips) > 0:
                    cx, cy = filtered_tips[0][0], filtered_tips[0][1]
                deadzone = 50
                if abs(cx - CAMERA_CENTER_X) > deadzone or abs(cy - CAMERA_CENTER_Y) > deadzone:
                    # mouse_x = np.interp(cx, [0, CAMERA_WIDTH], [0, screen_w])
                    # mouse_y = np.interp(cy, [0, CAMERA_HEIGHT], [0, screen_h])
                    # pyautogui.moveTo(mouse_x, mouse_y, duration=0.1)
                    move_x = cx - CAMERA_CENTER_X
                    move_y = cy - CAMERA_CENTER_Y

                    max_rate = 20
                    # Limit movement to ±100 pixels
                    move_x = max(-max_rate, min(max_rate, move_x))
                    move_y = max(-max_rate, min(max_rate, move_y))
                    pyautogui.move(move_x, move_y, duration=0.1)



                cv2.putText(display, f"Fingers detected (est.): {len(filtered_tips)}",
                            (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)
    else:
        cv2.putText(display, "Main non detectee", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)

    cv2.imshow("Result", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('s'):
        # snapshot for debugging
        cv2.imwrite("hand_snapshot.png", display)
        print("Snapshot saved: hand_snapshot.png")

cap.release()
cv2.destroyAllWindows()
