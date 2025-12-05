"""
Finger detection and counting logic
Separated from main CV detector for better organization
"""
import cv2
import numpy as np
import math


def count_fingers_from_contour(contour, defect_threshold=8000):
    """
    Count fingers using convexity defects
    
    Args:
        contour: Hand contour
        defect_threshold: Threshold for defect depth
        
    Returns:
        int: Number of fingers detected
    """
    try:
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(contour, hull)
            if defects is not None:
                finger_count = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])
                    
                    # Calculate angle between start-far-end
                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    
                    angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c))
                    
                    # If angle < 90 degrees and defect deep enough, count as finger
                    if angle <= math.pi / 2 and d > defect_threshold:
                        finger_count += 1
                
                return min(5, finger_count + 1)  # +1 for thumb, cap at 5
        return 1
    except:
        return 0


def smooth_finger_count(finger_history):
    """
    Smooth finger count using median filtering
    
    Args:
        finger_history: Deque of recent finger counts
        
    Returns:
        int: Smoothed finger count
    """
    if len(finger_history) >= 2:
        return int(np.median(list(finger_history)))
    elif len(finger_history) == 1:
        return int(finger_history[0])
    return 0


def map_fingers_to_gesture(finger_count):
    """
    Map finger count to gesture string (aligned with MediaPipe gestures)
    
    Args:
        finger_count: Number of fingers detected
        
    Returns:
        str: Gesture name
    """
    gesture_map = {
        0: "None",
        1: "Draw",              # 1 finger = Draw (like thumb+index in MediaPipe)
        2: "Erase",             # 2 fingers = Erase (like thumb+middle in MediaPipe)
        3: "Cycle Color",       # 3 fingers = Cycle Color (like thumb+ring in MediaPipe)
        4: "Increase Size",     # 4 fingers = Increase Size (like index+middle in MediaPipe)
        5: "Clear"              # 5 fingers = Clear (like thumb+pinky in MediaPipe)
    }
    return gesture_map.get(finger_count, "None")


def draw_finger_visualization(frame, contour, defects=None):
    """
    Draw visualization of finger detection
    
    Args:
        frame: Frame to draw on
        contour: Hand contour
        defects: Convexity defects
        
    Returns:
        Modified frame
    """
    # Draw convex hull in yellow
    hull_points = cv2.convexHull(contour)
    cv2.drawContours(frame, [hull_points], 0, (0, 255, 255), 2)
    
    if defects is not None:
        # Mark fingertips (peaks) with green circles
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            far = tuple(contour[f][0])
            
            # Draw fingertip (peak)
            cv2.circle(frame, start, 8, (0, 255, 0), -1)
            # Draw valley between fingers
            cv2.circle(frame, far, 5, (255, 0, 0), -1)
    
    return frame
