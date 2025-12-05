"""
Visualization and debug overlay for CV detector
"""
import cv2


def draw_debug_overlay(frame, metrics, finger_count, gesture, confidence, finger_history):
    """
    Draw comprehensive debug information on frame
    
    Args:
        frame: Frame to draw on
        metrics: Debug metrics dictionary
        finger_count: Current finger count
        gesture: Current gesture string
        confidence: Detection confidence
        finger_history: Deque of finger counts
        
    Returns:
        Modified frame
    """
    # Status overlay (top left)
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (635, 155), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Detection method indicator
    cv2.putText(frame, "Detection: CV", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Mode and status
    mode_text = "TRACKING" if metrics.get('tracking', False) else "DETECTING"
    mode_color = (0, 255, 255) if metrics.get('tracking', False) else (255, 255, 255)
    
    cv2.putText(frame, f"Status: {mode_text}", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
    
    # Confidence
    cv2.putText(frame, f"Confidence: {confidence}%", (10, 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if confidence > 50 else (255, 255, 0), 2)
    
    # Finger count
    cv2.putText(frame, f"Fingers: {finger_count}", (10, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Gesture with color coding
    gesture_colors = {
        "Draw": (0, 255, 0),
        "Erase": (0, 165, 255),
        "Cycle Color": (255, 255, 0),
        "Increase Size": (255, 165, 0),
        "Decrease Size": (255, 100, 0),
        "Clear": (0, 0, 255),
        "None": (128, 128, 128)
    }
    gesture_color = gesture_colors.get(gesture, (255, 255, 255))
    cv2.putText(frame, f"Gesture: {gesture}", (10, 125),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)
    
    # Detection status indicator
    status_text = "HAND DETECTED" if confidence > 0 else "NO HAND"
    status_color = (0, 255, 0) if confidence > 0 else (0, 0, 255)
    cv2.putText(frame, status_text, (10, 150),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
    
    # Finger mapping legend (top right)
    legend_x = 445
    cv2.rectangle(frame, (legend_x - 5, 5), (635, 135), (0, 0, 0), -1)
    cv2.putText(frame, "Finger Map:", (legend_x, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "1 = Draw", (legend_x, 45),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(frame, "2 = Erase", (legend_x, 63),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
    cv2.putText(frame, "3 = Color", (legend_x, 81),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    cv2.putText(frame, "4 = Size+", (legend_x, 99),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 165, 0), 1)
    cv2.putText(frame, "5 = Clear", (legend_x, 117),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.putText(frame, "0 = None", (legend_x, 133),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    
    return frame


def draw_extended_debug_metrics(frame, metrics, finger_history):
    """
    Draw extended debug metrics panel
    
    Args:
        frame: Frame to draw on
        metrics: Debug metrics dictionary
        finger_history: Finger count history
        
    Returns:
        Modified frame
    """
    debug_y = 150
    
    # Semi-transparent background for debug section
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (5, debug_y - 10), (635, debug_y + 180), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, "=== DEBUG METRICS ===", (10, debug_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    debug_y += 25
    
    # Detection rate
    detection_rate = (metrics['detected_frames'] / max(1, metrics['total_frames'])) * 100
    cv2.putText(frame, f"Detection Rate: {detection_rate:.1f}%", (10, debug_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    debug_y += 20
    
    cv2.putText(frame, f"Total Frames: {metrics['total_frames']}", (10, debug_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    debug_y += 20
    
    cv2.putText(frame, f"Gesture Changes: {metrics['gesture_changes']}", (10, debug_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    debug_y += 20
    
    # Contour area
    if metrics['contour_areas']:
        cv2.putText(frame, f"Contour Area: {int(metrics['contour_areas'][-1])}", (10, debug_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        debug_y += 20
    
    # Finger count history
    history_str = ' -> '.join([str(int(x)) for x in list(finger_history)[-5:]])
    cv2.putText(frame, f"Finger History: {history_str}", (10, debug_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    debug_y += 20
    
    # Recent gesture transitions
    cv2.putText(frame, "Recent Transitions:", (10, debug_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
    debug_y += 18
    
    for old_g, new_g, frame_num in list(metrics['finger_transitions'])[-3:]:
        trans_text = f"  {old_g} -> {new_g} (frame {frame_num})"
        cv2.putText(frame, trans_text, (10, debug_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        debug_y += 16
    
    return frame
