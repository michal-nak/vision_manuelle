"""
Detection pipeline methods for CV detector
Contains the core detection logic separated from the main class
"""
import cv2
from ...core.config import MIN_HAND_AREA, MAX_HAND_AREA
from .skin_detection import (detect_skin_ycrcb_hsv, apply_morphological_operations, 
                              find_largest_contour, filter_forearm_by_shape, 
                              filter_forearm_by_orientation, select_hand_contour_intelligent,
                              detect_wrist_and_crop, apply_top_priority_filter)
from .finger_detection import count_fingers_from_contour, smooth_finger_count, map_fingers_to_gesture, draw_finger_visualization
from .visualization import draw_debug_overlay, draw_extended_debug_metrics


def detect_hand_full_pipeline(frame, gray, bg_subtractor, state, color_bounds, show_debug, processing_params=None):
    """
    Full hand detection pipeline
    
    Args:
        frame: BGR frame
        gray: Grayscale frame
        bg_subtractor: Background subtractor object
        state: DetectorState object
        color_bounds: Tuple of (ycrcb_lower, ycrcb_upper, hsv_lower, hsv_upper)
        show_debug: Whether to show debug overlay
        processing_params: Dict with processing parameters (denoise_h, kernel_small, etc.)
    
    Returns:
        Dict with detection results or None if no hand detected
    """
    ycrcb_lower, ycrcb_upper, hsv_lower, hsv_upper = color_bounds
    
    # Get processing parameters
    if processing_params is None:
        processing_params = {
            'denoise_h': 10,
            'kernel_small': 3,
            'kernel_large': 7,
            'morph_iterations': 2,
            'min_contour_area': 1000,
            'max_contour_area': 50000
        }
    
    # Skin detection
    mask = detect_skin_ycrcb_hsv(frame, ycrcb_lower, ycrcb_upper, hsv_lower, hsv_upper, 
                                   processing_params['denoise_h'])
    
    # Motion filtering
    if state.frame_count > state.bg_learning_frames:
        fg_mask = bg_subtractor.apply(frame, learningRate=0.001)
        kernel_motion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fg_mask = cv2.dilate(fg_mask, kernel_motion, iterations=2)
        mask = cv2.bitwise_and(mask, fg_mask)
    else:
        bg_subtractor.apply(frame, learningRate=0.1)
    
    # Clean up mask
    mask = apply_morphological_operations(mask, 
                                          processing_params['kernel_small'],
                                          processing_params['kernel_large'],
                                          processing_params['morph_iterations'])
    
    # Calculate max area in pixels (MAX_HAND_AREA is a fraction of frame area)
    h, w = frame.shape[:2]
    max_area_pixels = int(w * h * MAX_HAND_AREA)
    
    # Find all contours (not just largest - need to filter forearms)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, mask
    
    # Filter contours by area first
    valid_contours = [c for c in contours 
                     if processing_params['min_contour_area'] < cv2.contourArea(c) < processing_params['max_contour_area']]
    
    if not valid_contours:
        if show_debug:
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                state.debug_metrics['largest_contour_area'] = area
                state.debug_metrics['contour_rejected'] = 'Area filter'
        return None, mask
    
    # Use intelligent selection to filter out forearms
    hand_contour = select_hand_contour_intelligent(valid_contours, frame.shape)
    
    if hand_contour is None:
        if show_debug:
            state.debug_metrics['contour_rejected'] = 'Forearm filter'
        return None, mask
    
    # Additional geometric validation
    if not filter_forearm_by_shape(hand_contour, h):
        if show_debug:
            state.debug_metrics['contour_rejected'] = 'Shape validation'
        return None, mask
    
    if not filter_forearm_by_orientation(hand_contour):
        if show_debug:
            state.debug_metrics['contour_rejected'] = 'Orientation filter'
        return None, mask
    
    # Try to detect wrist and crop forearm
    hand_contour = detect_wrist_and_crop(mask, hand_contour)
    
    return hand_contour, mask


def process_detected_hand(frame, gray, contour, state, show_debug):
    """
    Process a detected hand contour
    
    Args:
        frame: BGR frame
        gray: Grayscale frame
        contour: Hand contour
        state: DetectorState object
        show_debug: Whether to show debug overlay
    
    Returns:
        Dict with detection results
    """
    annotated = frame.copy()
    
    # Draw contour
    cv2.drawContours(annotated, [contour], 0, (0, 255, 0), 2)
    
    # Get hand center
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        h, w = frame.shape[:2]
        hand_center = (cx / w, cy / h)
    else:
        return None
    
    # Count fingers with debug info
    finger_count, debug_info = count_fingers_from_contour(contour, return_debug=True)
    state.finger_history.append(finger_count)
    smooth_count = smooth_finger_count(state.finger_history)
    
    # Map to gesture
    gesture = map_fingers_to_gesture(smooth_count)
    
    # Update metrics
    state.update_detection_metrics(contour, gesture)
    
    # Draw finger visualization
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(contour, hull)
        if defects is not None:
            annotated = draw_finger_visualization(annotated, contour, defects)
            state.debug_metrics['hull_defect_counts'].append(len(defects))
            if len(state.debug_metrics['hull_defect_counts']) > 30:
                state.debug_metrics['hull_defect_counts'].pop(0)
    
    # Draw debug overlay
    if show_debug:
        area = cv2.contourArea(contour)
        confidence = min(100, int((area / 10000) * 100))
        annotated = draw_debug_overlay(
            annotated, state.debug_metrics, smooth_count,
            gesture, confidence, state.finger_history
        )
        annotated = draw_extended_debug_metrics(
            annotated, state.debug_metrics, state.finger_history
        )
    
    return {
        'detected': True,
        'hand_center': hand_center,
        'hand_x': hand_center[0],
        'hand_y': hand_center[1],
        'finger_count': smooth_count,
        'gesture': gesture,
        'annotated_frame': annotated,
        'contour': contour,
        'debug_info': debug_info
    }


def create_no_detection_result(frame, mask, state, show_debug):
    """
    Create result when no hand is detected
    
    Args:
        frame: BGR frame
        mask: Binary mask (can be None)
        state: DetectorState object
        show_debug: Whether to show debug overlay
    
    Returns:
        Dict with no detection result
    """
    state.mark_no_tracking()
    
    # Show mask in corner for debugging
    annotated = frame.copy()
    if mask is not None and show_debug:
        # Resize mask to small preview
        h, w = frame.shape[:2]
        mask_preview_size = (160, 120)
        mask_resized = cv2.resize(mask, mask_preview_size)
        mask_colored = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        
        # Place in top-right corner
        x_offset = w - mask_preview_size[0] - 10
        y_offset = 10
        annotated[y_offset:y_offset+mask_preview_size[1], 
                 x_offset:x_offset+mask_preview_size[0]] = mask_colored
        
        # Label it
        cv2.putText(annotated, "Skin Mask", (x_offset, y_offset - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add "No hand detected" message
    if show_debug:
        # Top status panel
        overlay = annotated.copy()
        cv2.rectangle(overlay, (5, 5), (635, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        
        cv2.putText(annotated, "Detection: CV", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, "Status: NO HAND DETECTED", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(annotated, "Confidence: 0%", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show why detection failed
        if 'largest_contour_area' in state.debug_metrics:
            area = state.debug_metrics['largest_contour_area']
            cv2.putText(annotated, f"Largest contour: {int(area)} px (min: {MIN_HAND_AREA})", 
                       (10, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.putText(annotated, "Show your hand to the camera", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return {
        'detected': False,
        'hand_center': (0.5, 0.5),
        'hand_x': 0.5,
        'hand_y': 0.5,
        'finger_count': 0,
        'gesture': 'None',
        'annotated_frame': annotated
    }


def create_tracking_result(frame, bbox, center, state, show_debug):
    """
    Create result from tracking data
    
    Args:
        frame: BGR frame
        bbox: Bounding box (x, y, w, h)
        center: Center point (x, y)
        state: DetectorState object
        show_debug: Whether to show debug overlay
    
    Returns:
        Dict with tracking result
    """
    x, y, w, h = bbox
    annotated = frame.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
    
    # Use last known finger count
    finger_count = state.finger_history[-1] if state.finger_history else 0
    gesture = map_fingers_to_gesture(finger_count)
    
    state.update_tracking_metrics()
    
    if show_debug:
        annotated = draw_debug_overlay(
            annotated, state.debug_metrics, finger_count,
            gesture, 85, state.finger_history
        )
    
    frame_h, frame_w = frame.shape[:2]
    return {
        'detected': True,
        'hand_center': (center[0] / frame_w, center[1] / frame_h),
        'hand_x': center[0] / frame_w,
        'hand_y': center[1] / frame_h,
        'finger_count': finger_count,
        'gesture': gesture,
        'annotated_frame': annotated
    }
