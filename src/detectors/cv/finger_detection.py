"""
Finger detection and counting logic
Separated from main CV detector for better organization
Uses traditional computer vision techniques:
- Contour analysis and geometric features
- Distance transform (morphological operation)
- Adaptive thresholding based on hand geometry
"""
import cv2
import numpy as np
import math
from collections import deque


class FingerCountSmoother:
    """Temporal smoothing using exponential moving average"""
    def __init__(self, alpha=0.3, stability_threshold=3):
        self.history = deque(maxlen=10)
        self.alpha = alpha  # Smoothing factor
        self.smoothed_value = 0
        self.stability_threshold = stability_threshold
        
    def update(self, new_count):
        """Update with new measurement and return smoothed value"""
        self.history.append(new_count)
        
        if len(self.history) < 3:
            self.smoothed_value = new_count
            return new_count
        
        # Exponential moving average
        self.smoothed_value = self.alpha * new_count + (1 - self.alpha) * self.smoothed_value
        smoothed_int = int(round(self.smoothed_value))
        
        # Stability check: only change if consistent over last N frames
        recent = list(self.history)[-self.stability_threshold:]
        if len(recent) >= self.stability_threshold:
            # Check if most recent values are close to smoothed
            if sum(abs(x - smoothed_int) <= 1 for x in recent) >= self.stability_threshold - 1:
                return smoothed_int
        
        # Return last stable value if smoothed is not stable
        return int(self.history[-1])


# Global smoother instance (reused across frames)
_finger_smoother = FingerCountSmoother()


def count_fingers_from_contour(contour, defect_threshold=8000, return_debug=False):
    """
    Count fingers using hybrid approach with adaptive thresholding
    Combines convexity defects with geometric validation and fallback logic
    
    Args:
        contour: Hand contour
        defect_threshold: Base threshold for defect depth (will be adapted)
        return_debug: If True, return (count, debug_info) tuple
        
    Returns:
        int or tuple: Number of fingers detected, optionally with debug info
    """
    debug_info = {}
    try:
        # Calculate hand size for adaptive thresholding
        area = cv2.contourArea(contour)
        if area < 1000:
            debug_info['reason'] = 'area_too_small'
            if return_debug:
                return 0, debug_info
            return 0
        
        # Check for closed fist using solidity (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        debug_info['area'] = area
        debug_info['solidity'] = solidity
        debug_info['hull_area'] = hull_area
        
        # If solidity > 0.92, it's likely a closed fist (0 fingers)
        # Lowered from 0.95 for better closed fist detection
        if solidity > 0.92:
            debug_info['reason'] = 'closed_fist'
            if return_debug:
                return 0, debug_info
            return 0
        
        hand_radius = np.sqrt(area / np.pi)
        
        # Get hand center and bounding box
        M = cv2.moments(contour)
        if M["m00"] == 0:
            debug_info['reason'] = 'zero_moment'
            if return_debug:
                return 0, debug_info
            return 0
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        hand_center = (cx, cy)
        
        x, y, w, h = cv2.boundingRect(contour)
        hand_bbox = (x, y, w, h)
        
        # Method 1: Improved convexity defects with adaptive threshold
        count1, defect_debug = _count_by_convexity_defects_adaptive(contour, hand_center, hand_radius, return_debug=True)
        
        # Method 2: Contour extrema (topmost points)
        count2, extrema_debug = _count_by_extrema_points(contour, hand_center, hand_bbox, return_debug=True)
        
        # Method 3: Distance transform peaks
        count3 = _count_by_distance_transform(contour, hand_center)
        
        debug_info['method_counts'] = [count1, count2, count3]
        debug_info['hand_center'] = [int(hand_center[0]), int(hand_center[1])]
        debug_info['fingertips'] = defect_debug.get('fingertips', [])
        debug_info['valleys'] = defect_debug.get('valleys', [])
        debug_info['extrema_points'] = extrema_debug.get('points', [])
        
        # Check for open hand (5 fingers) using geometric features
        # Open hand has low solidity, large area, and many extrema points
        x, y, w, h = hand_bbox
        aspect_ratio = w / h if h > 0 else 1
        
        debug_info['aspect_ratio'] = aspect_ratio
        
        # Improved open hand detection with multiple criteria
        if (solidity < 0.75 and  # Not compact (relaxed from 0.7)
            area > 12000 and  # Large hand (lowered threshold)
            count2 >= 4 and  # Multiple extrema
            0.6 < aspect_ratio < 1.4):  # Roughly square (relaxed)
            # Likely open hand with 5 fingers
            final_count = 5
            debug_info['reason'] = 'open_hand_detection'
        # Alternative: very large area with multiple detection methods agreeing
        elif area > 20000 and count2 >= 4 and count3 >= 4:
            final_count = 5
            debug_info['reason'] = 'large_area_multi_detect'
        else:
            # Weighted voting: prioritize methods that found results
            # If convexity defects failed (returned 1), rely more on other methods
            if count1 == 1 and (count2 > 1 or count3 > 1):
                # Convexity defects failed, use average of other two
                final_count = int(np.mean([count2, count3]))
                debug_info['reason'] = 'defects_failed_use_avg'
            elif count1 == 0 or count2 == 0 or count3 == 0:
                # One method found 0, use max of the three
                final_count = max(count1, count2, count3)
                debug_info['reason'] = 'zero_found_use_max'
            else:
                # All methods found something, use median
                final_count = int(np.median([count1, count2, count3]))
                debug_info['reason'] = 'median_voting'
        
        # Apply temporal smoothing
        smoothed_count = _finger_smoother.update(final_count)
        debug_info['final_count'] = final_count
        debug_info['smoothed_count'] = smoothed_count
        
        if return_debug:
            return min(5, max(0, smoothed_count)), debug_info
        return min(5, max(0, smoothed_count))
        
    except Exception as e:
        debug_info['error'] = str(e)
        if return_debug:
            return 0, debug_info
        return 0


def _count_by_convexity_defects_adaptive(contour, hand_center, hand_radius, return_debug=False):
    """
    Improved convexity defects with adaptive thresholding
    Based on geometric analysis of hand contour
    """
    debug_info = {'fingertips': [], 'valleys': []}
    
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) <= 3:
        if return_debug:
            return 0, debug_info
        return 0
    
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0
    
    # Adaptive threshold scales with hand size
    # Set to 50 for benchmark (better detection, FPS not critical)
    adaptive_threshold = int(hand_radius * 50)  # Scale with hand radius
    
    valid_fingers = []
    
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        
        # Calculate triangle side lengths
        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(end) - np.array(far))
        
        # Skip if triangle is degenerate
        if b < 1 or c < 1:
            continue
        
        # Calculate angle at valley point (far)
        try:
            angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
        except:
            continue
        
        # Enhanced validation criteria  
        # Moderate angle of pi/2.0 (90°) to balance detection
        if (angle <= np.pi / 2.0 and  # Angle between finger sides
            d > adaptive_threshold and  # Deep enough defect
            b > hand_radius * 0.22 and  # Finger length
            c > hand_radius * 0.22 and
            start[1] < hand_center[1]):  # Finger above hand center
            
            # Check if not duplicate (far from existing fingers)
            is_duplicate = False
            for existing_finger in valid_fingers:
                if np.linalg.norm(np.array(start) - np.array(existing_finger)) < hand_radius * 0.4:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                valid_fingers.append(start)
                debug_info['fingertips'].append([int(start[0]), int(start[1])])
                debug_info['valleys'].append([int(far[0]), int(far[1])])
    
    # Count actual defects found
    count = len(valid_fingers)
    
    if count == 0:
        # No defects found - could mean:
        # - Closed fist (0 fingers) - checked by solidity in main function
        # - Single pointing finger (1 finger)
        # - Convexity defects failed to detect
        # Return 1 and let main function use other methods if needed
        if return_debug:
            return 1, debug_info
        return 1
    elif count >= 4:
        # Found 4+ defects = likely 5 fingers (open hand)
        if return_debug:
            return 5, debug_info
        return 5
    else:
        # Each defect represents gap between fingers, so add 1
        result = min(5, count + 1)
        if return_debug:
            return result, debug_info
        return result


def _count_by_extrema_points(contour, hand_center, hand_bbox, return_debug=False):
    """
    Count fingers by finding topmost extrema points
    Uses geometric clustering to identify distinct fingertips
    """
    debug_info = {'points': []}
    x, y, w, h = hand_bbox
    
    # Define upper region (top 50% of hand)
    top_region_y = y + h * 0.5
    
    # Extract all contour points
    points = contour.reshape(-1, 2)
    
    # Filter points in upper region
    top_points = points[points[:, 1] < top_region_y]
    
    if len(top_points) == 0:
        return 0
    
    # Sort by y-coordinate (topmost first)
    top_points = top_points[top_points[:, 1].argsort()]
    
    # Cluster nearby points (fingers are spatially separated)
    finger_tips = []
    min_separation = w / 5.5  # Slightly reduced from w/5
    
    for point in top_points[:30]:  # Check top 30 points (increased from 20)
        # Check if far enough from existing fingertips
        is_new_finger = True
        for tip in finger_tips:
            if np.linalg.norm(point - tip) < min_separation:
                is_new_finger = False
                break
        
        # Relaxed height requirement to detect more fingertips
        if is_new_finger and point[1] < hand_center[1]:
            finger_tips.append(point)
            debug_info['points'].append([int(point[0]), int(point[1])])
            
            # Stop if we found 5 fingertips
            if len(finger_tips) >= 5:
                break
    
    # Return count, ensuring it's at least 1 if hand detected
    result = max(1, min(5, len(finger_tips)))
    if return_debug:
        return result, debug_info
    return result


def _count_by_distance_transform(contour, hand_center):
    """
    Use distance transform to find finger peaks
    Distance transform is a morphological operation covered in vision courses
    """
    # Create binary mask from contour
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros((h + 20, w + 20), dtype=np.uint8)
    
    # Shift contour to fit in mask
    shifted_contour = contour.copy()
    shifted_contour[:, :, 0] -= (x - 10)
    shifted_contour[:, :, 1] -= (y - 10)
    
    cv2.drawContours(mask, [shifted_contour], 0, 255, -1)
    
    # Distance transform (distance from boundary)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # Normalize to 0-255
    dist_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Find peaks (local maxima) using morphological operations
    # Kernel size 6x6 is middle ground between 7x7 and 5x5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    dilated = cv2.dilate(dist_normalized, kernel)
    
    # Peaks are where original equals dilated (local maxima)
    # Threshold at 0.35 balances detection
    peaks = (dist_normalized == dilated) & (dist_normalized > 0.35 * dist_normalized.max())
    peaks = peaks.astype(np.uint8) * 255
    
    # Find connected components (each peak)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks, connectivity=8)
    
    # Filter peaks: only in upper part of hand
    valid_peaks = 0
    hand_center_local = (hand_center[0] - x + 10, hand_center[1] - y + 10)
    
    for i in range(1, num_labels):  # Skip background
        peak_y = centroids[i][1]
        peak_area = stats[i, cv2.CC_STAT_AREA]
        
        # Only count significant peaks above hand center
        # Relaxed height requirement and area filter
        if peak_y < hand_center_local[1] and peak_area > 3:
            valid_peaks += 1
    
    # Return at least 1 if hand detected, cap at 5
    return max(1, min(5, valid_peaks))


def smooth_finger_count(finger_history):
    """
    Smooth finger count using median filtering (legacy function)
    Note: New implementation uses FingerCountSmoother class for better temporal consistency
    
    Args:
        finger_history: Deque of recent finger counts
        
    Returns:
        int: Smoothed finger count
    """
    if len(finger_history) >= 3:
        return int(np.median(list(finger_history)))
    elif len(finger_history) >= 1:
        return int(finger_history[-1])
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
    Draw visualization of finger detection methods
    Shows convex hull and fingertip detections
    
    Args:
        frame: Frame to draw on
        contour: Hand contour
        defects: Convexity defects (optional, for backward compatibility)
        
    Returns:
        Modified frame
    """
    try:
        # Draw convex hull in yellow
        hull_points = cv2.convexHull(contour)
        cv2.drawContours(frame, [hull_points], 0, (0, 255, 255), 2)
        
        # Get hand geometry for visualization
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return frame
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Draw hand center
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)
        
        # Visualize convexity defects if provided
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                far = tuple(contour[f][0])
                
                # Draw fingertip candidates (peaks) - green
                cv2.circle(frame, start, 6, (0, 255, 0), -1)
                # Draw valleys between fingers - blue
                cv2.circle(frame, far, 4, (255, 0, 0), -1)
        
        # Visualize extrema points (topmost points in contour)
        x, y, w, h = cv2.boundingRect(contour)
        top_region_y = y + h * 0.5
        points = contour.reshape(-1, 2)
        top_points = points[points[:, 1] < top_region_y]
        
        if len(top_points) > 0:
            # Draw top 5 extrema points in cyan
            top_points_sorted = top_points[top_points[:, 1].argsort()][:5]
            for point in top_points_sorted:
                cv2.circle(frame, tuple(point), 4, (255, 255, 0), 1)
        
    except Exception as e:
        pass  # Fail silently for visualization
    
    return frame
