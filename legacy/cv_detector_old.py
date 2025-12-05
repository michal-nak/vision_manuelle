"""
Enhanced computer vision-based hand detection
Uses multiple techniques: YCrCb + HSV skin detection, adaptive filtering,
improved contour analysis, and temporal smoothing
"""
import cv2
import numpy as np
import math
from collections import deque
from .hand_detector_base import HandDetectorBase
from ..core.config import (
    YCRCB_LOWER, YCRCB_UPPER, HSV_LOWER, HSV_UPPER,
    BG_HISTORY, BG_VAR_THRESHOLD, BG_DETECT_SHADOWS,
    POSITION_SMOOTHING, FINGER_COUNT_SMOOTHING,
    MIN_HAND_AREA, MAX_HAND_AREA
)

class CVDetector(HandDetectorBase):
    def __init__(self):
        self.ycrcb_lower = np.array([128, 117, 28], dtype=np.uint8)
        self.ycrcb_upper = np.array([199, 163, 131], dtype=np.uint8)
        self.hsv_lower = np.array([3, 42, 135], dtype=np.uint8)
        self.hsv_upper = np.array([35, 255, 255], dtype=np.uint8)
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=BG_HISTORY, 
            varThreshold=BG_VAR_THRESHOLD, 
            detectShadows=BG_DETECT_SHADOWS
        )
        
        self.position_history = deque(maxlen=POSITION_SMOOTHING)
        self.finger_history = deque(maxlen=FINGER_COUNT_SMOOTHING)
        
        self.frame_count = 0
        self.bg_learning_frames = 30
        
        # Optical flow tracking mode (inspired by MediaPipe's tracking)
        self.tracking_mode = False
        self.tracked_points = None
        self.tracking_bbox = None
        self.tracking_lost_frames = 0
        self.max_tracking_lost = 10  # Switch back to detection after this many lost frames
        self.prev_gray = None
        
        # LK optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Debugging metrics
        self.debug_metrics = {
            'total_frames': 0,
            'detected_frames': 0,
            'tracking_frames': 0,
            'gesture_changes': 0,
            'last_gesture': 'None',
            'finger_transitions': [],
            'contour_areas': [],
            'hull_defect_counts': []
        }
        self.show_debug_overlay = True
    
    def process_frame(self, frame):
        """Process frame using enhanced CV methods with optical flow tracking"""
        self.frame_count += 1
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try optical flow tracking first if in tracking mode
        if self.tracking_mode and self.prev_gray is not None:
            tracking_result = self._try_optical_flow_tracking(frame, gray)
            if tracking_result is not None:
                self.prev_gray = gray.copy()
                return tracking_result
            else:
                # Tracking failed, switch back to detection
                self.tracking_mode = False
                self.tracked_points = None
                self.tracking_bbox = None
        
        # Full detection mode
        # Adaptive filtering for denoising
        denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        
        # YCrCb skin detection
        ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, self.ycrcb_lower, self.ycrcb_upper)
        
        # HSV skin detection
        hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # YCrCb && HSV
        mask_combined = cv2.bitwise_and(mask_ycrcb, mask_hsv)
        
        # Motion-based filtering (bias for backgrounds preference)
        if self.frame_count > self.bg_learning_frames:
            fg_mask = self.bg_subtractor.apply(frame, learningRate=0.001)
            
            # Dilate foreground mask to ensure hand is captured
            kernel_motion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            fg_mask = cv2.dilate(fg_mask, kernel_motion, iterations=2)
            
            # fg mask &&  skin mask
            mask_combined = cv2.bitwise_and(mask_combined, fg_mask)

        else:
            # Learn background in first frames
            self.bg_subtractor.apply(frame, learningRate=0.1)
        
        # Morphological operations
        # Use larger kernels
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        
        # Remove small noise
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel_small, iterations=2)
        # Fill holes in hand region
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel_large, iterations=3)
        # Final smoothing
        mask_combined = cv2.GaussianBlur(mask_combined, (5, 5), 0)
        
        # Contour selection
        contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hand_contour = self._select_best_hand_contour(contours, h, w)
        
        # Prepare output
        output = {
            'detected': False,
            'hand_x': 0.5,
            'hand_y': 0.5,
            'finger_count': 0,
            'annotated_frame': frame.copy()
        }
        
        if hand_contour is not None:
            # Draw contour and hull
            cv2.drawContours(output['annotated_frame'], [hand_contour], -1, (0, 255, 0), 2)
            hull = cv2.convexHull(hand_contour, returnPoints=True)
            cv2.drawContours(output['annotated_frame'], [hull], -1, (255, 0, 0), 2)
            
            # Get hand center with temporal smoothing
            M = cv2.moments(hand_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Apply temporal smoothing to position
                self.position_history.append((cx, cy))
                if len(self.position_history) >= 3:
                    smooth_cx = int(np.mean([p[0] for p in self.position_history]))
                    smooth_cy = int(np.mean([p[1] for p in self.position_history]))
                else:
                    smooth_cx, smooth_cy = cx, cy
                
                # Normalize position
                output['hand_x'] = smooth_cx / w
                output['hand_y'] = smooth_cy / h
                
                # Draw center
                cv2.circle(output['annotated_frame'], (smooth_cx, smooth_cy), 10, (0, 255, 0), -1)
            
            # Enhanced finger counting with multiple methods
            finger_count = self._count_fingers_enhanced(hand_contour, output['annotated_frame'])
            
            # Apply temporal smoothing to finger count
            self.finger_history.append(finger_count)
            if len(self.finger_history) >= 2:
                # Use median to avoid outliers
                smooth_finger_count = int(np.median(list(self.finger_history)))
            else:
                smooth_finger_count = finger_count
            
            output['finger_count'] = smooth_finger_count
            output['detected'] = True
            
            # Map finger count to gesture (for compatibility with gesture_paint.py)
            gesture_map = {
                0: "None",
                1: "Draw",      # 1 finger = Draw
                2: "Erase",     # 2 fingers = Erase
                3: "None",      # 3 fingers = None
                4: "None",      # 4 fingers = None
                5: "Clear"      # 5 fingers = Clear
            }
            output['gesture'] = gesture_map.get(smooth_finger_count, "None")
            
            # Initialize tracking mode when hand is well detected
            if not self.tracking_mode and len(self.position_history) >= 3:
                self._initialize_tracking(hand_contour, gray)
            
            # Enhanced visualization
            mode_text = "TRACKING" if self.tracking_mode else "DETECTING"
            mode_color = (0, 255, 255) if self.tracking_mode else (255, 255, 255)
            
            # Large overlay box for status
            overlay = output['annotated_frame'].copy()
            cv2.rectangle(overlay, (5, 5), (635, 130), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, output['annotated_frame'], 0.4, 0, output['annotated_frame'])
            
            # Mode and fingers
            cv2.putText(output['annotated_frame'], f"Mode: {mode_text}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
            cv2.putText(output['annotated_frame'], f"Fingers: {smooth_finger_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Gesture with color coding
            gesture_colors = {
                "Draw": (0, 255, 0),
                "Erase": (0, 165, 255),
                "Clear": (0, 0, 255),
                "None": (128, 128, 128)
            }
            gesture_color = gesture_colors.get(output['gesture'], (255, 255, 255))
            cv2.putText(output['annotated_frame'], f"Gesture: {output['gesture']}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, gesture_color, 2)
            
            # Confidence
            area = cv2.contourArea(hand_contour)
            confidence = min(100, int((area / 10000) * 100))
            cv2.putText(output['annotated_frame'], f"Confidence: {confidence}%", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Finger count legend on right side
            legend_x = 450
            cv2.rectangle(output['annotated_frame'], (legend_x-5, 5), (635, 110), (0, 0, 0), -1)
            cv2.putText(output['annotated_frame'], "Finger Map:", (legend_x, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(output['annotated_frame'], "1 = Draw", (legend_x, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(output['annotated_frame'], "2 = Erase", (legend_x, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
            cv2.putText(output['annotated_frame'], "5 = Clear", (legend_x, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            cv2.putText(output['annotated_frame'], "0/3/4 = None", (legend_x, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1)
            
            # Update debug metrics
            self.debug_metrics['total_frames'] += 1
            self.debug_metrics['detected_frames'] += 1
            if self.tracking_mode:
                self.debug_metrics['tracking_frames'] += 1
            
            # Track gesture changes
            if output['gesture'] != self.debug_metrics['last_gesture']:
                self.debug_metrics['gesture_changes'] += 1
                self.debug_metrics['finger_transitions'].append(
                    (self.debug_metrics['last_gesture'], output['gesture'], self.debug_metrics['total_frames'])
                )
                if len(self.debug_metrics['finger_transitions']) > 10:
                    self.debug_metrics['finger_transitions'].pop(0)
                self.debug_metrics['last_gesture'] = output['gesture']
            
            # Track contour metrics
            self.debug_metrics['contour_areas'].append(area)
            if len(self.debug_metrics['contour_areas']) > 30:
                self.debug_metrics['contour_areas'].pop(0)
            
            # Debug overlay with detailed metrics
            if self.show_debug_overlay:
                debug_y = 150
                # Semi-transparent background for debug section
                overlay2 = output['annotated_frame'].copy()
                cv2.rectangle(overlay2, (5, debug_y-10), (635, debug_y+180), (20, 20, 20), -1)
                cv2.addWeighted(overlay2, 0.7, output['annotated_frame'], 0.3, 0, output['annotated_frame'])
                
                cv2.putText(output['annotated_frame'], "=== DEBUG METRICS ===", 
                           (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                debug_y += 25
                
                detection_rate = (self.debug_metrics['detected_frames'] / max(1, self.debug_metrics['total_frames'])) * 100
                cv2.putText(output['annotated_frame'], f"Detection Rate: {detection_rate:.1f}%", 
                           (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                debug_y += 20
                
                cv2.putText(output['annotated_frame'], f"Total Frames: {self.debug_metrics['total_frames']}", 
                           (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                debug_y += 20
                
                cv2.putText(output['annotated_frame'], f"Gesture Changes: {self.debug_metrics['gesture_changes']}", 
                           (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                debug_y += 20
                
                cv2.putText(output['annotated_frame'], f"Contour Area: {int(area)}", 
                           (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                debug_y += 20
                
                # Finger count history
                history_str = ' -> '.join([str(int(x)) for x in list(self.finger_history)[-5:]])
                cv2.putText(output['annotated_frame'], f"Finger History: {history_str}", 
                           (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                debug_y += 20
                
                # Recent gesture transitions
                cv2.putText(output['annotated_frame'], "Recent Transitions:", 
                           (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
                debug_y += 18
                
                for old_g, new_g, frame in list(self.debug_metrics['finger_transitions'])[-3:]:
                    trans_text = f"  {old_g} -> {new_g} (frame {frame})"
                    cv2.putText(output['annotated_frame'], trans_text, 
                               (10, debug_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                    debug_y += 16
            
            # Draw convex hull and defects for finger detection visualization
            hull = cv2.convexHull(hand_contour, returnPoints=False)
            if len(hull) > 3:
                defects = cv2.convexityDefects(hand_contour, hull)
                if defects is not None:
                    # Draw convex hull in yellow
                    hull_points = cv2.convexHull(hand_contour)
                    cv2.drawContours(output['annotated_frame'], [hull_points], 0, (0, 255, 255), 2)
                    
                    # Mark fingertips (peaks) with green circles
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(hand_contour[s][0])
                        end = tuple(hand_contour[e][0])
                        far = tuple(hand_contour[f][0])
                        
                        # Draw fingertip (peak)
                        cv2.circle(output['annotated_frame'], start, 8, (0, 255, 0), -1)
                        # Draw valley between fingers
                        cv2.circle(output['annotated_frame'], far, 5, (255, 0, 0), -1)
                    
                    self.debug_metrics['hull_defect_counts'].append(len(defects))
                    if len(self.debug_metrics['hull_defect_counts']) > 30:
                        self.debug_metrics['hull_defect_counts'].pop(0)
        else:
            cv2.putText(output['annotated_frame'], "Hand not detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Clear history when hand is lost
            self.position_history.clear()
            self.finger_history.clear()
            self.tracking_mode = False
            self.tracked_points = None
        
        self.prev_gray = gray.copy()
        return output
    
    def _select_best_hand_contour(self, contours, frame_h, frame_w):
        """Select best hand contour using multiple criteria"""
        if not contours:
            return None
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < MIN_HAND_AREA or area > (frame_h * frame_w * MAX_HAND_AREA):
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter < 200:
                continue
            
            compactness = (4 * np.pi * area) / (perimeter ** 2)
            if compactness < 0.2 or compactness > 0.95:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio > 2.0 or aspect_ratio < 0.3:
                continue
            
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area) / hull_area
                if solidity < 0.4 or solidity > 0.9:
                    continue
            
            center_y = y + h / 2
            position_score = 1.0 - (center_y / frame_h) * 0.3
            
            score = area * position_score * compactness
            candidates.append((score, contour))
        
        if not candidates:
            return None
        
        return max(candidates, key=lambda x: x[0])[1]
    
    def _angle_between(self, a, b, c):
        """Calculate angle at b between points a-b-c"""
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
    
    def _count_fingers_enhanced(self, contour, debug_frame):
        """Enhanced finger counting using convex hull peak detection"""
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0
        
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        
        hull = cv2.convexHull(contour, returnPoints=True)
        if len(hull) < 5:
            return 0
        
        distances = []
        hull_points = []
        for i in range(len(hull)):
            pt = tuple(hull[i][0])
            dist = math.hypot(pt[0] - center_x, pt[1] - center_y)
            distances.append(dist)
            hull_points.append(pt)
        
        if not distances:
            return 0
        
        max_dist = max(distances)
        avg_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        distance_threshold = avg_dist + (std_dist * 0.5)
        
        peaks = []
        for i in range(len(distances)):
            if distances[i] > distance_threshold:
                prev_idx = (i - 1) % len(distances)
                next_idx = (i + 1) % len(distances)
                
                if distances[i] >= distances[prev_idx] and distances[i] >= distances[next_idx]:
                    peaks.append(i)
        
        candidate_fingers = []
        for peak_idx in peaks:
            pt = hull_points[peak_idx]
            
            if pt[1] < center_y + avg_dist * 0.3:
                
                is_duplicate = False
                for existing_pt, _ in candidate_fingers:
                    if math.hypot(pt[0] - existing_pt[0], pt[1] - existing_pt[1]) < 20:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    candidate_fingers.append((pt, distances[peak_idx]))
        
        candidate_fingers.sort(key=lambda x: x[0][0])
        
        final_fingers = []
        min_finger_spacing = 15
        
        for i, (pt, dist) in enumerate(candidate_fingers):
            if i == 0:
                final_fingers.append(pt)
            else:
                last_pt = final_fingers[-1]
                spacing = abs(pt[0] - last_pt[0])
                
                if spacing >= min_finger_spacing:
                    final_fingers.append(pt)
                elif dist > candidate_fingers[candidate_fingers.index((last_pt, distances[hull_points.index(last_pt)]))][1]:
                    final_fingers[-1] = pt
        
        for tip in final_fingers:
            cv2.circle(debug_frame, tip, 8, (255, 0, 255), -1)
        
        cv2.circle(debug_frame, (center_x, center_y), 5, (0, 255, 255), -1)
        
        finger_count = len(final_fingers)
        
        hull_idx = cv2.convexHull(contour, returnPoints=False)
        if len(hull_idx) > 3:
            defects = cv2.convexityDefects(contour, hull_idx)
            if defects is not None:
                deep_valleys = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    far = tuple(contour[f][0])
                    
                    if far[1] < center_y and d > 8000:
                        angle = self._angle_between(
                            tuple(contour[s][0]),
                            far,
                            tuple(contour[e][0])
                        )
                        if angle < 90:
                            deep_valleys += 1
                            cv2.circle(debug_frame, far, 5, (0, 0, 255), -1)
                
                if deep_valleys > 0 and finger_count > 0:
                    finger_count = min(deep_valleys + 1, finger_count)
        
        return max(0, min(finger_count, 5))
    
    def set_ycrcb_range(self, lower, upper):
        """Update YCrCb range for skin detection"""
        self.ycrcb_lower = np.array([128, 117, 28], dtype=np.uint8)
        self.ycrcb_upper = np.array([199, 163, 131], dtype=np.uint8)
    
    def set_hsv_range(self, lower, upper):
        """Update HSV range for skin detection"""
        self.hsv_lower = np.array([3, 42, 135], dtype=np.uint8)
        self.hsv_upper = np.array([35, 255, 255], dtype=np.uint8)
    
    def _initialize_tracking(self, contour, gray_frame):
        """Initialize optical flow tracking on detected hand"""
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        self.tracking_bbox = (x, y, w, h)
        
        # Select good features to track within the hand region
        mask = np.zeros(gray_frame.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Detect corner points in hand region
        corners = cv2.goodFeaturesToTrack(
            gray_frame,
            maxCorners=50,
            qualityLevel=0.01,
            minDistance=10,
            mask=mask
        )
        
        if corners is not None and len(corners) > 10:
            self.tracked_points = corners
            self.tracking_mode = True
            self.tracking_lost_frames = 0
            print("Tracking mode ACTIVATED")
    
    def _try_optical_flow_tracking(self, frame, gray):
        """Try to track hand using optical flow (Lucas-Kanade)"""
        if self.tracked_points is None or len(self.tracked_points) < 5:
            return None
        
        # Calculate optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.tracked_points,
            None,
            **self.lk_params
        )
        
        if new_points is None:
            self.tracking_lost_frames += 1
            if self.tracking_lost_frames > self.max_tracking_lost:
                return None
            return None
        
        # Select good points
        good_new = new_points[status.flatten() == 1]
        good_old = self.tracked_points[status.flatten() == 1]
        
        # Need at least 5 points to continue tracking
        if len(good_new) < 5:
            self.tracking_lost_frames += 1
            if self.tracking_lost_frames > self.max_tracking_lost:
                return None
            # Try to continue with fewer points
            good_new = new_points[status.flatten() == 1]
        else:
            self.tracking_lost_frames = 0
        
        # Update tracked points
        self.tracked_points = good_new.reshape(-1, 1, 2)
        
        # Compute hand center from tracked points
        if len(good_new) > 0:
            center_x = int(np.mean(good_new[:, 0]))
            center_y = int(np.mean(good_new[:, 1]))
            
            # Update bounding box
            x_coords = good_new[:, 0]
            y_coords = good_new[:, 1]
            x, y = int(x_coords.min()), int(y_coords.min())
            w, h = int(x_coords.max() - x), int(y_coords.max() - y)
            
            # Add margin
            margin = 30
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(frame.shape[1] - x, w + 2*margin)
            h = min(frame.shape[0] - y, h + 2*margin)
            
            self.tracking_bbox = (x, y, w, h)
            
            # Prepare output
            output = {
                'detected': True,
                'hand_x': center_x / frame.shape[1],
                'hand_y': center_y / frame.shape[0],
                'finger_count': self.finger_history[-1] if self.finger_history else 0,
                'annotated_frame': frame.copy()
            }
            
            # Draw tracking visualization
            for pt in good_new:
                cv2.circle(output['annotated_frame'], (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
            
            # Draw bounding box
            cv2.rectangle(output['annotated_frame'], (x, y), (x+w, y+h), (255, 255, 0), 2)
            
            # Draw center
            cv2.circle(output['annotated_frame'], (center_x, center_y), 10, (0, 255, 255), -1)
            
            # Draw motion vectors
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                a, b, c, d = int(a), int(b), int(c), int(d)
                cv2.line(output['annotated_frame'], (a, b), (c, d), (0, 255, 0), 1)
            
            # Try to estimate finger count from hand region
            hand_region = frame[y:y+h, x:x+w]
            if hand_region.size > 0:
                # Quick finger estimation from tracked region
                finger_count = self._estimate_fingers_from_region(hand_region, output['annotated_frame'], (x, y))
                if finger_count > 0:
                    self.finger_history.append(finger_count)
                    output['finger_count'] = int(np.median(list(self.finger_history))) if self.finger_history else finger_count
            
            return output
        
        return None
    
    def _estimate_fingers_from_region(self, region, debug_frame, offset):
        """Quick finger estimation from tracked hand region"""
        if region.shape[0] < 30 or region.shape[1] < 30:
            return 0
        
        # Quick skin detection
        ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, self.ycrcb_lower, self.ycrcb_upper)
        
        # Simple morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0
        
        # Get largest contour
        hand_contour = max(contours, key=cv2.contourArea)
        
        # Quick finger count using convex hull
        hull = cv2.convexHull(hand_contour, returnPoints=True)
        if len(hull) < 5:
            return 0
        
        # Count peaks
        M = cv2.moments(hand_contour)
        if M["m00"] == 0:
            return 0
        
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        
        # Find peaks in hull
        distances = [math.hypot(pt[0][0] - center_x, pt[0][1] - center_y) for pt in hull]
        avg_dist = np.mean(distances)
        
        peaks = 0
        for i in range(len(distances)):
            if distances[i] > avg_dist * 1.2:
                prev_idx = (i - 1) % len(distances)
                next_idx = (i + 1) % len(distances)
                if distances[i] >= distances[prev_idx] and distances[i] >= distances[next_idx]:
                    peaks += 1
        
        return min(peaks, 5)
    
    def reset_background(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        self.frame_count = 0
        self.tracking_mode = False
        self.tracked_points = None
    
    def cleanup(self):
        """Clean up resources"""
        self.position_history.clear()
        self.finger_history.clear()
        self.tracked_points = None
        self.prev_gray = None
