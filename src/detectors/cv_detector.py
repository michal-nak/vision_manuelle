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
    
    def process_frame(self, frame):
        """Process frame using enhanced CV methods"""
        self.frame_count += 1
        h, w = frame.shape[:2]
        
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
            
            # Add info text
            cv2.putText(output['annotated_frame'], f"Fingers: {smooth_finger_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show detection confidence
            area = cv2.contourArea(hand_contour)
            confidence = min(100, int((area / 10000) * 100))
            cv2.putText(output['annotated_frame'], f"Confidence: {confidence}%", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(output['annotated_frame'], "Hand not detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Clear history when hand is lost
            self.position_history.clear()
            self.finger_history.clear()
        
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
    
    def reset_background(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        self.frame_count = 0
    
    def cleanup(self):
        """Clean up resources"""
        self.position_history.clear()
        self.finger_history.clear()
