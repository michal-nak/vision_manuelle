"""
Enhanced computer vision-based hand detection
Uses multiple techniques: YCrCb + HSV skin detection, adaptive filtering,
improved contour analysis, and temporal smoothing
"""
import cv2
import numpy as np
import math
from collections import deque
from hand_detector_base import HandDetectorBase
from config import (
    YCRCB_LOWER, YCRCB_UPPER, HSV_LOWER, HSV_UPPER,
    BG_HISTORY, BG_VAR_THRESHOLD, BG_DETECT_SHADOWS,
    POSITION_SMOOTHING, FINGER_COUNT_SMOOTHING,
    MIN_HAND_AREA, MAX_HAND_AREA
)

class CVDetector(HandDetectorBase):
    def __init__(self):
        self.ycrcb_lower = np.array(YCRCB_LOWER, dtype=np.uint8)
        self.ycrcb_upper = np.array(YCRCB_UPPER, dtype=np.uint8)
        self.hsv_lower = np.array(HSV_LOWER, dtype=np.uint8)
        self.hsv_upper = np.array(HSV_UPPER, dtype=np.uint8)
        
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
        
        # Stage 1: Preprocessing with adaptive filtering
        # Reduce noise while preserving edges
        denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        
        # Stage 2: Multi-channel skin detection
        # YCrCb is more robust to lighting changes than HSV
        ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, self.ycrcb_lower, self.ycrcb_upper)
        
        # HSV for additional validation
        hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        
        # Combine masks (AND operation for higher confidence)
        mask_combined = cv2.bitwise_and(mask_ycrcb, mask_hsv)
        
        # Stage 3: Motion-based filtering (helps with static backgrounds)
        if self.frame_count > self.bg_learning_frames:
            fg_mask = self.bg_subtractor.apply(frame, learningRate=0.001)
            # Dilate foreground mask to ensure hand is captured
            kernel_motion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            fg_mask = cv2.dilate(fg_mask, kernel_motion, iterations=2)
            # Combine with skin mask
            mask_combined = cv2.bitwise_and(mask_combined, fg_mask)
        else:
            # Learn background in first frames
            self.bg_subtractor.apply(frame, learningRate=0.1)
        
        # Stage 4: Advanced morphological operations
        # Use larger kernels for better noise removal
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        
        # Remove small noise
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel_small, iterations=2)
        # Fill holes in hand region
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel_large, iterations=3)
        # Final smoothing
        mask_combined = cv2.GaussianBlur(mask_combined, (5, 5), 0)
        
        # Stage 5: Intelligent contour selection
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
        """Enhanced finger counting using multiple methods"""
        # Method 1: Convexity defects (primary method)
        hull_idx = cv2.convexHull(contour, returnPoints=False)
        if len(hull_idx) <= 3:
            return 0
        
        defects = cv2.convexityDefects(contour, hull_idx)
        if defects is None:
            return 0
        
        # Find the center of the hand for distance calculations
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        center = (center_x, center_y)
        
        # Calculate average distance from center to contour (hand radius)
        distances = [math.hypot(pt[0][0] - center_x, pt[0][1] - center_y) for pt in contour]
        avg_radius = np.mean(distances)
        
        # Detect finger tips and valleys
        finger_tips = []
        valleys = []
        
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            # Distance from center to defect point
            dist_start = math.hypot(start[0] - center_x, start[1] - center_y)
            dist_end = math.hypot(end[0] - center_x, end[1] - center_y)
            dist_far = math.hypot(far[0] - center_x, far[1] - center_y)
            
            # Angle at valley
            angle = self._angle_between(start, far, end)
            
            # Valid valley: angle < 90, far point close to center, depth significant
            if angle < 90 and dist_far < avg_radius * 0.8 and d > 5000:
                valleys.append(far)
                
                # Finger tips are at start/end if they're far from center
                if dist_start > avg_radius * 0.9:
                    finger_tips.append(start)
                if dist_end > avg_radius * 0.9:
                    finger_tips.append(end)
                
                # Draw valleys for debugging
                cv2.circle(debug_frame, far, 5, (0, 0, 255), -1)
        
        # Method 2: Distance-based finger detection (backup)
        hull_points = cv2.convexHull(contour, returnPoints=True)
        extreme_points = []
        
        for pt in hull_points:
            point = tuple(pt[0])
            dist = math.hypot(point[0] - center_x, point[1] - center_y)
            # Points far from center are potential finger tips
            if dist > avg_radius * 0.95:
                extreme_points.append(point)
        
        # Combine and filter finger tips
        all_tips = finger_tips + extreme_points
        
        # Remove duplicates (merge points within 30 pixels)
        filtered_tips = []
        for p in all_tips:
            if not any(math.hypot(p[0] - q[0], p[1] - q[1]) < 30 for q in filtered_tips):
                filtered_tips.append(p)
        
        # Draw finger tips
        for tip in filtered_tips:
            cv2.circle(debug_frame, tip, 8, (255, 0, 255), -1)
        
        # Finger count logic based on tips and valleys
        finger_count = len(filtered_tips)
        
        # Heuristic corrections:
        # - If we have good valleys, use valley count + 1
        # - Otherwise use tip count with reasonable bounds
        if len(valleys) > 0:
            finger_count = min(len(valleys) + 1, 5)
        else:
            finger_count = min(max(finger_count, 0), 5)
        
        # If very few tips/valleys detected, might be fist (0) or flat hand
        if finger_count <= 1 and cv2.contourArea(contour) > 8000:
            # Large area with few tips = likely flat hand (5 fingers)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 500:
                finger_count = 5
            else:
                finger_count = 0  # Fist
        
        return finger_count
    
    def set_ycrcb_range(self, lower, upper):
        """Update YCrCb range for skin detection"""
        self.ycrcb_lower = np.array([126, 117, 28], dtype=np.uint8)
        self.ycrcb_upper = np.array([203, 155, 129], dtype=np.uint8)
    
    def set_hsv_range(self, lower, upper):
        """Update HSV range for skin detection"""
        self.hsv_lower = np.array([12, 25, 121], dtype=np.uint8)
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
