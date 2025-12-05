"""
Main CV Detector - Refactored into modular components
Each logical piece is separated for better maintainability
"""
import cv2
import numpy as np
import json
from collections import deque
from pathlib import Path

from ..hand_detector_base import HandDetectorBase
from ...core.config import (
    YCRCB_LOWER, YCRCB_UPPER, HSV_LOWER, HSV_UPPER,
    BG_HISTORY, BG_VAR_THRESHOLD, BG_DETECT_SHADOWS,
    POSITION_SMOOTHING, FINGER_COUNT_SMOOTHING,
    MIN_HAND_AREA, MAX_HAND_AREA
)

from .skin_detection import detect_skin_ycrcb_hsv, apply_morphological_operations, find_largest_contour
from .finger_detection import count_fingers_from_contour, smooth_finger_count, map_fingers_to_gesture, draw_finger_visualization
from .tracking import OpticalFlowTracker
from .visualization import draw_debug_overlay, draw_extended_debug_metrics


class CVDetector(HandDetectorBase):
    """Enhanced CV-based hand detector with modular architecture"""
    
    def __init__(self, show_debug=False):
        # Load color space bounds from JSON config if available
        config_path = Path(__file__).parent.parent.parent.parent / 'skin_detection_config.json'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.ycrcb_lower = np.array(config['ycrcb_lower'], dtype=np.uint8)
                self.ycrcb_upper = np.array(config['ycrcb_upper'], dtype=np.uint8)
                self.hsv_lower = np.array(config['hsv_lower'], dtype=np.uint8)
                self.hsv_upper = np.array(config['hsv_upper'], dtype=np.uint8)
            except Exception as e:
                print(f"⚠️  Failed to load skin detection config: {e}")
                # Fall back to defaults
                self.ycrcb_lower = np.array(YCRCB_LOWER, dtype=np.uint8)
                self.ycrcb_upper = np.array(YCRCB_UPPER, dtype=np.uint8)
                self.hsv_lower = np.array(HSV_LOWER, dtype=np.uint8)
                self.hsv_upper = np.array(HSV_UPPER, dtype=np.uint8)
        else:
            # Use config.py defaults
            self.ycrcb_lower = np.array(YCRCB_LOWER, dtype=np.uint8)
            self.ycrcb_upper = np.array(YCRCB_UPPER, dtype=np.uint8)
            self.hsv_lower = np.array(HSV_LOWER, dtype=np.uint8)
            self.hsv_upper = np.array(HSV_UPPER, dtype=np.uint8)
        
        # Background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=BG_HISTORY,
            varThreshold=BG_VAR_THRESHOLD,
            detectShadows=BG_DETECT_SHADOWS
        )
        
        # History and smoothing
        self.position_history = deque(maxlen=POSITION_SMOOTHING)
        self.finger_history = deque(maxlen=FINGER_COUNT_SMOOTHING)
        
        # Frame tracking
        self.frame_count = 0
        self.bg_learning_frames = 30
        
        # Optical flow tracker
        self.tracker = OpticalFlowTracker()
        self.tracking_mode = False
        self.prev_gray = None
        
        # Debug metrics
        self.debug_metrics = {
            'total_frames': 0,
            'detected_frames': 0,
            'tracking_frames': 0,
            'gesture_changes': 0,
            'last_gesture': 'None',
            'finger_transitions': [],
            'contour_areas': [],
            'hull_defect_counts': [],
            'tracking': False
        }
        self.show_debug_overlay = show_debug
    
    def process_frame(self, frame):
        """Process frame using modular CV pipeline"""
        self.frame_count += 1
        self.debug_metrics['total_frames'] += 1
        
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try tracking first if enabled
        if self.tracking_mode:
            result = self._try_tracking(frame, gray)
            if result is not None:
                self.prev_gray = gray.copy()
                return result
            # Tracking failed, fall through to detection
            self.tracking_mode = False
            self.tracker.reset()
        
        # Full detection pipeline
        return self._detect_hand(frame, gray)
    
    def _try_tracking(self, frame, gray):
        """Attempt to track hand using optical flow"""
        success, bbox, center = self.tracker.track_frame(gray)
        
        if not success:
            return None
        
        # Create output with tracked data
        x, y, w, h = bbox
        annotated = frame.copy()
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        # Use last known finger count
        finger_count = self.finger_history[-1] if self.finger_history else 0
        gesture = map_fingers_to_gesture(finger_count)
        
        self.debug_metrics['tracking'] = True
        self.debug_metrics['tracking_frames'] += 1
        
        if self.show_debug_overlay:
            annotated = draw_debug_overlay(
                annotated, self.debug_metrics, finger_count,
                gesture, 85, self.finger_history
            )
        
        return {
            'detected': True,
            'hand_center': (center[0] / w, center[1] / h),
            'finger_count': finger_count,
            'gesture': gesture,
            'annotated_frame': annotated
        }
    
    def _detect_hand(self, frame, gray):
        """Full hand detection pipeline"""
        # Skin detection
        mask = detect_skin_ycrcb_hsv(
            frame, self.ycrcb_lower, self.ycrcb_upper,
            self.hsv_lower, self.hsv_upper
        )
        
        # Motion filtering
        if self.frame_count > self.bg_learning_frames:
            fg_mask = self.bg_subtractor.apply(frame, learningRate=0.001)
            kernel_motion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            fg_mask = cv2.dilate(fg_mask, kernel_motion, iterations=2)
            mask = cv2.bitwise_and(mask, fg_mask)
        else:
            self.bg_subtractor.apply(frame, learningRate=0.1)
        
        # Clean up mask
        mask = apply_morphological_operations(mask)
        
        # Calculate max area in pixels (MAX_HAND_AREA is a fraction of frame area)
        h, w = frame.shape[:2]
        max_area_pixels = int(w * h * MAX_HAND_AREA)
        
        # Find hand contour
        hand_contour = find_largest_contour(mask, MIN_HAND_AREA, max_area_pixels)
        
        if hand_contour is None:
            # Debug: show why no contour found
            if self.show_debug_overlay:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest)
                    self.debug_metrics['largest_contour_area'] = area
            return self._no_detection_result(frame, mask)
        
        # Process detected hand
        return self._process_detected_hand(frame, gray, hand_contour)
    
    def _process_detected_hand(self, frame, gray, contour):
        """Process detected hand contour"""
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
            return self._no_detection_result(frame)
        
        # Count fingers
        finger_count = count_fingers_from_contour(contour)
        self.finger_history.append(finger_count)
        smooth_count = smooth_finger_count(self.finger_history)
        
        # Map to gesture
        gesture = map_fingers_to_gesture(smooth_count)
        
        # Update metrics
        self._update_metrics(contour, gesture)
        
        # Draw finger visualization
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(contour, hull)
            if defects is not None:
                annotated = draw_finger_visualization(annotated, contour, defects)
                self.debug_metrics['hull_defect_counts'].append(len(defects))
                if len(self.debug_metrics['hull_defect_counts']) > 30:
                    self.debug_metrics['hull_defect_counts'].pop(0)
        
        # Draw debug overlay
        if self.show_debug_overlay:
            area = cv2.contourArea(contour)
            confidence = min(100, int((area / 10000) * 100))
            annotated = draw_debug_overlay(
                annotated, self.debug_metrics, smooth_count,
                gesture, confidence, self.finger_history
            )
            annotated = draw_extended_debug_metrics(
                annotated, self.debug_metrics, self.finger_history
            )
        
        # Initialize tracking if hand is stable
        if not self.tracking_mode and len(self.position_history) >= 3:
            self.tracker.initialize_tracking(contour, gray)
            self.tracking_mode = True
        
        self.prev_gray = gray.copy()
        
        return {
            'detected': True,
            'hand_center': hand_center,
            'hand_x': hand_center[0],
            'hand_y': hand_center[1],
            'finger_count': smooth_count,
            'gesture': gesture,
            'annotated_frame': annotated
        }
    
    def _no_detection_result(self, frame, mask=None):
        """Return result when no hand detected"""
        self.debug_metrics['tracking'] = False
        
        # Show mask in corner for debugging
        annotated = frame.copy()
        if mask is not None and self.show_debug_overlay:
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
        if self.show_debug_overlay:
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
            if 'largest_contour_area' in self.debug_metrics:
                area = self.debug_metrics['largest_contour_area']
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
    
    def _update_metrics(self, contour, gesture):
        """Update debug metrics"""
        self.debug_metrics['detected_frames'] += 1
        self.debug_metrics['tracking'] = False
        
        # Track gesture changes
        if gesture != self.debug_metrics['last_gesture']:
            self.debug_metrics['gesture_changes'] += 1
            self.debug_metrics['finger_transitions'].append(
                (self.debug_metrics['last_gesture'], gesture, self.debug_metrics['total_frames'])
            )
            if len(self.debug_metrics['finger_transitions']) > 10:
                self.debug_metrics['finger_transitions'].pop(0)
            self.debug_metrics['last_gesture'] = gesture
        
        # Track contour metrics
        area = cv2.contourArea(contour)
        self.debug_metrics['contour_areas'].append(area)
        if len(self.debug_metrics['contour_areas']) > 30:
            self.debug_metrics['contour_areas'].pop(0)
    
    def update_calibration(self, ycrcb_lower, ycrcb_upper, hsv_lower, hsv_upper):
        """Update color calibration bounds"""
        self.ycrcb_lower = ycrcb_lower
        self.ycrcb_upper = ycrcb_upper
        self.hsv_lower = hsv_lower
        self.hsv_upper = hsv_upper
        self.position_history.clear()
        self.finger_history.clear()
        self.tracker.reset()
    
    def cleanup(self):
        """Cleanup resources"""
        self.position_history.clear()
        self.finger_history.clear()
        self.tracker.reset()
