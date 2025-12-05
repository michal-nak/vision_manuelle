"""
Main CV Detector - Orchestrates modular components
Each logical piece is separated into dedicated modules
"""
import cv2
from ..hand_detector_base import HandDetectorBase
from ...core.config import BG_HISTORY, BG_VAR_THRESHOLD, BG_DETECT_SHADOWS

from .config_loader import load_skin_detection_config, load_processing_params
from .detector_state import DetectorState
from .tracking import OpticalFlowTracker
from .detection_pipeline import (
    detect_hand_full_pipeline,
    process_detected_hand,
    create_no_detection_result,
    create_tracking_result
)


class CVDetector(HandDetectorBase):
    """Enhanced CV-based hand detector with modular architecture"""
    
    def __init__(self, show_debug=False):
        # Load color calibration
        self.ycrcb_lower, self.ycrcb_upper, self.hsv_lower, self.hsv_upper = load_skin_detection_config()
        
        # Load processing parameters
        self.processing_params = load_processing_params()
        
        # Background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=BG_HISTORY,
            varThreshold=BG_VAR_THRESHOLD,
            detectShadows=BG_DETECT_SHADOWS
        )
        
        # State management
        self.state = DetectorState()
        
        # Optical flow tracker
        self.tracker = OpticalFlowTracker()
        self.tracking_mode = False
        self.prev_gray = None
        
        self.show_debug_overlay = show_debug
    
    def process_frame(self, frame):
        """Process frame using modular CV pipeline"""
        self.state.increment_frame()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try tracking first if enabled
        if self.tracking_mode:
            success, bbox, center = self.tracker.track_frame(gray)
            if success:
                result = create_tracking_result(frame, bbox, center, self.state, self.show_debug_overlay)
                self.prev_gray = gray.copy()
                return result
            # Tracking failed, fall through to detection
            self.tracking_mode = False
            self.tracker.reset()
        
        # Full detection pipeline
        color_bounds = (self.ycrcb_lower, self.ycrcb_upper, self.hsv_lower, self.hsv_upper)
        hand_contour, mask = detect_hand_full_pipeline(
            frame, gray, self.bg_subtractor, self.state, color_bounds, self.show_debug_overlay,
            self.processing_params
        )
        
        if hand_contour is None:
            return create_no_detection_result(frame, mask, self.state, self.show_debug_overlay)
        
        # Process detected hand
        result = process_detected_hand(frame, gray, hand_contour, self.state, self.show_debug_overlay)
        
        if result is None:
            return create_no_detection_result(frame, mask, self.state, self.show_debug_overlay)
        
        # Initialize tracking if hand is stable
        if not self.tracking_mode and len(self.state.position_history) >= 3:
            self.tracker.initialize_tracking(hand_contour, gray)
            self.tracking_mode = True
        
        self.prev_gray = gray.copy()
        return result

    
    def update_calibration(self, ycrcb_lower, ycrcb_upper, hsv_lower, hsv_upper):
        """Update color calibration bounds"""
        self.ycrcb_lower = ycrcb_lower
        self.ycrcb_upper = ycrcb_upper
        self.hsv_lower = hsv_lower
        self.hsv_upper = hsv_upper
        self.state.clear()
        self.tracker.reset()
    
    def reset_background(self):
        """Reset background subtractor"""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=BG_HISTORY,
            varThreshold=BG_VAR_THRESHOLD,
            detectShadows=BG_DETECT_SHADOWS
        )
        self.state.clear()
        self.tracker.reset()
    
    def cleanup(self):
        """Cleanup resources"""
        self.state.clear()
        self.tracker.reset()
