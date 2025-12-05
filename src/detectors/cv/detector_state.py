"""
State management for CV detector
Handles metrics, history buffers, and frame tracking
"""
import cv2
from collections import deque
from ...core.config import POSITION_SMOOTHING, FINGER_COUNT_SMOOTHING


class DetectorState:
    """Manages detector state including history and metrics"""
    
    def __init__(self):
        # History and smoothing
        self.position_history = deque(maxlen=POSITION_SMOOTHING)
        self.finger_history = deque(maxlen=FINGER_COUNT_SMOOTHING)
        
        # Frame tracking
        self.frame_count = 0
        self.bg_learning_frames = 30
        
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
    
    def increment_frame(self):
        """Increment frame counters"""
        self.frame_count += 1
        self.debug_metrics['total_frames'] += 1
    
    def update_detection_metrics(self, contour, gesture):
        """Update metrics after successful detection"""
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
    
    def update_tracking_metrics(self):
        """Update metrics for tracking mode"""
        self.debug_metrics['tracking'] = True
        self.debug_metrics['tracking_frames'] += 1
    
    def mark_no_tracking(self):
        """Mark that tracking is not active"""
        self.debug_metrics['tracking'] = False
    
    def clear(self):
        """Clear all history buffers"""
        self.position_history.clear()
        self.finger_history.clear()
