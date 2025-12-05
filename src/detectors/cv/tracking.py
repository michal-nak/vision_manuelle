"""
Optical flow tracking for hand movement
"""
import cv2
import numpy as np


class OpticalFlowTracker:
    """Tracks hand using Lucas-Kanade optical flow"""
    
    def __init__(self):
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.tracked_points = None
        self.tracking_bbox = None
        self.prev_gray = None
        self.tracking_lost_frames = 0
        self.max_tracking_lost = 10
    
    def initialize_tracking(self, contour, gray_frame):
        """
        Initialize tracking with detected hand contour
        
        Args:
            contour: Hand contour
            gray_frame: Grayscale frame
        """
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        self.tracking_bbox = (x, y, w, h)
        
        # Select good features to track within the hand region
        mask = np.zeros_like(gray_frame)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        self.tracked_points = cv2.goodFeaturesToTrack(
            gray_frame,
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7,
            mask=mask
        )
        
        self.prev_gray = gray_frame.copy()
        self.tracking_lost_frames = 0
    
    def track_frame(self, current_gray):
        """
        Track hand in current frame using optical flow
        
        Args:
            current_gray: Current grayscale frame
            
        Returns:
            (success, bbox, center) or (False, None, None)
        """
        if self.tracked_points is None or self.prev_gray is None:
            return False, None, None
        
        # Calculate optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            current_gray,
            self.tracked_points,
            None,
            **self.lk_params
        )
        
        if new_points is None:
            self.tracking_lost_frames += 1
            return False, None, None
        
        # Select good points
        good_new = new_points[status == 1]
        
        if len(good_new) < 10:  # Not enough points
            self.tracking_lost_frames += 1
            if self.tracking_lost_frames > self.max_tracking_lost:
                return False, None, None
            return True, self.tracking_bbox, self._get_bbox_center(self.tracking_bbox)
        
        # Update tracked points
        self.tracked_points = good_new.reshape(-1, 1, 2)
        
        # Calculate new bounding box
        x_coords = good_new[:, 0]
        y_coords = good_new[:, 1]
        
        x, y = int(x_coords.min()), int(y_coords.min())
        w = int(x_coords.max() - x)
        h = int(y_coords.max() - y)
        
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = w + 2 * padding
        h = h + 2 * padding
        
        self.tracking_bbox = (x, y, w, h)
        self.prev_gray = current_gray.copy()
        self.tracking_lost_frames = 0
        
        center = self._get_bbox_center(self.tracking_bbox)
        return True, self.tracking_bbox, center
    
    def reset(self):
        """Reset tracking state"""
        self.tracked_points = None
        self.tracking_bbox = None
        self.prev_gray = None
        self.tracking_lost_frames = 0
    
    @staticmethod
    def _get_bbox_center(bbox):
        """Get center of bounding box"""
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)
