"""
Base class for hand detection methods
"""
from abc import ABC, abstractmethod

class HandDetectorBase(ABC):
    
    @abstractmethod
    def process_frame(self, frame):
        """
        Process a frame and detect hand(s)
        
        Args:
            frame: BGR image from camera
            
        Returns:
            dict with keys:
                - 'detected': bool, whether hand was detected
                - 'hand_x': float (0-1), normalized x position
                - 'hand_y': float (0-1), normalized y position
                - 'finger_count': int, number of extended fingers
                - 'annotated_frame': BGR image with annotations
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        pass
