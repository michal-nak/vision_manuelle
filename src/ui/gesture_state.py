"""
Gesture handling and state management logic
"""
import time
from collections import deque


class GestureStateManager:
    """Manages gesture states and transitions"""
    
    def __init__(self):
        self.last_gesture = "None"
        self.current_gesture = "None"
        self.gesture_triggered = False
        self.gesture_log = deque(maxlen=50)
        self.frame_count = 0
        self.drawing_enabled = False
    
    def update_gesture(self, gesture):
        """Update current gesture and check for changes"""
        gesture_changed = (gesture != self.last_gesture)
        self.current_gesture = gesture
        return gesture_changed
    
    def log_gesture(self, log_entry, debug_mode=False):
        """Log gesture event"""
        if isinstance(log_entry, str):
            self.gesture_log.append(log_entry)
        else:
            # Original behavior for backwards compatibility
            gesture = log_entry
            additional_info = ""
            self.frame_count += 1
            timestamp = time.strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] Frame {self.frame_count}: {gesture}"
            if additional_info:
                log_entry += f" {additional_info}"
            
            if gesture != "None" or gesture != self.last_gesture:
                self.gesture_log.append(log_entry)
                if debug_mode:
                    print(log_entry)
        
        return log_entry
    
    def set_last_gesture(self, gesture):
        """Update last gesture for change detection"""
        self.last_gesture = gesture
    
    def enable_drawing(self, enabled):
        """Set drawing enabled state"""
        self.drawing_enabled = enabled
    
    def is_drawing_enabled(self):
        """Check if drawing is currently enabled"""
        return self.drawing_enabled
