"""
Gesture handling logic separated from UI
"""
import time
from collections import deque


class GestureHandler:
    """Handles gesture recognition and action mapping"""
    
    def __init__(self):
        self.current_gesture = "None"
        self.last_gesture = "None"
        self.gesture_log = deque(maxlen=50)
        self.frame_count = 0
        self.callbacks = {}
    
    def register_callback(self, gesture, callback):
        """Register a callback for a specific gesture"""
        self.callbacks[gesture] = callback
    
    def process_gesture(self, gesture):
        """
        Process detected gesture and execute callbacks
        
        Args:
            gesture: Gesture string from detector
            
        Returns:
            dict: Action information
        """
        self.frame_count += 1
        self.current_gesture = gesture
        gesture_changed = (gesture != self.last_gesture)
        
        # Log gesture event
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] Frame {self.frame_count}: {gesture}"
        
        # Execute callback if registered
        action = {"gesture": gesture, "changed": gesture_changed}
        
        if gesture in self.callbacks:
            result = self.callbacks[gesture](gesture_changed)
            if result:
                log_entry += f" -> {result}"
                action["result"] = result
        
        # Add to log if significant
        if gesture != "None" or gesture_changed:
            self.gesture_log.append(log_entry)
            print(log_entry)
        
        self.last_gesture = gesture
        return action
    
    def get_recent_log(self, count=8):
        """Get recent gesture log entries"""
        return list(self.gesture_log)[-count:]
    
    def clear_log(self):
        """Clear gesture log"""
        self.gesture_log.clear()
