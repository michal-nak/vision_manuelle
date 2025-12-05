"""
Utility functions for the hand detection system
"""
import cv2
import numpy as np
import time
import platform
from functools import wraps


def find_camera(max_attempts=5):
    """
    Find and open an available camera with OS-specific optimizations
    
    Args:
        max_attempts: Maximum number of camera indices to try
        
    Returns:
        cv2.VideoCapture object or None if no camera found
    """
    os_name = platform.system()
    
    if os_name == 'Darwin':
        backends = [cv2.CAP_AVFOUNDATION]
    elif os_name == 'Windows':
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    
    for camera_index in range(max_attempts):
        for backend in backends:
            test_cap = cv2.VideoCapture(camera_index, backend)
            if not test_cap.isOpened():
                test_cap.release()
                continue
                
            ret, frame = test_cap.read()
            if ret:
                return test_cap
            test_cap.release()
    
    return None


def setup_camera(cap, width=640, height=480, fps=30):
    """
    Configure camera with specified settings
    
    Args:
        cap: cv2.VideoCapture object
        width: Desired frame width
        height: Desired frame height
        fps: Desired frames per second
    """
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)


class FPSCounter:
    """Calculate and smooth FPS over time"""
    
    def __init__(self, smoothing=0.9):
        self.fps = 0
        self.prev_time = time.time()
        self.smoothing = smoothing
    
    def update(self):
        """Update FPS calculation"""
        curr_time = time.time()
        if curr_time - self.prev_time > 0:
            instant_fps = 1 / (curr_time - self.prev_time)
            self.fps = self.fps * self.smoothing + instant_fps * (1 - self.smoothing)
        self.prev_time = curr_time
        return self.fps
    
    def get_fps(self):
        """Get current smoothed FPS"""
        return int(self.fps)


class PerformanceTimer:
    """Track performance timing for different operations"""
    
    def __init__(self):
        self.timings = {}
        self.start_times = {}
    
    def start(self, label):
        """Start timing for a label"""
        self.start_times[label] = time.perf_counter()
    
    def stop(self, label):
        """Stop timing for a label and store the duration"""
        if label in self.start_times:
            duration = (time.perf_counter() - self.start_times[label]) * 1000
            self.timings[label] = duration
    
    def get(self, label):
        """Get timing for a label in milliseconds"""
        return self.timings.get(label, 0)
    
    def get_all(self):
        """Get all timings"""
        return self.timings.copy()


def draw_text_with_background(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX,
                              font_scale=1, text_color=(255, 255, 255),
                              bg_color=(0, 0, 0), thickness=2, padding=5):
    """
    Draw text with a background rectangle for better visibility
    
    Args:
        frame: Image to draw on
        text: Text to draw
        position: (x, y) tuple for text position
        font: OpenCV font type
        font_scale: Font scale
        text_color: Text color (B, G, R)
        bg_color: Background color (B, G, R)
        thickness: Text thickness
        padding: Padding around text in pixels
    """
    x, y = position
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    cv2.rectangle(frame,
                 (x - padding, y - text_h - padding),
                 (x + text_w + padding, y + padding),
                 bg_color, -1)
    
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)


def normalize_coordinates(x, y, width, height):
    """
    Normalize coordinates to 0-1 range
    
    Args:
        x, y: Pixel coordinates
        width, height: Image dimensions
        
    Returns:
        (normalized_x, normalized_y) tuple
    """
    return x / width, y / height


def denormalize_coordinates(norm_x, norm_y, width, height):
    """
    Convert normalized coordinates back to pixel coordinates
    
    Args:
        norm_x, norm_y: Normalized coordinates (0-1)
        width, height: Image dimensions
        
    Returns:
        (x, y) tuple in pixel coordinates
    """
    return int(norm_x * width), int(norm_y * height)


class TemporalSmoother:
    """Smooth values over time using a moving average"""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.values = []
    
    def add(self, value):
        """Add a new value"""
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
    
    def get(self):
        """Get smoothed value"""
        if not self.values:
            return None
        return sum(self.values) / len(self.values)
    
    def clear(self):
        """Clear all values"""
        self.values.clear()


def retry_on_failure(max_attempts=3, delay=0.1):
    """
    Decorator to retry a function on failure
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


def ensure_bgr(frame):
    """
    Ensure frame is in BGR format
    
    Args:
        frame: Input frame
        
    Returns:
        BGR frame
    """
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def safe_divide(numerator, denominator, default=0):
    """
    Safely divide two numbers, returning default if denominator is zero
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division by zero
        
    Returns:
        Result of division or default value
    """
    return numerator / denominator if denominator != 0 else default
