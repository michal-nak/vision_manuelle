"""
Camera frame processing and display logic
"""
import cv2
import time
import threading
from PIL import Image, ImageTk
from ..core.config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
from ..core.utils import find_camera, setup_camera, FPSCounter


class CameraManager:
    """Manages camera capture and frame processing"""
    
    def __init__(self):
        self.cap = None
        self.camera_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.fps_counter = FPSCounter()
        self.timing_info = {
            "capture": 0,
            "process": 0,
            "drawing": 0,
            "display": 0
        }
    
    def start_camera(self, error_callback):
        """Initialize and start camera capture"""
        # Setup camera synchronously to ensure it's ready
        self.cap = find_camera()
        if self.cap:
            setup_camera(self.cap, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
            self.camera_running = True
            return True
        else:
            error_callback("Could not open camera")
            return False
    
    def stop_camera(self, release_camera=False):
        """Stop camera capture"""
        self.camera_running = False
        time.sleep(0.1)
        if release_camera and self.cap:
            self.cap.release()
            self.cap = None
    
    def is_running(self):
        """Check if camera is currently running"""
        return self.camera_running
    
    def capture_frame(self):
        """Capture a single frame from camera"""
        if self.cap is None or not self.cap.isOpened():
            return None
        
        t0 = time.perf_counter()
        ret, frame = self.cap.read()
        t1 = time.perf_counter()
        
        if not ret:
            return None
        
        self.timing_info["capture"] = (t1 - t0) * 1000
        frame = cv2.flip(frame, 1)  # Mirror horizontally
        return frame
    
    def update_frame(self, annotated_frame):
        """Update current frame for display"""
        self.fps_counter.update()
        display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        with self.frame_lock:
            self.current_frame = display_frame
    
    def get_current_frame(self):
        """Get current frame safely"""
        with self.frame_lock:
            return self.current_frame
    
    def get_timing_info(self):
        """Get frame processing timing information"""
        return self.timing_info
    
    def update_timing(self, key, value):
        """Update timing information for a specific operation"""
        self.timing_info[key] = value
    
    def cleanup(self):
        """Clean up camera resources"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
    
    @staticmethod
    def prepare_frame_for_display(frame, display_width=380):
        """Convert and resize frame for Tkinter display"""
        if frame is None:
            return None
        
        img = Image.fromarray(frame)
        
        # Downscale to fit display width while maintaining aspect ratio
        aspect_ratio = img.height / img.width
        display_height = int(display_width * aspect_ratio)
        img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
        
        return ImageTk.PhotoImage(image=img)
