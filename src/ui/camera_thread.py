"""
Camera processing thread separated from UI
"""
import cv2
import threading
import time
from PIL import Image


class CameraThread:
    """Handles camera capture and processing in separate thread"""
    
    def __init__(self, detector, process_scale=0.5):
        self.detector = detector
        self.process_scale = process_scale
        
        self.cap = None
        self.running = False
        self.thread = None
        
        self.current_frame = None
        self.current_result = None
        self.frame_lock = threading.Lock()
    
    def start(self, camera_index=0):
        """Start camera thread"""
        if self.running:
            return False
        
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        self.thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.thread.start()
        return True
    
    def stop(self):
        """Stop camera thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
    
    def get_latest_result(self):
        """Get latest processing result (thread-safe)"""
        with self.frame_lock:
            return self.current_result
    
    def _camera_loop(self):
        """Main camera processing loop"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                # Process frame
                small_frame = cv2.resize(
                    frame,
                    None,
                    fx=self.process_scale,
                    fy=self.process_scale,
                    interpolation=cv2.INTER_LINEAR
                )
                
                result = self.detector.process_frame(small_frame)
                
                # Store result (thread-safe)
                with self.frame_lock:
                    self.current_result = result
                
                time.sleep(0.01)  # Small delay to prevent CPU overuse
                
            except Exception as e:
                print(f"Error in camera loop: {e}")
                time.sleep(0.1)
    
    def is_running(self):
        """Check if camera is running"""
        return self.running and self.cap is not None and self.cap.isOpened()
