# API Reference

## Core Interfaces

### HandDetectorBase

Abstract base class for all hand detectors.

```python
from src.detectors.hand_detector_base import HandDetectorBase

class HandDetectorBase:
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame and detect hand gestures.
        
        Args:
            frame: BGR image as numpy array (height, width, 3)
            
        Returns:
            Dictionary containing:
            - detected (bool): Whether hand was detected
            - hand_x (float): Normalized X position (0.0-1.0)
            - hand_y (float): Normalized Y position (0.0-1.0)
            - finger_count (int): Number of extended fingers
            - gesture (str): Detected gesture name
            - annotated_frame (np.ndarray): Frame with overlays
        """
        pass
    
    def cleanup(self):
        """Release resources and clean up."""
        pass
```

## Detector Implementations

### MediaPipeDetector

MediaPipe-based hand detection with landmark tracking.

```python
from src.detectors.mediapipe_detector import MediaPipeDetector

detector = MediaPipeDetector(show_debug=False)

# Process frame
result = detector.process_frame(frame)

# Access results
if result['detected']:
    x, y = result['hand_x'], result['hand_y']
    gesture = result['gesture']
    
# Cleanup when done
detector.cleanup()
```

**Gestures**:
- `"Draw"`: Thumb + Index touching
- `"Erase"`: Thumb + Middle touching
- `"Cycle Color"`: Thumb + Ring touching
- `"Clear"`: Thumb + Pinky touching
- `"Increase Size"`: Index + Middle touching
- `"Decrease Size"`: Middle + Ring touching
- `"None"`: No gesture detected

**Parameters**:
- `show_debug` (bool): Enable debug overlays on annotated_frame

### CVDetector

Computer vision-based hand detection using skin detection.

```python
from src.detectors.cv.cv_detector import CVDetector

detector = CVDetector(show_debug=False)

# Process frame
result = detector.process_frame(frame)

# Access additional CV-specific data
if result['detected']:
    hand_center = result['hand_center']  # (x, y) tuple
    finger_count = result['finger_count']
    
detector.cleanup()
```

**Gestures**:
- `"Draw"`: 1 finger extended
- `"Erase"`: 2 fingers extended
- `"Cycle Color"`: 3 fingers extended
- `"Increase Size"`: 4 fingers extended
- `"Clear"`: 5 fingers extended
- `"None"`: 0 fingers or no detection

**Parameters**:
- `show_debug` (bool): Enable debug overlays on annotated_frame

**Additional Methods**:
```python
# Access debug metrics
metrics = detector.debug_metrics
print(f"Detection rate: {metrics['detected_frames']}/{metrics['total_frames']}")
```

## UI Components

### GesturePaintApp

Main application class managing UI and gesture control.

```python
from src.ui.gesture_paint import GesturePaintApp
import tkinter as tk

root = tk.Tk()
app = GesturePaintApp(root, detection_mode="mediapipe")
root.mainloop()
```

**Parameters**:
- `root`: Tkinter root window
- `detection_mode` (str): "mediapipe" or "cv"

**Methods**:
```python
# Switch detection mode
app.switch_detection_mode("cv")

# Toggle debug mode programmatically
app.debug_mode = True
app.toggle_debug()

# Start/stop camera
app.start_camera()
app.stop()
```

### CanvasController

Reusable canvas drawing controller.

```python
from src.ui.canvas_controller import CanvasController

controller = CanvasController(canvas)

# Draw on canvas
controller.draw_at_position(
    x=0.5,          # Normalized X (0.0-1.0)
    y=0.5,          # Normalized Y (0.0-1.0)
    color="black",  # Optional, uses current if not specified
    size=5          # Optional, uses current if not specified
)

# Update cursor position
controller.update_cursor(x=0.5, y=0.5)

# Mode switching
controller.use_eraser()
controller.use_brush()

# Canvas operations
controller.clear_canvas()
controller.save_canvas()

# State management
controller.set_color("red")
controller.set_brush_size(10)
```

### GestureHandler

Maps gestures to canvas actions.

```python
from src.ui.gesture_handler import GestureHandler

handler = GestureHandler(canvas_controller)

# Process gesture
action = handler.process_gesture(
    gesture="Draw",
    x=0.5,
    y=0.5
)

# Check current state
if handler.is_drawing():
    # Drawing is active
    pass
```

**Gesture Actions**:
- `"Draw"`: Enable drawing, draw at position
- `"Erase"`: Enable eraser, erase at position
- `"Cycle Color"`: Change to next color
- `"Increase Size"`: Increase brush size by 2
- `"Decrease Size"`: Decrease brush size by 2
- `"Clear"`: Clear entire canvas
- `"None"`: Stop drawing/erasing

## Utility Modules

### Camera Management

```python
from src.core.camera import find_camera, setup_camera

# Find available camera
cap = find_camera()

if cap:
    # Configure camera
    setup_camera(cap, width=640, height=480, fps=30)
    
    # Read frame
    ret, frame = cap.read()
    
    # Release when done
    cap.release()
```

### Configuration

```python
from src.core.config import (
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    MIN_HAND_AREA,
    MAX_HAND_AREA,
    # ... and more
)

# Configuration is loaded automatically from:
# 1. skin_detection_config.json (if exists)
# 2. Default values in config.py
```

**Key Configuration Constants**:
```python
# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Hand detection thresholds
MIN_HAND_AREA = 5000      # Minimum contour area
MAX_HAND_AREA = 0.6       # Maximum as fraction of frame

# Color detection ranges
YCRCB_LOWER = [0, 133, 77]
YCRCB_UPPER = [255, 173, 127]
HSV_LOWER = [0, 48, 80]
HSV_UPPER = [20, 255, 255]

# MediaPipe settings
MP_MODEL_COMPLEXITY = 0                  # 0=lite, 1=full
MP_MIN_DETECTION_CONFIDENCE = 0.3
MP_MIN_TRACKING_CONFIDENCE = 0.3
```

### FPS Counter

```python
from src.utils.fps_counter import FPSCounter

counter = FPSCounter()

# Update each frame
fps = counter.update()

# Get current FPS
current_fps = counter.get_fps()
```

## Skin Detection (CV Mode)

### Skin Detection Functions

```python
from src.detectors.cv.skin_detection import (
    detect_skin_ycrcb_hsv,
    detect_skin_ycrcb,
    detect_skin_hsv
)

# Combined detection (recommended)
mask = detect_skin_ycrcb_hsv(
    frame,
    ycrcb_lower,
    ycrcb_upper,
    hsv_lower,
    hsv_upper
)

# YCrCb only
mask = detect_skin_ycrcb(frame, ycrcb_lower, ycrcb_upper)

# HSV only
mask = detect_skin_hsv(frame, hsv_lower, hsv_upper)
```

**HSV Hue Wrap-around**:
The system automatically handles HSV hue wrap-around:
```python
# Example: Red hues that wrap around 180°
hsv_lower = [170, 50, 50]  # Higher hue
hsv_upper = [10, 255, 255]  # Lower hue

# Internally creates: (H: 170-180 OR H: 0-10) AND (S: 50-255) AND (V: 50-255)
```

### Finger Detection

```python
from src.detectors.cv.finger_detection import (
    count_fingers_from_contour,
    smooth_finger_count,
    map_fingers_to_gesture
)

# Count fingers from contour
finger_count = count_fingers_from_contour(
    contour,
    frame,
    defect_threshold=0.2
)

# Smooth finger count over time
from collections import deque
finger_history = deque(maxlen=5)
finger_history.append(finger_count)
smoothed = smooth_finger_count(finger_history)

# Map to gesture
gesture = map_fingers_to_gesture(smoothed)
```

### Tracking

```python
from src.detectors.cv.tracking import HandTracker

tracker = HandTracker()

# Initialize tracking
tracker.initialize_tracking(hand_contour, gray_frame)

# Track in subsequent frames
success, bbox, center = tracker.track_frame(gray_frame)

if success:
    x, y, w, h = bbox
    cx, cy = center
else:
    # Tracking failed, fall back to detection
    tracker.reset()
```

## Debugging Tools

### Skin Tuner

Interactive tool for adjusting skin detection parameters.

```bash
python tools/skin_tuner.py
```

**Features**:
- Live trackbar adjustment
- 6-panel view (YCrCb, HSV, combined)
- Save configuration to JSON
- Auto-load existing config

### Debug Detection

Pipeline visualization tool showing all processing steps.

```bash
python tools/debug_detection.py
```

**Features**:
- 8-step pipeline visualization
- Toggle denoising/background subtraction
- Save configuration
- Real-time metrics display

## Data Structures

### Result Dictionary

Standard format returned by all detectors:

```python
result = {
    'detected': True,                    # bool
    'hand_x': 0.45,                      # float (0.0-1.0)
    'hand_y': 0.52,                      # float (0.0-1.0)
    'hand_center': (0.45, 0.52),         # tuple (CV only)
    'finger_count': 2,                   # int (0-5)
    'gesture': 'Erase',                  # str
    'annotated_frame': np.ndarray(...)   # BGR image
}
```

### Debug Metrics (CV Detector)

```python
metrics = {
    'total_frames': 1000,
    'detected_frames': 850,
    'gesture_changes': 23,
    'tracking_frames': 450,
    'position_history': deque([...]),
    'finger_transitions': [('Draw', 'None', 123), ...],
    'contour_areas': [15000, 15200, ...],
    'hull_defect_counts': [3, 4, 3, ...],
    'tracking': False,
    'largest_contour_area': 15432  # Only when detection fails
}
```

## Error Handling

### MediaPipe Timestamp Errors

Automatically handled by MediaPipeDetector:

```python
try:
    results = self.hands.process(rgb_frame)
except Exception as e:
    if "timestamp" in str(e).lower():
        # Returns last successful result
        return self.last_result
    raise
```

### Camera Initialization Failures

```python
from src.core.camera import find_camera

cap = find_camera()
if cap is None:
    print("Error: No camera available")
    exit(1)
```

## Extension Examples

### Custom Detector

```python
from src.detectors.hand_detector_base import HandDetectorBase
import cv2

class CustomDetector(HandDetectorBase):
    def __init__(self, show_debug=False):
        self.show_debug = show_debug
        # Your initialization
    
    def process_frame(self, frame):
        # Your detection logic
        
        return {
            'detected': detected,
            'hand_x': hand_x,
            'hand_y': hand_y,
            'finger_count': finger_count,
            'gesture': gesture,
            'annotated_frame': annotated
        }
    
    def cleanup(self):
        # Release resources
        pass
```

### Custom Gesture

```python
# In finger_detection.py
def map_fingers_to_gesture(finger_count):
    gesture_map = {
        0: "None",
        1: "Draw",
        2: "Erase",
        3: "Custom Gesture",  # Add custom mapping
        4: "Increase Size",
        5: "Clear"
    }
    return gesture_map.get(finger_count, "None")

# In gesture_handler.py
def process_gesture(self, gesture, x, y):
    if gesture == "Custom Gesture":
        # Implement custom action
        self.canvas_controller.custom_action()
```

## Performance Tips

- Use `show_debug=False` in production for better FPS
- Reduce camera resolution if experiencing lag
- Use CV mode for faster startup time
- Enable tracking mode in CV detector for stable hands
- Close debug windows when not needed

## Version Compatibility

- Python: 3.10+
- OpenCV: 4.5+
- MediaPipe: 0.10+
- NumPy: 1.21+
- Pillow: 9.0+
