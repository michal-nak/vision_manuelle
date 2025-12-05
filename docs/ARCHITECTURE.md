# Architecture Documentation

## Project Overview

Gesture Paint is a modular computer vision application that enables hands-free drawing through hand gesture recognition. The architecture follows clean code principles with separation of concerns and reusable components.

## Directory Structure

```
vision_manuelle/
├── src/
│   ├── core/              # Core configuration and utilities
│   │   ├── config.py      # Centralized configuration management
│   │   └── utils.py       # Camera initialization, FPS counter, utilities
│   ├── detectors/         # Hand detection implementations
│   │   ├── cv/            # Computer vision detector (skin detection)
│   │   │   ├── cv_detector.py        # Main CV detector class
│   │   │   ├── skin_detection.py     # YCrCb + HSV skin detection
│   │   │   ├── finger_detection.py   # Finger counting and gesture mapping
│   │   │   ├── tracking.py           # Optical flow tracking
│   │   │   └── visualization.py      # Debug overlays
│   │   ├── mediapipe_detector.py     # MediaPipe hand detector
│   │   └── hand_detector_base.py     # Base class for detectors
│   └── ui/                # User interface components
│       ├── gesture_paint.py          # Main application window
│       ├── camera_thread.py          # Camera capture thread management
│       ├── canvas_controller.py      # Canvas drawing logic
│       └── gesture_handler.py        # Gesture action mapping
├── tools/                 # Standalone debugging and tuning tools
│   ├── skin_tuner.py                 # Interactive skin detection tuner
│   └── debug_detection.py            # Pipeline visualization tool
├── docs/                  # Documentation
├── legacy/                # Deprecated code (for reference)
├── main.py                # Application entry point
├── requirements.txt       # Python dependencies
└── skin_detection_config.json  # Skin detection parameters
```

## Core Architecture

### 1. Detection Layer

**Base Class**: `HandDetectorBase`
- Defines common interface for all detectors
- Methods: `process_frame()`, `cleanup()`

**Implementations**:

#### MediaPipe Detector
- Uses Google's MediaPipe Hand solution
- Provides 21 3D hand landmarks
- Gesture detection via fingertip distances
- Handles timestamp mismatches gracefully

#### CV Detector
- Custom computer vision pipeline without neural networks
- **Dual color space detection**: YCrCb (illumination invariance) + HSV (hue wrapping)
- **Contour analysis**: Convex hull and convexity defects for finger counting
- **Hybrid tracking**: Switches between detection and optical flow
- **Configurable**: JSON-based color range tuning

### 2. UI Layer

**GesturePaintApp** (`gesture_paint.py`)
- Main Tkinter application window
- Manages camera feed display
- Coordinates between detector and canvas
- Provides mode switching (CV/MediaPipe)

**CanvasController** (`canvas_controller.py`)
- Reusable drawing logic
- Handles pen/eraser modes
- Manages colors and brush sizes
- Frame-to-canvas coordinate mapping

**GestureHandler** (`gesture_handler.py`)
- Maps gestures to actions
- Debouncing and state management
- Logging for debugging

**CameraThread** (`camera_thread.py`)
- Manages camera capture in separate thread
- Handles frame buffering and threading
- Thread-safe communication with UI

### 3. Configuration Layer

**Centralized Config** (`config.py`)
- Loads from `skin_detection_config.json`
- Provides default fallbacks
- Shared across all components

**Utilities** (`utils.py`)
- `find_camera()`: OS-aware camera backend selection
- `setup_camera()`: Optimal resolution and FPS configuration
- `FPSCounter`: Frame rate calculation and monitoring
- Multi-platform support (Windows/Linux/macOS)

## Data Flow

```
Camera → Detector → Result Dict → UI → Canvas
                ↓
           Debug Overlay
```

### Result Dictionary Format

All detectors return a standardized dictionary:

```python
{
    'detected': bool,           # Whether hand was detected
    'hand_x': float,            # X position (0.0-1.0, normalized)
    'hand_y': float,            # Y position (0.0-1.0, normalized)
    'hand_center': tuple,       # (x, y) in normalized coords (CV only)
    'finger_count': int,        # Number of extended fingers
    'gesture': str,             # Gesture name
    'annotated_frame': ndarray  # Frame with overlays drawn
}
```

## Design Patterns

### 1. Strategy Pattern
- `HandDetectorBase` defines interface
- Concrete detectors (`MediaPipeDetector`, `CVDetector`) implement strategies
- UI can switch strategies at runtime

### 2. Template Method Pattern
- Base detector defines processing flow
- Subclasses implement specific detection algorithms

### 3. Singleton-like Config
- Configuration loaded once on module import
- Shared immutable state across components

### 4. Observer Pattern (Implicit)
- Camera thread produces frames
- UI consumes and displays results
- Gesture handler reacts to gesture changes

## Threading Model

- **Main Thread**: Tkinter UI event loop
- **Camera Thread**: Continuous frame capture and processing
- **Synchronization**: Thread-safe updates via `root.after()`

## Error Handling

### MediaPipe Timestamp Errors
- Catches timestamp mismatch exceptions
- Returns last successful result
- Prevents application crashes

### Camera Failures
- Graceful degradation
- User-friendly error messages
- Automatic backend fallback

## Extension Points

### Adding New Detectors
1. Inherit from `HandDetectorBase`
2. Implement `process_frame(frame)` → result dict
3. Implement `cleanup()`
4. Register in UI mode switcher

### Adding New Gestures
1. Update detection logic in detector
2. Add mapping in `gesture_handler.py`
3. Update UI instructions
4. Update debug overlays

### Custom Actions
1. Define action in `CanvasController`
2. Map gesture to action in `GestureHandler`
3. Update documentation

## Performance Considerations

### Optimizations
- **Minimal processing**: Optical flow tracking when hand stable
- **Downsampled preview**: Full resolution processing, reduced display size
- **Static image mode**: MediaPipe processes each frame independently
- **Exponential smoothing**: 0.5 factor reduces jitter without lag

### Bottlenecks
- Camera capture rate (typically 30 FPS)
- MediaPipe processing (~10-15ms per frame)
- CV skin detection (~5-8ms per frame)
- UI rendering (~16ms for 60 FPS)

## Testing Strategy

### Tools
- `skin_tuner.py`: Interactive parameter adjustment
- `debug_detection.py`: 8-step pipeline visualization
- Debug mode toggle: Real-time metrics overlay

### Debugging
- Console logging with timestamps
- Visual overlays (confidence, finger count, contours)
- FPS monitoring
- Frame-by-frame inspection

## Dependencies

### Core
- `opencv-python`: Computer vision operations
- `mediapipe`: Hand landmark detection
- `numpy`: Numerical operations
- `Pillow`: Image handling for Tkinter

### Optional
- Platform-specific camera backends (automatically handled)

## Configuration Files

### skin_detection_config.json
```json
{
  "ycrcb_lower": [Y_min, Cr_min, Cb_min],
  "ycrcb_upper": [Y_max, Cr_max, Cb_max],
  "hsv_lower": [H_min, S_min, V_min],
  "hsv_upper": [H_max, S_max, V_max]
}
```

- Auto-loaded by detectors and tools
- Saved by tuning tools with timestamps
- Supports HSV hue wrap-around (e.g., 170-10 for red tones)

## Future Improvements

- Multi-hand support
- Gesture learning mode
- 3D drawing space
- Remote collaboration
- Recording and playback
- Custom gesture definitions via config
