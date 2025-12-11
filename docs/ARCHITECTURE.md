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
│   ├── calibration/       # Modular calibration system
│   │   ├── __init__.py               # Package initialization
│   │   ├── auto_calibrate.py         # Automatic color calibration
│   │   ├── auto_optimize.py          # Parameter optimization with MediaPipe validation
│   │   ├── manual_tune.py            # Interactive manual tuning
│   │   ├── performance_tune.py       # Performance-focused parameter adjustment
│   │   ├── config_io.py              # Configuration save/load utilities
│   │   └── ui_display.py             # Calibration UI components
│   ├── skin_tuner.py                 # Legacy: Interactive skin detection tuner
│   ├── debug_detection.py            # Pipeline visualization tool
│   └── cv_calibrate_with_mediapipe.py  # MediaPipe-based calibration tool
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
- **Configurable**: JSON-based color range and processing parameter tuning
- **Modular architecture**: Separated into specialized components:
  - `cv_detector.py`: Main detector class and orchestration
  - `detection_pipeline.py`: Complete detection pipeline with configurable parameters
  - `skin_detection.py`: Dual color space skin masking
  - `config_loader.py`: Configuration loading and validation
  - `detector_state.py`: State management for tracking mode
- **Auto-calibration**: MediaPipe-based intelligent calibration with IoU validation

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

### 4. Calibration System

**Modular Architecture** (`tools/calibration/`)
- Each calibration mode in separate module for maintainability
- Shared utilities in `config_io.py` and `ui_display.py`
- Clean separation of concerns: UI, logic, I/O

**Auto-Calibration Flow**:
1. **Color Calibration** (`auto_calibrate.py`):
   - MediaPipe detects hand automatically
   - Samples YCrCb/HSV values from full hand region
   - Uses 10th-90th percentile for robust color bounds
   - 10-second duration for adequate sampling

2. **Auto-Optimization** (`auto_optimize.py`):
   - Tests 6 parameter presets (Fast, Balanced, Quality, etc.)
   - Uses MediaPipe as ground truth (palm center mode)
   - Calculates IoU (Intersection over Union) for spatial validation
   - Visual feedback: Blue (MediaPipe) vs Green (CV) detections
   - Selects best preset based on F1 score
   - 18-second duration (3 seconds per preset)

3. **Configuration Management** (`config_io.py`):
   - Loads/saves JSON configuration
   - Validates parameter ranges
   - Provides sensible defaults
   - Timestamp tracking

**Manual Tuning** (`manual_tune.py`):
- Real-time trackbar adjustments
- 21 parameters: color thresholds, morphology, background subtraction
- HSV wrap-around visualization support
- Live preview with dual-mode display

**Performance Tuning** (`performance_tune.py`):
- Optimize for speed vs accuracy tradeoff
- Test parameter impact on FPS
- Preset management

**MediaPipe Integration**:
- Context-aware positioning: `use_palm_center` parameter
- **Calibration mode**: Palm center for spatial alignment with CV detector
- **Drawing mode**: Thumb tip for precise cursor control
- Prevents spatial mismatch during validation

## Data Flow

```
Camera → Detector → Result Dict → UI → Canvas
                ↓
           Debug Overlay

# Calibration Flow
Camera → MediaPipe (palm center) → Hand Region → Color Sampling → Config
                ↓
         Auto-Optimize → Test Presets → IoU Validation → Best Config
```

### Result Dictionary Format

All detectors return a standardized dictionary:

```python
{
    'detected': bool,           # Whether hand was detected
    'hand_x': float,            # X position (0.0-1.0, normalized)
                               # MediaPipe: Thumb tip (default) or palm center (calibration mode)
                               # CV: Hand centroid (center of contour)
    'hand_y': float,            # Y position (0.0-1.0, normalized)
    'hand_center': tuple,       # (x, y) in normalized coords (CV only, deprecated)
    'finger_count': int,        # Number of extended fingers
    'gesture': str,             # Gesture name
    'annotated_frame': ndarray  # Frame with overlays drawn
}
```

**MediaPipe Context-Aware Positioning**:
```python
# Drawing mode (default)
detector.process_frame(frame)  # Returns thumb tip position

# Calibration mode (for CV alignment)
detector.process_frame(frame, use_palm_center=True)  # Returns palm center
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
  "hsv_upper": [H_max, S_max, V_max],
  "processing_params": {
    "denoise_h": 10,              # Non-local means denoising strength
    "kernel_small": 3,            # Small morphology kernel size
    "kernel_large": 7,            # Large morphology kernel size
    "morph_iterations": 2,        # Morphological operation iterations
    "min_contour_area": 3000,     # Minimum hand area (pixels)
    "max_contour_area": 50000     # Maximum hand area (pixels)
  },
  "preset_name": "Balanced",      # Name of optimization preset used
  "timestamp": "2025-12-05 10:30:45"
}
```

- Auto-loaded by detectors and tools
- Saved by calibration system with timestamps
- Supports HSV hue wrap-around (e.g., 170-10 for red tones)
- Processing parameters set by auto-optimization
- Validated ranges prevent invalid configurations

## Command-Line Interface

### Main Application (`main.py`)

**Syntax**:
```bash
python main.py [mode] [flags]
```

**Modes**:
- `mediapipe` (default): MediaPipe hand tracking
- `cv`: Computer vision detector

**Flags**:
- `--skip-calibration`, `-s`: Use saved config, skip calibration (CV only)
- `--skip-optimization`, `-o`: Skip auto-optimization, color only (CV only)
- `--debug`, `-d`: Enable debug overlay

**Examples**:
```bash
python main.py                 # MediaPipe mode
python main.py cv              # Full CV calibration (28s)
python main.py cv -o           # Quick calibration (10s)
python main.py cv -s           # Skip calibration (instant)
python main.py cv -d           # Enable debug overlay
python main.py cv -od          # Quick calibration + debug
```

**Calibration Flow Logic**:
```python
if mode == 'cv':
    if skip_calibration:
        load_config()
    else:
        color_calibrate()  # 10 seconds
        if not skip_optimization:
            auto_optimize()  # 18 seconds
        save_config()
```

## Future Improvements

- Multi-hand support (multiple users simultaneously)
- Gesture learning mode (custom gesture definitions)
- 3D drawing space (depth perception)
- Remote collaboration (network drawing)
- Recording and playback (gesture sequences)
- Custom gesture definitions via config
- Adaptive IoU thresholds based on hand size
- Incremental calibration (update existing config)
- Lighting condition detection and auto-adjustment
