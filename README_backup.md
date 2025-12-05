# Gesture Paint - Hand-Controlled Drawing Application

A computer vision application that enables hands-free drawing and painting using hand gestures detected through your webcam. Control a full-featured paint application without touching your keyboard or mouse.

## 🎯 Features

### Core Functionality
- **Gesture-Based Controls**: Draw, erase, change colors, and adjust brush size using natural hand gestures
- **Dual Detection Modes**: 
  - **MediaPipe** (AI-based, high accuracy, gesture recognition with thumb visualization)
  - **Traditional CV** (no neural networks, customizable, with optical flow tracking)
- **Auto-Calibration**: Quick 5-second setup for optimal hand detection (runs automatically on CV mode startup)
- **Real-Time Performance**: Smooth drawing experience with exponential smoothing and continuous drawing
- **Full Paint Features**: Multiple colors, adjustable brush sizes, eraser, save/load images

### Advanced CV Enhancements
- **Hybrid Detection-Tracking System**: Switches between full detection and lightweight optical flow tracking (inspired by MediaPipe)
- **Enhanced Finger Counting**: Convex hull peak detection with valley validation for accurate finger count
- **Dual Color Space Detection**: YCrCb + HSV skin detection with adaptive filtering
- **Background Subtraction**: MOG2 algorithm for motion-based hand isolation
- **Temporal Smoothing**: Position and finger count smoothing to reduce jitter
- **OS-Aware Camera Backend**: Automatic selection of optimal camera backend (DirectShow/AVFoundation/V4L2)

### Optimization Features
- **Exponential Smoothing**: 0.5 smoothing factor for jitter-free drawing
- **Continuous Drawing Mode**: Eliminates gaps in drawn lines
- **FPS Optimizations**: Removed artificial FPS caps, optimized MediaPipe settings (model_complexity=0, low confidence thresholds)
- **Camera Display Downscaling**: Full resolution processing with downscaled preview for better UI

## 📋 Requirements

- Python 3.10+
- Webcam
- Windows/Linux/macOS

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/michal-nak/vision_manuelle.git
   cd vision_manuelle
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## ⚡ Quick Start

### 1. Launch the Application

**MediaPipe Mode (Default - Recommended):**
```bash
python main.py
```
or
```bash
python main.py mediapipe
```

**CV Mode (with automatic calibration):**
```bash
python main.py cv
```
- Automatic 5-second calibration will start
- Position your hand in the yellow box
- Press **SPACE** to begin calibration
- Press **ESC** to skip calibration (uses defaults)

The application window will open with:
- **Left panel**: Live camera feed showing hand detection (400px fixed width)
- **Right panel**: Resizable drawing canvas
- **Toolbar**: Color picker, brush size, save/load buttons

### 2. Use Hand Gestures

**MediaPipe Mode (Uses Thumb for Cursor):**

| Fingers Extended | Gesture | Action |
|-----------------|---------|--------|
| 1 finger | **Draw** | Enable drawing mode |
| 2 fingers | **Erase** | Enable eraser mode |
| 5 fingers (all) | **Clear** | Clear entire canvas |
| 0 fingers (fist) | **None** | No action |

**Visual Indicators:**
- Large **green circle** = Thumb position (cursor)
- Pink circle = Index finger reference
- Gesture name displayed on video feed
- Finger count and FPS shown in overlay

**CV Mode (Enhanced with Tracking):**
- Uses convex hull peak detection for finger counting
- Automatic switch to optical flow tracking when hand is stable
- Green dots = tracked feature points (in tracking mode)
- Yellow box = tracking bounding box
- "DETECTING" or "TRACKING" mode indicator

## 🛠️ Advanced Tools

### 1. Unified Calibration Tool (`tools/calibrate.py`)

Comprehensive calibration with multiple modes:

```bash
python tools/calibrate.py
```

**Options:**
1. **Auto-Calibrate** - Quick 5-second automatic setup with preview mode
2. **Manual Tuning** - Advanced trackbar controls for YCrCb/HSV ranges
3. **Performance Tuning** - 21 adjustable parameters for fine-tuning detection
4. **AUTO-OPTIMIZE** - Tests 6 presets automatically and picks the best

**6 Optimization Presets:**
- Balanced (default)
- Speed (low latency)
- Accuracy (best detection)
- Low Noise (noisy environments)
- Fast Weak (minimal processing)
- Heavy (maximum filtering)

### 2. MediaPipe-Based CV Calibration (`tools/cv_calibrate_with_mediapipe.py`)

Uses MediaPipe as ground truth to tune CV detector parameters:

```bash
python tools/cv_calibrate_with_mediapipe.py
```

**Features:**
- Side-by-side comparison (MediaPipe ground truth | CV tuning)
- Real-time IoU (Intersection over Union) calculation
- Auto-suggest color thresholds from MediaPipe hand regions
- Live statistics: detection rates, match rate, misses, false positives
- Auto-optimization: tests all presets and selects best

**Controls:**
- **'a'** - Auto-suggest thresholds from collected hand regions (needs 10+ samples)
- **'p'** - Cycle through optimization presets
- **'o'** - Auto-optimize: test all 6 presets (50 frames each) and apply best
- **'s'** - Save current parameters to `calibration_mediapipe.json`
- **'r'** - Reset statistics
- **'q'** - Quit

**Visual Feedback:**
- Green bbox = Good match (IoU > 0.3)
- Red bbox = Poor match (IoU < 0.3)
- Statistics overlay at bottom (semi-transparent to avoid masking)

### 3. Enhanced Demo Tool (`tools/demo.py`)

Test different detection modes and visualizations:

```bash
python tools/demo.py
```

**Options:**
1. **CV vs MediaPipe Comparison** - Side-by-side with FPS counters and finger detection
2. **Gesture Recognition Demo** - Full-screen gesture detection with color-coded indicators
3. **Live Detector Switching** - Switch between MediaPipe/CV with 'r' to reset background
4. **Edge Detection Demo** - Sobel edge detection visualization

**New Features:**
- Large gesture indicator boxes with color coding
- Dark overlays for better text readability
- FPS counters for both detectors
- Continuous loop option

### 4. Debug Detection Tool (`tools/debug_detection.py`)

Shows all 8 processing steps for CV detector:

```bash
python tools/debug_detection.py
```

**Displays:**
1. Original frame
2. Denoised frame
3. YCrCb skin mask
4. HSV skin mask
5. Combined mask
6. Background subtraction
7. Morphological operations
8. Final detection with contours

**Controls:**
- **'r'** - Reset background subtractor
- **'d'** - Toggle denoising
- **'b'** - Toggle background subtraction
- **'q'** - Quit

## 📖 Technical Implementation Details

### Detection Algorithms

#### MediaPipe Detector (`src/detectors/mediapipe_detector.py`)
- **Base Model**: MediaPipe Hands with optimized settings
- **Model Complexity**: 0 (fastest)
- **Confidence Thresholds**: 0.3 for detection and tracking (lowered for speed)
- **Static Image Mode**: False (enables tracking between frames)
- **Performance**: Uses `writeable=False` flag for pass-by-reference optimization
- **Cursor Position**: Thumb tip (landmark[4]) instead of index finger
- **Gesture Detection**: Integrated into `process_frame()` for single-pass efficiency
- **Visualizations**: 
  - Large green circle (15px) on thumb tip with center dot
  - Pink circle on index finger for reference
  - Gesture name and finger count overlay

#### CV Detector (`src/detectors/cv_detector.py`)
**Color Space Detection:**
- Dual color space: YCrCb + HSV with bitwise AND combination
- Adaptive thresholds via calibration
- Fast NL means denoising (h=10, templateWindowSize=7, searchWindowSize=21)

**Background Subtraction:**
- MOG2 algorithm (history=500, varThreshold=16)
- Learns background in first 30 frames
- Dilated foreground mask (15x15 ellipse kernel) for hand capture

**Morphological Operations:**
- Opening (5x5 kernel, 2 iterations) - removes noise
- Closing (11x11 kernel, 3 iterations) - fills hand holes
- Gaussian blur (5x5) - final smoothing

**Finger Counting Algorithm:**
1. Convex hull peak detection
2. Distance threshold: avg_dist + (std_dist × 0.5)
3. Peak validation: local maxima with 15px minimum spacing
4. Valley detection: convexity defects with depth > 8000 and angle < 90°
5. Temporal smoothing: median filter over history buffer

**Optical Flow Tracking Mode:**
- Inspired by MediaPipe's tracking approach
- Switches to tracking after 3 stable detection frames
- Lucas-Kanade optical flow on 50 feature points
- Tracks hand even when closed (fist)
- Falls back to detection if < 5 good points for 10 frames
- Parameters: winSize=(21,21), maxLevel=3

**Quality Filters:**
- Area: MIN_HAND_AREA (3000) to MAX_HAND_AREA (0.9 × frame)
- Compactness: 0.2 to 0.95 (circularity measure)
- Aspect ratio: 0.3 to 2.0
- Solidity: 0.4 to 0.9 (hand vs convex hull ratio)
- Position score: preference for upper frame regions

### UI Implementation (`src/ui/gesture_paint.py`)

**Threading Architecture:**
- Separate camera thread for frame capture
- Main thread handles UI updates (16ms interval)
- Thread-safe queue for frame passing

**Drawing System:**
- Exponential smoothing (factor=0.5): `new_pos = prev + 0.5 × (raw_pos - prev)`
- Continuous drawing: calls `draw_at_cursor()` every frame when enabled
- Canvas operations on PIL Image for efficiency

**Camera Display:**
- Processes at full resolution (640×480)
- Downscales to 380×285 for display (LANCZOS resampling)
- Fixed left panel width (400px with `pack_propagate(False)`)
- Resizable canvas panel

### Performance Optimizations

**MediaPipe:**
- Removed double processing bug (was running twice per frame)
- Single gesture detection pass integrated into `process_frame()`
- Removed artificial FPS cap (`time.sleep(0.033)`)
- Avoided unnecessary frame copies

**CV Detector:**
- Tracking mode reduces processing by ~60%
- Background learning only in first 30 frames
- Efficient contour selection with multi-criteria filtering

**Camera Backends:**
- OS-aware selection via `find_camera()` in `src/core/utils.py`
- Windows: `CAP_DSHOW` (DirectShow)
- macOS: `CAP_AVFOUNDATION`
- Linux: `CAP_V4L2`

## 🔧 Troubleshooting

**Hand not detected?**
- Run `python main.py cv` for auto-calibration
- Use `python tools/cv_calibrate_with_mediapipe.py` for advanced tuning
- Ensure good lighting conditions (avoid backlighting)
- Try switching detection modes (MediaPipe ↔ CV)
- Check camera is not in use by another application

**Jittery cursor?**
- Jitter reduced by exponential smoothing (factor=0.5)
- In CV mode, tracking mode provides smoother results
- Ensure stable lighting (avoid flickering lights)
- Keep hand steady during gestures

**Low frame rate?**
- CV mode typically faster (~40-50 FPS) vs MediaPipe (~25-30 FPS)
- Close other camera-using applications
- Ensure webcam supports 30 FPS
- Check System Monitor/Task Manager for CPU usage
- Try lowering MediaPipe confidence thresholds further

**Finger counting issues in CV mode?**
- Run calibration: `python tools/calibrate.py` → option 4 (AUTO-OPTIMIZE)
- Use `tools/cv_calibrate_with_mediapipe.py` to compare with ground truth
- Adjust lighting (avoid shadows on hand)
- Keep hand flat and fingers spread
- Tracking mode uses region-based estimation for closed hands

**Drawing not responsive?**
- Check that gesture is being detected (visible on video overlay)
- Ensure continuous drawing is enabled (should be by default)
- Try switching to tracking mode in CV (happens automatically)
- MediaPipe: ensure thumb tip is visible

**Camera not opening?**
- Check `find_camera()` in `src/core/utils.py` for OS-specific backends
- Try manually specifying camera index: `cv2.VideoCapture(0)`
- On macOS, grant camera permissions in System Preferences
- On Windows, check Camera Privacy settings

**Calibration values not saving?**
- Check write permissions in project directory
- Look for `calibration_mediapipe.json` or `calibration_backup.json`
- CV detector updates are applied at runtime (not persisted by default)

## 💡 Tips for Best Results

- Use in well-lit environments
- Keep background simple and uncluttered
- Position camera at eye level
- Calibrate once per lighting setup
- MediaPipe mode: Better gesture accuracy
- CV mode: Better performance in varied lighting

## 📂 Project Structure

```
vision_manuelle/
├── main.py                              # Main entry point with auto-calibration
├── requirements.txt                     # Dependencies (opencv, numpy, mediapipe, pillow)
├── README.md                           # This file
├── LICENSE                             # Project license
│
├── src/                                # Source code
│   ├── __init__.py
│   ├── core/                          # Core utilities and configuration
│   │   ├── __init__.py
│   │   ├── config.py                  # Centralized configuration constants
│   │   └── utils.py                   # Utility functions (find_camera, FPSCounter, etc.)
│   │
│   ├── detectors/                     # Hand detection implementations
│   │   ├── __init__.py
│   │   ├── hand_detector_base.py     # Abstract base class
│   │   ├── mediapipe_detector.py     # MediaPipe implementation (optimized)
│   │   └── cv_detector.py            # CV implementation (with optical flow tracking)
│   │
│   └── ui/                            # User interface
│       ├── __init__.py
│       └── gesture_paint.py          # Main Tkinter application (exponential smoothing)
│
├── tools/                             # Development and calibration tools
│   ├── __init__.py
│   ├── calibrate.py                  # Unified calibration (4 modes, AUTO-OPTIMIZE)
│   ├── cv_calibrate_with_mediapipe.py # MediaPipe ground truth calibration
│   ├── demo.py                       # Enhanced demo (4 visualization modes)
│   └── debug_detection.py            # Debug tool (shows 8 processing steps)
│
└── legacy/                            # Original implementations (archived)
    ├── detection_francois_avec_controle_souris.py
    ├── interface.py
    ├── interfaceEdward.py
    └── paint_controller.py
```

## 🎨 Application Controls

**Gesture Controls (Primary):**
- See "Use Hand Gestures" section above for gesture mappings
- Visual feedback on camera feed shows current gesture
- Finger count displayed in real-time

**Mouse Controls (Alternative):**
- Use toolbar buttons as alternative to gestures
- Click and drag for traditional mouse drawing
- Color picker for custom colors

**Keyboard Shortcuts:**
- No keyboard required for basic operation
- Save/Load via File menu (mouse)
- Toolbar accessible via mouse clicks

## 🔬 Key Improvements & Evolution

### Phase 1: Initial Setup
- Original simple HSV-based detection (~60% accuracy)
- Basic gesture recognition
- Single-threaded implementation

### Phase 2: CV Detector Enhancement
- Upgraded to YCrCb + HSV dual color space detection
- Added MOG2 background subtraction
- Implemented convex hull peak detection for finger counting
- **Result**: ~85% detection accuracy

### Phase 3: Calibration Systems
- 5-second auto-calibration with preview mode
- Manual tuning with 21 adjustable parameters
- AUTO-OPTIMIZE mode testing 6 presets
- Performance tuning mode

### Phase 4: MediaPipe-Based Ground Truth
- Created `cv_calibrate_with_mediapipe.py` tool
- Auto-suggestion of color thresholds from MediaPipe detections
- IoU-based quality metrics
- Real-time comparison and statistics

### Phase 5: Performance Optimization
- Removed MediaPipe double-processing bug
- Eliminated artificial FPS caps
- Optimized MediaPipe settings (model_complexity=0, low confidence)
- OS-aware camera backend selection
- **Result**: FPS increased from ~20 to ~40-50 (CV) and ~25-30 (MediaPipe)

### Phase 6: Drawing Experience
- Exponential smoothing (factor=0.5) for jitter reduction
- Continuous drawing mode (eliminates gaps)
- Canvas size fix and camera display downscaling
- Thumb-based cursor for MediaPipe (instead of index finger)

### Phase 7: Advanced Tracking
- Implemented optical flow tracking mode (inspired by MediaPipe)
- Lucas-Kanade algorithm on 50 feature points
- Automatic switching between detection and tracking
- Robust to closed fist and hand shape changes
- **Result**: Smooth tracking even with closed hands

### Phase 8: Visual Enhancements
- Large thumb visualization (15px green circle)
- Gesture name overlay on video feed
- Mode indicators (DETECTING/TRACKING)
- Motion vectors visualization in tracking mode
- Semi-transparent statistics overlay (bottom placement)

### Phase 9: Auto-Calibration Integration
- Integrated 5-second calibration into `main.py`
- Automatic startup calibration for CV mode
- ESC to skip with default fallback

## 💡 Tips for Best Results

### Environment Setup
- Use in **well-lit environments** (avoid backlighting)
- Keep background **simple and uncluttered**
- Position camera at **eye level** or slightly above
- Avoid **flickering lights** (causes jitter)
- Calibrate once per lighting setup (automatic in CV mode)

### Detection Mode Selection
**MediaPipe Mode (Recommended for Most Users):**
- ✅ Best gesture accuracy and reliability
- ✅ Works in varied lighting conditions
- ✅ Excellent finger counting
- ✅ Thumb-based cursor with visual indicators
- ⚠️ Lower FPS (~25-30)
- ⚠️ Requires more CPU

**CV Mode (For Advanced Users):**
- ✅ Higher FPS (~40-50)
- ✅ Customizable via calibration
- ✅ Optical flow tracking for smooth experience
- ✅ No neural network dependency
- ⚠️ Requires calibration for optimal results
- ⚠️ Sensitive to lighting changes

### Hand Position & Gestures
- Keep hand **flat and fingers spread** for best detection
- Position hand in **upper 2/3 of frame** (CV detector prefers this)
- For CV mode: wait for **"TRACKING" mode** indicator for best stability
- For MediaPipe: keep **thumb tip visible** (used as cursor)
- Hold gestures for **0.5-1 second** for reliable recognition

### Performance Tuning
1. **For highest FPS**: Use CV mode with "Speed" preset
2. **For best accuracy**: Use MediaPipe mode
3. **For best balance**: Use CV mode with "Balanced" preset (default)
4. **For noisy environments**: Use CV mode with "Low Noise" preset

### Calibration Strategy
1. **Quick start**: Just run `python main.py cv` (auto-calibration)
2. **Fine-tuning**: Use `python tools/cv_calibrate_with_mediapipe.py`
   - Collect 50+ hand regions (move hand around)
   - Press 'a' to auto-suggest thresholds
   - Press 'o' to auto-optimize processing parameters
   - Press 's' to save
3. **Troubleshooting**: Use `python tools/debug_detection.py` to see all processing steps

## 🎓 Academic Context

This project was developed as part of a computer vision course focused on:
- **Real-time hand detection algorithms**
- **Gesture recognition systems**
- **Performance optimization techniques**
- **Comparison of traditional CV vs ML-based approaches**

### Key Learning Outcomes
1. Implementation of dual color space skin detection (YCrCb + HSV)
2. Background subtraction using MOG2 algorithm
3. Convex hull analysis for finger counting
4. Optical flow tracking (Lucas-Kanade) for robust hand tracking
5. Integration of traditional CV with modern ML frameworks (MediaPipe)
6. Performance profiling and optimization techniques
7. UI/UX design for gesture-based interfaces

### Comparative Analysis: CV vs MediaPipe

| Metric | Traditional CV | MediaPipe |
|--------|---------------|-----------|
| **Detection Accuracy** | ~85% (after tuning) | ~95% |
| **FPS** | 40-50 | 25-30 |
| **Setup Complexity** | Requires calibration | Works out-of-box |
| **Lighting Sensitivity** | High | Low |
| **Closed Hand Detection** | Good (with tracking) | Excellent |
| **Finger Count Accuracy** | ~80% | ~95% |
| **CPU Usage** | Low-Medium | Medium-High |
| **Customizability** | High | Low |

### Novel Contributions
1. **Hybrid Detection-Tracking System**: Combines full detection with optical flow tracking for robustness to hand shape changes
2. **MediaPipe Ground Truth Calibration**: Uses MediaPipe as reference to automatically tune CV parameters
3. **Auto-Optimization Framework**: Tests multiple parameter presets and selects best based on real-time metrics
4. **Exponential Smoothing**: Reduces jitter while maintaining responsiveness
5. **OS-Aware Camera Backend Selection**: Automatic selection of optimal camera API per platform

## 📚 References & Technologies

### Core Technologies
- **OpenCV 4.8+**: Computer vision operations, camera capture, image processing
- **MediaPipe 0.10+**: Hand landmark detection and tracking
- **NumPy 1.24+**: Numerical operations and array processing
- **Pillow 10.0+**: Image manipulation for Tkinter canvas
- **Tkinter**: GUI framework (built-in with Python)

### Algorithms Implemented
1. **Skin Detection**: YCrCb and HSV color space thresholding
2. **Background Subtraction**: MOG2 (Mixture of Gaussians)
3. **Morphological Operations**: Opening, Closing, Dilation
4. **Contour Analysis**: Convex hull, convexity defects
5. **Finger Counting**: Peak detection with valley validation
6. **Optical Flow**: Lucas-Kanade pyramidal algorithm
7. **Temporal Smoothing**: Moving average and exponential smoothing
8. **Feature Detection**: Shi-Tomasi corner detection (goodFeaturesToTrack)

### Design Patterns
- **Strategy Pattern**: Interchangeable detector implementations (CV/MediaPipe)
- **Observer Pattern**: Camera thread updates UI asynchronously
- **Factory Pattern**: OS-aware camera backend selection
- **Template Method**: Base detector class defines processing pipeline

## 📝 License

See [LICENSE](LICENSE) for details.

## 👥 Contributors

- **Michal Naumiak** - Lead Developer, CV Optimization, Tracking System
- **Edward Leroux** - Initial Implementation, UI Design
- **François Gerbeau** - Original Detection System
- **Théo Lahmar** - Testing & Documentation

---

**Course**: Vision Numérique | **Semester**: Automne 2025-26

