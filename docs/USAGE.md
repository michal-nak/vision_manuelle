# Usage Guide

## Getting Started

### Running the Application

**Default mode (MediaPipe)**:
```bash
python main.py
```

**Computer Vision mode**:
```bash
python main.py cv
```

**Specify detection mode**:
```bash
python main.py [mediapipe|cv]
```

## Detection Modes

### MediaPipe Mode (Recommended)

**Advantages**:
- High accuracy hand tracking
- 21 landmark points for precise gesture detection
- Robust to lighting conditions
- No calibration needed

**Gestures**:
- **Thumb + Index**: Draw with current color
- **Thumb + Middle**: Erase
- **Thumb + Ring**: Cycle through colors
- **Thumb + Pinky**: Clear entire canvas
- **Index + Middle**: Increase brush size
- **Middle + Ring**: Decrease brush size

**Tips**:
- Hold fingertips close together (touching) to trigger gestures
- Green circle shows thumb position (main cursor)
- Purple circle shows index finger for reference

### CV Mode (Computer Vision)

**Advantages**:
- No neural network dependencies
- Customizable skin detection
- Educational value (understand the algorithm)
- Faster startup time

**Gestures**:
- **1 Finger**: Draw with current color
- **2 Fingers**: Erase
- **3 Fingers**: Cycle through colors
- **4 Fingers**: Increase brush size
- **5 Fingers**: Clear entire canvas
- **0 Fingers**: Idle (no action)

**Tips**:
- Ensure good lighting on your hand
- Keep hand in front of plain background
- Use skin tuner tool if detection is poor

## UI Controls

### Camera Feed Panel
- Shows live camera feed with hand detection overlays
- Green indicators = hand detected
- Red indicators = no hand detected

### Canvas Panel
- Main drawing area on the right side
- Real-time cursor preview shows your hand position
- Drawn content persists between gestures

### Control Elements

**Mode Selector**:
- Dropdown to switch between MediaPipe and CV modes
- Switching resets the detector

**Debug Mode Checkbox**:
- Toggle to show/hide detection metrics
- When enabled, displays:
  - Detection method (MediaPipe/CV)
  - Detection status
  - Confidence level
  - Finger count
  - Current gesture
  - Additional metrics (CV mode: detection rate, contour area, etc.)

**Color Palette**:
- Click colored buttons to select drawing color
- "Choose Color" button for custom color picker
- Current color shown in indicator box

**Brush Size Slider**:
- Adjust from 1 to 50 pixels
- Current size displayed above slider

**Action Buttons**:
- **Eraser**: Switch to eraser mode (draws with white)
- **Brush**: Return to normal drawing mode
- **Clear All**: Erase entire canvas
- **Save**: Save canvas as PNG image

**Drawing State Indicator**:
- Shows current state: IDLE, DRAWING, ERASING, or CLEARED
- Color-coded for quick reference

## Drawing Instructions Panel

Located below the camera feed, shows gesture controls for both modes:

**CV Mode Section**:
- Finger count → gesture mapping
- Always visible for quick reference

**MediaPipe Mode Section**:
- Fingertip touch combinations
- Detailed gesture descriptions

## Workflow Examples

### Basic Drawing
1. Launch application: `python main.py`
2. Show your hand to camera
3. Make "thumb + index" gesture (or 1 finger in CV mode)
4. Move hand to draw on canvas
5. Make "none" gesture to stop drawing

### Changing Colors
1. While drawing, make "thumb + ring" gesture (or 3 fingers in CV)
2. Color automatically cycles: Black → Red → Blue → Green → Yellow → Purple → Orange
3. Continue drawing with new color

### Adjusting Brush Size
1. Make "index + middle" gesture (or 4 fingers in CV) to increase
2. Make "middle + ring" gesture (MediaPipe only) to decrease
3. Or use the slider in the UI panel

### Erasing
1. Make "thumb + middle" gesture (or 2 fingers in CV)
2. Move hand over areas to erase
3. Return to draw gesture to resume drawing

### Clearing Canvas
1. Make "thumb + pinky" gesture (or 5 fingers in CV)
2. Entire canvas clears instantly
3. Drawing state indicator shows "CLEARED"

### Saving Your Work
1. Click "Save" button in UI
2. Choose filename and location
3. Image saved as PNG format

## Troubleshooting

### Hand Not Detected (MediaPipe)

**Check**:
- Hand is visible in camera frame
- Good lighting on hand
- Not too close or too far (arm's length optimal)
- MediaPipe initialized correctly (check console for errors)

**Solutions**:
- Adjust hand position
- Improve lighting
- Restart application
- Check if other webcam applications are running

### Hand Not Detected (CV Mode)

**Check**:
- Skin detection parameters are tuned
- Good contrast between hand and background
- Adequate lighting without strong shadows
- Hand is within camera frame

**Solutions**:
1. Run the skin tuner tool:
   ```bash
   python tools/skin_tuner.py
   ```
2. Adjust YCrCb and HSV sliders until hand is clearly visible
3. Press 's' to save configuration
4. Restart main application

### Jittery Cursor

**MediaPipe**:
- Keep hand steady
- Ensure stable lighting
- Check system performance (close other applications)

**CV Mode**:
- Check skin detection quality
- Reduce ambient motion
- Ensure hand is fully visible

### Gestures Not Recognized

**MediaPipe**:
- Bring fingertips closer together
- Hold gesture for 1-2 frames
- Check debug mode to see detected landmarks

**CV Mode**:
- Extend fingers clearly
- Check debug mode to see finger count
- Tune skin detection if hand outline is incomplete

### Performance Issues

**Reduce Load**:
- Close other applications
- Disable debug mode
- Use CV mode (lighter than MediaPipe)
- Reduce camera resolution (modify config.py)

**Check FPS**:
- Enable debug mode to see frame rate
- Should be >20 FPS for smooth experience
- Lower FPS indicates system bottleneck

## Advanced Features

### Debug Mode

Toggle the "Debug Mode" checkbox to see:

**MediaPipe**:
- Hand skeleton with 21 landmarks
- Fingertip positions
- Detection confidence (always 100% when detected)
- Current gesture and finger count

**CV Mode**:
- Skin mask preview (in separate window with skin_tuner)
- Contour visualization
- Finger count history
- Detection statistics (rate, total frames, gesture changes)
- Tracking vs Detection mode indicator

### Custom Skin Detection (CV Mode)

Use the skin tuner for precise calibration:

1. Run tool: `python tools/skin_tuner.py`
2. Place hand in camera view
3. Adjust sliders until hand is white in mask views
4. Check contour view - should outline hand cleanly
5. Press 's' to save configuration
6. Configuration auto-loads in main application

**Color Space Parameters**:
- **YCrCb**: Better for illumination invariance
- **HSV**: Better for hue-based skin detection
- Both are combined (AND operation) for robust detection

### Pipeline Visualization

Debug the entire CV pipeline step-by-step:

```bash
python tools/debug_detection.py
```

Shows 8 stages:
1. Original frame
2. YCrCb mask
3. HSV mask
4. Combined mask
5. Morphological operations
6. Background subtraction
7. Contours with finger detection
8. Final result with gesture

### Keyboard Shortcuts (in tools)

**Skin Tuner**:
- `s`: Save current settings to JSON
- `ESC` or close window: Exit

**Debug Detection**:
- `d`: Toggle denoising
- `b`: Toggle background subtraction
- `s`: Save current configuration
- `ESC` or close window: Exit

## Tips for Best Results

### Lighting
- Use bright, diffuse lighting
- Avoid direct sunlight or harsh shadows
- Consistent lighting temperature (avoid mixing daylight and indoor lights)

### Background
- Plain, contrasting background works best for CV mode
- Avoid skin-colored objects in background
- Keep background relatively static

### Hand Position
- Keep hand flat and fingers spread for finger counting
- Hold gestures steady for 1-2 frames
- Avoid fast movements (use smooth motions)
- Position hand centered in camera frame

### System Performance
- Close unnecessary applications
- Ensure camera is not used by other programs
- Update graphics drivers for optimal rendering
- Use USB 3.0 port for webcam if available

## Common Use Cases

### Teaching/Presentations
- Draw diagrams hands-free while explaining
- Highlight points on slides
- Create quick sketches without interrupting flow

### Accessibility
- Drawing without fine motor control
- Touchless interface for hygiene
- Alternative input method

### Creative Projects
- Experimental art creation
- Interactive installations
- Computer vision demonstrations

### Development/Learning
- Study computer vision algorithms
- Understand gesture recognition
- Explore hand tracking techniques
