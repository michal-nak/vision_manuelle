# Gesture Paint

Draw and paint using only your hand gestures! Control a full-featured paint application through your webcam without touching your keyboard or mouse.

![Demo](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Gesture-Based Drawing**: Draw, erase, change colors, and adjust brush size with hand gestures - completely hands-free!
- **Dual Detection Modes**: Choose between MediaPipe (AI-powered, ~15 FPS) or Computer Vision (traditional, ~1 FPS)
- **Real-Time Performance**: MediaPipe achieves ~15 FPS with optimized C++ implementation
- **Full Paint Features**: Multiple colors, adjustable brush sizes, save/load - all controlled by gestures
- **Debug Mode**: Toggle detailed detection metrics and visualization
- **Cross-Platform**: Works on Windows, Linux, and macOS

## Quick Demo

### MediaPipe Mode
- **Draw**: Touch thumb + index finger together
- **Erase**: Touch thumb + middle finger together  
- **Change Color**: Touch thumb + ring finger together
- **Clear Canvas**: Touch thumb + pinky together

### CV Mode
- **Draw**: Show 1 finger
- **Erase**: Show 2 fingers
- **Change Color**: Show 3 fingers
- **Increase Size**: Show 4 fingers
- **Clear Canvas**: Show 5 fingers (all)

## Installation

### Requirements
- Python 3.10+
- Webcam
- Windows/Linux/macOS

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/michal-nak/vision_manuelle.git
   cd vision_manuelle
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Launch the Application

**MediaPipe Mode (Recommended)**:
```bash
python main.py
```

**CV Mode** (with auto-calibration):
```bash
python main.py cv
```

**CV Mode** (skip calibration, use saved config):
```bash
python main.py cv --skip-calibration
# or shorthand:
python main.py cv -s
```

**CV Mode** (skip auto-optimization, faster startup):
```bash
python main.py cv --skip-optimization
# or shorthand:
python main.py cv -o
```

**CV Mode** (with debug overlay):
```bash
python main.py cv --debug
# or shorthand:
python main.py cv -d
```

**Combine flags**:
```bash
python main.py cv -od  # Skip optimization, enable debug
python main.py cv -s   # Skip all calibration
```

### Using Gestures

**MediaPipe Mode** - Touch fingertips together:
| Gesture | Action |
|---------|--------|
| Thumb + Index | Draw |
| Thumb + Middle | Erase |
| Thumb + Ring | Cycle Colors |
| Thumb + Pinky | Clear Canvas |
| Index + Middle | Increase Brush Size |
| Middle + Ring | Decrease Brush Size |

**CV Mode** - Show number of fingers:
| Fingers | Action |
|---------|--------|
| 1 | Draw |
| 2 | Erase |
| 3 | Cycle Colors |
| 4 | Increase Brush Size |
| 5 | Clear Canvas |
| 0 (fist) | No Action |

### UI Controls

**Gesture Controls** (Primary - Hands-Free):
- All drawing, erasing, color changes, and size adjustments done via gestures
- See gesture tables above for complete control mapping

**Toolbar** (Settings & File Operations):
- **Mode Selector**: Switch between MediaPipe and CV detection
- **Debug Mode**: Toggle to show detection metrics
- **Color Picker**: Alternative to 3-finger gesture for custom colors
- **Brush Size Slider**: Alternative to 4-finger gesture (1-50 pixels)
- **Clear/Save Buttons**: File operations (also available via 5-finger gesture for clear)

## Advanced Features

### Auto-Calibration System

**CV Mode now includes intelligent calibration**:
- **10-second color calibration**: MediaPipe detects your hand, system samples skin tones
- **18-second auto-optimization**: Tests 6 parameter presets against MediaPipe ground truth
- **IoU-based validation**: Uses Intersection over Union (30% threshold) for spatial accuracy
- **Palm center alignment**: During calibration, both detectors use palm center for accurate comparison
- **Visual feedback**: See both MediaPipe (blue) and CV (green) detections during optimization

**Command-line options**:
```bash
python main.py cv              # Full calibration (28s total)
python main.py cv -o           # Color only, skip optimization (10s)
python main.py cv -s           # Skip all, use saved config (instant)
python main.py cv -d           # Enable debug overlay
```

### Manual Calibration Tools

**Skin Detection Tuner**:
Fine-tune all 21 parameters for difficult lighting:
```bash
python tools/skin_tuner.py
```

**Pipeline Visualizer**:
See all 8 processing steps of the CV detector:
```bash
python tools/debug_detection.py
```

## Tips for Best Results

- Use **good lighting** (avoid backlighting)
- Keep **plain background** behind your hand
- Position hand **centered in frame**
- **First-time CV users**: Run full calibration once, then use `-s` flag for instant startup
- **Quick testing**: Use `-o` flag to skip 18-second optimization
- **MediaPipe** (Recommended): More accurate (95%), better performance (~15 FPS), works in varied conditions
- **CV Mode** (Educational): Learn traditional computer vision (~1 FPS), now with intelligent auto-calibration

### Performance Comparison

| Mode | FPS | Accuracy | Best For |
|------|-----|----------|----------|
| MediaPipe | ~15 | 95% | Production, demos, general use |
| CV | ~1 | 85% | Learning CV algorithms, research |

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)**: System design and technical details
- **[Usage Guide](docs/USAGE.md)**: Comprehensive user manual
- **[API Reference](docs/API.md)**: Developer documentation
- **[Contributing](docs/CONTRIBUTING.md)**: How to contribute
- **[Development History](docs/DEVELOPMENT.md)**: Project evolution and technical decisions

## Troubleshooting

**Hand not detected in CV mode?**
- Run fresh calibration: `python main.py cv` (without `-s` flag)
- Ensure good lighting during calibration
- Try manual tuning: `python tools/skin_tuner.py`
- Use debug mode to see detection issues: `python main.py cv -d`

**Calibration issues?**
- Keep hand in frame for full 10-second color sampling
- Move hand slowly during calibration for better color coverage
- If optimization fails (0% detection), use `-o` flag to skip it
- Check that MediaPipe detects your hand during preview

**Jittery cursor?**
- Keep hand steady
- Ensure stable lighting
- Application uses smoothing to reduce jitter

**Low frame rate?**
- Use MediaPipe mode for best performance (~15 FPS)
- CV mode is slower (~1 FPS) due to Python processing overhead
- Close other camera-using applications
- Check system performance and CPU usage

See [USAGE.md](docs/USAGE.md#troubleshooting) for detailed troubleshooting.

## Project Structure

```
vision_manuelle/
├── src/                    # Source code
│   ├── core/              # Configuration and utilities
│   ├── detectors/         # Hand detection implementations
│   │   ├── cv/           # Computer vision detector
│   │   └── mediapipe_detector.py
│   └── ui/               # User interface
├── tools/                 # Debugging and calibration tools
├── docs/                  # Documentation
├── main.py               # Application entry point
└── requirements.txt      # Dependencies
```

## Academic Context

This project was developed as part of a computer vision course, demonstrating:
- Real-time hand detection algorithms
- Gesture recognition systems
- Traditional CV vs ML-based approaches
- Performance optimization techniques

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for technical analysis and project evolution.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Quick Start for Contributors
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 👥 Team

- **Michal Naumiak** - Lead Developer, CV Optimization
- **Edward Leroux** - Initial Implementation, UI Design
- **François Gerbeau** - Original Detection System
- **Théo Lahmar** - Testing & Documentation

## Acknowledgments

- **OpenCV** - Computer vision library
- **MediaPipe** - Hand tracking solution
- **NumPy** - Numerical computing
- **Python Community** - Excellent ecosystem

## Contact

- **Repository**: [github.com/michal-nak/vision_manuelle](https://github.com/michal-nak/vision_manuelle)
- **Issues**: [Report a bug or request a feature](https://github.com/michal-nak/vision_manuelle/issues)

---

**Course**: Vision Numérique | **Semester**: Automne 2025-26
