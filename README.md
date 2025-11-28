# Gesture Paint - Hand-Controlled Drawing Application

A computer vision application that enables hands-free drawing and painting using hand gestures detected through your webcam. Control a full-featured paint application without touching your keyboard or mouse.

## 🎯 Features

- **Gesture-Based Controls**: Draw, erase, change colors, and adjust brush size using natural hand gestures
- **Dual Detection Modes**: 
  - MediaPipe (AI-based, high accuracy)
  - Traditional CV (no neural networks, customizable)
- **Auto-Calibration**: Quick 5-second setup for optimal hand detection
- **Real-Time Performance**: Smooth drawing experience with position smoothing
- **Full Paint Features**: Multiple colors, adjustable brush sizes, eraser, save/load images

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

### 1. Calibrate (First Time Setup)

Run the calibration tool for optimal hand detection:

```bash
python calibrate.py
```

- Select option **1** (Auto-Calibrate)
- Position your hand in the yellow box
- Press **SPACE** to start
- Move your hand slowly for 5 seconds
- Save when complete

### 2. Launch the Application

```bash
python gesture_paint.py
```

The application window will open with:
- **Left panel**: Live camera feed showing hand detection
- **Right panel**: Drawing canvas
- **Mode selector**: Switch between MediaPipe and CV detection

### 3. Use Hand Gestures

| Gesture | Action |
|---------|--------|
| 👆 Thumb + Index | **Draw** (pen mode) |
| 👆 Thumb + Middle | **Erase** |
| 👆 Thumb + Ring | **Cycle colors** |
| 👆 Thumb + Pinky | **Clear canvas** |
| 👆 Index + Middle | **Increase brush size** |
| 👆 Middle + Ring | **Decrease brush size** |

## 🛠️ Additional Tools

### Calibration Tool

Fine-tune hand detection for your lighting conditions:

```bash
python calibrate.py
```

**Options:**
1. **Auto-Calibrate** - Quick 5-second automatic setup (recommended)
2. **Manual Tuning** - Advanced trackbar controls for precise adjustment
3. **Verify** - Check current calibration status

### Demo & Testing

Test different detection modes and features:

```bash
python demo.py
```

**Options:**
1. **Detector Comparison** - Side-by-side old vs new detection
2. **Live Test** - Switch between MediaPipe and CV modes in real-time
3. **Edge Detection** - Sobel edge detection demonstration

## 📖 Detailed Documentation

For comprehensive documentation including technical details, troubleshooting, and advanced features, see [SIMPLIFIED_README.md](SIMPLIFIED_README.md).

## 🔧 Troubleshooting

**Hand not detected?**
- Run `python calibrate.py` and recalibrate
- Ensure good lighting conditions
- Try switching detection modes (MediaPipe ↔ CV)

**Jittery cursor?**
- Normal slight jitter is reduced by built-in smoothing
- Ensure stable lighting
- Keep hand steady during gestures

**Low frame rate?**
- Switch to CV mode (faster than MediaPipe)
- Close other camera-using applications
- Ensure your webcam supports 30 FPS

## 💡 Tips for Best Results

- Use in well-lit environments
- Keep background simple and uncluttered
- Position camera at eye level
- Calibrate once per lighting setup
- MediaPipe mode: Better gesture accuracy
- CV mode: Better performance in varied lighting

## 📂 Project Structure

```
├── gesture_paint.py      # Main application
├── calibrate.py          # Calibration tool
├── demo.py               # Testing & demos
├── cv_detector.py        # Traditional CV detector
├── mediapipe_detector.py # MediaPipe wrapper
└── hand_detector_base.py # Base detector interface
```

## 🎨 Application Controls

**Keyboard Shortcuts:**
- Drawing tools accessible via toolbar
- Save canvas via File menu
- Color picker for custom colors

**Mouse Controls:**
- Use toolbar buttons as alternative to gestures
- Click and drag for traditional mouse drawing

## 📝 License

See [LICENSE](LICENSE) for details.

## 👥 Team

- Edward Leroux
- Michal Naumiak
- François Gerbeau
- Théo Lahmar

