"""
Configuration constants for the hand detection system
"""

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Processing settings
PROCESSING_SCALE = 2
POSITION_SMOOTHING = 5
FINGER_COUNT_SMOOTHING = 3

# Color ranges (calibrated values)
YCRCB_LOWER = [126, 117, 28]
YCRCB_UPPER = [203, 155, 129]
HSV_LOWER = [12, 25, 121]
HSV_UPPER = [35, 255, 255]

# Default color ranges (fallback)
YCRCB_LOWER_DEFAULT = [0, 133, 77]
YCRCB_UPPER_DEFAULT = [255, 173, 127]
HSV_LOWER_DEFAULT = [0, 30, 60]
HSV_UPPER_DEFAULT = [20, 150, 255]

# Detection settings
MIN_HAND_AREA = 3000
MAX_HAND_AREA = 0.6

# Background subtractor settings
BG_HISTORY = 500
BG_VAR_THRESHOLD = 16
BG_DETECT_SHADOWS = False

# MediaPipe settings
MP_MODEL_COMPLEXITY = 0
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5

# Calibration settings
CALIBRATION_DURATION = 5
CALIBRATION_SAMPLE_INTERVAL = 3
CALIBRATION_MIN_SAMPLES = 100
CALIBRATION_RECT = (220, 140, 200, 200)

# UI settings
UI_WINDOW_WIDTH = 1200
UI_WINDOW_HEIGHT = 600
UI_MIN_WIDTH = 800
UI_MIN_HEIGHT = 400

# Color palette
COLOR_PALETTE = [
    "#000000", "#FF0000", "#00FF00", "#0000FF",
    "#FFFF00", "#FFA500", "#800080", "#00FFFF"
]

# Gestures
GESTURE_NAMES = {
    "thumb_index": "Draw",
    "thumb_middle": "Erase",
    "thumb_ring": "Cycle Color",
    "thumb_pinky": "Clear",
    "index_middle": "Increase Size",
    "middle_ring": "Decrease Size"
}

# File paths
CALIBRATION_FILE = "cv_detector.py"
CALIBRATION_BACKUP_FILE = "calibration_backup.json"
