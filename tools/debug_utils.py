"""
Shared utilities for debug tools
"""
import cv2
import numpy as np
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detectors.cv.cv_detector import CVDetector
from src.detectors.cv.skin_detection import detect_skin_ycrcb_hsv
from src.core.config import MIN_HAND_AREA, MAX_HAND_AREA


def init_camera(width=640, height=480):
    """Initialize camera with error handling"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot open camera")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def get_skin_mask(frame, detector):
    """Get skin detection mask from frame"""
    color_bounds = (detector.ycrcb_lower, detector.ycrcb_upper, 
                   detector.hsv_lower, detector.hsv_upper)
    mask = detect_skin_ycrcb_hsv(frame, *color_bounds, 
                                 denoise_h=10, enable_denoising=False)
    return mask


def draw_contour_info(frame, contours, min_area, max_area):
    """Draw contours with color coding based on area validity"""
    h, w = frame.shape[:2]
    max_area_pixels = int(w * h * max_area)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            color = (0, 0, 255)  # Red - too small
        elif area > max_area_pixels:
            color = (255, 0, 255)  # Magenta - too large
        else:
            color = (0, 255, 0)  # Green - valid
        cv2.drawContours(frame, [contour], -1, color, 2)
    return frame


def create_info_panel(width, height, title, info_dict):
    """Create an info panel with text overlay"""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    y = 30
    cv2.putText(panel, title, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (255, 255, 255), 2)
    y += 30
    
    for key, value in info_dict.items():
        text = f"{key}: {value}"
        cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (200, 200, 200), 1)
        y += 25
    return panel


def save_color_config(ycrcb_lower, ycrcb_upper, hsv_lower, hsv_upper, description=""):
    """Save color ranges to JSON config file"""
    config_path = Path('config/color_ranges.json')
    config_path.parent.mkdir(exist_ok=True)
    
    # Convert to lists and ensure Python int types (not numpy)
    def to_list(arr):
        if isinstance(arr, np.ndarray):
            return [int(x) for x in arr]
        return [int(x) for x in arr]
    
    config = {
        "timestamp": datetime.now().isoformat(),
        "ycrcb_lower": to_list(ycrcb_lower),
        "ycrcb_upper": to_list(ycrcb_upper),
        "hsv_lower": to_list(hsv_lower),
        "hsv_upper": to_list(hsv_upper),
        "description": description or "Skin detection color ranges for CV detector"
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Configuration saved to {config_path}")
    return config_path


def load_color_config():
    """Load color ranges from JSON config file"""
    config_path = Path('config/color_ranges.json')
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Config file corrupted, ignoring: {e}")
            return None
    return None


def calculate_skin_bounds(ycrcb_samples, hsv_samples, margin_factor=0.15):
    """
    Calculate skin color bounds from samples using percentile-based approach
    Supports HSV hue wrap-around detection
    
    Args:
        ycrcb_samples: List of YCrCb sample arrays
        hsv_samples: List of HSV sample arrays  
        margin_factor: Margin to add beyond percentiles (default 15%)
        
    Returns:
        tuple: (ycrcb_lower, ycrcb_upper, hsv_lower, hsv_upper)
    """
    ycrcb_all = np.vstack(ycrcb_samples)
    hsv_all = np.vstack(hsv_samples)
    
    # Use 5th and 95th percentiles (more robust than mean±std)
    ycrcb_p5 = np.percentile(ycrcb_all, 5, axis=0)
    ycrcb_p95 = np.percentile(ycrcb_all, 95, axis=0)
    ycrcb_range = ycrcb_p95 - ycrcb_p5
    ycrcb_lower = np.clip(ycrcb_p5 - margin_factor * ycrcb_range, 0, 255).astype(np.uint8)
    ycrcb_upper = np.clip(ycrcb_p95 + margin_factor * ycrcb_range, 0, 255).astype(np.uint8)
    
    # HSV bounds with hue wrap-around support
    hsv_p5 = np.percentile(hsv_all, 5, axis=0)
    hsv_p95 = np.percentile(hsv_all, 95, axis=0)
    
    # Handle hue wrap-around (0-180 in OpenCV)
    hue_values = hsv_all[:, 0]
    hue_median = np.median(hue_values)
    
    # Check if hue spans across 0/180 boundary
    hue_span = hsv_p95[0] - hsv_p5[0]
    if hue_span > 90:  # Likely wrapping around
        # Adjust for wrap-around by shifting values
        hue_adjusted = np.where(hue_values < hue_median, hue_values + 180, hue_values)
        hsv_p5[0] = np.percentile(hue_adjusted, 5) % 180
        hsv_p95[0] = np.percentile(hue_adjusted, 95) % 180
    
    hsv_range = hsv_p95 - hsv_p5
    hsv_range[0] = min(hsv_range[0], 180 - hsv_range[0])  # Ensure hue range is reasonable
    
    hsv_lower = np.clip(hsv_p5 - margin_factor * hsv_range, 0, [180, 255, 255]).astype(np.uint8)
    hsv_upper = np.clip(hsv_p95 + margin_factor * hsv_range, 0, [180, 255, 255]).astype(np.uint8)
    
    # Print metrics
    print(f"\n📊 Calibration Metrics:")
    print(f"  YCrCb Range: Y={ycrcb_upper[0]-ycrcb_lower[0]:3d} Cr={ycrcb_upper[1]-ycrcb_lower[1]:3d} Cb={ycrcb_upper[2]-ycrcb_lower[2]:3d}")
    print(f"  HSV Range:   H={hsv_upper[0]-hsv_lower[0]:3d} S={hsv_upper[1]-hsv_lower[1]:3d} V={hsv_upper[2]-hsv_lower[2]:3d}")
    if hsv_lower[0] > hsv_upper[0]:
        print(f"  ⚠️  Hue wrap-around detected: {hsv_lower[0]} → 180 → 0 → {hsv_upper[0]}")
    
    return ycrcb_lower, ycrcb_upper, hsv_lower, hsv_upper
