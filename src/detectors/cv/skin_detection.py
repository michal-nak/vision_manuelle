"""
Skin detection using multiple color spaces
"""
import cv2
import numpy as np


def detect_skin_ycrcb_hsv(frame, ycrcb_lower, ycrcb_upper, hsv_lower, hsv_upper):
    """
    Detect skin using both YCrCb and HSV color spaces
    
    Args:
        frame: Input BGR frame
        ycrcb_lower: Lower bound for YCrCb
        ycrcb_upper: Upper bound for YCrCb
        hsv_lower: Lower bound for HSV
        hsv_upper: Upper bound for HSV
        
    Returns:
        Combined binary mask
    """
    # Adaptive filtering for denoising
    denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    
    # YCrCb skin detection
    ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
    mask_ycrcb = cv2.inRange(ycrcb, ycrcb_lower, ycrcb_upper)
    
    # HSV skin detection with hue wrap-around handling
    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    
    # Handle hue wrap-around (e.g., 170-180 and 0-10 for red/pink skin tones)
    if hsv_lower[0] > hsv_upper[0]:
        # Wrapped range: combine two masks (H from lower to 180 OR H from 0 to upper)
        mask_hsv1 = cv2.inRange(hsv, hsv_lower, np.array([180, hsv_upper[1], hsv_upper[2]], dtype=np.uint8))
        mask_hsv2 = cv2.inRange(hsv, np.array([0, hsv_lower[1], hsv_lower[2]], dtype=np.uint8), hsv_upper)
        mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
    else:
        # Normal range
        mask_hsv = cv2.inRange(hsv, hsv_lower, hsv_upper)
    
    # Combine masks (intersection for better precision)
    mask_combined = cv2.bitwise_and(mask_ycrcb, mask_hsv)
    
    return mask_combined


def apply_morphological_operations(mask):
    """
    Apply morphological operations to clean up mask
    
    Args:
        mask: Binary mask
        
    Returns:
        Cleaned mask
    """
    # Morphological operations to clean up mask
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    mask = cv2.erode(mask, kernel_erode, iterations=2)
    mask = cv2.dilate(mask, kernel_dilate, iterations=2)
    
    # Gaussian blur to smooth edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    return mask


def find_largest_contour(mask, min_area=1000, max_area=300000):
    """
    Find largest valid contour in mask
    
    Args:
        mask: Binary mask
        min_area: Minimum contour area
        max_area: Maximum contour area
        
    Returns:
        Largest contour or None
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filter by area and find largest
    valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    
    if not valid_contours:
        return None
    
    return max(valid_contours, key=cv2.contourArea)
