"""
Skin detection using multiple color spaces
"""
import cv2
import numpy as np


def detect_skin_ycrcb_hsv(frame, ycrcb_lower, ycrcb_upper, hsv_lower, hsv_upper, denoise_h=10, enable_denoising=True):
    """
    Detect skin using both YCrCb and HSV color spaces
    
    Args:
        frame: Input BGR frame
        ycrcb_lower: Lower bound for YCrCb
        ycrcb_upper: Upper bound for YCrCb
        hsv_lower: Lower bound for HSV
        hsv_upper: Upper bound for HSV
        denoise_h: Denoising strength (default 10)
        enable_denoising: Enable expensive denoising (default False for performance)
        
    Returns:
        Combined binary mask
    """
    # Optional denoising (disabled by default - adds 200-300ms latency)
    # Enable only if dealing with very noisy camera input
    if enable_denoising:
        denoised = cv2.fastNlMeansDenoisingColored(frame, None, denoise_h, denoise_h, 7, 21)
    else:
        denoised = frame
    
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
    
    # Combine masks (union for better coverage)
    # Changed from bitwise_and to bitwise_or - AND was creating holes in palm interior
    # OR detects skin if EITHER color space matches (better for uniform palm regions)
    mask_combined = cv2.bitwise_or(mask_ycrcb, mask_hsv)
    
    return mask_combined


def apply_morphological_operations(mask, kernel_small=3, kernel_large=7, iterations=2):
    """
    Apply morphological operations to clean up mask
    
    Args:
        mask: Binary mask
        kernel_small: Size of small kernel for erosion
        kernel_large: Size of large kernel for dilation
        iterations: Number of iterations for operations
        
    Returns:
        Cleaned mask
    """
    # Morphological operations to clean up mask
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_small, kernel_small))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_large, kernel_large))
    
    mask = cv2.erode(mask, kernel_erode, iterations=iterations)
    mask = cv2.dilate(mask, kernel_dilate, iterations=iterations)
    
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


def filter_forearm_by_shape(contour, frame_height):
    """
    Filter out forearm by analyzing contour shape properties
    Uses geometric features to distinguish hand from forearm
    
    Args:
        contour: Input contour
        frame_height: Height of frame (for position filtering)
        
    Returns:
        bool: True if contour is likely a hand (not forearm)
    """
    if contour is None or len(contour) < 5:
        return False
    
    # Get contour properties
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if area < 100 or perimeter < 10:
        return False
    
    # 1. Aspect Ratio Check (hands are roughly square, forearms are elongated)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # Hand aspect ratio typically 0.6 to 1.5, forearm > 2.0 or < 0.5
    if aspect_ratio > 2.0 or aspect_ratio < 0.4:
        return False  # Too elongated (likely forearm)
    
    # 2. Compactness Check (circularity)
    # Compactness = 4π * Area / Perimeter²
    # Circle = 1.0, elongated shape < 0.5
    compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    
    if compactness < 0.3:  # Too elongated
        return False
    
    # 3. Solidity Check (hand has fingers = convex hull much larger than contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Hand solidity: 0.6-0.9 (fingers create gaps)
    # Forearm solidity: 0.9-1.0 (mostly solid/convex)
    if solidity > 0.95:  # Too solid, likely forearm
        return False
    
    # 4. Position Check (filter bottom 20% of frame - likely forearm)
    contour_center_y = y + h / 2
    if contour_center_y > frame_height * 0.8:  # In bottom 20%
        return False
    
    # 5. Extent Check (ratio of contour area to bounding rect area)
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0
    
    # Hand extent: 0.5-0.8 (irregular with fingers)
    # Forearm extent: 0.7-0.95 (more rectangular)
    if extent > 0.90:  # Too rectangular
        return False
    
    return True


def filter_forearm_by_orientation(contour):
    """
    Filter forearm by analyzing contour orientation
    Forearms tend to be vertical/diagonal, hands are more varied
    
    Args:
        contour: Input contour
        
    Returns:
        bool: True if likely a hand, False if likely forearm
    """
    if len(contour) < 5:
        return True
    
    try:
        # Fit ellipse to get orientation
        ellipse = cv2.fitEllipse(contour)
        angle = ellipse[2]  # Rotation angle
        
        # Normalize angle to 0-180
        angle = angle % 180
        
        # Forearms are typically vertical (near 90°) or diagonal (30-60°, 120-150°)
        # Hands have more varied orientations
        
        # Check if strongly vertical (likely forearm entering from bottom)
        if 80 < angle < 100:  # Nearly vertical
            # Get aspect ratio of fitted ellipse
            width, height = ellipse[1]
            if height > width * 2.5:  # Very elongated vertically
                return False
        
        return True
        
    except:
        return True  # If fitting fails, assume it's valid


def crop_to_upper_region(mask, contour, crop_ratio=0.7):
    """
    Crop contour to upper region to remove forearm
    Traditional CV technique: region of interest (ROI) selection
    
    Args:
        mask: Binary mask
        contour: Input contour
        crop_ratio: Keep top X% of contour (default 70%)
        
    Returns:
        Cropped contour or None
    """
    if contour is None or len(contour) < 5:
        return None
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Define crop line (keep upper portion)
    crop_y = int(y + h * crop_ratio)
    
    # Create cropped mask (only upper region)
    cropped_mask = mask.copy()
    cropped_mask[crop_y:, :] = 0  # Zero out lower region
    
    # Find contour in cropped region
    contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Return largest contour in cropped region
    return max(contours, key=cv2.contourArea)


def select_hand_contour_intelligent(contours, frame_shape):
    """
    Intelligently select hand contour from multiple candidates
    Filters out forearms using multiple geometric criteria
    
    Args:
        contours: List of contours
        frame_shape: (height, width) of frame
        
    Returns:
        Best hand contour or None
    """
    if not contours:
        return None
    
    h, w = frame_shape[:2]
    
    # Score each contour
    scored_contours = []
    
    for contour in contours:
        if len(contour) < 5:
            continue
        
        area = cv2.contourArea(contour)
        if area < 1000:  # Too small
            continue
        
        # Calculate multiple features
        x, y, cw, ch = cv2.boundingRect(contour)
        
        # Feature 1: Aspect ratio (hands are squarish)
        aspect_ratio = float(cw) / ch if ch > 0 else 0
        aspect_score = 1.0 if 0.6 < aspect_ratio < 1.5 else 0.3
        
        # Feature 2: Position (hands in upper/middle, forearms in lower)
        center_y = y + ch / 2
        position_score = 1.0 - (center_y / h)  # Higher score for upper positions
        
        # Feature 3: Compactness (hands less compact due to fingers)
        perimeter = cv2.arcLength(contour, True)
        compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        compact_score = 1.0 if 0.3 < compactness < 0.7 else 0.5
        
        # Feature 4: Solidity (hands have lower solidity due to finger gaps)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        solidity_score = 1.0 if 0.6 < solidity < 0.9 else 0.5
        
        # Feature 5: Area size (prefer medium-sized contours)
        size_score = min(1.0, area / 15000)  # Normalize around typical hand size
        
        # Weighted score
        total_score = (
            aspect_score * 0.25 +
            position_score * 0.25 +
            compact_score * 0.20 +
            solidity_score * 0.20 +
            size_score * 0.10
        )
        
        scored_contours.append((total_score, area, contour))
    
    if not scored_contours:
        return None
    
    # Sort by score (descending), then by area (descending)
    scored_contours.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    # Return best scoring contour
    return scored_contours[0][2]


def detect_wrist_and_crop(mask, contour):
    """
    Detect wrist position using horizontal edge analysis
    Traditional CV: edge detection + geometric analysis
    
    Args:
        mask: Binary mask of hand region
        contour: Hand contour
        
    Returns:
        Cropped contour (hand only, no forearm) or original if wrist not found
    """
    if contour is None or len(contour) < 5:
        return contour
    
    try:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract region of interest
        roi = mask[y:y+h, x:x+w]
        
        if roi.shape[0] < 10 or roi.shape[1] < 10:
            return contour
        
        # Analyze horizontal cross-sections (find narrowest point = wrist)
        min_width = w
        wrist_y_local = h // 2  # Default to middle
        
        # Scan from bottom 60% to top 40% (wrist typically in this range)
        start_scan = int(h * 0.4)
        end_scan = int(h * 0.9)
        
        for i in range(start_scan, end_scan):
            if i >= roi.shape[0]:
                break
            
            # Count white pixels in this row (hand width at this height)
            row_width = np.sum(roi[i, :] > 0)
            
            if row_width < min_width and row_width > w * 0.1:  # Ignore noise
                min_width = row_width
                wrist_y_local = i
        
        # If wrist found (narrow region in lower part)
        if min_width < w * 0.6 and wrist_y_local > h * 0.5:
            # Crop at wrist position (keep upper part only)
            crop_y_global = y + wrist_y_local - int(h * 0.05)  # Keep small margin
            
            # Create cropped mask
            cropped_mask = mask.copy()
            cropped_mask[crop_y_global:, :] = 0
            
            # Find contour in cropped region
            contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Return largest contour in cropped region
                return max(contours, key=cv2.contourArea)
        
        return contour
        
    except Exception as e:
        return contour


def apply_top_priority_filter(mask, frame_height):
    """
    Apply region-based filtering to prioritize upper part of frame
    Geometric approach: weight mask by vertical position
    
    Args:
        mask: Binary mask
        frame_height: Height of frame
        
    Returns:
        Filtered mask with upper regions weighted higher
    """
    # Create vertical gradient (top = 1.0, bottom = 0.3)
    gradient = np.linspace(1.0, 0.3, frame_height).reshape(-1, 1)
    gradient = np.tile(gradient, (1, mask.shape[1]))
    
    # Apply gradient weighting to mask
    weighted_mask = (mask.astype(float) * gradient).astype(np.uint8)
    
    # Threshold to binary
    _, binary_mask = cv2.threshold(weighted_mask, 127, 255, cv2.THRESH_BINARY)
    
    return binary_mask
