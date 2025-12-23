"""
Simple Skin Detection Tuner
Focus on adjusting color ranges to match your skin tone
"""
import cv2
import numpy as np
from debug_utils import init_camera, save_color_config, load_color_config

def nothing(x):
    pass

def main():
    print("=" * 70)
    print("SKIN DETECTION TUNER")
    print("=" * 70)
    print("Adjust the trackbars to make your HAND WHITE in the masks")
    print("The goal is to have your hand clearly visible while filtering out background")
    print("\nTips:")
    print("  - Start by adjusting YCrCb ranges (usually more important)")
    print("  - Then fine-tune with HSV")
    print("  - Press 's' to save the current values")
    print("  - Press 'q' to quit")
    print("=" * 70)
    
    from src.detectors import CVDetector
    
    cap = init_camera()
    if not cap:
        return
    
    detector = CVDetector()
    config = load_color_config()
    
    # Use config values if available
    if config:
        detector.ycrcb_lower = np.array(config['ycrcb_lower'], dtype=np.uint8)
        detector.ycrcb_upper = np.array(config['ycrcb_upper'], dtype=np.uint8)
        detector.hsv_lower = np.array(config['hsv_lower'], dtype=np.uint8)
        detector.hsv_upper = np.array(config['hsv_upper'], dtype=np.uint8)
        print(f"Loaded saved configuration")
    else:
        print(f"No config found, using default values")
    
    # Create main display window (resizable) - size based on 2 camera views side by side
    # Camera is 640x480, so 2 side by side with half size each = 640x480 total per row, 3 rows
    cv2.namedWindow('Skin Detection Tuner', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Skin Detection Tuner', 640, 720 + 110)  # 3 rows of 240px height + header/footer
    
    # Create separate trackbar window
    cv2.namedWindow('Adjust Values', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Adjust Values', 400, 400)
    cv2.moveWindow('Adjust Values', 50, 50)
    
    # YCrCb trackbars
    cv2.createTrackbar('Y Min', 'Adjust Values', detector.ycrcb_lower[0], 255, nothing)
    cv2.createTrackbar('Y Max', 'Adjust Values', detector.ycrcb_upper[0], 255, nothing)
    cv2.createTrackbar('Cr Min', 'Adjust Values', detector.ycrcb_lower[1], 255, nothing)
    cv2.createTrackbar('Cr Max', 'Adjust Values', detector.ycrcb_upper[1], 255, nothing)
    cv2.createTrackbar('Cb Min', 'Adjust Values', detector.ycrcb_lower[2], 255, nothing)
    cv2.createTrackbar('Cb Max', 'Adjust Values', detector.ycrcb_upper[2], 255, nothing)
    
    # HSV trackbars
    cv2.createTrackbar('H Min', 'Adjust Values', detector.hsv_lower[0], 180, nothing)
    cv2.createTrackbar('H Max', 'Adjust Values', detector.hsv_upper[0], 180, nothing)
    cv2.createTrackbar('S Min', 'Adjust Values', detector.hsv_lower[1], 255, nothing)
    cv2.createTrackbar('S Max', 'Adjust Values', detector.hsv_upper[1], 255, nothing)
    cv2.createTrackbar('V Min', 'Adjust Values', detector.hsv_lower[2], 255, nothing)
    cv2.createTrackbar('V Max', 'Adjust Values', detector.hsv_upper[2], 255, nothing)
    
    print("\nTrackbar window created. Show your hand and adjust sliders!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Get trackbar values (with error handling)
        try:
            ycrcb_lower = np.array([
                cv2.getTrackbarPos('Y Min', 'Adjust Values'),
                cv2.getTrackbarPos('Cr Min', 'Adjust Values'),
                cv2.getTrackbarPos('Cb Min', 'Adjust Values')
            ], dtype=np.uint8)
            
            ycrcb_upper = np.array([
                cv2.getTrackbarPos('Y Max', 'Adjust Values'),
                cv2.getTrackbarPos('Cr Max', 'Adjust Values'),
                cv2.getTrackbarPos('Cb Max', 'Adjust Values')
            ], dtype=np.uint8)
            
            hsv_lower = np.array([
                cv2.getTrackbarPos('H Min', 'Adjust Values'),
                cv2.getTrackbarPos('S Min', 'Adjust Values'),
                cv2.getTrackbarPos('V Min', 'Adjust Values')
            ], dtype=np.uint8)
            
            hsv_upper = np.array([
                cv2.getTrackbarPos('H Max', 'Adjust Values'),
                cv2.getTrackbarPos('S Max', 'Adjust Values'),
                cv2.getTrackbarPos('V Max', 'Adjust Values')
            ], dtype=np.uint8)
        except cv2.error:
            # If trackbar window closed, use last known values
            continue
        
        # Apply skin detection
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, ycrcb_lower, ycrcb_upper)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Handle hue wrap-around (e.g., 170-180 and 0-10 for red/pink skin tones)
        if hsv_lower[0] > hsv_upper[0]:
            # Wrapped range: combine two masks
            mask_hsv1 = cv2.inRange(hsv, hsv_lower, np.array([180, hsv_upper[1], hsv_upper[2]], dtype=np.uint8))
            mask_hsv2 = cv2.inRange(hsv, np.array([0, hsv_lower[1], hsv_lower[2]], dtype=np.uint8), hsv_upper)
            mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
        else:
            # Normal range
            mask_hsv = cv2.inRange(hsv, hsv_lower, hsv_upper)
        
        mask_combined = cv2.bitwise_and(mask_ycrcb, mask_hsv)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Create visualizations
        mask_ycrcb_vis = cv2.cvtColor(mask_ycrcb, cv2.COLOR_GRAY2BGR)
        mask_hsv_vis = cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR)
        mask_combined_vis = cv2.cvtColor(mask_combined, cv2.COLOR_GRAY2BGR)
        mask_clean_vis = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
        
        # Apply mask to original frame
        result = cv2.bitwise_and(frame, frame, mask=mask_clean)
        
        # Use half size to keep aspect ratio (640x480 -> 320x240)
        view_width = w // 2
        view_height = h // 2
        
        def resize_view(img):
            return cv2.resize(img, (view_width, view_height), interpolation=cv2.INTER_AREA)
        
        # Add labels helper
        def add_label(img, text, color=(0, 255, 255)):
            labeled = resize_view(img)
            cv2.rectangle(labeled, (0, 0), (view_width, 30), (0, 0, 0), -1)
            cv2.putText(labeled, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            return labeled
        
        # Create 3x2 grid of views
        row1 = np.hstack([
            add_label(frame, "Original"),
            add_label(mask_ycrcb_vis, "YCrCb Mask", (255, 200, 200)),
        ])
        
        row2 = np.hstack([
            add_label(mask_hsv_vis, "HSV Mask", (200, 255, 200)),
            add_label(mask_combined_vis, "Combined AND", (255, 255, 0)),
        ])
        
        row3 = np.hstack([
            add_label(mask_clean_vis, "After Morph", (0, 255, 255)),
            add_label(result, "Result", (0, 255, 0)),
        ])
        
        # Stack rows
        display = np.vstack([row1, row2, row3])
        
        # Add instructions at top
        instructions = np.zeros((60, display.shape[1], 3), dtype=np.uint8)
        cv2.putText(instructions, "Adjust trackbars so HAND is WHITE in masks", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(instructions, "'s'=save | 'q'=quit", 
                   (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        display = np.vstack([instructions, display])
        
        # Add current ranges at bottom
        range_text = np.zeros((50, display.shape[1], 3), dtype=np.uint8)
        cv2.putText(range_text, f"YCrCb: [{ycrcb_lower[0]},{ycrcb_lower[1]},{ycrcb_lower[2]}]->[{ycrcb_upper[0]},{ycrcb_upper[1]},{ycrcb_upper[2]}]", 
                   (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 200), 1)
        cv2.putText(range_text, f"HSV: [{hsv_lower[0]},{hsv_lower[1]},{hsv_lower[2]}]->[{hsv_upper[0]},{hsv_upper[1]},{hsv_upper[2]}]", 
                   (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)
        
        display = np.vstack([display, range_text])
        
        cv2.imshow('Skin Detection Tuner', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_color_config(ycrcb_lower, ycrcb_upper, hsv_lower, hsv_upper)
            print("\n" + "=" * 70)
            print("CURRENT OPTIMIZED VALUES:")
            print("=" * 70)
            print(f"YCRCB_LOWER = {ycrcb_lower.tolist()}")
            print(f"YCRCB_UPPER = {ycrcb_upper.tolist()}")
            print(f"HSV_LOWER = {hsv_lower.tolist()}")
            print(f"HSV_UPPER = {hsv_upper.tolist()}")
            print("=" * 70 + "\n")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
