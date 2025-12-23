"""
Debug Detection Tool
Shows all processing steps to help diagnose why hand isn't being detected
"""
import cv2
import numpy as np
from debug_utils import init_camera, save_color_config
from src.detectors import CVDetector
from src.detectors.cv.skin_detection import (
    select_hand_contour_intelligent,
    filter_forearm_by_shape,
    filter_forearm_by_orientation,
    detect_wrist_and_crop
)

def main():
    print("=" * 70)
    print("DETECTION DEBUG TOOL - ENHANCED")
    print("=" * 70)
    print("This shows each processing step to help diagnose detection issues")
    print("\nControls:")
    print("  'r' - Reset background subtractor")
    print("  'd' - Toggle denoising (to speed up)")
    print("  'b' - Toggle background subtraction")
    print("  't' - Open trackbar window for live color adjustment")
    print("  's' - Save current color ranges to config")
    print("  'q' - Quit")
    print("=" * 70)
    
    cap = init_camera()
    if not cap:
        return
    
    detector = CVDetector()
    
    # Debug settings
    use_denoising = True
    use_bg_subtraction = True
    show_trackbars = False
    
    # Create trackbar window function
    def create_trackbars():
        cv2.namedWindow('Color Range Adjustment', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Color Range Adjustment', 400, 600)
        cv2.createTrackbar('YCrCb Y Min', 'Color Range Adjustment', detector.ycrcb_lower[0], 255, lambda x: None)
        cv2.createTrackbar('YCrCb Y Max', 'Color Range Adjustment', detector.ycrcb_upper[0], 255, lambda x: None)
        cv2.createTrackbar('YCrCb Cr Min', 'Color Range Adjustment', detector.ycrcb_lower[1], 255, lambda x: None)
        cv2.createTrackbar('YCrCb Cr Max', 'Color Range Adjustment', detector.ycrcb_upper[1], 255, lambda x: None)
        cv2.createTrackbar('YCrCb Cb Min', 'Color Range Adjustment', detector.ycrcb_lower[2], 255, lambda x: None)
        cv2.createTrackbar('YCrCb Cb Max', 'Color Range Adjustment', detector.ycrcb_upper[2], 255, lambda x: None)
        
        cv2.createTrackbar('HSV H Min', 'Color Range Adjustment', detector.hsv_lower[0], 180, lambda x: None)
        cv2.createTrackbar('HSV H Max', 'Color Range Adjustment', detector.hsv_upper[0], 180, lambda x: None)
        cv2.createTrackbar('HSV S Min', 'Color Range Adjustment', detector.hsv_lower[1], 255, lambda x: None)
        cv2.createTrackbar('HSV S Max', 'Color Range Adjustment', detector.hsv_upper[1], 255, lambda x: None)
        cv2.createTrackbar('HSV V Min', 'Color Range Adjustment', detector.hsv_lower[2], 255, lambda x: None)
        cv2.createTrackbar('HSV V Max', 'Color Range Adjustment', detector.hsv_upper[2], 255, lambda x: None)
        return True
    
    def update_ranges_from_trackbars():
        try:
            detector.ycrcb_lower = np.array([
                cv2.getTrackbarPos('YCrCb Y Min', 'Color Range Adjustment'),
                cv2.getTrackbarPos('YCrCb Cr Min', 'Color Range Adjustment'),
                cv2.getTrackbarPos('YCrCb Cb Min', 'Color Range Adjustment')
            ], dtype=np.uint8)
            
            detector.ycrcb_upper = np.array([
                cv2.getTrackbarPos('YCrCb Y Max', 'Color Range Adjustment'),
                cv2.getTrackbarPos('YCrCb Cr Max', 'Color Range Adjustment'),
                cv2.getTrackbarPos('YCrCb Cb Max', 'Color Range Adjustment')
            ], dtype=np.uint8)
            
            detector.hsv_lower = np.array([
                cv2.getTrackbarPos('HSV H Min', 'Color Range Adjustment'),
                cv2.getTrackbarPos('HSV S Min', 'Color Range Adjustment'),
                cv2.getTrackbarPos('HSV V Min', 'Color Range Adjustment')
            ], dtype=np.uint8)
            
            detector.hsv_upper = np.array([
                cv2.getTrackbarPos('HSV H Max', 'Color Range Adjustment'),
                cv2.getTrackbarPos('HSV S Max', 'Color Range Adjustment'),
                cv2.getTrackbarPos('HSV V Max', 'Color Range Adjustment')
            ], dtype=np.uint8)
        except cv2.error:
            return False
        return True
    
    # Create main display window (resizable)
    cv2.namedWindow('Detection Debug', cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if main window still exists
        if cv2.getWindowProperty('Detection Debug', cv2.WND_PROP_VISIBLE) < 1:
            print("\nWindow closed by user")
            break
        
        h, w = frame.shape[:2]
        
        # Step 1: Denoising
        if use_denoising:
            denoised = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        else:
            denoised = frame.copy()
        
        # Step 2: YCrCb skin detection
        ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, detector.ycrcb_lower, detector.ycrcb_upper)
        
        # Step 3: HSV skin detection with hue wrap-around handling
        hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
        
        # Handle hue wrap-around (e.g., 170-180 and 0-10 for red/pink skin tones)
        if detector.hsv_lower[0] > detector.hsv_upper[0]:
            # Wrapped range: combine two masks
            mask_hsv1 = cv2.inRange(hsv, detector.hsv_lower, np.array([180, detector.hsv_upper[1], detector.hsv_upper[2]], dtype=np.uint8))
            mask_hsv2 = cv2.inRange(hsv, np.array([0, detector.hsv_lower[1], detector.hsv_lower[2]], dtype=np.uint8), detector.hsv_upper)
            mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
        else:
            # Normal range
            mask_hsv = cv2.inRange(hsv, detector.hsv_lower, detector.hsv_upper)
        
        # Step 4: Combine masks
        mask_combined = cv2.bitwise_and(mask_ycrcb, mask_hsv)
        
        # Step 5: Background subtraction
        fg_mask = None
        if use_bg_subtraction and detector.state.frame_count > detector.state.bg_learning_frames:
            fg_mask = detector.bg_subtractor.apply(frame, learningRate=0.001)
            kernel_motion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            fg_mask_dilated = cv2.dilate(fg_mask, kernel_motion, iterations=2)
            mask_with_bg = cv2.bitwise_and(mask_combined, fg_mask_dilated)
        else:
            if use_bg_subtraction:
                detector.bg_subtractor.apply(frame, learningRate=0.1)
            mask_with_bg = mask_combined.copy()
            fg_mask = np.zeros_like(mask_combined)
        
        detector.state.frame_count += 1
        
        # Step 6: Morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        
        mask_morph = cv2.morphologyEx(mask_with_bg, cv2.MORPH_OPEN, kernel_small, iterations=2)
        mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, kernel_large, iterations=3)
        
        # Step 7: Find contours with NEW FOREARM FILTERING
        contours, _ = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours and info
        result_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw ALL contours in red (before filtering)
        all_valid_by_area = []
        rejection_reasons = {}
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 3000 < area < w * h * 0.5:  # Basic area filter
                all_valid_by_area.append(cnt)
                cv2.drawContours(result_frame, [cnt], -1, (0, 0, 255), 1)  # Red = rejected/unfiltered
        
        # Apply intelligent forearm filtering
        hand_contour = None
        filter_stage = "None"
        
        if all_valid_by_area:
            # Stage 1: Intelligent selection (geometric scoring)
            hand_contour = select_hand_contour_intelligent(all_valid_by_area, frame.shape)
            
            if hand_contour is not None:
                filter_stage = "Intelligent Selection ✓"
                
                # Stage 2: Shape validation
                if not filter_forearm_by_shape(hand_contour, h):
                    rejection_reasons["shape"] = "Failed shape validation (aspect/solidity/compactness)"
                    hand_contour = None
                    filter_stage = "Shape Filter ✗"
                
                # Stage 3: Orientation validation
                elif not filter_forearm_by_orientation(hand_contour):
                    rejection_reasons["orientation"] = "Failed orientation (vertical forearm detected)"
                    hand_contour = None
                    filter_stage = "Orientation Filter ✗"
                
                # Stage 4: Wrist detection and crop
                elif hand_contour is not None:
                    original_contour = hand_contour.copy()
                    hand_contour = detect_wrist_and_crop(mask_morph, hand_contour)
                    filter_stage = "All Filters Passed + Wrist Crop ✓"
            else:
                filter_stage = "Intelligent Selection ✗ (low score)"
                rejection_reasons["selection"] = "All contours scored poorly (likely forearms)"
        
        # Draw final selected contour in GREEN
        valid_contours = []
        if hand_contour is not None:
            valid_contours.append(hand_contour)
            cv2.drawContours(result_frame, [hand_contour], -1, (0, 255, 0), 3)  # Green = accepted
            
            # Show area and geometric metrics
            area = cv2.contourArea(hand_contour)
            x, y, w_box, h_box = cv2.boundingRect(hand_contour)
            aspect = w_box / float(h_box) if h_box > 0 else 0
            hull = cv2.convexHull(hand_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            M = cv2.moments(hand_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(result_frame, f"Area: {int(area)}", (cx-60, cy-30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Aspect: {aspect:.2f}", (cx-60, cy-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Solid: {solidity:.2f}", (cx-60, cy+10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        

        
        # Enhanced info overlay with all metrics
        # Semi-transparent background
        overlay = result_frame.copy()
        cv2.rectangle(overlay, (5, 5), (635, 230), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, result_frame, 0.4, 0, result_frame)
        
        info_y = 30
        
        # Detection status
        detected = len(valid_contours) > 0
        status_text = "HAND DETECTED" if detected else "NO HAND DETECTED"
        status_color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.putText(result_frame, status_text, (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        info_y += 30
        
        # Forearm Filter Stage
        cv2.putText(result_frame, f"Filter Stage: {filter_stage}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 2)
        info_y += 25
        
        # Contour counts
        cv2.putText(result_frame, f"Raw Contours: {len(all_valid_by_area)} | Final: {len(valid_contours)}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        info_y += 25
        
        # Rejection reasons
        if rejection_reasons:
            for reason_key, reason_text in rejection_reasons.items():
                cv2.putText(result_frame, f"Rejected: {reason_text[:50]}", (10, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
                info_y += 20
        
        # If hand detected, show additional metrics
        if valid_contours:
            largest = max(valid_contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            cv2.putText(result_frame, f"Largest Area: {int(area)} px", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            info_y += 25
            
            # Confidence based on area
            confidence = min(100, int((area / 10000) * 100))
            cv2.putText(result_frame, f"Confidence: {confidence}%", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            info_y += 25
        
        # Processing settings
        cv2.putText(result_frame, f"Denoise: {'ON' if use_denoising else 'OFF'} (d) | BG Sub: {'ON' if use_bg_subtraction else 'OFF'} (b)", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        info_y += 20
        
        cv2.putText(result_frame, f"Frame: {detector.state.frame_count} | Trackbars: {'OPEN' if show_trackbars else 'CLOSED'} (t)", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        info_y += 20
        
        # Gesture mapping legend
        legend_y = info_y + 10
        cv2.putText(result_frame, "Controls: 't'=trackbars | 's'=save | 'd'=denoise | 'b'=bg-sub | 'r'=reset | 'q'=quit", 
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 255), 1)
        
        # Show current color ranges
        info_y += 35
        cv2.putText(result_frame, f"YCrCb: [{detector.ycrcb_lower[0]},{detector.ycrcb_lower[1]},{detector.ycrcb_lower[2]}] to [{detector.ycrcb_upper[0]},{detector.ycrcb_upper[1]},{detector.ycrcb_upper[2]}]", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 200), 1)
        info_y += 15
        cv2.putText(result_frame, f"HSV: [{detector.hsv_lower[0]},{detector.hsv_lower[1]},{detector.hsv_lower[2]}] to [{detector.hsv_upper[0]},{detector.hsv_upper[1]},{detector.hsv_upper[2]}]", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)
        
        # Create visualization grid
        mask_ycrcb_vis = cv2.cvtColor(mask_ycrcb, cv2.COLOR_GRAY2BGR)
        mask_hsv_vis = cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR)
        mask_combined_vis = cv2.cvtColor(mask_combined, cv2.COLOR_GRAY2BGR)
        fg_mask_vis = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        mask_with_bg_vis = cv2.cvtColor(mask_with_bg, cv2.COLOR_GRAY2BGR)
        mask_morph_vis = cv2.cvtColor(mask_morph, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        def add_label(img, text):
            labeled = img.copy()
            cv2.putText(labeled, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return labeled
        
        row1 = np.hstack([
            add_label(frame, "1. Original"),
            add_label(mask_ycrcb_vis, "2. YCrCb Mask"),
            add_label(mask_hsv_vis, "3. HSV Mask")
        ])
        
        row2 = np.hstack([
            add_label(mask_combined_vis, "4. Combined (Y&H)"),
            add_label(fg_mask_vis, "5. FG Motion"),
            add_label(mask_with_bg_vis, "6. With BG Filter")
        ])
        
        row3 = np.hstack([
            add_label(mask_morph_vis, "7. Morphology"),
            add_label(result_frame, "8. Final Result"),
            np.zeros_like(frame)  # Placeholder
        ])
        
        grid = np.vstack([row1, row2, row3])
        
        # Resize to fit screen
        scale = 0.5
        grid = cv2.resize(grid, None, fx=scale, fy=scale)
        
        cv2.imshow('Detection Debug', grid)
        
        # Update trackbars if open
        if show_trackbars:
            if not update_ranges_from_trackbars():
                print("\nTrackbar window closed")
                show_trackbars = False
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=False
            )
            detector.state.frame_count = 0
            print("Background subtractor reset")
        elif key == ord('d'):
            use_denoising = not use_denoising
            print(f"Denoising: {'ON' if use_denoising else 'OFF'}")
        elif key == ord('b'):
            use_bg_subtraction = not use_bg_subtraction
            print(f"Background subtraction: {'ON' if use_bg_subtraction else 'OFF'}")
        elif key == ord('t'):
            if not show_trackbars:
                show_trackbars = create_trackbars()
                print("Trackbar window opened - adjust sliders to tune detection")
            else:
                cv2.destroyWindow('Color Range Adjustment')
                show_trackbars = False
                print("Trackbar window closed")
        elif key == ord('s'):
            save_color_config(detector.ycrcb_lower, detector.ycrcb_upper, 
                            detector.hsv_lower, detector.hsv_upper)
            print("\n" + "=" * 70)
            print("SAVED TO CONFIG FILE")
            print("=" * 70)
            print(f"YCrCb Lower: {detector.ycrcb_lower.tolist()}")
            print(f"YCrCb Upper: {detector.ycrcb_upper.tolist()}")
            print(f"HSV Lower: {detector.hsv_lower.tolist()}")
            print(f"HSV Upper: {detector.hsv_upper.tolist()}")
            print("=" * 70)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
