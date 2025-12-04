"""
Debug Detection Tool
Shows all processing steps to help diagnose why hand isn't being detected
"""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detectors import CVDetector

def main():
    print("=" * 70)
    print("DETECTION DEBUG TOOL")
    print("=" * 70)
    print("This shows each processing step to help diagnose detection issues")
    print("\nControls:")
    print("  'r' - Reset background subtractor")
    print("  'd' - Toggle denoising (to speed up)")
    print("  'b' - Toggle background subtraction")
    print("  'q' - Quit")
    print("=" * 70)
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    detector = CVDetector()
    
    # Debug settings
    use_denoising = True
    use_bg_subtraction = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
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
        
        # Step 3: HSV skin detection
        hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, detector.hsv_lower, detector.hsv_upper)
        
        # Step 4: Combine masks
        mask_combined = cv2.bitwise_and(mask_ycrcb, mask_hsv)
        
        # Step 5: Background subtraction
        fg_mask = None
        if use_bg_subtraction and detector.frame_count > detector.bg_learning_frames:
            fg_mask = detector.bg_subtractor.apply(frame, learningRate=0.001)
            kernel_motion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            fg_mask_dilated = cv2.dilate(fg_mask, kernel_motion, iterations=2)
            mask_with_bg = cv2.bitwise_and(mask_combined, fg_mask_dilated)
        else:
            if use_bg_subtraction:
                detector.bg_subtractor.apply(frame, learningRate=0.1)
            mask_with_bg = mask_combined.copy()
            fg_mask = np.zeros_like(mask_combined)
        
        detector.frame_count += 1
        
        # Step 6: Morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        
        mask_morph = cv2.morphologyEx(mask_with_bg, cv2.MORPH_OPEN, kernel_small, iterations=2)
        mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, kernel_large, iterations=3)
        
        # Step 7: Find contours
        contours, _ = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours and info
        result_frame = frame.copy()
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 3000:
                valid_contours.append(cnt)
                cv2.drawContours(result_frame, [cnt], -1, (0, 255, 0), 2)
                
                # Show area
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(result_frame, f"Area: {int(area)}", (cx-50, cy), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Info overlay
        info_y = 30
        cv2.putText(result_frame, f"Valid Contours: {len(valid_contours)}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if len(valid_contours) > 0 else (0, 0, 255), 2)
        info_y += 30
        cv2.putText(result_frame, f"Denoise: {'ON' if use_denoising else 'OFF'} (d)", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(result_frame, f"BG Sub: {'ON' if use_bg_subtraction else 'OFF'} (b)", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(result_frame, f"Frame: {detector.frame_count}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
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
        
        cv2.imshow('Detection Pipeline Debug', grid)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=False
            )
            detector.frame_count = 0
            print("✅ Background subtractor reset")
        elif key == ord('d'):
            use_denoising = not use_denoising
            print(f"✅ Denoising: {'ON' if use_denoising else 'OFF'}")
        elif key == ord('b'):
            use_bg_subtraction = not use_bg_subtraction
            print(f"✅ Background subtraction: {'ON' if use_bg_subtraction else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
