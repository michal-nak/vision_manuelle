"""
Unified Debug Tool Launcher
Centralizes all debug and diagnostic tools with widen color range option
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def show_menu():
    """Display main menu"""
    print("\n" + "="*70)
    print("🔧 DEBUG TOOLS LAUNCHER")
    print("="*70)
    print("\n📊 DIAGNOSTIC TOOLS:")
    print("  1. Pipeline Visualizer - See skin mask, morphology, contours")
    print("  2. Skin Color Tuner - Adjust YCrCb/HSV ranges with trackbars")
    print("  3. Full Debug Tool - Complete pipeline with toggles and filters")
    
    print("\n🎯 BENCHMARK TOOLS:")
    print("  4. Finger Counting Benchmark - CV vs MediaPipe comparison")
    print("  5. Performance Benchmark - FPS and latency comparison")
    
    print("\n⚙️  CALIBRATION TOOLS:")
    print("  6. MediaPipe Calibration - Auto-calibrate with MediaPipe")
    print("  7. Widen Color Ranges - Expand detection bounds by X%")
    
    print("\n  0. Exit")
    print("="*70)


def widen_color_ranges():
    """Widen color detection ranges by a percentage"""
    from src.core.config import YCRCB_LOWER, YCRCB_UPPER, HSV_LOWER, HSV_UPPER
    from debug_utils import save_color_config
    
    print("\n" + "="*70)
    print("WIDEN COLOR RANGES")
    print("="*70)
    print("\nCurrent ranges:")
    print(f"  YCrCb: {YCRCB_LOWER} - {YCRCB_UPPER}")
    print(f"  HSV:   {HSV_LOWER} - {HSV_UPPER}")
    
    try:
        percent = float(input("\nEnter percentage to widen (e.g., 20 for 20%): "))
    except ValueError:
        print("❌ Invalid input")
        return
    
    factor = percent / 100.0
    
    # Calculate ranges
    ycrcb_range = [YCRCB_UPPER[i] - YCRCB_LOWER[i] for i in range(3)]
    hsv_range = [HSV_UPPER[i] - HSV_LOWER[i] for i in range(3)]
    
    # Expand by percentage
    new_ycrcb_lower = [max(0, int(YCRCB_LOWER[i] - ycrcb_range[i] * factor)) for i in range(3)]
    new_ycrcb_upper = [min(255, int(YCRCB_UPPER[i] + ycrcb_range[i] * factor)) for i in range(3)]
    new_hsv_lower = [max(0, int(HSV_LOWER[i] - hsv_range[i] * factor)) for i in range(3)]
    new_hsv_upper = [min(255 if i > 0 else 180, int(HSV_UPPER[i] + hsv_range[i] * factor)) for i in range(3)]
    
    print(f"\nNew ranges (+{percent}%):")
    print(f"  YCrCb: {new_ycrcb_lower} - {new_ycrcb_upper}")
    print(f"  HSV:   {new_hsv_lower} - {new_hsv_upper}")
    
    confirm = input("\nSave these ranges? (y/n): ")
    if confirm.lower() != 'y':
        print("❌ Cancelled")
        return
    
    save_color_config(new_ycrcb_lower, new_ycrcb_upper, new_hsv_lower, new_hsv_upper,
                     description=f"Widened by {percent}% from previous ranges")
    print("   Restart the application to use new ranges")


def run_pipeline_visualizer():
    """Simple pipeline visualizer"""
    from debug_utils import init_camera, get_skin_mask, draw_contour_info
    from src.detectors.cv.cv_detector import CVDetector
    from src.detectors.cv.skin_detection import apply_morphological_operations
    from src.core.config import MIN_HAND_AREA, MAX_HAND_AREA
    import cv2
    import numpy as np
    
    print("\n🔍 PIPELINE VISUALIZER - Press 'q' to quit\n")
    
    cap = init_camera()
    if not cap:
        return
    
    detector = CVDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        h, w = frame.shape[:2]
        
        # Step 1: Skin mask
        mask = get_skin_mask(frame, detector)
        
        # Step 2: Morphology
        mask_morph = apply_morphological_operations(mask, 3, 7, 1)
        
        # Step 3: Contours
        contours, _ = cv2.findContours(mask_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_contours = frame.copy()
        frame_contours = draw_contour_info(frame_contours, contours, MIN_HAND_AREA, MAX_HAND_AREA)
        
        # Step 4: Final result
        result = detector.process_frame(frame.copy())
        frame_result = frame.copy()
        if result['detected']:
            cv2.putText(frame_result, f"Fingers: {result['finger_count']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame_result, "NO DETECTION", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Create 2x2 grid
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_morph_color = cv2.cvtColor(mask_morph, cv2.COLOR_GRAY2BGR)
        
        top = np.hstack([mask_color, mask_morph_color])
        bottom = np.hstack([frame_contours, frame_result])
        grid = np.vstack([top, bottom])
        
        # Add labels
        cv2.putText(grid, "1. Skin Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(grid, "2. Morphology", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(grid, "3. Contours", (10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(grid, "4. Result", (w+10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Pipeline Visualizer', grid)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    while True:
        show_menu()
        choice = input("\nSelect tool (0-7): ").strip()
        
        if choice == '0':
            print("\n👋 Goodbye!")
            break
        elif choice == '1':
            run_pipeline_visualizer()
        elif choice == '2':
            import skin_tuner
            skin_tuner.main()
        elif choice == '3':
            import debug_detection
            debug_detection.main()
        elif choice == '4':
            import finger_counting_benchmark
            finger_counting_benchmark.main()
        elif choice == '5':
            import benchmark_comparison
            benchmark_comparison.main()
        elif choice == '6':
            import cv_calibrate_with_mediapipe
            cv_calibrate_with_mediapipe.main()
        elif choice == '7':
            widen_color_ranges()
        else:
            print("❌ Invalid choice")


if __name__ == '__main__':
    main()
