"""
Unified Calibration Tool for CV Hand Detector
Main orchestrator that delegates to specialized calibration modules
"""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.utils import find_camera
from calibration.auto_calibrate import auto_calibrate
from calibration.manual_tune import manual_tune
from calibration.performance_tune import performance_tuning
from calibration.auto_optimize import auto_optimize
from calibration.config_io import save_calibration, verify_calibration


def main():
    """Main entry point - presents menu and delegates to calibration modules"""
    print("=" * 70)
    print("UNIFIED CALIBRATION TOOL")
    print("=" * 70)
    print("\n1. Auto-Calibrate (5 seconds, recommended)")
    print("2. Manual Tuning (trackbars)")
    print("3. Performance Tuning (manual FPS optimization)")
    print("4. AUTO-OPTIMIZE (automatic best settings - NEW)")
    print("5. Verify Current Calibration")
    print("6. Exit")
    print("=" * 70)
    
    choice = input("\nChoice (1-6): ").strip()
    
    if choice == '5':
        verify_calibration()
        return
    elif choice == '6':
        return
    
    # Open camera for calibration modes
    cap = find_camera()
    if not cap:
        print("Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    calibration = None
    
    # Delegate to appropriate calibration module
    if choice == '1':
        calibration = auto_calibrate(cap)
    elif choice == '2':
        calibration = manual_tune(cap)
    elif choice == '3':
        calibration = performance_tuning(cap)
    elif choice == '4':
        calibration = auto_optimize(cap)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save if calibration was successful
    if calibration:
        save_choice = input("\nSave calibration? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_calibration(calibration)
            print("\nCalibration saved! Run gesture_paint.py to test.")


if __name__ == "__main__":
    main()
