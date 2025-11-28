"""
Hand Detection System - Main Entry Point
Launch the gesture paint application
"""
import sys
from src.ui.gesture_paint import GesturePaintApp
import tkinter as tk

def main():
    root = tk.Tk()
    
    # Get detection mode from command line or default to mediapipe
    detection_mode = sys.argv[1] if len(sys.argv) > 1 else "mediapipe"
    if detection_mode not in ["mediapipe", "cv"]:
        print("Usage: python main.py [mediapipe|cv]")
        detection_mode = "mediapipe"
    
    app = GesturePaintApp(root, detection_mode=detection_mode)
    root.mainloop()

if __name__ == "__main__":
    main()
